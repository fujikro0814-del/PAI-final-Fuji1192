"""手順の完了判定。立方体の真値は使わず、3 つの条件をセンサと知覚から作る。

    j = JudgeV2(setup, perception, thr, params)
    j.reset(color)
    done = j.update(sensor, wm, hand_pos)       # 20 Hz。sensor は画像つきの SensorFrame、wm は知覚の最新の結果

完了 = 次が hold_s（1.0 s）続いた:
  (1) 箱の中に手順の色がある:
      - 俯瞰の画像で、**知覚した箱**の内寸の四隅を、箱の中の立方体の上面の高さで**実行系が持つ較正**で投影した領域に、
        その色の画素が box_min_pixels 以上
      - 手首の画像で、同じ四隅を撮った時刻の関節角の順運動学と実行系が持つ hand-eye で投影した領域の内側の画素が wrist_min_pixels
        以上で、その色の画素全体のうち内側の割合が wrist_min_fraction 以上
      - 深度: 知覚した目標の立方体が箱の中で、上面の高さが「箱の底＋一辺＋height_tol」以下（別の立方体の上に乗ったものを落とす）
  (2) グリッパが開: 測った開き幅が gripper_open_m 以上
  (3) 手が待機位置で静止: 測った関節角の順運動学の手先が待機位置から retreat_tol_m 以内で、その速さが still_speed 未満
"""
import math

import cv2
import numpy as np

from recovla.runtime import cue as C


class JudgeV2:
    def __init__(self, setup, perception, thr: C.Thresholds, p: dict):
        self.setup, self.per, self.thr, self.p = setup, perception, thr, dict(p)
        self.reset(None)

    def reset(self, color) -> None:
        self.color = color
        self.prev = None
        self.held = 0.0
        self.last = None

    def _box_corners(self, wm) -> np.ndarray:
        s = self.setup
        b = wm.box
        h = 0.5 * s.box_outer - s.box_wall
        z = wm.table_z + s.box_floor + s.cube_size
        ct, st = math.cos(b["yaw"]), math.sin(b["yaw"])
        pts = []
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            dx, dy = sx * h, sy * h
            pts.append([b["xy"][0] + ct * dx - st * dy, b["xy"][1] + st * dx + ct * dy, z])
        return np.array(pts)

    def _region(self, corners_world, R, t, intr):
        pc = (corners_world - t) @ R                          # 世界 → カメラ
        if np.any(-pc[:, 2] <= 1e-6):
            return None
        u = pc[:, 0] / -pc[:, 2] * intr.fx + intr.cx
        v = -pc[:, 1] / -pc[:, 2] * intr.fy + intr.cy
        m = np.zeros((intr.height, intr.width), np.uint8)
        cv2.fillPoly(m, [np.round(np.stack([u, v], 1) * 16).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=4)
        return m.astype(bool)

    def conditions(self, sensor, wm, hand_pos, speed) -> dict:
        p, color, s = self.p, self.color, self.setup
        out = {"box_pixels": 0, "wrist_in_pixels": 0, "wrist_fraction": 0.0, "depth_ok": False}
        if wm is not None and wm.box.get("ok") and color is not None:
            corners = self._box_corners(wm)
            ov = sensor.cameras.get("overhead")
            if ov is not None:
                R, t = self.per.camera_pose("overhead", ov.t_capture, sensor.gripper.width)   # 机の面での補正を含む
                reg = self._region(corners, R, t, s.cameras["overhead"].rgb_crop)
                if reg is not None:
                    out["box_pixels"] = int((C.color_mask(ov.rgb, color, self.thr) & reg).sum())
            wr = sensor.cameras.get("wrist")
            if wr is not None:
                R, t = self.per.camera_pose("wrist", wr.t_capture, sensor.gripper.width)
                reg = self._region(corners, R, t, s.cameras["wrist"].rgb_crop)
                m = C.color_mask(wr.rgb, color, self.thr)
                n_all = int(m.sum())
                n_in = int((m & reg).sum()) if reg is not None else 0
                out.update(wrist_in_pixels=n_in, wrist_fraction=(n_in / n_all if n_all else 0.0))
            e = wm.cubes.get(color)
            if e is not None and e.in_box and e.status in ("seen", "held"):
                top = e.pos[2] - wm.table_z + 0.5 * s.cube_size
                out["depth_ok"] = bool(top <= s.box_floor + s.cube_size + float(p["height_tol_m"]))
        out["in_box"] = bool(out["box_pixels"] >= p["box_min_pixels"] and out["wrist_in_pixels"] >= p["wrist_min_pixels"]
                             and out["wrist_fraction"] >= p["wrist_min_fraction"] and out["depth_ok"])
        out["gripper_open"] = bool(sensor.gripper.width >= float(p["gripper_open_m"]))
        d = float(np.linalg.norm(np.asarray(hand_pos) - np.asarray(s.retreat_pose)))
        out.update(retreat_dist_m=d, at_retreat=d <= float(p["retreat_tol_m"]),
                   still=speed is not None and speed < float(p["still_speed"]))
        return out

    def update(self, sensor, wm, hand_pos) -> bool:
        t = float(sensor.t)
        x = np.asarray(hand_pos, float)
        speed, dt = None, 0.0
        if self.prev is not None and t > self.prev[0]:
            dt = t - self.prev[0]
            speed = float(np.linalg.norm(x - self.prev[1]) / dt)
        c = self.conditions(sensor, wm, x, speed)
        ok = c["in_box"] and c["gripper_open"] and c["at_retreat"] and c["still"]
        self.held = self.held + dt if ok else 0.0
        self.prev = (t, x)
        self.last = {**c, "ok": ok, "held_s": self.held}
        return self.held >= float(self.p["hold_s"]) - 1e-9
