"""手順の完了判定（俯瞰と手首の画像、ロボット自身の状態から）。立方体の真値は使わない。

    j = CompletionJudge(regions)
    j.reset(color)
    done = j.update(t, raw_overhead, gripper_closed, x_des, raw_wrist, wrist_mask)   # フレームごと（20 Hz）
    # wrist_mask = planner.detect.wrist_box_mask(model, data, W, H)（ロボットの関節から決まる手首のカメラの位置で投影）

完了 = 次の 3 つが completion_hold_s（1.0 s）続いた:
  (1) 箱の中に手順の色が見える:
      俯瞰の画像で箱の領域に手順の色の画素が presence.box_min_pixels 以上、かつ
      手首の画像で箱の領域の内側の画素が presence.wrist_min_pixels 以上で、その色の画素全体のうち内側の割合が
      presence.wrist_min_fraction 以上（箱の縁に乗った立方体は、俯瞰では箱の内側に重なって見えるが、方向の違う手首の
      カメラでは外にはみ出す。閾値は学習用の配置だけで決めた）
  (2) グリッパが開（ロボット自身の状態）
  (3) 手が待機位置で静止: 参照位置 x_des が expert.retreat_pose から retreat_tol_m（5 cm）以内で、
      x_des の速さが still_speed（1 cm/s）未満
"""
import numpy as np

from recovla.common import config
from recovla.planner.detect import wrist_counts

_CFG = config.load()


def conditions(regions, img, color, gripper_closed, x_des, speed, wrist_img=None, wrist_mask=None, cfg=None) -> dict:
    cfg = cfg or _CFG
    p = cfg["planner"]["judge"]
    box_px = regions.counts(img)["box"][color]
    w_in, w_all = wrist_counts(wrist_img, wrist_mask, color, regions.thr) if wrist_img is not None else (0, 0)
    w_frac = w_in / w_all if w_all else 0.0
    wrist_ok = (wrist_img is not None and wrist_mask is not None and w_in >= regions.wrist_min
                and w_frac >= regions.wrist_frac)
    d = float(np.linalg.norm(np.asarray(x_des, float) - np.asarray(cfg["expert"]["retreat_pose"], float)))
    return {"box_pixels": box_px, "wrist_in_pixels": w_in, "wrist_fraction": w_frac,
            "in_box_image": box_px >= regions.box_min and wrist_ok, "gripper_open": not bool(gripper_closed),
            "retreat_dist_m": d, "at_retreat": d <= float(p["retreat_tol_m"]),
            "still": speed is not None and speed < float(p["still_speed"])}


class CompletionJudge:
    def __init__(self, regions, cfg: dict = None):
        self.regions = regions
        self.cfg = cfg or _CFG
        self.hold_s = float(self.cfg["planner"]["completion_hold_s"])
        self.reset(None)

    def reset(self, color) -> None:
        self.color = color
        self.prev = None
        self.held = 0.0
        self.last = None

    def update(self, t, img, gripper_closed, x_des, wrist_img=None, wrist_mask=None) -> bool:
        x = np.asarray(x_des, float)
        speed = None
        dt = 0.0
        if self.prev is not None and t > self.prev[0]:
            dt = t - self.prev[0]
            speed = float(np.linalg.norm(x - self.prev[1]) / dt)
        c = conditions(self.regions, img, self.color, gripper_closed, x, speed, wrist_img, wrist_mask, self.cfg)
        ok = c["in_box_image"] and c["gripper_open"] and c["at_retreat"] and c["still"]
        self.held = self.held + dt if ok else 0.0
        self.prev = (float(t), x)
        self.last = {**c, "ok": ok, "held_s": self.held}
        return self.held >= self.hold_s - 1e-9
