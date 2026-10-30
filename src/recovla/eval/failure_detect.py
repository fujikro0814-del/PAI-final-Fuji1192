"""方策が自分で起こした失敗の検出。試行の記録を後から分類するときと、走行中に見て生成プログラムへ切り替えるとき
（自分の失敗から学び直すデータの収集）で同じ定義を使う。

    det = FailureDetector(target_index, cfg)
    ev = det.update(t, fingertip, fingers, gripper_closed, cube_pos, cube_in_box)   # フレームごとに。確定したら Event、なければ None
    det.first                                                                        # 最初に確定した失敗（なければ None）
    events = detect_trial(arrays, t_end=None)                                        # 試行の記録（npz の配列）に当てる

| 種類 | 確定 |
|---|---|
| grasp_miss（掴み損ね） | グリッパが閉じてから grasp_window_s 以内に目標が lift_m 持ち上がらない → 閉じてから grasp_window_s の時点 |
| drop_table（落下） | 目標が lift_m 以上持ち上がった後、箱の外・把持されていない状態で、速さ rest_speed 未満が rest_hold_s 続いた。机の上（静止の高さ + table_z_tol_m 未満）で静止 |
| drop_other（その他の落下） | 同上で、机の上以外（箱の縁、他の立方体の上など）で other_rest_hold_s 静止 |
| wrong_color（誤った色へ接近） | グリッパが開で、指先の中心と、目標外で箱の外の机上の立方体の中心の距離（3 次元）が wrong_color_dist_m 未満 |
| stall（停滞） | 指先の速さ stall_hand_speed 未満かつ指の速さ stall_finger_speed 未満が stall_s 続いた |

- 速さはフレームの差分から（記録は 20 fps、走行中は呼ぶ間隔）。呼ぶ間隔が違うと差分の細かさが違う点に注意（走行中の切り替えは 10 fps）
- 成功の判定はここではしない（呼ぶ側が成功の時刻より前だけを見る）
"""
import dataclasses

import numpy as np

from recovla.common import config
from recovla.sim import frames

_CFG = config.load()
KINDS = ("grasp_miss", "drop_table", "drop_other", "wrong_color", "stall")


@dataclasses.dataclass
class Event:
    kind: str
    t: float
    info: dict


class FailureDetector:
    def __init__(self, target: int, cfg: dict = None):
        c = (cfg or _CFG)["eval"]["failure_detect"]
        self.c = {k: float(v) for k, v in c.items()}
        self.target = int(target)
        self.events = []
        self.first = None
        self._prev = None                  # (t, fingertip, fingers)
        self._closed_prev = None
        self._close_t = None
        self._was_lifted = False
        self._base_rise = 0.0              # 落下の数え直しの基準の高さ（静止の高さからの上がり）
        self._rest_s = 0.0
        self._stall_s = 0.0
        self._fired = set()                # 種類ごとに 1 回（同じ状態が続く間は数え直さない）

    def _emit(self, kind, t, **info):
        ev = Event(kind, float(t), info)
        self.events.append(ev)
        if self.first is None:
            self.first = ev
        return ev

    def update(self, t, fingertip, fingers, gripper_closed, cube_pos, cube_in_box):
        c = self.c
        t = float(t)
        tip = np.asarray(fingertip, float)
        fing = np.asarray(fingers, float)
        cubes = np.asarray(cube_pos, float)
        inbox = np.asarray(cube_in_box, bool)
        closed = bool(gripper_closed)
        tp = cubes[self.target]
        rise = float(tp[2] - frames.CUBE_REST_Z)
        out = None
        dt = None
        hand_v = fing_v = tgt_v = None
        if self._prev is not None:
            dt = t - self._prev[0]
            if dt > 0:
                hand_v = float(np.linalg.norm(tip - self._prev[1]) / dt)
                fing_v = float(np.sum(np.abs(fing - self._prev[2])) / dt)
                tgt_v = float(np.linalg.norm(tp - self._prev[3]) / dt)
        # (i) 掴み損ね
        if closed and self._closed_prev is False:
            self._close_t = t
        if self._close_t is not None:                        # 途中で開いても、閉じてから 2 s で持ち上がっていなければ掴み損ね
            if rise >= c["lift_m"]:
                self._close_t = None
            elif t - self._close_t >= c["grasp_window_s"] - 1e-9:
                out = out or self._emit("grasp_miss", t, t_close=self._close_t)
                self._close_t = None
        # (ii) 落下（持ち上げた後、箱の外で、把持されずに静止）
        if rise >= self._base_rise + c["lift_m"]:              # 最後に静止した高さから lift_m（最初は机の上から）
            self._was_lifted = True
        held = closed and float(np.linalg.norm(tip - tp)) < c["held_dist_m"]
        if self._was_lifted and not inbox[self.target] and not held and tgt_v is not None and tgt_v < c["rest_speed"]:
            self._rest_s += dt
        else:
            self._rest_s = 0.0
        on_table = rise < c["table_z_tol_m"]
        if self._rest_s >= (c["rest_hold_s"] if on_table else c["other_rest_hold_s"]) - 1e-9:
            kind = "drop_table" if on_table else "drop_other"
            out = out or self._emit(kind, t, target_pos=[float(v) for v in tp])
            self._was_lifted, self._rest_s = False, 0.0          # 次に持ち上げたら数え直す
            self._base_rise = max(0.0, rise)
        # (iii) 誤った色へ接近
        if not closed:
            for i in range(len(cubes)):
                if i == self.target or inbox[i] or cubes[i][2] - frames.CUBE_REST_Z >= c["table_z_tol_m"]:
                    continue
                d = float(np.linalg.norm(tip - cubes[i]))
                if d < c["wrong_color_dist_m"] and ("wrong_color", i) not in self._fired:
                    self._fired.add(("wrong_color", i))
                    out = out or self._emit("wrong_color", t, cube=i, dist_m=d)
        # (iv) 停滞
        if hand_v is not None and hand_v < c["stall_hand_speed"] and fing_v < c["stall_finger_speed"]:
            self._stall_s += dt
        else:
            self._stall_s = 0.0
            self._fired.discard("stall")
        if self._stall_s >= c["stall_s"] - 1e-9 and "stall" not in self._fired:
            self._fired.add("stall")
            out = out or self._emit("stall", t, hand=[float(v) for v in tip])
        self._prev = (t, tip, fing, tp.copy())
        self._closed_prev = closed
        return out


def detect_trial(arrays, t_end: float = None, cfg: dict = None) -> list:
    """試行の記録（scene_trial の配列）に当てる。t_end より後（成功の時刻など）は見ない。"""
    target = int(np.asarray(arrays["target"])[0])
    det = FailureDetector(target, cfg)
    t = np.asarray(arrays["sim_time"], float)
    for f in range(len(t)):
        if t_end is not None and t[f] > t_end + 1e-9:
            break
        det.update(t[f], arrays["fingertip"][f], arrays["fingers"][f], arrays["gripper_closed"][f],
                   arrays["cube_pos"][f], arrays["cube_in_box"][f])
    return det.events
