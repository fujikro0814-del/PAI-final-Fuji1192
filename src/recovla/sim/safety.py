"""安全フィルタ: 手・指が障害物に近づきすぎないよう、手先の参照位置の増分を制限する。

IK の手前で、参照位置の増分 Δx（並進 3 つ）だけを制限する。グリッパの指令と手先の向きには触れない。

    f = SafetyFilter(rig)                  # rig.meter の距離のモデル（native CCD）を使う
    f.start_trial(target)                  # 手順ごとに目標の色（障害物の組は読み取りごとに作り直す）
    f.gate = True                          # 方策が指令を出している間だけ真（失敗の意図的な失敗の上書き中・生成プログラムでは偽）
    f.begin_read(data, x_cmd)              # 入力の読み取りごと（50 Hz）: 距離 d と向き n を測る
    dx = f.filter(x_cmd, dx)               # 物理ステップごと（500 Hz）: VelocityCommandIntegrator.command_filter

- 障害物: 机上の目標外の立方体と箱の壁 4 枚（ContactMeter.obstacle_mask、接触の数え方と同じ組）
- 距離 d: 手・指の全ての衝突形状と障害物の最短距離（ContactMeter と同じ距離のモデル）。n は障害物から手へ向かう単位の向き
- 手先の遅延: 参照位置での距離を d + n·(x_cmd − x_hand) と見積もり（x_hand は制御器が desired_pos に追わせる
  hand の原点）、実際の距離 d との小さい方を d_ref とする（手が参照より障害物側にある場合もあるため、保守側に取る）。
  読み取りの間は d_ref に n·(x_cmd の動き) を足して一次で見積もる
- 不等式: d_ref < d_detect の障害物について n·Δx ≥ −γ (d_ref − d_min)。これを全部満たす最も近い Δx に置き換える
  （変数 3 つ・不等式は高々 7 本の二次計画。有効な組を数え上げ、KKT を満たす解を取る。外部の解法器は使わない）
"""
import mujoco
import numpy as np

from recovla.common import config
from recovla.runtime.qp import project  # noqa: F401（二次計画は実行系と共通）
from recovla.sim import contact

_CFG = config.load()
_EPS = 1e-12


class SafetyFilter:
    def __init__(self, rig, cfg: dict = None):
        c = (cfg or _CFG)["safety_filter"]
        self.enabled = bool(c["enabled"])
        self.d_min = float(c["d_min_m"]) if c["d_min_m"] is not None else None
        self.d_detect = float(c["d_detect_m"]) if c["d_detect_m"] is not None else None
        self.gamma = float(c["gamma"])
        self.rig = rig
        self.meter = rig.meter
        self.gate = False
        self.target = None
        self._fromto = np.zeros(6)
        self._rows = []                                 # 読み取りの時点の [(n, d_ref, 列)]
        self._x0 = None
        self._last_n = {}
        self.reset_trial()

    # ------------------------------------------------------------------ 状態
    def reset_trial(self) -> None:
        self.gate = False
        self._rows = []
        self._x0 = None
        self._last_n = {}
        self.window_active = False
        self.n_active_steps = 0
        self.total_change_m = 0.0
        self.infeasible_steps = 0

    def start_trial(self, target: str) -> None:
        self.reset_trial()
        self.target = target

    def on(self) -> bool:
        return self.enabled and self.gate and self.target is not None

    def take_window(self) -> bool:
        a, self.window_active = self.window_active, False
        return a

    def summary(self) -> dict:
        return {"enabled": self.enabled, "d_min_m": self.d_min, "d_detect_m": self.d_detect, "gamma": self.gamma,
                "active_physics_steps": self.n_active_steps, "total_change_m": self.total_change_m,
                "infeasible_steps": self.infeasible_steps}

    # ------------------------------------------------------------ 入力の読み取りごと
    def begin_read(self, data, x_cmd) -> None:
        """距離のモデルに今の qpos を写し、障害物ごとの最短距離と向きを測る。"""
        self._rows = []
        self._x0 = np.array(x_cmd, float)
        if not self.on():
            return
        m = self.meter
        dd = m.ddata
        np.copyto(dd.qpos, data.qpos)
        mujoco.mj_kinematics(m.dmodel, dd)
        x_hand = dd.xpos[self.rig.hand_id].copy()
        lag = self._x0 - x_hand
        mask = contact.ContactMeter.obstacle_mask(self.target, self.rig.cubes_in_box(data))
        dmax = self.d_detect + 0.05                     # 遅延（最大 17 mm）の分を足して測る
        for k in np.flatnonzero(mask):
            og = int(m.column_geoms[k])
            rows_k = []
            # 手・指の形状ごとに 1 本（立方体を開いた指の間に挟むと、左右の指で向きが逆の制約になる）
            for rg in m.distance_geoms:
                best = mujoco.mj_geomDistance(m.dmodel, dd, int(rg), og, dmax, self._fromto)
                if best >= dmax:
                    continue
                ft = self._fromto
                v = ft[:3] - ft[3:]                     # 障害物の最近点 → ロボットの最近点
                nv = np.linalg.norm(v)
                key = (int(k), int(rg))
                if abs(best) > 1e-4 and nv > 1e-9:
                    n = v / nv if best > 0 else -v / nv     # 食い込んでいれば、組の向きは障害物の内側を向くので反転
                    self._last_n[key] = n
                elif key in self._last_n:               # ちょうど触れている: 直前の向きを使う
                    n = self._last_n[key]
                else:
                    continue
                # 参照位置での見積もりと、実際の手の距離の小さい方。IK は参照と手の間に動かない差（主に +z、5〜17 mm、
                # 横にも数 mm）を残すので、手が参照より障害物側にあるときは実際の距離の方が小さい（1〜2 mm）
                d_ref = min(best, best + float(n @ lag))
                if d_ref < self.d_detect:
                    rows_k.append((n, d_ref, int(k)))
            rows_k.sort(key=lambda r: r[1])
            for r in rows_k:                            # 向きがほぼ同じ（cos > 0.95）組は近い方だけ残す
                if all(float(r[0] @ q[0]) <= 0.95 for q in self._rows if q[2] == r[2]):
                    self._rows.append(r)

    # ------------------------------------------------------- 物理の 1 ステップごと
    def filter(self, x_cmd, dx):
        if not self._rows or not self.on():
            return dx
        moved = np.asarray(x_cmd, float) - self._x0
        A = np.array([n for n, _, _ in self._rows])
        d_now = np.array([d + float(n @ moved) for n, d, _ in self._rows])
        b = -self.gamma * (d_now - self.d_min)
        x, ok = project(dx, A, b)
        ch = float(np.linalg.norm(x - dx))
        if ch > 0.0:
            self.window_active = True
            self.n_active_steps += 1
            self.total_change_m += ch
        if not ok:
            self.infeasible_steps += 1
        return x
