"""評価で失敗をわざと起こす（意図的な失敗）。P1 = 掴み損ね、P2 = 運ぶ途中の落下、P3 = 置き損ね。
数値は configs の eval.P1〜P3、記録の形は docs/interfaces/trial_record.md の induce。

    ind = Inducer("P2", seed, layout, target, rig)      # パラメータは試行の開始時に意図的な失敗の乱数列（stream(seed, "induce")）から全部引く
    a = ind.filter(k, a, truth)                          # 10 fps の行動を 1 つ上書きする（scene_trial.run_trial が呼ぶ）
    ind.after(k, truth)                                  # その行動を実行した後の状態で、実際の失敗を判定する
    ind.record                                         # meta["induce"]

| 意図的な失敗 | 発動 | 上書き | 実際の失敗の条件 |
|---|---|---|---|
| P1 | 方策のグリッパの行動が初めて正（閉）になった | 閉じるのを delay_s 遅らせ、その間に手先の指令を世界の x（指の閉じる向きと直交）へ ±2.5〜3.5 cm ずらしてから閉じる。ずれは戻さない | 閉じてから confirm_window_s 以内に目標が confirm_lift_m 以上持ち上がらなかった |
| P2 | 目標を min_lift_m 以上持ち上げた（掴んでいる） | r0＝その時点の箱の中心までの水平距離。距離が min + u·(r0 − min) を切ったら指を開き、立方体が机に着いて静止するまで開いたまま保つ | 箱の外の机上に正常に着地した（expert.inject.landing_check） |
| P3 | 掴んで箱の上にいるとき、方策が指を開く行動を出した | 開かずに、搬送の高さへ上がり、ずらし先（配置の範囲の一様、他の立方体・箱から隙間 3 cm 以上）の真上へ運び、グリッパを開く高さへ下ろしてから開く | P2 と同じ |

- 上書きは 10 fps の行動（手先参照位置の差分 xyz とグリッパ）の形で行う。生成プログラム・方策・意図的な失敗は同じ指令の入口を通る
- 実際には失敗しなかった試行と、発動しなかった試行も記録に残す（reason）。乱数は試行の始めに全部引き、途中では引かない
"""
import numpy as np

from recovla.common import config, seeds
from recovla.common.seeds import COLORS
from recovla.expert import inject as INJ
from recovla.expert import script as S
from recovla.sim import frames

_CFG = config.load()
KINDS = ("P1", "P2", "P3")
ACTION_DT = float(_CFG["sim"]["stride"]) * float(_CFG["sim"]["record_every"]) * float(_CFG["sim"]["timestep"])
MOVE_SPEED = 0.18              # P3 の運びの速さ [m/s]（生成プログラムの speed_ref.xy と同じ）
MOVE_TOL = 0.004               # P3 のずらし先・高さに着いたとみなす距離 [m]


class Inducer:
    def __init__(self, kind: str, seed: int, layout, target: str, rig, cfg: dict = None):
        if kind not in KINDS:
            raise ValueError(f"kind {kind!r} not in {KINDS}")
        self.cfg = cfg or _CFG
        self.kind, self.seed, self.rig, self.target = kind, int(seed), rig, target
        ev = self.cfg["eval"]
        self.ev = ev
        self.pp = S.PhaseParams.from_config(self.cfg)
        rng = seeds.stream(seed, "induce")              # 種類によらず同じ順で全部引く
        mag = float(rng.uniform(*ev["P1"]["lateral_offset_m"]))
        sign = 1.0 if rng.random() < 0.5 else -1.0
        u = float(rng.random())
        drop, tries = INJ._sample_drop(rng, layout, target, self.cfg)
        self.params = {"P1": {"offset_m": mag * sign, "direction_rad": 0.0 if sign > 0 else float(np.pi)},
                       "P2": {"u": u},
                       "P3": {"offset_xy": [float(drop[0]), float(drop[1])], "tries": int(tries)}}[kind]
        self.fired = False
        self.t_fire = None
        self.established = False
        self.t_established = None
        self.t_failure = None
        self.reason = None
        self.info = {}
        self.stage = None
        self.active = False            # この行動を意図的な失敗が上書きしたか（記録の induce_active）
        self._n = 0

    # ------------------------------------------------------------------ 補助
    def _row(self, xyz, grip: float) -> np.ndarray:
        a = np.zeros(7)
        a[:3] = xyz
        a[6] = grip
        return a

    def _toward(self, x_cmd, goal) -> np.ndarray:
        d = np.asarray(goal, float) - np.asarray(x_cmd, float)
        n = np.linalg.norm(d)
        step = MOVE_SPEED * ACTION_DT
        return d if n <= step else d * (step / n)

    # ------------------------------------------------------------------ 行動の上書き
    def filter(self, k: int, a: np.ndarray, truth: S.Truth) -> np.ndarray:
        self.active = False
        a = np.asarray(a, float).copy()
        return {"P1": self._p1, "P2": self._p2, "P3": self._p3}[self.kind](k, a, truth)

    def _p1(self, k, a, tr):
        p = self.ev["P1"]
        if not self.fired:
            if a[6] > 0.0 and not tr.gripper_closed:
                self.fired, self.t_fire = True, round(tr.t, 3)
                self._n = max(1, int(round(float(p["delay_s"]) / ACTION_DT)))
                self.stage = "shift"
            else:
                return a
        if self.stage == "shift" and self._n > 0:
            self._n -= 1
            self.active = True
            return self._row([self.params["offset_m"] / max(1, int(round(float(p["delay_s"]) / ACTION_DT))), 0, 0], -1.0)
        if self.stage == "shift":                 # ずらし終えた: 方策が合わせ直す前に、ここで閉じる
            self.stage = "closed"
            self.active = True
            return self._row([0, 0, 0], 1.0)
        return a

    def _p2(self, k, a, tr):
        p = self.ev["P2"]
        rise = tr.target_pos[2] - frames.CUBE_REST_Z
        if not self.fired:
            if self.stage is None and S.holding(tr, self.pp) and rise >= float(p["min_lift_m"]):
                r0 = float(np.hypot(*(tr.target_pos[:2] - tr.box[:2])))
                dmin = float(p["min_dist_from_box_m"])
                self.info.update(r0_m=r0)
                self.params["r0_m"] = r0
                if r0 < dmin:
                    self.stage, self.reason = "not_applicable", "r0_below_min"
                    return a
                self._thr = dmin + float(self.params["u"]) * (r0 - dmin)
                self.info["threshold_m"] = self._thr
                self.stage = "armed"
            if self.stage == "armed" and S.holding(tr, self.pp):
                if float(np.hypot(*(tr.target_pos[:2] - tr.box[:2]))) < self._thr:
                    self.fired, self.t_fire = True, round(tr.t, 3)
                    self.stage, self._rest = "dropped", 0.0
            if not self.fired:
                return a
        if self.stage == "dropped":          # 着いて静止するまで開いたまま（手先は方策のまま）
            self.active = True
            a[6] = -1.0
        return a

    def _p3(self, k, a, tr):
        e = self.cfg["expert"]
        if not self.fired:
            over = frames.over_box_interior(tr.x_cmd, tr.box)
            if a[6] <= 0.0 and tr.gripper_closed and S.holding(tr, self.pp) and S.lifted(tr, self.pp) and over:
                self.fired, self.t_fire = True, round(tr.t, 3)
                self.stage = "rise"
            else:
                return a
        if self.stage == "done":                  # 実際の失敗の判定が済んだ: 方策に戻す
            return a
        self.active = True
        if self.stage == "dropped":
            a[:3] = 0.0
            a[6] = -1.0
            return a
        if not S.holding(tr, self.pp):
            self.stage, self.reason = "dropped", "slipped_while_moving"
            self._rest = 0.0
            return self._row([0, 0, 0], -1.0)
        x = np.asarray(tr.x_cmd, float)
        carry_z, release_z = float(e["carry_z"]), float(e["release_z"])
        dest = np.asarray(self.params["offset_xy"], float)
        aim = x[:2] + (dest - tr.target_pos[:2])          # 立方体の中心をずらし先に合わせる
        if self.stage == "rise":
            if abs(x[2] - carry_z) <= MOVE_TOL:
                self.stage = "move"
            else:
                return self._row(self._toward(x, [x[0], x[1], carry_z]), 1.0)
        if self.stage == "move":
            if np.hypot(*(dest - tr.target_pos[:2])) > MOVE_TOL:
                return self._row(self._toward(x, [aim[0], aim[1], carry_z]), 1.0)
            self.stage = "lower"
        if abs(x[2] - release_z) > MOVE_TOL:
            return self._row(self._toward(x, [aim[0], aim[1], release_z]), 1.0)
        self.stage, self._rest = "dropped", 0.0
        self.info["drop_point"] = [float(v) for v in tr.target_pos]
        return self._row([0, 0, 0], -1.0)

    # ------------------------------------------------------------ 実際の失敗の判定
    def after(self, k: int, tr: S.Truth) -> None:
        """行動を実行した後の状態で実際の失敗を判定する（10 fps）。"""
        if not self.fired or self.established or self.reason is not None and self.stage != "dropped":
            return
        if self.kind == "P1":
            p = self.ev["P1"]
            if tr.gripper_closed and "t_closed" not in self.info:
                self.info["t_closed"] = round(tr.t, 3)
            if "t_closed" in self.info and self.reason is None:
                rise = tr.target_pos[2] - frames.CUBE_REST_Z
                if rise >= float(p["confirm_lift_m"]):
                    self.reason = "lifted_anyway"
                elif tr.t - self.info["t_closed"] >= float(p["confirm_window_s"]) - 1e-9:
                    self.established, self.t_established = True, round(tr.t, 3)
                    self.t_failure = self.info["t_closed"]
            return
        if self.stage != "dropped":
            return
        lc = self.cfg["inject"]["landing"]
        self._rest = self._rest + ACTION_DT if tr.target_speed < float(lc["rest_speed"]) else 0.0
        on_table = tr.target_pos[2] - frames.CUBE_REST_Z < self.pp.grasped_lift_min
        if on_table and self._rest >= float(lc["rest_hold_s"]) - 1e-9 and not tr.gripper_closed:
            land = INJ.landing_check(self.rig, tr, self.cfg)
            self.info["landing"] = land
            if land["landing_ok"]:
                self.established, self.t_established = True, round(tr.t, 3)
                self.t_failure = self.t_established
                self.reason = None
            else:
                self.reason = "landing_invalid:" + ",".join(land["landing_fail"])
            self.stage = "done"
        elif tr.t - self.t_fire > float(lc["timeout_s"]):
            # 原因を分けて記録する: 他の立方体の上に乗って止まった／それ以外（動き続けた・宙に留まった）
            t_ = tr.target
            on_cube = [COLORS[i] for i in range(len(COLORS)) if i != t_
                       and np.hypot(*(tr.target_pos[:2] - tr.cube_pos[i, :2])) < frames.CUBE_SIZE
                       and tr.target_pos[2] > tr.cube_pos[i, 2] + frames.CUBE_HALF]
            self.info["timeout_state"] = {"target_pos": [float(v) for v in tr.target_pos],
                                          "target_speed": float(tr.target_speed), "on_cube": on_cube}
            self.reason = "landed_on_cube" if on_cube else "no_rest"
            self.stage = "done"

    @property
    def releasing(self) -> bool:
        """P2・P3 で、落ちてから静止するまで開いたまま保っている間（方策に戻す前）。"""
        return self.kind in ("P2", "P3") and self.stage == "dropped"

    def record(self) -> dict:
        if not self.fired and self.reason is None:
            self.reason = "not_fired"
        return {"kind": self.kind, "params": self.params, "fired": self.fired, "t_fire": self.t_fire,
                "established": self.established, "t_established": self.t_established, "t_failure": self.t_failure,
                "reason": self.reason, "info": self.info}
