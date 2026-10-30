"""運ぶ途中の落下の意図的な失敗の、実機に近い版（引き抜き）。運んでいる最中に、外から立方体を指の間から引き抜く。

    ind = PullOutInducer(seed, layout, target, world)      # 発動の時刻の決め方は P2 と同じ（同じ u）
    a = ind.filter(k, a, truth)                            # 行動は上書きしない（ハンドは方策の指令のまま閉じ続ける）
    ind.pre_physics_step(world)                            # 物理の 1 ステップごと（harness/loop.py が呼ぶ）。引き抜く
    ind.after(k, truth)                                    # 実際の失敗の判定（P2 と同じ: 箱の外の机の上に正常に着地して静止）

この形にした理由:
- P2 は指を開いて落とす（行動の 7 番目を開に上書き）。実機で落ちるときは grasp の指令が閉じ続けるので、指は
  空を掴んで開き幅が 0 へ向かう。P2 はこの点が実機と逆。
- 実機の立方体は 4 cm の PLA（約 50 g）。Franka Hand の grasp は数十 N で握るので、運ぶだけでは滑らない（約 1 N で足りる）。
  摩擦を下げて滑らせるのは現実的でない。実機で起こるのは、人が手で立方体を引き抜く・はたき落とす場合で、それを真似る。
- 引き抜き: 発動から、物理の 1 ステップごとに、指に対する立方体の位置と速さの「引く向きの成分」を、pull_speed で離れていく道に
  合わせる（立方体の位置と速度を直接書く。ほかの向きの成分・回転・重力・指との接触は物理のまま）。指の摩擦はこれに
  逆らう力として腕と指にかかる（人が引くのと同じ向き）。立方体の中心が指先の中心から引く向きに exit_m 離れたら手を離す
  （max_s で打ち切り）。向きは指の溝の向き（hand の x 軸を水平にしたもの。指の閉じる向きと直交）の ±に、下向きを
  down_deg で混ぜたもの。
- 力で引く形（xfrc_applied、ばね・ダンパでも）と、速さだけを書く形は使わない。速さだけでは、指の摩擦が 1 ステップで
  3 m/s ぶんの速さを打ち消して動かない。力で引くと、静止摩擦（40 N/本 × 2 × μ 1 = 約 80 N）が切れた瞬間に、
  硬い当て板の接触の押し込みが戻って立方体が 6〜10 m/s で飛ぶ（2 ms の陽な力では抑えられない）。
- 引き抜きの向き・下向きの角・速さは、試行のシードから別の乱数列（seed_sequence(seed, "induce", 2)）で引く。P2 の乱数列
  （"induce"）は触らないので、同じシードなら目標の色・配置・発動のしきい値（u・r0）は P2 と同じになり、対にして比べられる
  （tests/test_reflex.py で確かめる）。
- 実際の失敗は P2 と同じ（Inducer.after。着地の検査）。ただし「指が開いている」の代わりに「立方体が手の中にない
  （指先の中心から drop_dist より遠い）」を使う（指は閉じたままなので）。
- 記録: meta["induce"] の kind は "P2"（集計は P2 として読む）、variant="pull_out"、info["pull"] に
  引いた向き・抜けた時刻・理由・抜けたときの指に対する速さ・位置を書き換えた量の最大、人の手が与えた力積（引く向き）と
  1 ステップあたりの力の最大、指と立方体の接触力（法線・接線の和の最大、引く向きに逆らう成分の最大）、指先の動き（最大・終わり）、
  腕の関節トルクの引き始めからの変化の最大、立方体の質量。
ほかの実機で落ちる原因（作っていない）: ヨーのずれによる角・縁の掴み（立方体のヨーは ±30°、指のヨーは固定）、浅い掴み、
箱の壁や他の立方体への衝突。
"""
import dataclasses

import mujoco
import numpy as np

from recovla.common import seeds
from recovla.eval.induce import Inducer
from recovla.sim import frames, scene

PULL = {"speed_m_s": (0.2, 0.5),   # 引き抜く速さ（指に対する立方体の速さ）の範囲 [m/s]
        "down_deg": (0.0, 60.0),   # 下向きを混ぜる角の範囲 [°]（0 = 水平に溝の向き、90 = 真下）
        "exit_m": 0.045,           # 指先の中心から引く向きにこれだけ離れたら手を離す [m]（立方体の半分 2 cm＋当て板）
        "max_s": 0.4}              # 引く時間の上限 [s]


class PullOutInducer(Inducer):
    VARIANT = "pull_out"
    VERSION = "P2S-v1"

    def __init__(self, seed: int, layout, target: str, rig, cfg: dict = None, pull: dict = None):
        super().__init__("P2", seed, layout, target, rig, cfg)           # u は P2 と同じ列から
        self.pc = dict(PULL, **(pull or {}))
        rng = seeds.generator(seeds.seed_sequence(seed, "induce", 2))
        side = 1.0 if rng.random() < 0.5 else -1.0
        down = float(rng.uniform(*self.pc["down_deg"]))
        speed = float(rng.uniform(*self.pc["speed_m_s"]))
        self.params["pull"] = {"side": side, "down_deg": down, "speed_m_s": speed}
        self.pull = {"t_start": None, "t_end": None, "end_reason": None, "dir": None, "exit_speed_m_s": None,
                     "max_dp_m": None, "impulse_ns": None, "hand_force_max_n": 0.0, "contact_normal_max_n": 0.0,
                     "contact_tangential_max_n": 0.0, "contact_resist_max_n": 0.0, "tip_move_max_m": 0.0,
                     "tip_move_end_m": None, "arm_torque_dev_max_nm": 0.0, "mass_kg": None}
        self._on = False
        self._fingers = self._tau0 = self._tip0 = None
        self._imp = 0.0
        self._tip_prev = None
        self._p0 = None

    # ------------------------------------------------------------------ 行動（上書きしない）
    def _p2(self, k, a, tr):
        g = float(a[6])
        a = super()._p2(k, a, tr)              # 発動の判定は P2 と同じ（落とした後に開に上書きする部分は戻す）
        if self.fired:
            a[6] = g
            self.active = False
            if self.pull["t_start"] is None:
                self._start(tr)
        return a

    def _start(self, tr) -> None:
        d = self.rig.data
        R = d.xmat[self.rig.hand_id].reshape(3, 3)
        x = np.array([R[0, 0], R[1, 0], 0.0])
        x /= max(np.linalg.norm(x), 1e-9)
        p = self.params["pull"]
        th = np.radians(p["down_deg"])
        u = np.cos(th) * p["side"] * x + np.sin(th) * np.array([0.0, 0.0, -1.0])
        self._dir = u / np.linalg.norm(u)
        self.pull.update(t_start=round(float(tr.t), 3), dir=[round(float(v), 4) for v in self._dir],
                         mass_kg=round(float(self.rig.model.body_mass[self.rig.cube_ids[self.rig_target]]), 4))
        self._on, self._tip_prev, self._p0, self._t0 = True, None, None, float(d.time)

    # ------------------------------------------------------------------ 物理の 1 ステップごと
    def _measure(self, world, b, tip) -> None:
        """引いている間の記録（直前の物理の 1 ステップの後の状態から）: 指と立方体の接触力、引く向きの成分、指先の動き、
        腕の関節トルクの、引き始めからの変化。"""
        d, m = world.data, world.model
        if self._fingers is None:
            self._fingers = {m.body(n).id for n in ("left_finger", "right_finger")}
        fn = ft = fdir = 0.0
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if not ((b1 == b and b2 in self._fingers) or (b2 == b and b1 in self._fingers)):
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            fn += abs(float(f6[0]))
            ft += float(np.hypot(f6[1], f6[2]))
            fw = np.asarray(c.frame, float).reshape(3, 3).T @ f6[:3]       # 接触の座標から世界へ（geom1 が geom2 に及ぼす力）
            fdir += float(np.dot(fw if b2 == b else -fw, self._dir))       # 立方体が受ける力の、引く向きの成分
        tau = np.asarray(d.qfrc_actuator[world.arm_vadr], float)
        if self._tau0 is None:
            self._tau0, self._tip0 = tau.copy(), tip.copy()
        P = self.pull
        P["contact_normal_max_n"] = round(max(P["contact_normal_max_n"], fn), 2)
        P["contact_tangential_max_n"] = round(max(P["contact_tangential_max_n"], ft), 2)
        P["contact_resist_max_n"] = round(max(P["contact_resist_max_n"], -fdir), 2)     # 引く向きに逆らう摩擦の最大
        P["tip_move_max_m"] = round(max(P["tip_move_max_m"], float(np.linalg.norm(tip - self._tip0))), 4)
        P["arm_torque_dev_max_nm"] = round(max(P["arm_torque_dev_max_nm"], float(np.max(np.abs(tau - self._tau0)))), 3)

    def pre_physics_step(self, world) -> None:
        if not self._on:
            return
        d, m = world.data, world.model
        ti = self.rig_target
        b, va = world.cube_ids[ti], int(world.cube_vadr[ti])
        tip = frames.fingertip_center(d, world.hand_id)
        dt = float(m.opt.timestep)
        v_tip = np.zeros(3) if self._tip_prev is None else (tip - self._tip_prev) / dt
        self._tip_prev = tip.copy()
        self._measure(world, b, tip)
        p = float(np.dot(d.xpos[b] - tip, self._dir))
        el = float(d.time) - self._t0
        if p > self.pc["exit_m"] or el > self.pc["max_s"]:
            self._on = False
            v_rel = float(np.dot(d.qvel[va:va + 3] - v_tip, self._dir))
            self.pull.update(t_end=round(float(d.time), 4), end_reason="exit" if p > self.pc["exit_m"] else "max_s",
                             exit_speed_m_s=round(v_rel, 3), tip_move_end_m=round(float(np.linalg.norm(tip - self._tip0)), 4),
                             impulse_ns=round(self._imp, 4))
            return
        vt = float(self.params["pull"]["speed_m_s"])
        if self._p0 is None:
            self._p0 = p
        qa = scene.cube_qpos_adr(m, self.target)[0]
        dp = (self._p0 + vt * el) - p               # 人の手: 指に対する立方体の位置を、引く向きに vt で動かす（他の向きは物理のまま）
        d.qpos[qa:qa + 3] = d.qpos[qa:qa + 3] + dp * self._dir
        v = d.qvel[va:va + 3]
        dv = (float(np.dot(v_tip, self._dir)) + vt) - float(np.dot(v, self._dir))
        d.qvel[va:va + 3] = v + dv * self._dir
        mass = float(m.body_mass[b])
        self._imp += mass * dv                      # 人の手が立方体に与えた運動量（引く向き）＝外からの力積の見積もり
        P = self.pull
        P["max_dp_m"] = round(max(float(P.get("max_dp_m") or 0.0), abs(dp)), 5)
        P["hand_force_max_n"] = round(max(P["hand_force_max_n"], mass * dv / dt), 2)   # 1 ステップの力積 / 1 ステップの時間
    @property
    def rig_target(self) -> int:
        from recovla.common.seeds import COLORS
        return COLORS.index(self.target)

    # ------------------------------------------------------------------ 実際の失敗（P2 と同じ。指の開きの代わりに「手の中にない」）
    def after(self, k: int, tr) -> None:
        if self.fired and self.stage == "dropped":
            in_hand = float(np.linalg.norm(tr.target_pos - tr.fingertip)) <= self.pp.drop_dist
            tr = dataclasses.replace(tr, gripper_closed=bool(in_hand))
        super().after(k, tr)

    @property
    def releasing(self) -> bool:
        return False                           # 指を開いたまま保つことはしない

    def record(self) -> dict:
        if self._on:                           # 試行が引いている途中で終わった
            self._on = False
            self.pull.update(t_end=round(float(self.rig.data.time), 4), end_reason="trial_end", impulse_ns=round(self._imp, 4))
        r = super().record()
        r["info"] = dict(r["info"], pull=dict(self.pull))
        r["variant"], r["version"] = self.VARIANT, self.VERSION
        return r
