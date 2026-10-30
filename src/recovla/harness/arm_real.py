"""腕の関節の位置のサーボを、実機の Panda で測った追従に合わせる（configs/real_v1.yaml の real_v1.arm_servo）。

実機で生成プログラムの関節の指令を再生して記録した追従に合わせる。
既定のサーボ（panda.xml の利得 kp 4500〜2000・kv = kp × 0.1）は、作動器の減衰が関節の速さそのものに掛かるので、
一定の速さで動くときの遅延が kv/kp = 0.1 s になる（シミュレーションの応答は指令に約 98 ms 遅延した）。実機は約 6 ms。
利得を大きくして kv/kp を縮めると、物理の 1 ステップ 2 ms では kp が既定の 5 倍を超えると発散する。そこで、
実機の内部の制御器（関節のインピーダンス＋指令の速さの先送り）と同じ形にする: 位置の指令に、指令の速さ × kv/kp × vel_ff を
足して作動器に渡す。kp(q_c + vel_ff·(kv/kp)·dq_c − q) − kv·dq = kp(q_c − q) + kv(vel_ff·dq_c − dq) なので、vel_ff = 1 で
減衰が速さのずれに掛かる。

重力の補償: WorldRig はモデルを作った後に body_gravcomp を 1 にするが、MuJoCo はモデルを作る時点の
ngravcomp（重力の補償をする体の数）が 0 だと補償を計算しない（mujoco 3.2.3 で qfrc_gravcomp = 0）。
そのため既定のシミュレーションの腕は、重力の分だけ位置のサーボがずれる（開始姿勢で関節 2 が 2.9 mrad・
関節 4 が −5.5 mrad、低い姿勢で関節 2 が 5.1 mrad）。gravity_comp = true で、腕の体の重力を打ち消す関節の力を毎手
qfrc_applied に入れる（実機は内部の制御器が重力を補償する）。

変えるのは次の 5 つだけ。どれもモデル（MjModel）の値と指令の口の手前の処理で、harness/world.py は変えない
（既定の設定では動きは変わらない）。監査（CommandAudit）には元の指令を渡す。

- kp: 関節ごとの位置の利得 [Nm/rad]（actuator_gainprm[0]・biasprm[1] = −kp）。空なら変えない
- kv: 関節ごとの速さの利得 [Nm·s/rad]（biasprm[2] = −kv）。空なら変えない
- vel_ff: 指令の速さの先送りの割合（0〜1）。指令の速さは物理の 1 ステップに来る 2 つの指令（1 ms 違い）の差
- cmd_delay_steps: 指令を物理の何手ぶん遅らせて効かせるか（1 ステップ = 2 ms。実機の libfranka の濾波と通信の遅延に当たる）
- gravity_comp: 腕の体の重力を打ち消す関節の力を、毎手 qfrc_applied に入れる（作動器の力の上限の外）

    servo = ArmServo.from_config(cfg["real_v1"]["arm_servo"])
    saved = install(rig, servo)  # WorldRig（とその子）の腕を差し替える。rig.reset の前に呼ぶ（gravity_comp では reset が
                                 # 腕を最初の指令へ置き直し、遅延の列を空にする）
    restore(rig, saved)
"""
import collections
import dataclasses

import numpy as np

CMD_DT = 1e-3                                       # 1 つの物理の手に来る 2 つの指令の間隔 [s]


@dataclasses.dataclass(frozen=True)
class ArmServo:
    kp: tuple = ()
    kv: tuple = ()
    vel_ff: float = 0.0
    cmd_delay_steps: int = 0

    gravity_comp: bool = False

    @classmethod
    def from_config(cls, cfg: dict) -> "ArmServo":
        cfg = cfg or {}
        kp = tuple(float(v) for v in cfg.get("kp", ()))
        kv = tuple(float(v) for v in cfg.get("kv", ()))
        if (kp and len(kp) != 7) or (kv and len(kv) != 7):
            raise ValueError("arm_servo の kp・kv は 7 つ")
        return cls(kp, kv, float(cfg.get("vel_ff", 0.0)), int(cfg.get("cmd_delay_steps", 0)),
                   bool(cfg.get("gravity_comp", False)))

    def is_default(self) -> bool:
        return (not self.kp and not self.kv and self.vel_ff == 0.0 and self.cmd_delay_steps == 0
                and not self.gravity_comp)


def gravity_torque(rig) -> np.ndarray:
    """腕の体（link1〜指）に掛かる重力を打ち消す関節の力（腕の 7 関節ぶん）。−Σ J_b^T m_b g。"""
    import mujoco
    from recovla.harness.world import ROBOT_BODIES
    m, d = rig.model, rig.data
    jac = np.zeros((3, m.nv))
    tau = np.zeros(m.nv)
    for b in ROBOT_BODIES:
        bid = m.body(b).id
        mujoco.mj_jacBodyCom(m, d, jac, None, bid)
        tau -= m.body_mass[bid] * (jac.T @ m.opt.gravity)
    return tau[rig.arm_vadr]


def arm_actuators(rig) -> list:
    return [int(a[0]) for a in rig.controller.arm]


def place_arm(rig, q) -> None:
    """腕を関節角 q（シミュレーションの座標）に止めて置き、位置のサーボの指令も q にする。"""
    import mujoco
    q = np.array(q, float)
    rig.data.qpos[rig.arm_qadr] = q
    rig.data.qvel[rig.arm_vadr] = 0.0
    rig.data.ctrl[arm_actuators(rig)] = q
    rig._last_cmd = q.copy()
    mujoco.mj_forward(rig.model, rig.data)


def install(rig, servo: ArmServo) -> dict:
    """モデルの腕の作動器の利得を差し替え、指令の先送り・遅延を入れる。差し替える前の値を返す（restore に渡す）。"""
    m = rig.model
    acts = arm_actuators(rig)
    saved = {"gainprm": m.actuator_gainprm[acts].copy(), "biasprm": m.actuator_biasprm[acts].copy()}
    for k, a in enumerate(acts):
        if servo.kp:
            m.actuator_gainprm[a, 0] = servo.kp[k]
            m.actuator_biasprm[a, 1] = -servo.kp[k]
        if servo.kv:
            m.actuator_biasprm[a, 2] = -servo.kv[k]
    fifo = collections.deque()
    if servo.gravity_comp:
        cls_step, cls_reset = type(rig).physics_step, type(rig).reset

        def physics_step(on_step=None):
            rig.data.qfrc_applied[rig.arm_vadr] = gravity_torque(rig)
            cls_step(rig, on_step)

        def reset(layout):
            # 開始姿勢は重力の補償なしで落ち着かせた状態（重力の分だけずれている）。実機は指令の開始姿勢に
            # ぴったり止まっているので、腕を最初の指令（制限層の位置。なければ位置のサーボの指令）へ置き直す
            cls_reset(rig, layout)
            fifo.clear()
            lim = getattr(getattr(rig, "motion", None), "limiter", None)
            place_arm(rig, lim.q if lim is not None else rig._last_cmd)
        rig.physics_step, rig.reset = physics_step, reset
        saved["gravity_comp"] = True
    if servo.vel_ff == 0.0 and servo.cmd_delay_steps == 0:
        return saved
    lead = servo.vel_ff * (-m.actuator_biasprm[acts, 2]) / m.actuator_gainprm[acts, 0]     # vel_ff × kv/kp [s]

    def apply(cmds):
        cmds = [np.array(c, float) for c in cmds]
        fifo.append(cmds)
        if len(fifo) <= servo.cmd_delay_steps:                    # 始めは最初の指令を保つ（止まっている開始姿勢）
            use = [fifo[0][0], fifo[0][0]]
        else:
            use = fifo.popleft()
        # 指令の監査（Panda の上限）には制限層が出した元の指令を書く。遅延は実機の中の振る舞いのモデルで、
        # 実機の口に入るのは遅らせる前の列（遅らせた列だと、動き出しで見かけの加速度・躍度の超過が出る）
        for q in cmds:
            rig.audit.joint_command(q)
        dq = (use[-1] - use[-2]) / CMD_DT if len(use) >= 2 else np.zeros(len(acts))
        rig._last_cmd = use[-1] + lead * dq
    rig.apply_joint_commands = apply
    saved["fifo"] = fifo
    return saved


def restore(rig, saved: dict) -> None:
    m = rig.model
    acts = arm_actuators(rig)
    m.actuator_gainprm[acts] = saved["gainprm"]
    m.actuator_biasprm[acts] = saved["biasprm"]
    rig.__dict__.pop("apply_joint_commands", None)
    if saved.get("gravity_comp"):
        rig.__dict__.pop("physics_step", None)
        rig.__dict__.pop("reset", None)
        rig.data.qfrc_applied[rig.arm_vadr] = 0.0
