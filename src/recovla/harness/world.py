"""シミュレーションの世界（真の状態を持つ側）。SimRig に、実機と同じ口の作動のモデルを入れる。

    w = WorldRig(render=True)
    w.reset(layout)                      # 開始姿勢・立方体の配置。ハンドは開く
    w.apply_joint_commands([q1, q2])     # 1 kHz 相当の関節の位置の指令（500 Hz の 1 ステップぶんで 2 つ）。最後の値を位置のサーボへ
    w.physics_step                     # ハンドの 1 ステップと mj_step（世界は止まらない。物理の周期ごとに必ず 1 回）

作動のモデル:
- 腕: 関節の位置のサーボ（panda.xml の利得）に、重力の補償を作動器を通して足す（実機の内部の制御器と同じく、指令の位置を
  保つのに重力の分のずれが要らない）。関節の作動器の力は 87・12 Nm（Panda の公称）で切る（重力の補償を含む）
- ハンド: recovla.harness.hand.FrankaHand（速度のサーボと力の上限。指の当て板の接触の剛さは設定の finger_pad_solref）
- 監査: 口を通った指令を recovla.harness.audit.CommandAudit に渡す
"""
import numpy as np
import mujoco

from recovla.common import config
from recovla.harness.audit import CommandAudit
from recovla.harness.hand import FrankaHand, HandParams
from recovla.sim import control
from recovla.sim.rig import SimRig

_CFG = config.load("sensor_v1")                   # センサと作動のモデル v1（configs/sensor_v1.yaml）
ROBOT_BODIES = ("link1", "link2", "link3", "link4", "link5", "link6", "link7", "hand", "left_finger", "right_finger")
ARM_FRC = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])


def actuation_config(cfg=None) -> dict:
    return (cfg or _CFG)["actuation"]


class WorldRig(SimRig):
    def __init__(self, render: bool = True, cfg: dict = None, audit_fk: bool = True, gravcomp: bool = None):
        super().__init__(render=render, cfg=cfg or _CFG)
        act = actuation_config(self.cfg)
        m = self.model
        if gravcomp is None:
            gravcomp = bool(act["gravity_compensation"])
        # 重力の補償（作動器を通す）と関節の力の上限
        for b in (ROBOT_BODIES if gravcomp else ()):             # gravcomp=False は診断用
            m.body_gravcomp[m.body(b).id] = 1.0
        for i in range(1, 8):
            j = m.joint(f"joint{i}").id
            m.jnt_actgravcomp[j] = 1
            m.jnt_actfrclimited[j] = 1
            m.jnt_actfrcrange[j] = (-ARM_FRC[i - 1], ARM_FRC[i - 1])
        # 指の当て板の接触の剛さ
        sol = [float(v) for v in act["finger_pad_solref"]]
        for g in range(m.ngeom):
            if m.body(m.geom_bodyid[g]).name in ("left_finger", "right_finger") and m.geom_contype[g]:
                m.geom_solref[g] = sol
        self.hand = FrankaHand(m, self.data, self.grip_act, self.finger_qadr, self.finger_vadr,
                               HandParams.from_config(act.get("hand")))
        self.controller.gripper_indices = []                 # 世界の側の制御器（開始姿勢の静止だけに使う）は指に触れない
        # 重力の補償を入れたモデルで開始姿勢を作り直す（ハンドは開く速さの指令）
        import contextlib
        import io
        retreat = np.array(self.cfg["expert"]["retreat_pose"], dtype=float)
        with contextlib.redirect_stdout(io.StringIO()):
            self.starts = {"home": control.settle_start_state(m), "retreat": control.settle_start_state(m, pos=retreat)}
        for st in self.starts.values():
            st.ctrl[self.grip_act] = 0.5 * 0.9 * 0.1
        self.arm_dof = self.arm_vadr
        self.audit = CommandAudit(self._make_fk() if audit_fk else None)
        self._last_cmd = None

    def _make_fk(self):
        md = mujoco.MjData(self.model)
        qadr, hid = self.arm_qadr, self.hand_id

        def fk(q):
            md.qpos[qadr] = q
            mujoco.mj_kinematics(self.model, md)
            return md.xpos[hid].copy()
        return fk

    def reset(self, layout) -> None:
        super().reset(layout)
        self.hand.reset(opened=True)
        self.audit.reset()
        self._last_cmd = self.data.ctrl[[a[0] for a in self.controller.arm]].copy()

    # ------------------------------------------------------------------ 指令
    def apply_joint_commands(self, cmds) -> None:
        for q in cmds:
            self.audit.joint_command(q)
        self._last_cmd = np.array(cmds[-1], float)

    def physics_step(self, on_step=None) -> None:
        d = self.data
        for k, (act_idx, _, _, _) in enumerate(self.controller.arm):
            d.ctrl[act_idx] = self._last_cmd[k]
        self.hand.step()
        mujoco.mj_step(self.model, d)
        self.step += 1
        self.meter.on_step(d)
        self.audit.torque(d.qfrc_actuator[self.arm_dof])
        self.audit.finger_speed(np.abs(d.qvel[self.finger_vadr]).max())
        if on_step is not None:
            on_step(self)

    def audit_summary(self) -> dict:
        self.audit.hand_commands(self.hand.log)
        return self.audit.summary()
