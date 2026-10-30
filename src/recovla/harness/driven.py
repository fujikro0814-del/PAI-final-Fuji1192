"""生成プログラムを、実行時と同じ指令の口に通して動かす世界。

生成プログラムの入口 rig.pad_read(vel, press) を、実行系と同じ経路に付け替える:
  手先の速さ → recovla.runtime.motion.Motion（x_cmd の積分・追従器・IK・制限層）→ 関節の位置の指令（1 kHz 相当）→ 世界
  グリッパの押下（開閉の切り替え）→ ハンドの grasp / move（把持力・速さは作動のモデル）
生成プログラムは真の状態を読む。関節角も真の値をそのまま Motion に渡す。
"""
import dataclasses

import numpy as np

from recovla.harness import setup as H
from recovla.harness.world import WorldRig
from recovla.runtime.motion import Motion
from recovla.runtime.types import JointState


class DrivenRig(WorldRig):
    def __init__(self, render: bool = False, limiter_enabled: bool = True, cfg: dict = None):
        super().__init__(render=render, cfg=cfg)
        act = self.cfg["actuation"]
        self.setup_info = H.nominal_setup(self.cfg)
        self.motion = Motion(self.setup_info, limiter_enabled=limiter_enabled, margin=float(act["limiter_margin"]),
                             xcmd_leash_m=(self.cfg.get("runtime_v2") or {}).get("xcmd_leash_m"),
                             cart_margin=(self.cfg.get("runtime_v2") or {}).get("cart_margin"))
        self.grasp_args = (float(act["grasp_force"]), float(act["grasp_eps"]))
        self.gripper_speed = float(act["gripper_speed"])
        self.closed = False

    def joint_state(self) -> JointState:
        d = self.data
        return JointState(float(d.time), d.qpos[self.arm_qadr].copy(), d.qvel[self.arm_vadr].copy(),
                          self._last_cmd.copy())

    def reset(self, layout) -> None:
        super().reset(layout)
        self.motion.reset(self.joint_state())
        self.closed = False

    def truth(self, target: str, data=None):
        tr = super().truth(target, data)
        return dataclasses.replace(tr, x_cmd=self.motion.x_cmd, gripper_closed=bool(self.closed))

    def pad_read(self, vel=None, press: bool = False, on_step=None) -> None:
        if press:
            self.closed = not self.closed
            if self.closed:
                force, eps = self.grasp_args
                self.hand.grasp(self.setup_info.cube_size, self.gripper_speed, force, eps, eps)
            else:
                self.hand.move(0.08, self.gripper_speed)
        self.motion.set_velocity(np.zeros(3) if vel is None else vel)
        if self.safety.on():
            self.safety.begin_read(self.data, self.motion.integrator.x_cmd)
            self.motion.set_command_filter(self.safety.filter)
        else:
            self.motion.set_command_filter(None)
        for _ in range(self.steps_per_read):
            self.apply_joint_commands(self.motion.step(self.joint_state()))
            self.physics_step(on_step)
