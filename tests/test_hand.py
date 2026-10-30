"""Franka Hand のモデル（harness/hand.py）の grasp の止まり方の確かめ。"""
import numpy as np

from recovla.common import config
from recovla.harness.world import WorldRig
from recovla.sim import scene


def test_grasp_on_air_stops_without_chatter():
    """空を掴むと、開き幅 0 で止まり（下限を越えて押し込まない）、指の速さが静止の閾値を下回り続ける。"""
    cfg = config.load_v2()
    act = cfg["actuation"]
    rig = WorldRig(render=False, cfg=cfg, audit_fk=False)
    try:
        rig.reset(scene.sample_layout(59950, "empty", start="home"))
        rig.apply_joint_commands([rig.data.qpos[rig.arm_qadr].copy()])
        dt = float(rig.model.opt.timestep)
        for _ in range(int(0.5 / dt)):
            rig.physics_step()
        rig.hand.grasp(0.04, float(act["gripper_speed"]), float(act["grasp_force"]), 0.005, 0.005)
        widths, speeds = [], []
        for _ in range(int(3.0 / dt)):
            rig.physics_step()
            widths.append(rig.hand.width())
            speeds.append(float(np.sum(np.abs(rig.data.qvel[rig.finger_vadr]))))
        tail = slice(int(2.0 / dt), None)
        assert min(widths[tail]) > -1e-4
        assert max(speeds[tail]) < float(cfg["expert"]["finger_rest_speed"])
        assert not rig.hand.is_grasped()
    finally:
        rig.close()


def test_grasp_command_unchanged_at_cube_width():
    """立方体の幅の近く（開き幅 > speed/k_pos）では、閉じる速さの指令は −speed のまま（目標の近くだけ減速する）。"""
    cfg = config.load_v2()
    rig = WorldRig(render=False, cfg=cfg, audit_fk=False)
    try:
        h = rig.hand
        h.grasp(0.04, 0.08, 40.0)
        for w in (0.08, 0.04, 0.035, 0.005):
            rig.data.qpos[h.qadr] = w / 2.0
            h.step()
            assert rig.data.ctrl[h.act] * 2.0 == -0.08
    finally:
        rig.close()
