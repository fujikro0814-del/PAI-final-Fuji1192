"""腕のサーボを実機の Panda で測った追従に合わせる重ね（harness/arm_real.py、configs/real_v1.yaml の real_v1.arm_servo）の確かめ。"""
import numpy as np


def _rig(servo=None):
    from recovla.common import config
    from recovla.harness import arm_real
    from recovla.harness.world import WorldRig
    from recovla.sim import scene
    rig = WorldRig(render=False, cfg=config.load_v2(), audit_fk=False)
    saved = arm_real.install(rig, servo) if servo is not None else None
    rig.reset(scene.sample_layout(59950, "empty", start="home"))
    return rig, saved


def _ramp(rig, v=0.2, joint=0, seconds=1.0):
    """関節 1 つを一定の速さ v [rad/s] で動かす指令（1 kHz、物理の 1 ステップに 2 つ）。測った関節角の列を返す。"""
    dt = float(rig.model.opt.timestep)
    q0 = rig.data.qpos[rig.arm_qadr].copy()
    out, cmd = [], []
    for n in range(int(seconds / dt)):
        c = [q0.copy(), q0.copy()]
        c[0][joint] += v * (2 * n + 1) * 1e-3
        c[1][joint] += v * (2 * n + 2) * 1e-3
        rig.apply_joint_commands(c)
        rig.physics_step()
        out.append(rig.data.qpos[rig.arm_qadr].copy())
        cmd.append(c[1])
    return np.array(out), np.array(cmd)


def test_default_servo_is_bit_identical():
    from recovla.harness.arm_real import ArmServo, restore
    base, _ = _rig()
    try:
        g0, b0 = base.model.actuator_gainprm.copy(), base.model.actuator_biasprm.copy()
        q_base, _ = _ramp(base)
    finally:
        base.close()
    same, saved = _rig(ArmServo())
    try:
        assert ArmServo().is_default()
        assert np.array_equal(same.model.actuator_gainprm, g0) and np.array_equal(same.model.actuator_biasprm, b0)
        assert "apply_joint_commands" not in same.__dict__ and "physics_step" not in same.__dict__
        q_same, _ = _ramp(same)
        restore(same, saved)
    finally:
        same.close()
    assert np.array_equal(q_base, q_same)


def test_gravity_comp_flag_is_inactive_in_world_and_overlay_holds_pose():
    """WorldRig は作った後に body_gravcomp を立てるが、MuJoCo は ngravcomp = 0 なら補償しない。重ねの重力補償では姿勢を保つ。"""
    from recovla.harness.arm_real import ArmServo
    base, _ = _rig()
    try:
        assert base.model.ngravcomp == 0
        q0 = base._last_cmd.copy()
        for _ in range(500):
            base.apply_joint_commands([q0, q0])
            base.physics_step()
        sag = np.abs(base.data.qpos[base.arm_qadr] - q0).max()
    finally:
        base.close()
    gc, _ = _rig(ArmServo(gravity_comp=True))
    try:
        q0 = gc._last_cmd.copy()
        assert np.allclose(gc.data.qpos[gc.arm_qadr], q0)        # reset で最初の指令へ置き直す
        for _ in range(500):
            gc.apply_joint_commands([q0, q0])
            gc.physics_step()
        held = np.abs(gc.data.qpos[gc.arm_qadr] - q0).max()
    finally:
        gc.close()
    assert sag > 1e-3 and held < 1e-4


def test_velocity_feedforward_and_delay_set_the_lag():
    from recovla.harness.arm_real import ArmServo
    kp = np.array([4500.0, 4500.0, 3500.0, 3500.0, 2000.0, 2000.0, 2000.0])
    v = 0.2
    lags = {}
    for name, servo in (("default", None), ("ff", ArmServo(tuple(kp), tuple(kp * 0.05), 1.0, 0, True)),
                        ("ff_delay3", ArmServo(tuple(kp), tuple(kp * 0.05), 1.0, 3, True))):
        rig, _ = _rig(servo)
        try:
            q, c = _ramp(rig, v=v)
        finally:
            rig.close()
        lags[name] = float(np.mean((c[-100:, 0] - q[-100:, 0]) / v))   # 一定の速さでの遅延 [s]
    assert 0.09 < lags["default"] < 0.11                              # kv/kp = 0.1 s
    assert abs(lags["ff"]) < 0.002
    assert abs(lags["ff_delay3"] - 0.006) < 0.002                     # 3 ステップ = 6 ms


def test_delay_audits_original_commands_and_keeps_motion():
    """遅延を入れても、動きの制限の監査（G3）には制限層の元の指令が入り、サーボへは遅らせた指令の列が届く。"""
    from recovla.harness.arm_real import ArmServo
    from recovla.runtime.limiter import JointLimiter
    kp = np.array([4500.0, 4500.0, 3500.0, 3500.0, 2000.0, 2000.0, 2000.0])
    rig, _ = _rig(ArmServo(tuple(kp), tuple(kp * 0.05), 1.0, 4, True))
    try:
        rig.audit.reset()
        lim = JointLimiter()
        lim.reset(rig._last_cmd.copy())
        sent, held = [], []
        for _ in range(300):
            c = [lim.step(np.full(7, 0.5)), lim.step(np.full(7, 0.5))]
            sent += c
            rig.apply_joint_commands(c)
            held.append(rig._last_cmd.copy())
            rig.physics_step()
        assert np.array_equal(np.array(rig.audit.q), np.array(sent))
        g3 = rig.audit.summary()
        assert g3["violations"]["joint_acc"] == 0 and g3["violations"]["joint_jerk"] == 0
        assert np.allclose(held[0], held[3]) and not np.allclose(held[4], held[3])   # 4 ステップは最初の指令を保ち、5 ステップ目から動く
    finally:
        rig.close()


def test_config_values_parse():
    from recovla.common import config
    from recovla.harness.arm_real import ArmServo
    cfg = config.load(*[n for n in config.V2_LAYERS if (config.CONFIG_DIR / f"{n}.yaml").is_file()], "expert_v3", "real_v1")
    s = ArmServo.from_config(cfg["real_v1"]["arm_servo"])
    assert len(s.kp) == 7 and len(s.kv) == 7 and s.gravity_comp and 0.0 < s.vel_ff <= 1.0 and s.cmd_delay_steps >= 0
