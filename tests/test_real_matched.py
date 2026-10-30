"""実機の Panda で測った応答に合わせたシミュレーション（v3_real。harness/real_matched.py・expert/inject_real.py・
expert/generate_real.py・configs/v3_real.yaml、項目は docs/real/v3_real.md）の確かめ。合わせる前の v3 の生成が変わらないことと、
v3_real の項目の入り・切りが 1 つずつ効くこと。シード 58700 は v3_real の動作確認用。"""
import dataclasses
import importlib.util
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEED = 58700
OFF = {k: False for k in ("arm_servo", "hand_timing", "width_sensor", "expert_width", "closed_fix", "b_slip", "contact_std")}


def _v3_rig():
    from recovla.expert import generate as G
    from recovla.harness.gen_sensed import SensedDrivenRig
    return SensedDrivenRig(G.rig_config("v3"))


def _real_rig(**sw):
    from recovla.harness import real_matched as V
    return V.V3RealGenRig(V.load_config("v3_real", generation=True, switches=sw or None))


def _attempt(rig, kind, real: bool):
    from recovla.expert import generate as G
    from recovla.expert import generate_real as GR
    from recovla.sim import scene
    lay = scene.sample_layout(SEED)
    color = lay.table_colors[SEED % len(lay.table_colors)]
    run = GR.run_attempt if real else G.run_attempt
    return run(rig, G.EpisodeSpec(SEED, color, kind=kind), 0, None, False)


# ------------------------------------------------------------------ 合わせる前の生成は変わらない
def test_default_configs_do_not_see_v3_real():
    from recovla.common import config
    from recovla.expert import generate as G
    for cfg in (G.rig_config("v3"), config.load_v2(), config.load()):
        assert "v3_real" not in cfg and "real_v1" not in cfg
    assert G.rig_config("v3")["sensor"]["gripper"] == {"rate_hz": 10, "width_sigma": 1.0e-4, "width_unit": 1.0e-4}
    assert G.rig_config("v3_real") is None                             # generate.py の rig_config は v3_real を知らない


def test_patches_are_scoped_and_restored(tmp_path):
    from recovla.expert import generate as G
    from recovla.expert import generate_real as GR
    from recovla.expert import inject
    orig = (G._init_worker, G.rig_config, inject.Injector)
    with GR.patched():
        assert inject.Injector is not orig[2] and G._init_worker is GR._init_worker
        assert G.rig_config("v3")["sensor"]["gripper"]["rate_hz"] == 10     # v3 はそのまま元の rig_config へ
    assert (G._init_worker, G.rig_config, inject.Injector) == orig
    from recovla.sim import scene
    color = scene.sample_layout(SEED).table_colors[0]
    res = GR.generate([G.EpisodeSpec(SEED, color, kind="n")], tmp_path / "g", render=False, max_retry=0)
    assert (G._init_worker, G.rig_config, inject.Injector) == orig and len(res) == 1
    run = __import__("json").loads((tmp_path / "g" / "run.json").read_text(encoding="utf-8"))
    assert run["rig_kind"] == "v3_real"


def test_make_injector_passes_through_unless_b_slip():
    from recovla.expert import inject, inject_real
    from recovla.harness import real_matched as V
    from recovla.sim import scene
    rig = _v3_rig()
    try:
        lay = scene.sample_layout(SEED)
        for kind in "ABC":
            ip = inject.sample_params(kind, np.random.default_rng(0), lay, lay.table_colors[0], rig.cfg)
            assert type(inject_real.make_injector(ip, None, rig, rig.cfg)) is inject_real.ORIG_INJECTOR
        ip = inject.sample_params("B", np.random.default_rng(0), lay, lay.table_colors[0], rig.cfg)
        on = V.load_config("v3_real", True, {"b_slip": True})
        assert type(inject_real.make_injector(ip, None, rig, V.load_config("v3_real", True))) is inject_real.ORIG_INJECTOR
        inj = inject_real.make_injector(ip, None, rig, on)
        assert isinstance(inj, inject_real.SlipInjector)
        lo, hi = on["v3_real"]["b_slip"]["overlap_m"]
        assert lo <= inj.p.overlap_m <= hi and inj.p.side in (1.0, -1.0)
        assert abs(inject_real.pad_half_width(rig.model) - 0.0085) < 1e-12        # panda.xml の当て板
        assert abs(inj.p.shift_m - (0.02 + 0.0085 - inj.p.overlap_m)) < 1e-6
        assert inj.p.to_json()["variant"] == "shallow_slip"
    finally:
        rig.close()


@pytest.mark.parametrize("kind", ["n", "A"])
def test_all_switches_off_is_bit_identical_to_v3(kind):
    """入り・切りを全部切った v3_real の生成の世界（と差し替えた意図的な失敗）は、合わせる前の rig v3 とフレーム単位でビット一致する。"""
    a = _v3_rig()
    try:
        ra = _attempt(a, kind, real=False)
    finally:
        a.close()
    b = _real_rig(**OFF)
    try:
        rb = _attempt(b, kind, real=True)
    finally:
        b.close()
    fa, fb = ra.pop("_frames"), rb.pop("_frames")
    for r in (ra, rb):
        r.pop("wall_s")
    assert ra == rb
    assert len(fa) == len(fb) and len(fa) > 100
    for x, y in zip(fa, fb):
        assert x.keys() == y.keys()
        for k in x:
            assert np.array_equal(np.asarray(x[k]), np.asarray(y[k])), k


# ------------------------------------------------------------------ v3_real の入り・切り
def test_each_switch_turns_its_item_on():
    from recovla.harness import real_matched as V
    from recovla.harness.hand import FrankaHand
    from recovla.harness.hand_real import RealTimingHand
    from recovla.harness.sensors import SensorSuite
    rig = _real_rig()
    try:
        cfg = rig.cfg
        assert V.Switches.from_config(cfg) == V.Switches(True, True, True, True, True, False, False)   # b_slip・contact_std は切（docs/real/prereg_regen_v1_amend1.md）
        acts = [int(x[0]) for x in rig.controller.arm]
        assert list(rig.model.actuator_gainprm[acts, 0]) == [9000.0, 9000.0, 7000.0, 7000.0, 4000.0, 4000.0, 4000.0]
        assert "apply_joint_commands" in rig.__dict__ and "physics_step" in rig.__dict__     # 先送り・遅延・重力の補償
        assert isinstance(rig.hand, RealTimingHand) and rig.hand.timing.dead_time_s == 0.15 and rig.hand.p.k_pos == 9.1
        assert isinstance(rig.suite, V.RealGripperSuite) and rig.suite.delay_s == 0.08
        assert cfg["sensor"]["gripper"]["rate_hz"] == 6.5 and cfg["sensor"]["gripper"]["width_unit"] == 6.6e-7
        assert cfg["v3_real"]["closed_fix"]["delay_s"] == 0.23
        from recovla.sim import scene
        rig.reset(scene.sample_layout(59950, "empty", start="home"))
        meta = rig.episode_meta()["world"]
        assert meta["world"] == "v3_real" and meta["actuation"]["hand_timing"]["dead_time_s"] == 0.15
    finally:
        rig.close()
    for name in ("arm_servo", "hand_timing", "width_sensor"):
        sw = {name: False}
        if name == "width_sensor":
            sw["expert_width"] = False
        r = _real_rig(**sw)
        try:
            if name == "arm_servo":
                acts = [int(x[0]) for x in r.controller.arm]
                assert r.model.actuator_gainprm[acts[0], 0] == 4500.0 and "apply_joint_commands" not in r.__dict__
            if name == "hand_timing":
                assert type(r.hand) is FrankaHand
            if name == "width_sensor":
                assert type(r.suite) is SensorSuite and r.cfg["sensor"]["gripper"]["rate_hz"] == 10
        finally:
            r.close()
    with pytest.raises(ValueError):
        V.load_config("v3_real", switches={"width_sensor": False})      # expert_width だけ残すと読む値がない
    with pytest.raises(ValueError):
        V.load_config("v3_real", switches={"corner_basin": True})       # 作っていない項目は断る


def _contact_items(m):
    return (m.opt.cone, m.opt.impratio, m.opt.noslip_iterations, m.geom_condim.copy(), m.geom_friction.copy(),
            m.geom_solref.copy(), m.geom_solimp.copy(), m.geom_size.copy(), m.body_mass.copy())


def test_contact_std_on_and_off():
    """contact_std 入り: 当て板の condim 4・楕円の錐・impratio 10（ほかの値は v3 のまま）。切: v3 と同じ。"""
    import mujoco
    from recovla.harness import real_matched as V
    a = _v3_rig()
    try:
        base = _contact_items(a.model)
    finally:
        a.close()
    on = _real_rig(contact_std=True)
    try:
        m = on.model
        pads = V.pad_geoms(m)
        assert base[0] == mujoco.mjtCone.mjCONE_PYRAMIDAL and base[1] == 1.0 and set(base[3][pads].tolist()) == {3}
        assert len(pads) == 12
        assert {m.body(m.geom_bodyid[g]).name for g in pads} == {"left_finger", "right_finger"}
        assert all(m.geom_condim[g] == 4 for g in pads)
        rest = [g for g in range(m.ngeom) if g not in pads]
        assert np.array_equal(m.geom_condim[rest], base[3][rest])        # 立方体などは 3 のまま
        assert m.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC and m.opt.impratio == 10.0 and m.opt.noslip_iterations == 0
        got = _contact_items(m)
        for x, y in zip(got[4:], base[4:]):                               # 摩擦・solref・solimp・形・質量は同じ
            assert np.array_equal(x, y)
        assert on.v3_real_contact == {"pad_geoms": 12, "pad_condim": 4, "cone": "elliptic", "impratio": 10.0}
    finally:
        on.close()
    off = _real_rig(contact_std=False)
    try:
        got = _contact_items(off.model)
        assert got[0] == base[0] and got[1] == base[1] and got[2] == base[2]
        for x, y in zip(got[3:], base[3:]):
            assert np.array_equal(x, y)
        assert off.v3_real_contact is None
    finally:
        off.close()
    world, suite = V.make_eval_world(V.load_config("v3_real", False, {"contact_std": True}))   # 評価の世界も同じ
    try:
        assert world.model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC and world.model.opt.impratio == 10.0
        assert all(world.model.geom_condim[g] == 4 for g in V.pad_geoms(world.model))
    finally:
        suite.close()
        world.close()


def _close_and_watch(rig, seconds=1.2):
    """指が開き切って止まるまで 1 s 待ってから閉じる指令を出し、物理の 1 ステップごとの（指令からの時刻, 真の幅, 届いている幅,
    生成プログラムが読む指の速さの和, 生成プログラムが読む指の幅）を返す。"""
    from recovla.sim import scene
    from recovla.sim.rig import quiet
    rig.reset(scene.sample_layout(59950, "empty", start="home"))
    n = rig.steps_per_read * rig.timestep
    out = []

    def on_step(r):
        tr = r.truth("red")
        out.append((float(r.data.time) - t0, float(r.data.qpos[r.finger_qadr].sum()), float(r.suite.grip_state.width),
                    float(np.sum(np.abs(tr.finger_vel))), float(np.sum(tr.fingers))))
    with quiet():
        for _ in range(int(1.0 / n)):
            rig.pad_read(np.zeros(3), False)
        t0 = float(rig.data.time)
        rig.pad_read(np.zeros(3), True, on_step)
        for _ in range(int(seconds / n)):
            rig.pad_read(np.zeros(3), False, on_step)
    return np.array(out), t0


def test_width_sensor_rate_delay_and_hold():
    rig = _real_rig()
    try:
        x, t0 = _close_and_watch(rig)
        s = np.array([(a - t0, b, c - t0) for a, b, c in rig.suite.samples if a >= t0 - 0.5])
    finally:
        rig.close()
    t, true, got = x[:, 0], x[:, 1], x[:, 2]
    d = np.diff(s[1:, 0])
    assert np.all(np.abs(d - 1.0 / 6.5) <= 0.0021)                    # 6.5 Hz（物理の 1 ステップ 2 ms に丸める）
    assert np.allclose(s[1:, 0] - s[1:, 2], 0.08, atol=1e-9)        # 取る値は 0.08 s 前
    late = s[:, 2] >= t[0]
    ref = np.interp(s[late, 2], t, true)
    assert late.sum() >= 6 and np.abs(s[late, 1] - ref).max() < 3e-6        # 雑音 1.4e-7・刻み 6.6e-7
    k = np.searchsorted(s[:, 0], t + 1e-9, side="right") - 1
    assert np.array_equal(got, s[k, 1])                              # 次の値まで保つ
    onset_true = t[np.argmax(true < true[0] - 0.0003)]
    onset_seen = t[np.argmax(got < got[0] - 0.0003)]
    assert 0.149 <= onset_true <= 0.175                              # 指の動き出し 0.15 s（hand_timing、実機の Panda で測った値）
    assert 0.08 <= onset_seen - onset_true <= 0.08 + 1.0 / 6.5 + 0.004


def test_base_sensor_reads_true_width_without_delay():
    """合わせる前の v3 の生成の幅は 10 Hz のモデル: 読んだ時刻の真値（雑音 0.1 mm・刻み 0.1 mm）で、遅延はない。"""
    from recovla.sim import scene
    from recovla.sim.rig import quiet
    rig = _v3_rig()
    try:
        rig.reset(scene.sample_layout(59950, "empty", start="home"))
        rows = []
        with quiet():
            rig.pad_read(np.zeros(3), True)
            for i in range(60):
                rig.pad_read(np.zeros(3), False)
                if i % 2 == 0:                                       # 記録のフレーム（20 Hz）と同じ頻度で読む
                    g = rig.suite.sense(rig.data, rig.hand, rig._last_cmd, False).gripper
                    rows.append((g.t, g.width, rig.hand.width(), float(rig.data.time)))
    finally:
        rig.close()
    r = np.array(rows)
    fresh = r[:, 0] == r[:, 3]
    assert fresh.sum() >= 10 and np.abs(r[fresh, 1] - r[fresh, 2]).max() < 6e-4


def test_closed_fix_restarts_the_rest_count():
    on = _real_rig()
    try:
        x, _ = _close_and_watch(on, 0.6)
    finally:
        on.close()
    moving = x[:, 3] > 0.005
    t = x[:, 0]
    assert moving[t < 0.229].all()                                    # 指令から 0.23 s は動いているとみなす
    off = _real_rig(closed_fix=False)
    try:
        y, _ = _close_and_watch(off, 0.6)
    finally:
        off.close()
    # closed_fix を切ると、指が動き始める（0.15 s）前と、センサに動きが届く前は「止まっている」と読める
    assert (y[y[:, 0] < 0.15, 3] < 0.005).all()


def test_expert_reads_sensor_width():
    rig = _real_rig()
    try:
        x, _ = _close_and_watch(rig, 0.6)
    finally:
        rig.close()
    assert np.array_equal(x[:, 4], x[:, 2])                          # 生成プログラムが読む指の幅 = 届いている幅
    off = _real_rig(expert_width=False)
    try:
        y, _ = _close_and_watch(off, 0.6)
    finally:
        off.close()
    assert np.allclose(y[:, 4], y[:, 1])                             # 切れば真値


def test_eval_world_and_96_wrapper():
    from recovla.harness import real_matched as V
    from recovla.harness.hand_real import RealTimingHand
    world, suite = V.make_eval_world(V.load_config("v3_real", generation=False))
    try:
        assert isinstance(world.hand, RealTimingHand) and isinstance(suite, V.RealGripperSuite)
        assert "expert_v3" not in str(world.cfg.get("expert", {}).get("final_approach"))   # 評価は expert_v3 を重ねない
        assert world.cfg["expert"].get("final_approach") is None
    finally:
        suite.close()
        world.close()
    spec = importlib.util.spec_from_file_location("ev3r", ROOT / "scripts" / "real" / "eval_real_matched.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.split_argv(["--world", "v3_real", "--switch", "b_slip=on", "run", "--x", "1"]) == ("v3_real", {"b_slip": True},
                                                                                              ["run", "--x", "1"])
    m96 = m.patch(m.load_96(), "v3_real")
    eng = m96.Engine.__new__(m96.Engine)
    eng.a = type("A", (), {"diag_no_gravcomp": False})()
    eng.cfg, eng.suite = V.load_config("v3_real", generation=False), None
    eng.new_world()
    try:
        assert isinstance(eng.world.hand, RealTimingHand) and isinstance(eng.suite, V.RealGripperSuite)
        assert eng.world.v3_real_actuation["arm_servo"]["cmd_delay_steps"] == 4
    finally:
        eng.suite.close()
        eng.world.close()


def test_b_slip_start_condition_shape():
    from recovla.expert import inject_real
    from recovla.harness import real_matched as V
    cfg = V.load_config("v3_real", True, {"b_slip": True})
    n = 2
    data = {"cube_pos": np.zeros((n, 3, 3)), "fingers": np.zeros((n, 2)), "step": np.array([100, 125]),
            "gripper_closed": np.array([True, True]), "fingertip": np.array([[0.0, 0.0, 0.2]] * n),
            "cube_linvel": np.zeros((n, 3, 3))}
    data["cube_pos"][:, 0] = [0.4, 0.0, 0.02]
    meta = {"target": "red", "record_start_step": 100, "inject": {"info": {"rest_s": 0.3}}}
    c = inject_real.start_condition(data, meta, cfg)
    assert c["ok"], c
    data["gripper_closed"] = np.array([False, False])
    assert not inject_real.start_condition(data, meta, cfg)["ok"]
