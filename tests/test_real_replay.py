"""生成プログラムの関節の指令を実機の Panda で再生して比べる道具（scripts/real/export_replay.py・compare_replay.py・
real_to_ds.py）と、実機で測ったグリッパの遅延・速さのモデル（harness/hand_real.py）の確かめ。"""
import math
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "real"))

import compare_replay as C  # noqa: E402
import export_replay as X  # noqa: E402
import real_to_ds as D  # noqa: E402

HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])


@pytest.fixture(scope="module")
def fk():
    from recovla.common import config
    from recovla.harness import setup as H
    from recovla.runtime.motion import Motion
    return X.FK(Motion(H.nominal_setup(config.load_v2())))


# ------------------------------------------------------------------ q1 のずらしと手先の決まり

def test_to_real_shifts_only_q1():
    q = np.tile(HOME, (3, 1))
    r = X.to_real(q)
    assert np.allclose(r[:, 0], HOME[0] - math.radians(130.0))
    assert np.array_equal(r[:, 1:], q[:, 1:])
    assert np.array_equal(q[:, 0], np.full(3, HOME[0]))          # 元を変えない


def test_fk_rotates_with_q1_and_matches_real_home(fk):
    e0, _ = fk(HOME)
    e1, _ = fk(X.to_real(HOME))
    assert np.allclose(e1, D.rz(X.Q1_SHIFT) @ e0, atol=1e-9)
    # 実機の Panda で測った値: ホーム（q1 −130°）の O_T_EE を作業座標に戻すと (0.3072, 0.0000, 0.4869) m
    assert np.allclose(e0, [0.3072, 0.0, 0.4869], atol=1e-3)


# ------------------------------------------------------------------ 上限の確かめ

def _smooth(n=3000, amp=0.2):
    t = np.arange(1, n + 1) * X.DT
    s = 0.5 * (1 - np.cos(np.pi * np.clip(t / (n * X.DT), 0, 1)))
    q = np.tile(X.to_real(HOME), (n, 1))
    q[:, 3] += amp * s
    hold = np.tile(q[-1], (300, 1))
    return X.to_real(HOME), np.vstack([q, hold])


def test_check_stream_accepts_smooth_and_rejects_bad():
    start, q = _smooth()
    ok = X.check_stream(q, start, [0.04, 0.08])
    assert ok["ok"], ok
    jump = q.copy()
    jump[1000:, 2] += 0.01                                       # 1 刻みで 10 mrad の段差
    bad = X.check_stream(jump, start)
    assert not bad["ok"] and any("dddq" in r for r in bad["reasons"])
    out = X.check_stream(q, start, q1_range=(-2.80, -2.30))      # q1 −2.269 は範囲の外
    assert not out["ok"] and any("q1" in r for r in out["reasons"])
    moving = X.check_stream(q[:2000], start)                     # 動いている途中で終わる
    assert not moving["ok"] and any("止まって" in r for r in moving["reasons"])
    assert not X.check_stream(q, start, [0.09])["ok"]


# ------------------------------------------------------------------ CSV

def test_csv_round_trip(tmp_path):
    rows = [[f"{0.001 * (i + 1):.3f}", 0, 1] + [X.fmt(v) for v in np.random.default_rng(i).normal(size=4)] for i in range(5)]
    cols = ["t", "tail", "recorded", "a", "b", "c", "d"]
    sha = X.write_csv(tmp_path / "x.csv", rows, cols)
    assert len(sha) == 64
    r = D.read_csv(tmp_path / "x.csv")
    assert list(r) == cols
    assert np.allclose(r["a"], [float(x[3]) for x in rows], atol=1e-10)
    assert np.allclose(r["t"], [0.001, 0.002, 0.003, 0.004, 0.005])


# ------------------------------------------------------------------ 比べ

def _traj(t):
    return np.stack([0.4 + 0.1 * np.sin(2 * np.pi * 0.3 * t), -0.3 + 0.05 * np.cos(2 * np.pi * 0.5 * t),
                     0.2 + 0.02 * np.sin(2 * np.pi * 0.7 * t)], axis=1)


def _synthetic(sim_lag=0.005, real_lag=0.008, real_offset=0.0):
    t = np.arange(1, 6001) * X.DT
    cmd, sim = _traj(t), _traj(t - sim_lag)
    q_down = np.array([0.0, 1.0, 0.0, 0.0])
    replay = {"t": t, "width_sim": np.where(t < 2.0, 0.08, np.where(t < 4.0, 0.039, 0.08))}
    for i, a in enumerate("xyz"):
        replay[f"ee_cmd_{a}"], replay[f"ee_sim_{a}"] = cmd[:, i], sim[:, i]
    for i, k in enumerate(("ee_sim_qw", "ee_sim_qx", "ee_sim_qy", "ee_sim_qz")):
        replay[k] = np.full(len(t), q_down[i])
    real_pos = _traj(t - real_lag) + np.array([real_offset, 0, 0])
    R = C.quat_to_R(q_down)
    real = {"t": t, "success_rate": np.full(len(t), 0.98)}
    T = np.zeros((len(t), 4, 4))
    T[:, :3, :3] = R
    T[:, :3, 3] = real_pos
    T[:, 3, 3] = 1.0
    flat = T.transpose(0, 2, 1).reshape(len(t), 16)               # 列ごと（libfranka の並び）
    for i in range(16):
        real[f"O_T_EE{i}"] = flat[:, i]
    events = [{"t": 1.0, "kind": "grasp", "width": 0.04}, {"t": 3.0, "kind": "move", "width": 0.08}]
    return replay, real, events


def test_compare_passes_on_small_lag_and_fails_on_offset():
    replay, real, events = _synthetic()
    res = C.compare(replay, real, events, None, {"errors": []})
    cr = res["criteria"]
    assert abs(cr["lag"]["real_s"] - 0.008) < 0.0011 and abs(cr["lag"]["sim_s"] - 0.005) < 0.0011
    assert cr["lag"]["pass"] and cr["orientation"]["pass"] and cr["errors"]["pass"] and cr["success_rate"]["pass"]
    assert cr["position"]["pass"], cr["position"]
    assert res["verdict"] == "incomplete"                        # グリッパーの記録がない
    replay, real, events = _synthetic(real_offset=0.006)
    res = C.compare(replay, real, events, None, {"errors": []})
    assert not res["criteria"]["position"]["pass"] and res["verdict"] == "fail"
    replay, real, events = _synthetic(real_lag=0.030)
    assert not C.compare(replay, real, events, None, {"errors": []})["criteria"]["lag"]["pass"]


def test_compare_gripper_uses_open_start_when_widths_differ():
    replay, real, events = _synthetic()
    t = real["t"]
    # 実機は 0.3 s 遅延して動き、立方体なしで閉じ切る（0）。開く指令の時点の幅が違うので開き始めで比べる
    w = np.full(len(t), 0.08)
    w[(t >= 1.3)] = np.clip(0.08 - 0.05 * (t[t >= 1.3] - 1.3), 0.0, 0.08)
    up = t >= 3.3
    w[up] = np.clip(0.0 + 0.05 * (t[up] - 3.3), 0.0, 0.08)
    res = C.compare(replay, real, events, {"t": t, "width": w}, {"errors": []})
    g = res["criteria"]["gripper"]
    assert [e["used"] for e in g["events"]] == ["close_start", "open_start"]
    assert g["pass"] is False                                    # 0.3 s の遅延は基準 0.1 s を超える


# ------------------------------------------------------------------ 学習データの列（10 Hz）

def test_vendored_state_matches_repo():
    from recovla.data import vla_state
    rng = np.random.default_rng(0)
    q = rng.normal(size=(20, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    assert np.array_equal(D.orientation_deviation(q), vla_state.orientation_deviation(q))
    pos, f, j = rng.normal(size=(20, 3)), rng.uniform(0, 0.04, (20, 2)), rng.normal(size=(20, 7))
    assert np.array_equal(D.policy_state(pos, q, f, j), vla_state.policy_state(pos, q, f, j))
    assert D.STATE_NAMES == list(vla_state.STATE_NAMES)
    from recovla.data import convert
    assert D.ACTION_NAMES == convert.ACTION_NAMES


def test_flange_from_otee_and_real_to_ds(fk):
    """実機の記録（O_T_EE・q・幅）から作った表が、同じ値からの学習データの定義と一致する。フレームは t ≤ T の最後の行。"""
    from recovla.data import vla_state
    from recovla.runtime.motion import Motion
    from recovla.common import config
    from recovla.harness import setup as H
    mo = Motion(H.nominal_setup(config.load_v2()))
    t = np.arange(1, 501) * X.DT
    qs = np.tile(X.to_real(HOME), (len(t), 1))
    qs[:, 3] += 0.1 * np.sin(t)
    arm = {"t": t}
    otees, hands = [], []
    for q in qs:
        pos, quat = mo.hand_pose(q)
        e, _ = fk(q)
        R = np.zeros((4, 4))
        R[:3, :3] = mo.data.xmat[mo.hand_id].reshape(3, 3)       # hand_pose が読み込んだ姿勢の hand の向き
        R[:3, 3] = e
        R[3, 3] = 1.0
        otees.append(R.T.reshape(16))
        hands.append((pos, quat))
    for i in range(7):
        arm[f"q{i}"] = qs[:, i]
    for i in range(16):
        arm[f"O_T_EE{i}"] = np.array([o[i] for o in otees])
    grip = {"t": t[::66], "width": np.linspace(0.08, 0.04, len(t[::66]))}
    T = np.array([0.0995, 0.1995, 0.3005])
    rds = {"frame_index": np.arange(3), "timestamp": np.arange(3) * 0.1, "t_stream": T}
    for c in D.ACTION_NAMES:
        rds[c] = np.zeros(3)
    rows, cols = D.real_to_ds(arm, grip, rds, "base")
    got = np.array([[float(x) for x in r[3:18]] for r in rows])
    ia = D.last_at_or_before(t, T)
    assert list(ia) == [98, 198, 299]                            # t = 0.099・0.199・0.300
    ig = D.last_at_or_before(grip["t"], T)
    want = vla_state.policy_state(np.array([hands[i][0] for i in ia]), np.array([hands[i][1] for i in ia]),
                                  np.stack([0.5 * grip["width"][ig]] * 2, axis=1), qs[ia])
    assert np.allclose(got, want, atol=2e-6)
    assert cols[:25] == D.DS_COLS


def test_ds_from_dataset_rotation_round_trip():
    """学習データの行（シミュレーションの座標）を台座の座標へ回したものが、回した値から直接計算したものと一致する。"""
    from recovla.data import vla_state
    rng = np.random.default_rng(1)
    n = 10
    small = rng.normal(scale=0.05, size=(n, 3))
    ang = np.linalg.norm(small, axis=1)
    d = np.concatenate([np.cos(ang / 2)[:, None], np.sin(ang / 2)[:, None] * small / ang[:, None]], axis=1)
    quat = D.quat_mul(d, vla_state.Q_DOWN)
    pos, f, j = rng.normal(size=(n, 3)), rng.uniform(0, 0.04, (n, 2)), rng.normal(scale=0.3, size=(n, 7))
    st = vla_state.policy_state(pos, quat, f, j)
    act = rng.normal(size=(n, 7)).astype(np.float32)
    out = X.ds_from_dataset(st, act, np.arange(n) * 0.1)
    qb = D.quat_mul(np.broadcast_to(D.rz_quat(X.Q1_SHIFT), quat.shape).copy(), quat)
    want = vla_state.policy_state(pos @ D.rz(X.Q1_SHIFT).T, qb, f, X.to_real(j))
    assert np.allclose(out["state"], want, atol=2e-6)
    assert np.allclose(out["action"][:, :3], act[:, :3] @ D.rz(X.Q1_SHIFT).T, atol=1e-6)
    assert np.array_equal(out["action"][:, 3:], act[:, 3:])
    # task へ戻すと元の値（real_to_ds の --frame task と同じ回し方）
    p2, q2, j2 = D.to_frame(pos @ D.rz(X.Q1_SHIFT).T, qb, X.to_real(j), "task")
    assert np.allclose(vla_state.policy_state(p2, q2, f, j2), st, atol=2e-6)


# ------------------------------------------------------------------ ハンドの遅延・速さの上限（hand_real）

def _close_widths(rig, seconds=1.5):
    dt = float(rig.model.opt.timestep)
    rig.hand.grasp(0.04, 0.08, 40.0, 0.005, 0.005)
    out = []
    for _ in range(int(seconds / dt)):
        rig.physics_step()
        out.append(rig.hand.width())
    return np.array(out)


def _rig(timing=None):
    from recovla.common import config
    from recovla.harness import hand_real
    from recovla.harness.world import WorldRig
    from recovla.sim import scene
    rig = WorldRig(render=False, cfg=config.load_v2(), audit_fk=False)
    if timing is not None:
        hand_real.install(rig, timing)
    rig.reset(scene.sample_layout(59950, "empty", start="home"))
    rig.apply_joint_commands([rig.data.qpos[rig.arm_qadr].copy()])
    for _ in range(100):
        rig.physics_step()
    return rig


def test_hand_real_defaults_bit_identical_and_timing():
    from recovla.harness.hand_real import RealTiming
    base = _rig()
    try:
        w0 = _close_widths(base)
    finally:
        base.close()
    same = _rig(RealTiming())
    try:
        w1 = _close_widths(same)
        assert len(same.hand.log) == 1
    finally:
        same.close()
    assert np.array_equal(w0, w1)
    late = _rig(RealTiming(dead_time_s=0.3, speed_cap=0.05))
    try:
        w2 = _close_widths(late)
        dt = float(late.model.opt.timestep)
        assert len(late.hand.log) == 1                           # 記録は受けた時刻の 1 行だけ
    finally:
        late.close()
    start = lambda w: int(np.argmax(w < w[0] - 0.001))
    assert start(w2) * dt >= start(w0) * dt + 0.29
    speed = np.abs(np.diff(w2)) / dt
    assert speed.max() <= 0.05 * 1.05


def _open_tail_s(rig):
    """閉じ切ってから 8 cm へ開き、75 mm から 79 mm までにかかった時間（目標の手前の減速の強さ k_pos で決まる）。"""
    _close_widths(rig, 1.6)
    dt = float(rig.model.opt.timestep)
    rig.hand.move(0.08, 0.08)
    w = []
    for _ in range(int(2.0 / dt)):
        rig.physics_step()
        w.append(rig.hand.width())
    w = np.array(w)
    return (int(np.argmax(w > 0.079)) - int(np.argmax(w > 0.075))) * dt, w


def test_hand_real_config_values_and_shape():
    from recovla.common import config
    from recovla.harness.hand_real import RealTiming
    t = RealTiming.from_config(config.load("real_v1")["real_v1"]["gripper"])
    assert (t.dead_time_s, t.dead_time_sd, t.speed_scale, t.speed_cap, t.k_pos) == (0.15, 0.0, 1.0, 0.0, 9.1)
    assert RealTiming().is_identity and not t.is_identity
    base = _rig()
    try:
        tail0, _ = _open_tail_s(base)
    finally:
        base.close()
    real = _rig(t)
    try:
        dt = float(real.model.opt.timestep)
        w = _close_widths(real, 1.0)
        onset = int(np.argmax(w < w[0] - 0.0003)) * dt
        assert 0.15 <= onset <= 0.17
        k75, k50 = int(np.argmax(w < 0.075)), int(np.argmax(w < 0.050))
        assert abs(0.025 / ((k50 - k75) * dt) - 0.08) < 0.003         # 動いている間は指令の速さのまま（上限で切らない）
        real.hand.reset(opened=True)
        for _ in range(400):
            real.physics_step()
        tail1, _ = _open_tail_s(real)
    finally:
        real.close()
    # 75→79 mm は 指数の減速なら ln(5/1)/k_pos: 実機 9.1 /s で約 0.18 s（実機の幅のセンサの値では 0.19〜0.20 s）、hand.py の 20 /s で約 0.08 s
    assert tail0 < 0.11 and 0.15 < tail1 < 0.21


def test_hand_real_dead_time_draws_are_seeded_and_clipped():
    from recovla.harness.hand_real import RealTiming, RealTimingHand
    h = object.__new__(RealTimingHand)                             # モデルなしで遅延の引き方だけを確かめる
    h.timing = RealTiming(dead_time_s=0.2, dead_time_sd=0.1, dead_time_s_range=(0.1, 0.3), seed=3)
    h.reseed(7)
    a = [h._dead_time() for _ in range(200)]
    h.reseed(7)
    b = [h._dead_time() for _ in range(200)]
    assert a == b and min(a) >= 0.1 and max(a) <= 0.3 and len(set(a)) > 50
    h.timing = RealTiming(dead_time_s=0.2)
    assert h._dead_time() == 0.2


# ------------------------------------------------------------------ 遅い版（replay_v1_slow）

def test_stretch_halves_speed_and_keeps_path():
    import export_replay_slow as S
    start, q = _smooth()
    slow = S.stretch(q, start, 2.0)
    assert len(slow) == 2 * len(q)
    assert np.allclose(slow[1::2], q, atol=1e-9)                 # 行 2i+1（時刻 2(i+1) ms）は元の行 i
    a = X.check_stream(q, start)["ratio_max_all"]
    b = X.check_stream(slow, start)["ratio_max_all"]
    assert b["dq"] <= 0.5 * a["dq"] * 1.01 and b["ddq"] <= 0.25 * a["ddq"] * 1.05
    assert np.allclose(S.to_sim(X.to_real(q)), q)


def test_compare_flags_slow_variant(tmp_path):
    import json
    replay, real, events = _synthetic()
    cols = ["t", "width_sim"] + [f"ee_cmd_{a}" for a in "xyz"] + [f"ee_sim_{a}" for a in "xyz"] + \
        ["ee_sim_qw", "ee_sim_qx", "ee_sim_qy", "ee_sim_qz"]
    rows = [[replay[c][i] for c in cols] for i in range(len(replay["t"]))]
    D.write_csv(tmp_path / "x.csv", rows, cols)
    D.write_csv(tmp_path / "x_gripper.csv", [[1.0, "grasp", 0.04], [3.0, "move", 0.08]], ["t", "kind", "width"])
    rcols = ["t", "success_rate"] + [f"O_T_EE{i}" for i in range(16)]
    D.write_csv(tmp_path / "arm.csv", [[real[c][i] for c in rcols] for i in range(len(real["t"]))], rcols)
    (tmp_path / "manifest.json").write_text(json.dumps({"variant": "slow_0.5x"}), encoding="utf-8")
    out = tmp_path / "res.json"
    C.main(["--replay", str(tmp_path / "x.csv"), "--real", str(tmp_path / "arm.csv"), "--out", str(out)])
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["variant"] == "slow_0.5x" and res["slow_variant"] is True
