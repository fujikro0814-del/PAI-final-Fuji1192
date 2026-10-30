"""掴み損ね・落下に反応して止める反射（src/recovla/runtime/reflex.py）と、その実行系への組み込み（runtime/runner.py）、
手から引き抜く失敗の意図的な失敗 P2S（src/recovla/eval/induce_slip.py）、評価の本体 eval_resumable.py の --reflex の確かめ。
CPU だけ（GPU・評価の配置のシミュレーションは使わない）。

読むもの: src の上の 4 ファイル、scripts/eval_resumable.py・scripts/check_real_boundary.py（importlib）。書くもの: pytest の一時フォルダだけ。
"""
import importlib.util
import json
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from recovla.runtime import runner as RUN
from recovla.runtime.reflex import GraspLossDetector, GraspLossReflex, ReflexParams

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------- 検出（合成の信号）
def feed(det, seq, dt=0.1, t0=0.0):
    """seq = [(width, intent_closed, hand_z)]。発火の事象の並びと、持ち始め・放しの並びを返す。"""
    fires, marks = [], []
    for j, (w, c, z) in enumerate(seq):
        ev = det.update(t0 + j * dt, w, c, [0.0, 0.0, z])
        if det.last is not None:
            marks.append(det.last["kind"])
        if ev is not None:
            fires.append(ev)
    return fires, marks


def grasp_and_lift(n_close=6, n_hold=10, w_hold=0.0388, z0=0.02, lift=0.10):
    """開いた指を閉じて立方体を握り、持ち上げて運ぶまで（閉の指令のまま）。"""
    seq = [(0.08, False, z0)] * 3
    seq += [(0.08 - (0.08 - w_hold) * (i + 1) / n_close, True, z0) for i in range(n_close)]
    seq += [(w_hold, True, z0)] * 3
    seq += [(w_hold, True, z0 + lift * (i + 1) / n_hold) for i in range(n_hold)]
    return seq


def test_hold_is_established_after_lift_and_no_fire_while_carrying():
    det = GraspLossDetector(ReflexParams())
    rng = np.random.default_rng(0)
    seq = grasp_and_lift() + [(0.0388 + float(rng.normal(0, 5e-4)), True, 0.12)] * 40
    fires, marks = feed(det, seq)
    assert fires == [] and marks == ["hold"] and det.state == "holding"


def test_no_hold_without_lift():
    det = GraspLossDetector(ReflexParams())
    seq = grasp_and_lift(lift=0.0) + [(0.0, True, 0.02)] * 4          # 持ち上げないまま開き幅が 0 へ（0.3 s 続く）
    fires, marks = feed(det, seq)
    assert "hold" not in marks
    assert [f["reason"] for f in fires] == ["miss"]                      # 持つ前に空になった = 空掴みの扱い


def test_slip_close_fires_collapse():
    det = GraspLossDetector(ReflexParams())
    seq = grasp_and_lift() + [(0.0388, True, 0.12)] * 3
    seq += [(max(0.0, 0.0388 - 0.008 * (i + 1)), True, 0.12) for i in range(6)]
    fires, _ = feed(det, seq)
    assert len(fires) == 1 and fires[0]["reason"] == "collapse" and fires[0]["w"] < fires[0]["w_ref"] - 0.004
    assert det.state == "fired"


def test_slip_open_fires_open_and_can_be_switched_off():
    seq = grasp_and_lift() + [(0.0388, True, 0.12)] * 3 + [(0.0388 + 0.008 * (i + 1), True, 0.12) for i in range(5)]
    fires, _ = feed(GraspLossDetector(ReflexParams()), seq)
    assert [f["reason"] for f in fires] == ["open"]
    fires, _ = feed(GraspLossDetector(ReflexParams(fire_on_open=False)), seq)
    assert fires == []


def test_intended_release_does_not_fire():
    det = GraspLossDetector(ReflexParams())
    seq = grasp_and_lift() + [(0.0388, True, 0.12)] * 3
    seq += [(0.0388 + 0.008 * (i + 1), False, 0.12) for i in range(5)] + [(0.08, False, 0.12)] * 5
    fires, marks = feed(det, seq)
    assert fires == [] and marks == ["hold", "release"]


def test_regrasp_after_release_holds_again_without_fire():
    det = GraspLossDetector(ReflexParams())
    seq = grasp_and_lift() + [(0.08, False, 0.12)] * 4 + grasp_and_lift(z0=0.12)
    fires, marks = feed(det, seq)
    assert fires == [] and marks == ["hold", "release", "hold"]


def test_diagonal_grasp_that_settles_before_lift_does_not_fire():
    """斜めに握った（5.5 cm）あと、持ち上げる前に平らに座り直す（4.0 cm）。持つ判定は持ち上げの後なので発火しない。"""
    det = GraspLossDetector(ReflexParams())
    seq = [(0.08, False, 0.02)] * 2 + [(0.0554, True, 0.02)] * 6 + [(0.040, True, 0.02)] * 3
    seq += [(0.0388, True, 0.02 + 0.01 * (i + 1)) for i in range(8)] + [(0.0388, True, 0.10)] * 10
    fires, marks = feed(det, seq)
    assert fires == [] and marks == ["hold"]


def test_miss_needs_persistence_and_can_be_switched_off():
    p = ReflexParams()
    closing = [(0.08, False, 0.02)] + [(w, True, 0.02) for w in (0.07, 0.06, 0.05, 0.04, 0.03, 0.02)]
    quick = closing + [(0.0, True, 0.02)] * 1 + [(0.01, False, 0.02)] * 3       # 方策がすぐ開き直す（0.1 s）
    fires, _ = feed(GraspLossDetector(p), quick)
    assert fires == []
    slow = closing + [(0.0, True, 0.02)] * 4
    fires, _ = feed(GraspLossDetector(p), slow)
    assert [f["reason"] for f in fires] == ["miss"] and fires[0]["closing_s"] > 0
    fires, _ = feed(GraspLossDetector(ReflexParams(fire_on_miss=False)), slow)
    assert fires == []


def test_params_validation_and_roundtrip():
    p = ReflexParams.from_dict({"stop": "freeze", "loss_m": 0.005})
    assert ReflexParams.from_dict(p.to_dict()) == p
    for bad in ({"nope": 1}, {"stop": "teleport"}, {"hold_lo_m": 0.05, "hold_hi_m": 0.04}, {"miss_m": 0.04}):
        with pytest.raises(ValueError):
            ReflexParams.from_dict(bad)


# ---------------------------------------------------------------- 発火の後（止め方・待ち・再開）
def test_reflex_hold_action_wait_and_resume():
    rx = GraspLossReflex(ReflexParams(cue_stable_m=0.01))
    seq = grasp_and_lift() + [(0.0388, True, 0.12)] * 3
    t = 0.0
    for w, c, z in seq:
        assert not rx.observe(t, 0, w, c, hand=[0.3, 0.0, z], x_cmd=[0.33, 0.0, z])
        t += 0.1
    assert rx.observe(t, 7, 0.030, True, hand=[0.30, 0.0, 0.12], x_cmd=[0.335, 0.0, 0.12], sample_t=t - 0.02)
    assert rx.overriding and rx.events[0]["reason"] == "collapse" and rx.events[0]["target"] == [0.3, 0.0, 0.12]
    a = rx.hold_action([0.335, 0.0, 0.12])                       # 測った手先へ戻す（上限 0.25 m/s × 0.1 s）
    assert a[6] == -1.0 and np.allclose(a[:3], [-0.025, 0.0, 0.0])
    assert np.allclose(rx.hold_action([0.301, 0.0, 0.12])[:3], [-0.001, 0.0, 0.0])
    tf = t
    assert rx.blocking(tf + 0.2)                                 # 待ちの最短 0.5 s の前
    rx.observe(tf + 0.6, 8, 0.05, True)
    assert rx.blocking(tf + 0.6)                                 # 指がまだ開いていない
    rx.observe(tf + 0.9, 9, 0.078, True)
    assert not rx.blocking(tf + 0.9) and rx.mode == "wait_chunk"
    assert not rx.accept_chunk(tf + 1.2, 12, 40, cue=[0.4, 0.1, 1.0])     # 目標位置の入力の 1 つ目は比べる相手がない
    assert not rx.accept_chunk(tf + 1.5, 15, 41, cue=[0.43, 0.1, 1.0])    # 3 cm 動いた（まだ転がっている）
    assert rx.accept_chunk(tf + 1.8, 18, 42, cue=[0.432, 0.1, 1.0])
    ev = rx.events[0]
    assert ev["n_rejected"] == 2 and ev["resume_chunk"] == 42 and ev["settle_reason"] == "opened"
    assert rx.mode == "normal" and rx.det.state == "open"
    s = rx.summary()
    assert s["n_fire"] == 1 and s["n_hold"] == 1 and json.dumps(s)


def test_reflex_freeze_stop_and_max_wait():
    rx = GraspLossReflex(ReflexParams(stop="freeze", max_wait_s=1.0))
    for j, (w, c, z) in enumerate(grasp_and_lift() + [(0.0388, True, 0.12)] * 3):
        rx.observe(j * 0.1, j, w, c, hand=[0.3, 0, z], x_cmd=[0.33, 0, z])
    assert rx.observe(5.0, 50, 0.02, True, hand=[0.3, 0, 0.12], x_cmd=[0.33, 0, 0.12])
    assert np.allclose(rx.hold_action([0.33, 0, 0.12])[:3], 0.0)            # 参照をその場で止める
    assert rx.blocking(5.5)                                                 # 開き幅は 0.02 のまま
    assert not rx.blocking(6.0) and rx.events[0]["settle_reason"] == "max_wait"


# ---------------------------------------------------------------- 実行系への組み込み（偽の口・偽の方策）
class FakeMotion:
    def __init__(self):
        self.x = np.array([0.4, 0.0, 0.02])
        self.vel = np.zeros(3)
        self.lag_stats = {}
        self.n_cart_clipped = self.n_xcmd_leashed = self.n_cart_unresolved = 0
        self.xcmd_leash_m = None
        self.limiter = SimpleNamespace(n_pos_clipped=0, n_clipped=0)

    def reset(self, joints):
        self.x = np.asarray(joints.q[:3], float).copy()

    def set_velocity(self, v):
        self.vel = np.asarray(v, float).copy()

    def set_command_filter(self, fn):
        pass

    @property
    def x_cmd(self):
        return self.x.copy()

    def hand_pose(self, q):
        return np.asarray(q[:3], float).copy(), np.array([1.0, 0, 0, 0])

    def step(self, joints):
        self.x = self.x + self.vel * 0.002
        q = np.zeros(7)
        q[:3] = self.x - 0.3 * self.vel * 0.1                  # 腕は参照に少し遅延する
        return [q, q]


class FakeFuture:
    def __init__(self, t_ready, value):
        self.t_ready, self.latency, self.wall_s, self.value = t_ready, 0.3, 0.0, value

    def ready(self, t):
        return t >= self.t_ready - 1e-9

    def result(self, t):
        assert self.ready(t)
        return self.value


class FakeIO:
    """2 ms ごとに進む時計、10 Hz で読めるハンド（立方体があれば 3.88 cm で止まる。grasp は閉じ続ける）。"""

    def __init__(self, drop_at=None):
        self.t, self.q = 0.0, np.r_[0.4, 0.0, 0.02, np.zeros(4)]
        self.w, self.mode, self.cube = 0.08, "move", True
        self.drop_at = drop_at
        self.grip = SimpleNamespace(t=0.0, width=0.08, is_grasped=False)
        self.cmds, self.grip_cmds, self.n_compute = [], [], 0

    def now(self):
        return self.t

    def advance(self):
        dt = 0.002
        if self.drop_at is not None and self.t >= self.drop_at:
            self.cube = False
        goal = 0.08 if self.mode == "move" else (0.0388 if self.cube else 0.0)
        self.w += float(np.clip(goal - self.w, -0.08 * dt, 0.08 * dt))
        self.t = round(self.t + dt, 6)
        if abs((self.t * 10) % 1.0 - 0.5) < 1e-6:            # 10 Hz（位相 0.05 s）
            self.grip = SimpleNamespace(t=self.t, width=round(self.w, 4), is_grasped=False)

    def sense(self, cameras=True):
        cams = {n: SimpleNamespace(t_capture=self.t) for n in ("overhead", "wrist")} if cameras else {}
        return SimpleNamespace(t=self.t, joints=SimpleNamespace(q=self.q.copy(), dq=np.zeros(7), q_d=None),
                               gripper=self.grip, cameras=cams)

    def command_joints(self, q):
        self.q = np.asarray(q, float).copy()
        self.cmds.append(self.q.copy())

    def gripper_grasp(self, *a):
        self.mode = "grasp"
        self.grip_cmds.append((self.t, "grasp"))

    def gripper_move(self, *a):
        self.mode = "move"
        self.grip_cmds.append((self.t, "move"))

    def compute(self, kind, fn):
        self.n_compute += 1
        return FakeFuture(self.t + 0.3, fn())


class FakePolicy:
    """閉じたまま上がって箱の向き（+x）へ運ぶ行動列を返す偽の方策。"""

    def __init__(self):
        self.last_raw, self.last_cue = None, None

    def start_trial(self, seed):
        pass

    def reset_cue(self):
        pass

    def observe(self, sensor, task):
        return {"observation.state": np.zeros(8)}

    def infer(self, obs, gen, rtc):
        post = np.zeros((50, 7))
        post[:, 6] = 1.0
        post[:, 2] = 0.006
        post[10:, 0] = 0.012
        self.last_raw = post.copy()
        self.last_cue = np.array([0.4, 0.0, 1.0])
        return post


def run_fake(reflex="absent", drop_at=None, seconds=6.0, monkeypatch=None):
    monkeypatch.setattr(RUN, "noise_generator", lambda seed, i: None)
    io = FakeIO(drop_at)
    setup = SimpleNamespace(cube_size=0.04, gripper_speed=0.08, grasp_force=40.0)
    kw = {} if reflex == "absent" else {"reflex": reflex}
    rt = RUN.PolicyRuntime(io, setup, FakePolicy(), mode="naive", s=6, motion=FakeMotion(), **kw)
    rt.start("pick up the red cube", 1)
    for _ in range(int(round(seconds / 0.002))):
        rt.tick()
        io.advance()
    return rt, io


def _acts(rt):
    return [(k, round(t, 6), c, h, tuple(np.round(a, 12))) for k, t, c, h, a in rt.log_act]


def test_default_off_is_identical(monkeypatch):
    a, ia = run_fake("absent", drop_at=3.0, monkeypatch=monkeypatch)
    b, ib = run_fake(None, drop_at=3.0, monkeypatch=monkeypatch)
    assert _acts(a) == _acts(b) and ia.grip_cmds == ib.grip_cmds
    assert all(np.array_equal(x, y) for x, y in zip(ia.cmds, ib.cmds)) and len(ia.cmds) == len(ib.cmds)
    assert "reflex" not in a.trace() and json.dumps(a.trace(), default=str) == json.dumps(b.trace(), default=str)


def test_on_without_loss_changes_nothing_but_the_record(monkeypatch):
    off, io_off = run_fake(None, drop_at=None, monkeypatch=monkeypatch)
    on, io_on = run_fake(GraspLossReflex(ReflexParams()), drop_at=None, monkeypatch=monkeypatch)
    assert _acts(off) == _acts(on) and io_off.grip_cmds == io_on.grip_cmds
    assert all(np.array_equal(x, y) for x, y in zip(io_off.cmds, io_on.cmds))
    tr = on.trace()
    assert tr["reflex"]["n_fire"] == 0 and tr["reflex"]["n_hold"] == 1
    assert json.dumps({k: v for k, v in tr.items() if k != "reflex"}, default=str) == json.dumps(off.trace(), default=str)


def test_on_drop_discards_chunk_stops_opens_and_replans(monkeypatch):
    rx = GraspLossReflex(ReflexParams())
    rt, io = run_fake(rx, drop_at=3.0, seconds=6.0, monkeypatch=monkeypatch)
    ev = rx.events[0]
    assert ev["reason"] == "collapse" and 3.0 < ev["t"] < 3.3
    assert ev["discarded"]["active"] is not None
    assert any(c == "move" and abs(t - ev["t"]) < 0.003 for t, c in io.grip_cmds)        # その場で指を開く
    t_fire = ev["t"]
    rows = [(t, c, h, a) for k, t, c, h, a in rt.log_act if t_fire < t < ev["t_resume"] - 1e-9]
    assert rows and all(h and c is None and a[6] == -1.0 for t, c, h, a in rows)        # 待ちの間は止めて開いたまま
    starts = [e["t_obs"] for e in rt.log_inf if e["t_obs"] > t_fire]
    assert starts and starts[0] >= ev["t_settled"] - 1e-9 >= t_fire + 0.5 - 1e-9        # 待ちの後に新しい観測から推論
    assert ev["settled_w"] >= 0.075 and ev["hold_s"] > 0.8
    assert abs(np.asarray(ev["target"]) - np.asarray(ev["hand"])).max() < 1e-12          # 測った手先で止める
    tr = rt.trace()["reflex"]
    assert tr["n_fire"] >= 1 and tr["events"][0]["t_resume"] is not None
    # 行動の記録は実際に出した指令（止めの指令）。発火の時刻の行があり、捨てた方策の行動ではない
    fire_rows = [r for r in rt.log_act if abs(r[1] - t_fire) < 1e-9]
    assert len(fire_rows) == 1 and fire_rows[0][3] and fire_rows[0][2] is None
    assert np.allclose(fire_rows[0][4], ev["a_sent"]) and fire_rows[0][4][6] == -1.0
    assert ev["log"] in ("appended_row", "replaced_boundary_row")


def test_fire_on_a_boundary_tick_replaces_the_logged_policy_action(monkeypatch):
    """区切りと同じ手で発火したら、その区切りの記録（捨てた方策の行動）を止めの指令に置き換える。"""
    monkeypatch.setattr(RUN, "noise_generator", lambda seed, i: None)
    io = FakeIO(None)
    rx = GraspLossReflex(ReflexParams())
    rt = RUN.PolicyRuntime(io, SimpleNamespace(cube_size=0.04, gripper_speed=0.08, grasp_force=40.0), FakePolicy(),
                           mode="naive", s=6, motion=FakeMotion(), reflex=rx)
    rt.start("pick up the red cube", 1)
    for _ in range(int(round(3.0 / 0.002))):
        rt.tick()
        io.advance()
    assert rx.det.state == "holding"
    while rt.n_tick % RUN.STEPS_PER_ACTION != 0:                          # 次の手が区切りになるまで進める
        rt.tick()
        io.advance()
    io.grip = SimpleNamespace(t=io.t, width=0.0, is_grasped=False)        # 区切りの手に、空になった読みが届く
    rt.tick()                                                             # 区切りの行動を出した後、同じ手で発火
    ev = rx.events[0]
    assert ev["log"] == "replaced_boundary_row"
    last = rt.log_act[-1]
    assert last[0] == rt.k and abs(last[1] - ev["t"]) < 1e-9 and last[3] and last[2] is None and last[4][6] == -1.0
    assert sum(1 for r in rt.log_act if r[0] == rt.k) == 1


def test_runtime_reflex_file_passes_g1_boundary():
    g = _load(ROOT / "scripts" / "check_real_boundary.py", "check_g1_boundary_reflex")
    src = (ROOT / "src" / "recovla" / "runtime" / "reflex.py").read_text(encoding="utf-8")
    assert not g.scan_runtime(src, "src/recovla/runtime/reflex.py")
    runner_src = (ROOT / "src" / "recovla" / "runtime" / "runner.py").read_text(encoding="utf-8")
    assert not g.scan_runtime(runner_src, "src/recovla/runtime/runner.py")
    assert "src/recovla/runtime/reflex.py" in g.runtime_v2_files(None)


# ---------------------------------------------------------------- eval_resumable.py の --reflex
@pytest.fixture(scope="module")
def r96():
    return _load(ROOT / "scripts" / "eval_resumable.py", "s4_resume_reflex_test")


BASE_ARGS = ["run", "--experiment", "X", "--condition", "c", "--model", "R1v3", "--trials", "natural:1:1"]


def test_96_reflex_off_keeps_spec_and_run_json(r96):
    a = r96.build_parser().parse_args(BASE_ARGS)
    a.world_per_trial = True
    assert r96.reflex_config(a) is None and "reflex" not in r96.make_spec(a)
    assert "reflex" not in json.loads(r96.run_json_text(a, "run", [], 0.0, 6, {"time_limit_s": 60.0}))


def test_96_reflex_on_records_config(r96):
    a = r96.build_parser().parse_args(BASE_ARGS + ["--reflex", "full", "--reflex-set", "stop=freeze", "--reflex-set", "settle_s=0.6"])
    a.world_per_trial = True
    cfg = r96.reflex_config(a)
    p = cfg["params"]
    assert cfg["preset"] == "full" and cfg["overrides"] == {"stop": "freeze", "settle_s": 0.6}
    assert p["stop"] == "freeze" and p["settle_s"] == 0.6 and p["loss_m"] == ReflexParams().loss_m and p["fire_on_miss"]
    assert r96.make_spec(a)["reflex"] == cfg
    assert json.loads(r96.run_json_text(a, "run", [], 0.0, 6, {"time_limit_s": 60.0}))["reflex"] == cfg
    for bad in (["--reflex-set", "x=1"], ["--reflex-set", "novalue"]):
        with pytest.raises(SystemExit):
            r96.reflex_config(r96.build_parser().parse_args(BASE_ARGS + ["--reflex", "full"] + bad))
    with pytest.raises(SystemExit):                                         # 名前のない --reflex は受け付けない
        r96.build_parser().parse_args(BASE_ARGS + ["--reflex"])


def test_96_drop_only_preset_is_explicit_in_run_json(r96):
    """主に使う設定: 落下だけ（collapse・open、持ち上げの確認の後）。空掴み（miss）は使えるが切。"""
    a = r96.build_parser().parse_args(BASE_ARGS + ["--reflex", "drop_only"])
    a.world_per_trial = True
    d = json.loads(r96.run_json_text(a, "run", [], 0.0, 6, {"time_limit_s": 60.0}))["reflex"]
    assert d["preset"] == "drop_only" and d["params"]["fire_on_miss"] is False and d["overrides"] == {}
    assert d["params"]["fire_on_open"] is True and d["params"]["min_lift_m"] > 0
    from recovla.runtime.reflex import PRESETS, preset_params
    assert set(PRESETS) == {"drop_only", "full"} and preset_params("full").fire_on_miss
    with pytest.raises(ValueError):
        preset_params("nope")
    with pytest.raises(SystemExit):                                         # --reflex なしの --reflex-set
        r96.reflex_config(r96.build_parser().parse_args(BASE_ARGS + ["--reflex-set", "stop=freeze"]))


def test_96_spec_check_refuses_mixing_reflex_on_and_off(r96, tmp_path):
    off = r96.build_parser().parse_args(BASE_ARGS)
    on = r96.build_parser().parse_args(BASE_ARGS + ["--reflex", "drop_only"])
    full = r96.build_parser().parse_args(BASE_ARGS + ["--reflex", "full"])
    for x in (off, on, full):
        x.world_per_trial = True
    r96.check_spec(off, tmp_path, r96.make_spec(off))                       # 控えを書く（反射なし）
    r96.check_spec(off, tmp_path, r96.make_spec(off))                       # 同じなら通る
    with pytest.raises(SystemExit):
        r96.check_spec(on, tmp_path, r96.make_spec(on))                     # 控えに無い反射を入れて続けない
    d2 = tmp_path / "on"
    d2.mkdir()
    r96.check_spec(on, d2, r96.make_spec(on))
    with pytest.raises(SystemExit):
        r96.check_spec(full, d2, r96.make_spec(full))                       # 設定の名前が違っても混ぜない


# ---------------------------------------------------------------- 引き抜きの意図的な失敗 P2S（小さなシミュレーション）
def test_pullout_params_and_no_action_override():
    from recovla.eval.induce import Inducer
    from recovla.eval.induce_slip import PullOutInducer
    from recovla.sim import scene
    lay = scene.sample_layout(53000, "empty", start="home")
    t = lay.table_colors[0]
    a, b = PullOutInducer(53000, lay, t, rig=None), PullOutInducer(53000, lay, t, rig=None)
    assert a.params == b.params and a.params["u"] == Inducer("P2", 53000, lay, t, rig=None).params["u"]
    p = a.params["pull"]
    assert p["side"] in (-1.0, 1.0) and 0.0 <= p["down_deg"] <= 60.0 and 0.2 <= p["speed_m_s"] <= 0.5
    assert a.record()["variant"] == "pull_out" and a.record()["kind"] == "P2"


def _carry_truths(target=0, n=60):
    """立方体を持って箱（+y）へ運ぶ真値の並び（発動の判定だけに使う。シミュレーションなし）。"""
    from recovla.expert import script as S
    from recovla.sim import frames
    out = []
    for j in range(n):
        y = -0.25 + 0.45 * j / (n - 1)
        tip = np.array([0.45, y, frames.CUBE_REST_Z + 0.10])
        cp = np.array([[0.30, 0.10, frames.CUBE_REST_Z], [0.60, -0.10, frames.CUBE_REST_Z], [0.35, -0.20, frames.CUBE_REST_Z]])
        cp[target] = tip
        out.append(S.Truth(t=round(0.1 * j, 3), target=target, cube_pos=cp, cube_quat=np.tile([1.0, 0, 0, 0], (3, 1)),
                           cube_linvel=np.zeros((3, 3)), fingers=np.array([0.0194, 0.0194]), finger_vel=np.zeros(2),
                           gripper_closed=True, hand_pos=tip + [0, 0, 0.1034], hand_vel=np.zeros(3), fingertip=tip,
                           x_cmd=tip + [0, 0, 0.1034], box=np.array([0.45, 0.20, 0.0])))
    return out


def test_p2_and_p2s_fire_at_the_same_point_on_the_same_seed():
    """同じシード・同じ並びなら、P2 と P2S は同じ u・同じしきい値で、同じ区切りに発動する（対にできる）。P2 は指を開き、P2S は開かない。"""
    from recovla.common.seeds import COLORS
    from recovla.eval.induce import Inducer
    from recovla.eval.induce_slip import PullOutInducer
    from recovla.sim import scene
    fake = SimpleNamespace(data=SimpleNamespace(xmat=np.tile(np.eye(3).reshape(1, 9), (4, 1)), time=0.0), hand_id=0,
                           model=SimpleNamespace(body_mass=np.full(4, 0.05)), cube_ids=np.array([1, 2, 3]))
    for seed in (191510, 191511, 191512, 191513):
        lay = scene.sample_layout(seed)
        tgt = COLORS[0]
        p2, p2s = Inducer("P2", seed, lay, tgt, fake), PullOutInducer(seed, lay, tgt, fake)
        assert p2.params["u"] == p2s.params["u"]
        k2 = ks = None
        for k, tr in enumerate(_carry_truths(COLORS.index(tgt))):
            a = np.zeros(7)
            a[6] = 1.0
            b2, bs = p2.filter(k, a, tr), p2s.filter(k, a, tr)
            if p2.fired and k2 is None:
                k2 = k
                assert b2[6] == -1.0
            if p2s.fired and ks is None:
                ks = k
                assert bs[6] == 1.0 and not p2s.active
        assert k2 is not None and k2 == ks and p2.t_fire == p2s.t_fire
        assert p2.info["threshold_m"] == p2s.info["threshold_m"] and p2.params["r0_m"] == p2s.params["r0_m"]


def test_96_p2_and_p2s_get_the_same_trial_list():
    """eval_resumable.py の --trials induced:<シード>:<数> は意図的な失敗の種類によらない（目標の色・配置が同じ）。P2 と P2S を同じ指定で回せば対になる。"""
    r41 = _load(ROOT / "scripts" / "eval_subtask_direct.py", "r41_reflex")
    a = r41.trial_list("induced:191510:6")
    assert [(s, t) for s, _, t in a] == [(s, t) for s, _, t in r41.trial_list("induced:191510:6")]

def test_pullout_on_tiny_sim_fingers_close_on_empty():
    """立方体を握った手（家の姿勢、空中）から引き抜く: grasp の指令はそのままで開き幅が 0 へ向かい、立方体は机に落ちる。"""
    import dataclasses
    import mujoco
    from recovla.common import config
    from recovla.eval.induce_slip import PullOutInducer
    from recovla.harness.world import WorldRig
    from recovla.sim import frames, scene
    cfg = config.load_v2()
    act = cfg["actuation"]
    rig = WorldRig(render=False, cfg=cfg, audit_fk=False)
    try:
        m, d = rig.model, rig.data
        seed = 59951
        lay = scene.sample_layout(seed, "empty", start="home")
        rig.reset(lay)
        rig.apply_joint_commands([d.qpos[rig.arm_qadr].copy()])
        tgt = lay.table_colors[0]
        ind = PullOutInducer(seed, lay, tgt, rig, cfg=config.load())
        qa, va = scene.cube_qpos_adr(m, tgt)
        for _ in range(int(0.3 / rig.timestep)):
            rig.physics_step()
        d.qpos[qa:qa + 3] = frames.fingertip_center(d, rig.hand_id)
        d.qpos[qa + 3:qa + 7] = d.xquat[rig.hand_id]
        d.qvel[va:va + 6] = 0
        mujoco.mj_forward(m, d)
        pose = d.qpos[qa:qa + 7].copy()
        rig.hand.grasp(0.04, float(act["gripper_speed"]), float(act["grasp_force"]), 0.005, 0.005)
        for i in range(int(1.0 / rig.timestep)):
            if i < int(0.7 / rig.timestep):                           # 指が閉じるまで立方体を空中に留める
                d.qpos[qa:qa + 7] = pose
                d.qvel[va:va + 6] = 0
            rig.physics_step()
        assert rig.hand.is_grasped() and 0.035 < rig.hand.width() < 0.045

        def truth():
            return dataclasses.replace(rig.truth(tgt), gripper_closed=True)
        ind.fired, ind.t_fire, ind.stage, ind._rest = True, round(float(d.time), 3), "dropped", 0.0
        ind._start(truth())
        w = []
        for i in range(int(2.0 / rig.timestep)):
            ind.pre_physics_step(rig)
            rig.physics_step()
            w.append(rig.hand.width())
            if rig.step % 50 == 0:
                ind.after(0, truth())
        rec = ind.record()
        pull = rec["info"]["pull"]
        assert rig.hand.mode == "grasp"                                   # ハンドの指令は変えていない
        assert pull["end_reason"] == "exit" and pull["t_end"] - pull["t_start"] < 0.3
        assert abs(pull["exit_speed_m_s"] - rec["params"]["pull"]["speed_m_s"]) < 0.1   # 飛ばさない
        assert min(w) < 0.002 and w[int(0.3 / rig.timestep)] < 0.035      # 空を掴んで閉じ切る
        assert abs(d.xpos[rig.cube_ids[ind.rig_target]][2] - frames.CUBE_REST_Z) < 0.005
        assert rec["established"] and rec["reason"] is None and rec["version"] == "P2S-v1"
        # 引いている間の記録（外からの力積・接触力・指先の動き・腕のトルクの変化）
        assert pull["mass_kg"] > 0 and pull["impulse_ns"] > 0 and pull["hand_force_max_n"] > 0
        assert pull["contact_normal_max_n"] > 40.0 and pull["contact_resist_max_n"] > 0     # 40 N/本で握っていた
        assert 0 <= pull["tip_move_max_m"] < 0.02 and pull["arm_torque_dev_max_nm"] > 0
        json.dumps(rec)
    finally:
        rig.close()
