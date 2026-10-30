"""task API の失敗の検知（見張り、api/monitors）と、それを使う実行器の包み（api/executor_monitored）の確かめ。
合成した幅の列（握る→運ぶ→幅が 0 へ／開く／意図した放し／空掴み）で GraspDropMonitor の出来事の有無と時刻、カメラの確かめの
あり・なし、LegacyInHandDropMonitor が executor_replan_early と同じ判定であることを固定する。CPU だけ（偽の io・知覚・判定・LLM）。"""
import importlib.util
import json
import pathlib
import sys
import types

import numpy as np
import pytest

from recovla.api import executor_monitored as XL
from recovla.api import relations as REL
from recovla.api import monitors as M
from recovla.api import perception as AP
from recovla.api import senses as AS
from recovla.api import task_spec as TS
from recovla.planner import replan_event as RL
from recovla.planner import replan_base as R4
from recovla.runtime import executor_replan_early as L2
from recovla.runtime.reflex import ReflexParams
from recovla.runtime.types import GripperState, JointState, SensorFrame

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEC = TS.load_task()
RED = TS.StepSpec("pick_place", "red", "box")
TABLE_RED = np.array([0.45, -0.10, 0.02])
from .kin_world import box_region  # noqa: E402

BOX = AP.PlaceState(np.array([0.45, 0.25]), 0.0, True, None, box_region())
SIG = 0.001


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


T = _load(ROOT / "tests" / "test_replan.py", "t_s4_u4_for_api_monitors")


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.setattr(R4, "CACHE", tmp_path / "llm_cache")


# ---------------------------------------------------------------- 合成の列
def grasp_seq(after, t_event=3.0, lift=True, xy=(0.45, -0.10), n=60):
    """0.1 s ごとの (t, 幅, 閉の指令, 手先)。0.5 s で閉じ、0.6 s から 3.9 cm で止まり、1.0 s から手先が上がる。
    t_event 以降は after(t) の (幅, 閉の指令) にする。"""
    out = []
    for i in range(n):
        t = round(0.1 * i, 3)
        z = 0.115 + (min(0.1, 0.02 * (i - 9)) if (lift and i >= 10) else 0.0)
        hand = np.array([xy[0], xy[1], z])
        if t < 0.5:
            w, closed = 0.08, False
        elif t < 0.6:
            w, closed = 0.06, True
        elif t < t_event - 1e-9:
            w, closed = 0.039, True
        else:
            w, closed = after(t)
        out.append((t, w, closed, hand))
    return out


def senses_at(t, w, hand):
    q = np.zeros(7)
    f = SensorFrame(t, JointState(t, q, np.zeros(7)), GripperState(t, w, False), {})
    return AS.Senses.from_sensor_frame(f, lambda _q, h=hand: (h, h + np.array([0.0, 0.0, -0.1034])))


def est_of(obj_pos, status="seen", in_box=False, near_hand=None, place=BOX, t=0.0, size=0.04):
    objs = {} if obj_pos is None else {"red": AP.ObjState(np.asarray(obj_pos, float), 0.0, status, {"box": in_box}, near_hand, t,
                                                          np.full(3, size), np.eye(3), SIG ** 2 * np.eye(3))}
    return AP.WorldEstimate(t, objs, {"box": place}, 0.0, "estimated")


def feed(mon, seq, est_fn=None, step=RED):
    """列を見張りに与えて、(出来事, 時刻) の一覧を返す。est_fn(t, 手先) は推定（既定は無し）。"""
    mon.start(SPEC, step)
    out = []
    for t, w, closed, hand in seq:
        est = est_fn(t, hand) if est_fn else None
        for ev in mon.observe(senses_at(t, w, hand), est, M.Intent(step, closed, "run"), t):
            out.append((ev.kind, ev.t, ev))
    return out


def test_collapse_while_held_is_dropped_without_camera():
    mon = M.GraspDropMonitor(ReflexParams(), confirm_s=None)
    assert mon.needs == ("gripper", "joints") and not mon.camera
    evs = feed(mon, grasp_seq(lambda t: (0.039 - 0.03 * min(1.0, (t - 2.9) * 2), True)))
    assert [(k, t) for k, t, _ in evs] == [("dropped", 3.0)]
    ev = evs[0][2]
    assert ev.obj == "red" and ev.monitor == "grasp_drop" and ev.evidence["fire"]["reason"] == "collapse"
    assert ev.evidence["fire"]["t_hold"] is not None and ev.evidence["camera"] is None
    assert mon.n_hold == 1 and mon.record()["events"][0]["kind"] == "dropped"
    json.dumps(mon.record())


def test_open_while_closed_command_is_dropped():
    evs = feed(M.GraspDropMonitor(confirm_s=None), grasp_seq(lambda t: (0.06, True)))
    assert [(k, t) for k, t, _ in evs] == [("dropped", 3.0)] and evs[0][2].evidence["fire"]["reason"] == "open"
    evs = feed(M.GraspDropMonitor(ReflexParams(fire_on_open=False), confirm_s=None), grasp_seq(lambda t: (0.06, True)))
    assert evs == []


def test_intended_release_is_not_an_event():
    mon = M.GraspDropMonitor(confirm_s=None)
    assert feed(mon, grasp_seq(lambda t: (0.039 + 0.01 * (t - 2.9), False))) == []
    assert mon.n_release == 1 and mon.n_hold == 1


def test_empty_grasp_is_missed_not_dropped():
    seq = [(t, (0.0 if t >= 0.7 else w), c, h) for t, w, c, h in grasp_seq(lambda t: (0.0, True))]
    evs = feed(M.GraspDropMonitor(confirm_s=None), seq)
    assert [(k, t) for k, t, _ in evs] == [("missed", 1.0)]                 # 0.7 s から 0.3 s（miss_hold_s）続いた
    assert evs[0][2].evidence["fire"]["reason"] == "miss"
    assert feed(M.GraspDropMonitor(ReflexParams(fire_on_miss=False), confirm_s=None), seq) == []


def test_no_lift_means_not_holding():
    """手先が上がらない（持ち上げていない）間は「持っている」にならず、幅が少し落ちても落としたにしない。"""
    mon = M.GraspDropMonitor(confirm_s=None)
    assert feed(mon, grasp_seq(lambda t: (0.03, True), lift=False)) == [] and mon.n_hold == 0


def test_collapse_over_the_place_is_suppressed():
    mon = M.GraspDropMonitor(confirm_s=None)
    seq = grasp_seq(lambda t: (0.0, True), xy=(0.47, 0.27))
    assert feed(mon, seq, lambda t, h: est_of(None)) == []
    assert mon.n_suppressed == 1 and mon.record()["log"][-1]["why"] == "over_place"


def test_camera_confirm_waits_confirm_s_on_the_table():
    mon = M.GraspDropMonitor(confirm_s=1.0, camera_confirm=True)
    assert mon.needs == ("gripper", "joints", "camera/overhead") and mon.camera

    def est(t, hand):
        if t < 3.0:
            return est_of(hand + [0, 0, -0.1034], status="occluded", near_hand=True)          # 手の中
        return est_of(TABLE_RED)                                                         # 机の上に見える
    evs = feed(mon, grasp_seq(lambda t: (0.0, True)), est)
    assert [(k, t) for k, t, _ in evs] == [("dropped", 4.0)]
    ev = evs[0][2]
    assert ev.evidence["t_fire"] == 3.0 and ev.evidence["t_since"] == 3.0 and ev.evidence["camera"]["status"] == "seen"
    assert ev.evidence["camera"]["dist_tip_m"] > ev.evidence["camera"]["radius_m"] == pytest.approx(REL.radius(est_of(TABLE_RED)
                                                                                                         .objects["red"]), abs=1e-4)


def test_camera_confirm_needs_the_object_away_from_the_hand_and_outside_the_place():
    seq = grasp_seq(lambda t: (0.0, True))
    # 発火の後も推定が指先のそば（手の中に写っている）: 出さない
    assert feed(M.GraspDropMonitor(), seq, lambda t, h: est_of(h + [0, 0, -0.1034])) == []
    # 推定が見えない（lost）: 出さない
    assert feed(M.GraspDropMonitor(), seq, lambda t, h: est_of(TABLE_RED, status="lost")) == []
    # 確かめの途中で箱の中に入った: 取り消す
    mon = M.GraspDropMonitor()
    evs = feed(mon, seq, lambda t, h: est_of(TABLE_RED) if t < 3.5 else est_of(TABLE_RED, in_box=True))
    assert evs == [] and mon.n_cancelled == 1
    # 途中で一度見えなくなる: 数え直す（3.0〜3.3 見える、3.4 lost、3.5 から見える → 4.5）
    evs = feed(M.GraspDropMonitor(), seq, lambda t, h: est_of(TABLE_RED, status="lost" if abs(t - 3.4) < 1e-6 else "seen"))
    assert [(k, t) for k, t, _ in evs] == [("dropped", 4.5)]


def test_monitor_set_runs_only_monitors_whose_senses_exist():
    real = AS.standard_registry("real")
    cam, nocam = M.GraspDropMonitor(), M.GraspDropMonitor(confirm_s=None)
    ms = M.MonitorSet([cam, M.LegacyInHandDropMonitor()], real)
    assert ms.active == [] and [s["name"] for s in ms.skipped] == ["grasp_drop", "legacy_in_hand_drop"]
    assert ms.skipped[0]["missing"] == ["camera/overhead"]
    ms = M.MonitorSet([nocam], real)                                        # カメラなしの実機でも、使える見張りは動く
    assert ms.active == [nocam]
    ms.start(SPEC, RED)
    out = []
    for t, w, c, h in grasp_seq(lambda t: (0.0, True)):
        out += ms.observe(senses_at(t, w, h), None, M.Intent(RED, c, "run"), t)
    assert [e.kind for e in out] == ["dropped"]
    rec = ms.record()
    assert rec["target"] == "real" and rec["events"][0]["step"]["obj"] == "red" and rec["monitors"]["grasp_drop"]["n_hold"] == 1
    json.dumps(rec)


def test_detector_restarts_after_fire_when_opened():
    """発火の後に開いたら数え直し、次の把持の落下も見つける（同じ試みの中で 2 回）。"""
    first = grasp_seq(lambda t: (0.0, True), n=35)
    second = [(round(t + 3.5, 3), w, c, h) for t, w, c, h in grasp_seq(lambda t: (0.0, True), n=40)]
    evs = feed(M.GraspDropMonitor(confirm_s=None), first + second)
    assert [(k, t) for k, t, _ in evs] == [("dropped", 3.0), ("dropped", 6.5)]


# ---------------------------------------------------------------- executor_replan_early と同じ判定
def _wm_of(states: dict):
    cubes = {c: types.SimpleNamespace(pos=np.array([0.4, 0.0, 0.02]), yaw=0.0, status=st, in_box=ib, t_seen=0.0)
             for c, (st, ib) in states.items()}
    return types.SimpleNamespace(t=0.0, cubes=cubes, box={"xy": np.array([0.45, 0.25]), "yaw": 0.0, "ok": True})


def test_legacy_monitor_matches_drop_seen_on_random_sequences():
    rng = np.random.default_rng(7)
    finder = AP.ColorWorldModelFinder.from_spec(SPEC)
    n_fire = 0
    for trial in range(300):
        ex = object.__new__(L2.EarlyReplanTaskRuntime)
        ex.prt, ex.steps, ex.drop_confirm_s = types.SimpleNamespace(wm=None), [{"color": "red"}], L2.DROP_CONFIRM_S
        ex._watch_reset()
        mon = M.LegacyInHandDropMonitor()
        mon.start(SPEC, RED)
        p = rng.dirichlet(np.ones(4))
        for k in range(150):
            t = round(0.02 * k * 5, 4)
            u = rng.random()
            if u < 0.03:
                wm = None
            elif u < 0.06:
                wm = _wm_of({"green": ("seen", False)})
            else:
                st = rng.choice(["seen", "held", "in_hand", "lost"], p=p)
                wm = _wm_of({"red": (str(st), bool(rng.random() < 0.15)), "blue": ("seen", False)})
            ex.prt.wm = wm
            legacy = ex._drop_seen(t)
            evs = mon.observe(None, finder.estimate(wm), None, t)
            assert bool(evs) == legacy, (trial, k)
            if legacy:
                assert evs[0].evidence["t_since"] == pytest.approx(ex._drop_since)
                n_fire += 1
                break
    assert n_fire > 30                                                      # 発火する列も十分に含む


# ---------------------------------------------------------------- 実行器の水準（executor_replan_early と executor_monitored）
DOWN = np.array([0.0, 1.0, 0.0, 0.0])


class IO(T.FakeIO):
    """SensorFrame を返す偽の io（幅と時刻は prt の表から）。"""

    def __init__(self, clock):
        super().__init__(clock)
        self.width, self.w_t, self.n_sense = 0.08, 0.0, 0

    def sense(self, cameras=True):
        self.n_sense += 1
        t = self.clock.t
        cams = {n: types.SimpleNamespace(t_capture=t) for n in ("overhead", "wrist")} if cameras else {}
        return SensorFrame(t, JointState(t, np.zeros(7), np.zeros(7)), GripperState(self.w_t, self.width, False), cams)


class Motion(T.FakeMotion):
    def hand_pose(self, q):
        return self.x_cmd.copy(), DOWN.copy()


class SchedPRT(T.FakePRT):
    """手順と時刻の表で、狙いの色の知覚の状態・ハンドの幅・閉の指令・手先の高さを動かす（時刻は試みの始めから）。
    sched: {(手順, 色): [(始め, 終わり, {"status", "width", "closed", "z"})]}。"""
    tip_offset = 0.1034

    def __init__(self, clock, io, sched):
        super().__init__(clock)
        self.motion = Motion(clock)
        for c, e in self.wm.cubes.items():
            e.pos, e.yaw, e.t_seen = np.array([0.45, -0.1, 0.02]), 0.0, 0.0
        self.wm.t = 0.0
        self.io, self.sched, self.ex = io, sched, None

    def tick(self):
        super().tick()
        ex = self.ex
        if ex is None or not ex.steps or ex.phase != "run":
            return
        c = ex.steps[-1]["color"]
        dt = self.clock.t - ex.steps[-1]["attempts"][-1]["t_begin"]
        for a, b, d in self.sched.get((ex.j, c), ()):
            if a <= dt < b:
                if "status" in d:
                    self.wm.cubes[c].status = d["status"]
                if "closed" in d:
                    self.closed = d["closed"]
                if "z" in d:
                    self.motion.x_cmd[2] = d["z"]
                if "width" in d and int(round(self.clock.t / 0.002)) % 50 == 0:
                    self.io.width, self.io.w_t = d["width"], self.clock.t
                return


def make_ex(kind, sched, replan, monitors=None, step_timeout=5.0, max_replans=2):
    clock = T.Clock()
    io = IO(clock)
    prt = SchedPRT(clock, io, sched)
    judge = T.FakeJudge(prt.wm, {"green", "blue"})
    setup = types.SimpleNamespace(retreat_pose=np.array([0.4, 0.0, 0.3]), gripper_speed=0.1, truth=T.Forbidden())
    cfg = json.loads(json.dumps(T.PLANNER_CFG))
    cfg["step_timeout_s"] = step_timeout

    def decompose(text, table, box):
        return {"steps": [c for c in ("red", "green", "blue") if c in table], "reply": "赤・緑・青の順に入れます"}
    from recovla.runtime.executor_replan import ReplanTaskRuntime
    ex = ReplanTaskRuntime(io, setup, prt, judge, cfg, T.MP, decompose, "put the {color} cube in the box", replan,
                           max_replans=max_replans)
    if kind == "L2":
        ex = L2.install(ex, replan, max_replans=max_replans)
    else:
        ex = XL.install(ex, replan, monitors, SPEC, max_replans=max_replans)
    prt.ex = ex
    return ex, io, prt


def replan_fn():
    cli = T.FakeClient(T.raw("next", "red", [], report="赤の立方体を運ぶ途中で落としました。もう一度赤を入れます。"))
    rp = T.recording(lambda **kw: RL.replan_step(**kw, cli=cli, model=RL.DEFAULT_MODEL, use_cache=False))
    rp.takes_event = True
    return rp


DROP_RED = {(0, "red"): [(0.1, 0.5, {"status": "in_hand"}), (0.5, 99, {"status": "seen"})]}


def test_l2v2_with_legacy_monitor_matches_l2():
    out = {}
    for kind in ("L2", "L2v2"):
        rp = replan_fn()
        ms = M.MonitorSet([M.LegacyInHandDropMonitor()]) if kind == "L2v2" else None
        ex, io, prt = make_ex(kind, DROP_RED, rp, ms)
        out[kind] = (T.run(ex), rp.calls)
    (a, ca), (b, cb) = out["L2"], out["L2v2"]
    key = ("event", "t", "t_since", "step", "attempt", "color")
    assert [{k: e[k] for k in key} for e in a["early_events"]] == [{k: e[k] for k in key} for e in b["early_events"]]
    assert a["early_events"] and a["early_events"][0]["event"] == "dropped"
    assert [r["action"] for r in a["replans"]] == [r["action"] for r in b["replans"]]
    assert [r.get("event") for r in a["replans"]] == [r.get("event") for r in b["replans"]]
    assert [s["color"] for s in a["steps"]] == [s["color"] for s in b["steps"]]
    assert [c["event"] for c in ca] == [c["event"] for c in cb]
    assert b["early_events"][0]["monitor"]["monitor"] == "legacy_in_hand_drop"
    assert b["monitors"]["active"] == ["legacy_in_hand_drop"]


def test_l2v2_asks_replan_on_grasp_drop_from_width():
    """幅で見た落下（WorldModel は seen のまま＝executor_replan_early では見つけられない形）で、時間切れを待たずに LLM の組み直しを頼む。"""
    sched = {(0, "red"): [(0.0, 0.3, {"closed": False, "width": 0.08}),
                          (0.3, 0.4, {"closed": True, "width": 0.06}),
                          (0.4, 0.8, {"closed": True, "width": 0.039}),
                          (0.8, 2.0, {"closed": True, "width": 0.039, "z": 0.2}),
                          (2.0, 99, {"closed": True, "width": 0.0, "z": 0.2})]}
    rp = replan_fn()
    ex, io, prt = make_ex("L2v2", sched, rp, M.MonitorSet([M.GraspDropMonitor(confirm_s=None)]))
    rec = T.run(ex)
    ev = rec["early_events"][0]
    assert ev["event"] == "dropped" and ev["step"] == 0 and ev["color"] == "red"
    assert 2.0 <= ev["t"] - 0.0 and ev["t"] < 2.0 + 0.15 + 0.05                  # 幅の読み（10 Hz）の次の制御の刻み
    assert ev["monitor"]["evidence"]["fire"]["reason"] == "collapse"
    assert rp.calls[0]["event"] == "dropped" and rec["replans"][0]["event"] == "dropped"
    assert rec["monitors"]["monitors"]["grasp_drop"]["n_hold"] >= 1
    # 同じ表を executor_replan_early に通すと、WorldModel が in_hand にならないので早い検知は働かない（時間切れの組み直しだけ）
    ex2, _, _ = make_ex("L2", sched, replan_fn())
    assert T.run(ex2)["early_events"] == []


def test_l2v2_does_not_watch_after_replan_limit():
    ex, io, prt = make_ex("L2v2", DROP_RED, T.recording(lambda **kw: pytest.fail("上限なのに呼んだ")),
                          M.MonitorSet([M.LegacyInHandDropMonitor()]), step_timeout=0.4, max_replans=0)
    rec = T.run(ex)
    assert rec["early_events"] == [] and rec["stopped"]["step"] == 0


# ---------------------------------------------------------------- 相対の閾値（api/relations）と HeldEstimateDropMonitor
def test_relations_scale_with_the_estimated_size_and_uncertainty():
    tip = np.array([0.45, -0.10, 0.08])
    small, big = est_of([0.45, -0.10, 0.02]).objects["red"], est_of([0.45, -0.10, 0.02], size=0.08).objects["red"]
    assert REL.radius(small) == pytest.approx(0.02 * 3 ** 0.5) and REL.height(big) == pytest.approx(0.08)
    assert REL.near_hand(small, tip) is False and REL.near_hand(big, tip) is True        # 6 cm 離れ: 大きい物なら手に近い
    assert REL.on_table(small, 0.0) is True and REL.on_table(est_of([0.45, -0.1, 0.06]).objects["red"], 0.0) is False
    assert REL.on_table(est_of([0.45, -0.1, 0.06], size=0.08).objects["red"], 0.0) is True
    noisy = AP.ObjState(small.pos, 0.0, "seen", {}, None, 0.0, small.extent, small.R, (0.02 ** 2) * np.eye(3))
    assert REL.near_hand(noisy, tip) is True                                             # 不確かさが大きいと余白も広い
    inbox = est_of([0.47, 0.27, 0.03]).objects["red"]
    edge = est_of([0.45 + 0.081, 0.25, 0.03]).objects["red"]                             # 内側の領域の 1 mm 外
    assert REL.inside(inbox, BOX) is True and REL.outside(inbox, BOX) is False
    assert REL.inside(edge, BOX) is False and REL.outside(edge, BOX) is False             # 余白（2σ ＋ 2σ_箱）の中: 外と言わない
    assert REL.outside(small, BOX) is True
    bare = AP.ObjState(small.pos, 0.0, "seen", {}, None)
    assert REL.near_hand(bare, tip) is None and REL.on_table(bare, 0.0) is None          # 大きさの無い推定は判定しない
    assert REL.outside(small, AP.PlaceState(np.zeros(2), 0.0, True, 0.08)) is None


def held_est(t_drop=3.0, after="table", near_until=None):
    """推定の列: t_drop まで物は指先（手の中）、その後は after（"table" 机の上・箱の外、"box" 箱の中、"air" 宙、"hand" 手の中のまま）。"""
    near_until = t_drop if near_until is None else near_until

    def f(t, hand):
        tip = hand + [0, 0, -0.1034]
        if t < near_until - 1e-9 or after == "hand":
            return est_of(tip)
        return est_of({"table": TABLE_RED, "box": [0.47, 0.27, 0.03], "air": [0.30, 0.0, 0.10]}[after])
    return f


def test_held_estimate_drop_after_hold_and_on_table_outside():
    mon = M.HeldEstimateDropMonitor()
    assert mon.needs == ("gripper", "joints", "camera/overhead")
    evs = feed(mon, grasp_seq(lambda t: (0.039, True)), held_est())                       # 幅は 3.9 cm のまま（幅では分からない）
    assert [(k, t) for k, t, _ in evs] == [("dropped", 3.5)]                               # 机の上で箱の外が 0.5 s
    ev = evs[0][2]
    assert ev.evidence["t_since"] == 3.0 and ev.evidence["t_armed"] == pytest.approx(2.9) and ev.evidence["depth_m"] < 0
    assert ev.evidence["radius_m"] == pytest.approx(0.0346, abs=1e-4)
    json.dumps(mon.record())


def test_held_estimate_catches_a_drop_with_a_short_open():
    """方策が一瞬指を開いて落とす形（記録の測定で落下の 2/3）。幅の見張りは意図した放しとみなして出さない。"""
    seq = grasp_seq(lambda t: (0.06, abs(t - 3.0) > 0.15))                               # 2.9〜3.1 s だけ開の指令
    assert feed(M.GraspDropMonitor(confirm_s=None), seq, held_est()) == []
    evs = feed(M.HeldEstimateDropMonitor(), seq, held_est())
    assert [(k, t) for k, t, _ in evs] == [("dropped", 3.5)]


@pytest.mark.parametrize("after", ["box", "air", "hand"])
def test_held_estimate_no_event_when_placed_in_air_or_still_held(after):
    assert feed(M.HeldEstimateDropMonitor(), grasp_seq(lambda t: (0.039, True)), held_est(after=after)) == []


def test_held_estimate_needs_hold_and_near_hand_first():
    # 幅では持っていても、推定が一度も手に近くない（別の物を掴んだ）: 出さない
    assert feed(M.HeldEstimateDropMonitor(), grasp_seq(lambda t: (0.039, True), lift=True),
                lambda t, h: est_of([0.30, 0.10, 0.02])) == []
    # 机の上が on_table_s に届く前に手に戻った: 出さない
    def back(t, h):
        return est_of(TABLE_RED) if 3.0 <= t < 3.3 else est_of(h + [0, 0, -0.1034])
    assert feed(M.HeldEstimateDropMonitor(), grasp_seq(lambda t: (0.039, True)), back) == []
