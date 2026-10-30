"""条件 CL2（真値を読まない、手で書いたスクリプト方策）の確かめ: 高さの地図と経路（api/heightmap）、持っているか・落下・完了の
自己判定（api/skills_script_self）、実行器（api/executor_script）、入口（scripts/api/ladder.py）、既知の寸法を読まないこと。
合成の点群・偽の io・偽の世界だけ（CPU）。"""
import ast
import importlib.util
import json
import math
import os
import pathlib
import subprocess
import sys
import types

import numpy as np
import pytest

from recovla.api import executor_script as XC
from recovla.api import geometry as G
from recovla.api import heightmap as HM
from recovla.api import perception as AP
from recovla.api import skills_script_self as CL2
from recovla.api import task_spec as TS
from recovla.common import config

from . import kin_world as K
from . import synth_points as S

ROOT = pathlib.Path(__file__).resolve().parents[2]
CFG = config.load()
E = CFG["expert"]
RULES = CL2.CL2Rules.from_config(E, E["phase"])
SPEC = TS.load_task()
FMT = CFG["convert"]["instruction"]
FRAME = G.TableFrame.from_plane([0.0, 0.0, 1.0], 0.0)
PROF = (HM.Part(0.045, -0.012), HM.Part(0.11, 0.035))           # 指（指先より 1.2 cm 下）と手のひら（3.5 cm 上）


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------- 1. 高さの地図
def table_points(rng, noise=0.0015, step=0.004, x=(0.25, 0.70), y=(-0.20, 0.50)):
    X, Y = np.meshgrid(np.arange(*x, step), np.arange(*y, step))
    return np.column_stack([X.ravel(), Y.ravel(), rng.normal(0, noise, X.size)])


def scene_map(rng, yaw=math.radians(45), roll=math.radians(8), pitch=0.0, outer=(0.085, 0.085), height=0.06, cubes=()):
    """机・箱・立方体の点群（机の座標＝世界）から作った地図と、箱の真の値。"""
    Pb, T = S.container(rng, (0.45, 0.25), yaw, outer, 0.005, height, roll=roll, pitch=pitch)
    pts = [table_points(rng), Pb]
    for c in cubes:
        pts.append(S.cuboid(rng, np.asarray(c, float), np.eye(3), (0.04, 0.04, 0.04)))
    P = np.vstack(pts)
    hm = HM.HeightMap(FRAME, cell_k=2.0, noise=0.0015)
    hm.update([{"pw": P, "h": P[:, 2].copy(), "name": "overhead"}], 0.0)
    return hm, T


def true_rim_max(T, a, b, radius, outer):
    """箱の縁の上面（真の値、雑音なし）のうち、a→b から radius 以内の点の最大の高さ。"""
    hx, hy = outer
    u, v = np.meshgrid(np.linspace(-hx, hx, 120), np.linspace(-hy, hy, 120))
    loc = np.column_stack([u.ravel(), v.ravel(), np.zeros(u.size)])
    rim = loc[np.maximum(np.abs(loc[:, 0]) / hx, np.abs(loc[:, 1]) / hy) >= 1 - 0.005 / min(hx, hy)]
    w = rim @ T["R"].T + T["rim_center"]
    ab = np.asarray(b) - np.asarray(a)
    s = np.clip((w[:, :2] - a) @ ab / (ab @ ab), 0, 1)
    d = np.linalg.norm(w[:, :2] - (np.asarray(a) + s[:, None] * ab), axis=1)
    return float(w[d <= radius, 2].max())


@pytest.mark.parametrize("yaw_deg, roll_deg, outer, height", [(0, 0, (0.085, 0.085), 0.06), (45, 8, (0.085, 0.085), 0.06),
                                                              (20, -8, (0.10, 0.07), 0.045), (70, 5, (0.07, 0.07), 0.08)])
def test_pass_height_over_the_wall_is_rim_plus_object_plus_margin(yaw_deg, roll_deg, outer, height):
    rng = np.random.default_rng(1)
    hm, T = scene_map(rng, math.radians(yaw_deg), math.radians(roll_deg), outer=outer, height=height)
    a, b = np.array([0.45, 0.05]), np.array([0.45, 0.25])             # 箱の外 → 箱の中
    r_held, below = 0.0346, 0.03
    got = HM.pass_tip_height(hm, a, b, (HM.Part(0.0, 0.0),), r_held, below, unc_k=3.0)
    want = true_rim_max(T, a, b, r_held, outer) + below + 3.0 * hm.sigma
    assert got == pytest.approx(want, abs=0.004)
    assert got > T["rim_center"][2] + below                           # 縁を越える


def test_cell_is_a_multiple_of_the_point_spacing():
    rng = np.random.default_rng(0)
    P = table_points(rng, noise=0.0, step=0.004)
    hm = HM.HeightMap(FRAME, cell_k=2.0)
    hm.update([{"pw": P, "h": P[:, 2].copy()}], 0.0)
    assert hm.cell == pytest.approx(0.008, rel=0.01)


def test_tilted_box_is_decided_by_the_high_side():
    rng = np.random.default_rng(2)
    hm, T = scene_map(rng, yaw=0.0, roll=math.radians(8))              # x 軸まわりに傾けた: +y の縁が高い
    lo = HM.pass_tip_height(hm, (0.45, 0.25), (0.45, 0.25 - 0.12), (HM.Part(0.0, 0.0),))
    hi = HM.pass_tip_height(hm, (0.45, 0.25), (0.45, 0.25 + 0.12), (HM.Part(0.0, 0.0),))
    assert hi - lo == pytest.approx(2 * 0.085 * math.sin(math.radians(8)), abs=0.006)


def test_hidden_cells_keep_their_height_and_cleared_cells_drop():
    rng = np.random.default_rng(3)
    cube = (0.35, 0.05, 0.02)
    hm, _ = scene_map(rng, yaw=0.0, roll=0.0, cubes=(cube,))
    wall = ((0.45 - 0.085, 0.25), (0.45 - 0.085, 0.25))
    h_wall = hm.band_max(*wall, 0.01)
    h_cube = hm.band_max(cube[:2], cube[:2], 0.01)
    assert h_wall == pytest.approx(0.06, abs=0.004) and h_cube == pytest.approx(0.04, abs=0.004)
    # 次の時刻: 腕に隠れて壁の一部に点が無い。立方体は持ち去られ、その下の机が見える
    Pt = table_points(rng)
    Pb, _ = S.container(rng, (0.45, 0.25), 0.0, (0.085, 0.085), 0.005, 0.06)
    Pb = Pb[np.linalg.norm(Pb[:, :2] - np.array(wall[0]), axis=1) > 0.04]
    Pt = Pt[np.linalg.norm(Pt[:, :2] - np.array(wall[0]), axis=1) > 0.04]
    P = np.vstack([Pt, Pb])
    hm.update([{"pw": P, "h": P[:, 2].copy(), "name": "overhead"}], 0.1)
    assert hm.band_max(*wall, 0.01) == pytest.approx(h_wall)                   # 前の値を保つ
    assert hm.band_max(cube[:2], cube[:2], 0.01) < 0.005                       # 机に戻った
    snap = hm.snapshot()
    hm.clear()
    assert snap.band_max(*wall, 0.01) == pytest.approx(h_wall)                 # 写しは別の物


def test_held_object_points_are_excluded():
    rng = np.random.default_rng(4)
    tip = np.array([0.40, -0.05, 0.12])
    P = np.vstack([table_points(rng), S.cuboid(rng, tip, np.eye(3), (0.04, 0.04, 0.04))])
    hm = HM.HeightMap(FRAME, noise=0.0015)
    hm.update([{"pw": P, "h": P[:, 2].copy()}], 0.0, exclude=[(tip, 0.04)])
    assert hm.band_max(tip[:2], tip[:2], 0.01) < 0.005


def test_gripper_profile_from_the_robot_model():
    from recovla.harness.setup import nominal_setup                           # テストの道具（既知の情報の公称値）
    from recovla.runtime import robot_model
    su = nominal_setup(config.load(*config.V2_LAYERS))
    prof = HM.gripper_profile(robot_model.load(su.robot_xml), 0.1034)
    below = HM.finger_below(prof)
    assert 0.005 < below < 0.03                                                # 指の先は指先の中心より少し下
    assert max(p.radius for p in prof if p.low < 0) == pytest.approx(0.0675, abs=0.01)   # 開いた指
    closed = HM.gripper_profile(robot_model.load(su.robot_xml), 0.1034, width=0.04)
    assert max(p.radius for p in closed if p.low < 0) < max(p.radius for p in prof if p.low < 0)


# ---------------------------------------------------------------- 2. 経路
def fly(x, goal, hm, r_held=None, below=0.0, profile=PROF, steps=4000, dt=0.005):
    """経路の決まり（毎刻み、今の位置から通る高さを決め直す）で x（指先＝手先とみなす）を goal へ動かし、段の列と最大の高さ。"""
    x = np.asarray(x, float).copy()
    stages, z_move = [], []
    for _ in range(steps):
        pz = HM.pass_tip_height(hm, x[:2], goal[:2], profile, r_held, below)
        g, st = HM.path_goal(x, goal, pz, 0.004, 0.004)
        if not stages or stages[-1] != st:
            stages.append(st)
        if st == "move":
            z_move.append((x.copy(), pz))
        v = 4.0 * (g - x)
        n = np.linalg.norm(v[:2])
        v[:2] *= min(1.0, 0.18 / max(n, 1e-9))
        v[2] = np.clip(v[2], -0.09, 0.09)
        x = x + v * dt
        if np.linalg.norm(x - goal) < 0.001:
            break
    return stages, z_move, x


@pytest.mark.parametrize("start, goal, held", [((0.45, 0.05, 0.025), (0.45, 0.25, 0.09), True),       # 箱の外 → 中
                                               ((0.45, 0.25, 0.05), (0.35, 0.05, 0.03), False),      # 箱の中 → 外
                                               ((0.30, 0.05, 0.03), (0.40, 0.05, 0.03), False)])     # 立方体をまたぐ
def test_path_rises_moves_lowers_above_the_map(start, goal, held):
    rng = np.random.default_rng(5)
    hm, _ = scene_map(rng, yaw=0.0, roll=0.0, cubes=((0.35, 0.05, 0.02),))
    stages, mv, x = fly(start, np.array(goal), hm, 0.0346 if held else None, 0.03 if held else 0.0)
    assert stages == ["rise", "move", "lower"]
    assert np.linalg.norm(x - goal) < 0.002
    band_top = hm.band_max(np.array(start)[:2], np.array(goal)[:2], 0.045)
    first_z = mv[0][0][2]
    assert first_z >= band_top - PROF[0].low - 0.002                          # ②の高さはシード範囲の地図の最大より上
    for p, pz in mv:                                                          # 動いている間ずっと、残りのシード範囲を越えている
        rest = hm.band_max(p[:2], np.array(goal)[:2], 0.045)
        assert p[2] >= rest - PROF[0].low - 0.003


def test_path_goal_stages():
    g, st = HM.path_goal([0.3, 0.0, 0.10], [0.5, 0.0, 0.05], 0.15, 0.005, 0.004)
    assert st == "rise" and np.allclose(g, [0.3, 0.0, 0.15])
    g, st = HM.path_goal([0.3, 0.0, 0.15], [0.5, 0.0, 0.05], 0.15, 0.005, 0.004)
    assert st == "move" and np.allclose(g, [0.5, 0.0, 0.15])
    g, st = HM.path_goal([0.5, 0.0, 0.15], [0.5, 0.0, 0.05], 0.15, 0.005, 0.004)
    assert st == "lower" and np.allclose(g, [0.5, 0.0, 0.05])
    g, st = HM.path_goal([0.3, 0.0, 0.10], [0.5, 0.0, 0.30], 0.15, 0.005, 0.004)      # 行き先が高い: その高さを通る
    assert st == "rise" and g[2] == pytest.approx(0.30)


# ---------------------------------------------------------------- 3・4. 持っているか・落下・完了の自己判定
BOX = AP.PlaceState(np.array([0.45, 0.25]), 0.0, True, None, K.box_region())
CUBE = np.array([0.45, -0.10, 0.02])


def cube(pos, in_box=False, near=None, status="seen"):
    return AP.ObjState(np.asarray(pos, float), 0.0, status, {"box": in_box}, near, 0.0, np.full(3, 0.04), np.eye(3),
                       K.SIGMA ** 2 * np.eye(3))


class W:
    """地図だけの偽の世界（kin_world.heightmap_of を使う）。"""
    held = None

    def __init__(self, cubes):
        self.cubes, self.t = {k: np.asarray(v, float) for k, v in cubes.items()}, 0.0


def obs(t, hand, target, width, closed, cubes=None, held=None):
    hand = np.asarray(hand, float)
    w = W(cubes or {"red": target.pos})
    w.held = held
    return CL2.CL2Obs(t=t, target=target, others={}, place=BOX, place_name="box", closed=closed, width=width, width_t=t,
                      hand=hand, tip=hand + [0, 0, -0.1034], x_cmd=hand, table_z=0.0, est_t=t, hmap=K.heightmap_of(w),
                      home=np.array([0.33, 0.15, 0.30]), ws_z=(0.115, 0.35))


def run_brain(seq, brain=None):
    b = brain or CL2.CL2Brain(RULES, 0.02, lambda w: PROF)
    out = []
    for o in seq:
        out.append(b.step(o).phase)
    return b, out


def grasp_and_carry(t_drop=None, open_at=None, lift=True):
    """0.02 s ごと: 把持の高さで閉じ（0.1 s ごとに幅が 0.039 へ）、1.0 s から持ち上げ、t_drop から幅が 0.01（閉のまま）、
    open_at から開の指令。"""
    seq = []
    for i in range(150):
        t = round(0.02 * i, 3)
        z = 0.1234 + (min(0.1, 0.05 * (t - 1.0)) if (lift and t >= 1.0) else 0.0)
        hand = np.array([0.45, -0.10, z])
        width = 0.08 if t < 0.2 else (0.06 if t < 0.3 else 0.039)
        closed = True
        if t_drop is not None and t >= t_drop:
            width = 0.01
        if open_at is not None and t >= open_at:
            closed, width = False, 0.08
        tgt = cube(hand + [0, 0, -0.1034], near=True, status="occluded") if (t_drop is None or t < t_drop) else cube(CUBE)
        seq.append(obs(t, hand, tgt, width, closed, cubes={"red": CUBE}, held="red" if tgt.status == "occluded" else None))
    return seq


def test_drop_while_carrying_is_recorded_once():
    b, ph = run_brain(grasp_and_carry(t_drop=2.4))
    assert len(b.drops) == 1
    d = b.drops[0]
    assert d["t"] == pytest.approx(2.4) and d["phase_before"] in ("carry", "lift") and d["width"] == 0.01
    assert d["on_table"] is True and d["in_place"] is False and d["obj_w"] == pytest.approx(0.04)
    assert "reopen" in [p.name for p in ph]                                   # 落としたら開いて掴み直しへ


def test_no_drop_on_close_motion_intended_release_or_missed_grasp():
    b, ph = run_brain(grasp_and_carry())                                      # 閉じた直後（指がまだ動いている）も含めて落とさない
    assert b.drops == [] and CL2.CL2Phase.carry in ph
    b, _ = run_brain(grasp_and_carry(open_at=2.4))                            # 放す指令（意図した放し）
    assert b.drops == []
    b, _ = run_brain(grasp_and_carry(t_drop=0.5, lift=False))                 # 持ち上げる前に幅が落ちた（掴み損ね）
    assert b.drops == []


def test_holding_is_relative_to_the_estimated_width():
    b = CL2.CL2Brain(RULES, 0.02, lambda w: PROF)
    big = AP.ObjState(CUBE, 0.0, "seen", {"box": False}, True, 0.0, np.full(3, 0.08), np.eye(3), K.SIGMA ** 2 * np.eye(3))
    o = obs(0.0, CUBE + [0, 0, 0.1034], big, 0.039, True)
    b.step(o)
    assert not b.holding(o)                                                   # 8 cm の物に 3.9 cm の幅: 持っていない
    b2 = CL2.CL2Brain(RULES, 0.02, lambda w: PROF)
    o2 = obs(0.0, CUBE + [0, 0, 0.1034], cube(CUBE, near=True), 0.039, True)
    b2.step(o2)
    assert b2.holding(o2)


def placed_obs(t, hand_z=0.20, width=0.08, closed=False, cube_pos=(0.49, 0.29, 0.03), in_box=True):
    c = np.asarray(cube_pos, float)
    return obs(t, np.array([c[0], c[1], hand_z]), cube(c, in_box=in_box), width, closed, cubes={"red": c})


def settle_phases(make, n=40):
    b = CL2.CL2Brain(RULES, 0.02, lambda w: PROF)
    b.obj_w = 0.04
    return [b.step(make(round(0.02 * i, 3))).phase for i in range(n)], b


def test_step_done_when_inside_rested_open_and_hand_clear():
    ph, b = settle_phases(lambda t: placed_obs(t))
    assert ph[-1] == CL2.CL2Phase.done and b.step_done
    assert ph.index(CL2.CL2Phase.done) * 0.02 >= RULES.rest_hold_s - 0.02      # rest_hold_s だけ止まってから


def test_step_not_done_when_moving_on_the_rim_or_hand_near():
    ph, _ = settle_phases(lambda t: placed_obs(t, cube_pos=(0.49, 0.29, 0.03 + 0.05 * t)))       # まだ動いている
    assert CL2.CL2Phase.done not in ph
    ph, _ = settle_phases(lambda t: placed_obs(t, cube_pos=(0.45 + 0.083, 0.25, 0.08), in_box=False))   # 縁の上
    assert CL2.CL2Phase.done not in ph
    ph, b = settle_phases(lambda t: placed_obs(t, hand_z=0.03 + 0.1034))                          # 手が物のすぐ上
    assert CL2.CL2Phase.done not in ph and ph[-1] == CL2.CL2Phase.clear
    ph, _ = settle_phases(lambda t: placed_obs(t, width=0.042))                                   # 指がまだ開き切っていない
    assert CL2.CL2Phase.done not in ph


# ---------------------------------------------------------------- 5. 実行器
class FakeSkill:
    def __init__(self, clock, est, done_after=None):
        self.clock, self.est = clock, est
        self.done_after = done_after                    # 手順を始めてから済むまでの時間（None なら済まない）
        self.ready, self.stop_reason, self.external = True, None, True
        self.wm = types.SimpleNamespace(geometry=est)
        self.motion = types.SimpleNamespace(set_velocity=lambda v: None, hand_pose=lambda q: (np.zeros(3), None))
        self.tasks, self.drops, self.homed, self.at_home = [], [], False, False
        self.t_task = None

    def start(self, task, seed):
        self.tasks.append(("start", task))

    def set_task(self, task, seed):
        self.tasks.append(("set", task))
        self.t_task = self.clock.t

    def estimate(self):
        return self.est

    @property
    def step_done(self):
        return (self.done_after is not None and self.t_task is not None and not self.external
                and self.clock.t - self.t_task >= self.done_after)

    def go_home(self):
        self.homed = True

    def tick(self):
        self.clock.t += 0.002
        if self.homed:
            self.at_home = True


class FakeIO:
    def __init__(self, clock):
        self.clock, self.calls = clock, []

    def now(self):
        return self.clock.t

    def compute(self, kind, fn):
        self.calls.append(kind)
        return K.Ready(fn())

    def sense(self, cameras=True):
        return types.SimpleNamespace(cameras={"overhead": 1, "wrist": 1} if cameras else {},
                                     joints=types.SimpleNamespace(q=np.zeros(7)))


class FakeJudge:
    def __init__(self, rng):
        self.rng, self.last, self.n = rng, {}, 0

    def reset(self, color):
        pass

    def update(self, sensor, wm, hand):
        self.n += 1
        ok = bool(self.rng.random() < 0.5)
        self.last = {"ok": ok, "held_s": 0.0}
        return ok


def est3(box_ok=True, colors=("red", "green", "blue"), in_box=()):
    objs = {c: cube(CUBE, in_box=c in in_box) for c in colors}
    return AP.WorldEstimate(0.0, objs, {"box": AP.PlaceState(np.zeros(2), 0.0, box_ok, None, K.box_region() if box_ok else None)})


def make_ex(done_after=2.0, judge=None, est=None):
    clock = types.SimpleNamespace(t=0.0)
    sk = FakeSkill(clock, est or est3(), done_after)
    io = FakeIO(clock)
    calls = []

    def decompose(text, table, box):
        calls.append((text, list(table), list(box)))
        return {"steps": [c for c in ("red", "green", "blue") if c in table], "reply": "x"}
    ex = XC.CL2TaskRuntime(io, None, sk, judge, {"step_timeout_s": 30.0, "retry": 1}, decompose, FMT, SPEC, RULES)
    return ex, sk, io, calls, clock


def run_ex(ex, clock, limit=200.0):
    while not ex.finished and clock.t < limit:
        ex.tick()


def test_executor_startup_plan_three_steps_and_home():
    ex, sk, io, calls, clock = make_ex()
    ex.start("全部片付けて", 7)
    run_ex(ex, clock)
    rec = ex.record()
    assert ex.finished and rec["stopped"] is None and rec["ended"]["kind"] == "home" and sk.homed
    assert calls == [("全部片付けて", ["blue", "green", "red"], [])] and io.calls == ["llm"]
    assert [s["color"] for s in rec["steps"]] == ["red", "green", "blue"]
    assert all(s["judged_complete"] and s["end"] == "self" for s in rec["steps"])
    assert [t for k, t in sk.tasks] == [FMT.format(color="red")] + [FMT.format(color=c) for c in ("red", "green", "blue")]
    assert all(2.0 <= s["t_end"] - s["t_start"] < 2.1 for s in rec["steps"])
    assert set(rec) >= {"text", "detected", "plan", "steps", "returns", "stopped", "judge_rows", "startup", "replans",
                        "early_events", "interventions", "cl2"}
    json.dumps(rec, default=str)


def test_executor_moves_on_at_the_time_budget():
    ex, sk, io, calls, clock = make_ex(done_after=None)
    ex.start("全部片付けて", 7)
    run_ex(ex, clock, limit=400.0)
    rec = ex.record()
    assert [s["end"] for s in rec["steps"]] == ["budget"] * 3 and not any(s["judged_complete"] for s in rec["steps"])
    for s in rec["steps"]:
        assert s["t_end"] - s["t_start"] == pytest.approx(60.0, abs=0.03)    # 30 s × （やり直し 1 ＋ 1）
    assert rec["cl2"]["step_budget_s"] == 60.0


def test_judge_v2_is_recorded_but_does_not_change_the_flow():
    base, *_ = make_ex()
    base.start("全部片付けて", 7)
    run_ex(base, base.prt.clock)
    for seed in (1, 2):
        ex, sk, io, calls, clock = make_ex(judge=FakeJudge(np.random.default_rng(seed)))
        ex.start("全部片付けて", 7)
        run_ex(ex, clock)
        rec, ref = ex.record(), base.record()
        assert [(s["color"], s["end"], round(s["t_end"], 3)) for s in rec["steps"]] == \
               [(s["color"], s["end"], round(s["t_end"], 3)) for s in ref["steps"]]
        assert rec["judge_rows"] and all(r["record_only"] for r in rec["judge_rows"])


def test_executor_stops_only_without_box_or_objects():
    ex, *_rest = make_ex(est=est3(box_ok=False))
    clock = ex.prt.clock
    ex.start("全部片付けて", 7)
    run_ex(ex, clock)
    assert "箱らしい物が見つからない" in ex.record()["stopped"]["reply"]
    assert clock.t == pytest.approx(RULES.startup_wait_s, abs=0.03)          # 待つ上限まで待ってから
    ex, *_rest = make_ex(est=est3(colors=()))
    ex.start("全部片付けて", 7)
    run_ex(ex, ex.prt.clock)
    assert "物が 1 つも見えない" in ex.record()["stopped"]["reply"]
    ex, sk, io, calls, clock = make_ex(est=est3(in_box=("green",)))          # 箱の中の色は分解で外す
    ex.start("全部片付けて", 7)
    run_ex(ex, clock)
    assert calls == [("全部片付けて", ["blue", "red"], ["green"])]


def test_cl2_skill_and_executor_in_the_kinematic_world():
    """偽の世界（運動学だけ・地図つき）で、CL2Skill と CL2TaskRuntime が 3 個を別々の置き場所に入れ、始めの姿勢へ戻る。
    1 個目を運ぶ途中で落とすと、落下を記録して掴み直す。"""
    w = K.KinWorld({"red": (0.45, -0.10, 0.02), "green": (0.40, 0.0, 0.02), "blue": (0.50, -0.05, 0.02)})
    w.with_map = True
    sk = CL2.CL2Skill(w, K.setup(), RULES, SPEC, FMT, motion=K.KinMotion(w), profile=PROF)
    ex = XC.CL2TaskRuntime(w, K.setup(), sk, None, {"step_timeout_s": 30.0, "retry": 1},
                           lambda text, table, box: {"steps": ["red", "green", "blue"], "reply": "x"}, FMT, SPEC, RULES)
    st = {"dropped": None}

    def drop(world):
        if st["dropped"] is None and world.held == "red" and sk.last_phase == CL2.CL2Phase.carry and world.hand[2] > 0.19:
            world.drop()
            st["dropped"] = world.t
    ex.start("全部片付けて", 5)
    K.run(ex, w, 150, every=drop, skill=sk, until=lambda: ex.finished)
    rec = ex.record()
    assert ex.finished and rec["stopped"] is None and rec["ended"]["at_home"]
    assert all(s["end"] == "self" for s in rec["steps"])
    for c in ("red", "green", "blue"):
        assert np.max(np.abs(w.cubes[c][:2] - K.BOX_XY)) < K.INNER, c
    assert len({tuple(np.round(w.cubes[c][:2], 2)) for c in w.cubes}) == 3
    assert st["dropped"] is not None and len(rec["cl2"]["drops"]) == 1
    assert rec["cl2"]["drops"][0]["on_table"] is True
    assert np.linalg.norm(w.hand - sk.home) < 0.01
    tr = sk.trace()
    assert tr["script"]["skill"] == "script_cl2" and tr["cl2"]["drops"] == rec["cl2"]["drops"]


def test_cl2_perception_adds_the_map_and_keeps_the_world_model(monkeypatch):
    from recovla.harness.setup import nominal_setup                           # テストの道具（既知の情報の公称値）
    from recovla.runtime import cue as C
    from recovla.runtime.perception import Params, Perception
    from recovla.api import estimated as EST
    cfg = config.load(*config.V2_LAYERS)
    su = nominal_setup(cfg)
    thr = C.Thresholds.from_dict(config.color_detect(cfg))
    rng = np.random.default_rng(9)
    tip = np.array([0.40, -0.05, 0.12])
    held = S.cuboid(rng, tip, np.eye(3), (0.04, 0.04, 0.04))
    Pb, _ = S.container(rng, (0.45, 0.25), 0.0, (0.085, 0.085), 0.005, 0.06)
    pw = np.vstack([table_points(rng), Pb, held])
    col = np.r_[np.full(len(pw) - len(Pb) - len(held), -1), np.full(len(Pb), 3), np.zeros(len(held), int)]
    P = {"pw": pw, "h": pw[:, 2].copy(), "col": col, "plane": (np.array([0.0, 0.0, 1.0]), np.zeros(3))}
    empty = {"pw": np.zeros((0, 3)), "h": np.zeros(0), "col": np.zeros(0, int), "plane": P["plane"]}
    monkeypatch.setattr(Perception, "points", lambda self, fr, w, margin=None: P if fr.name == "overhead" else empty)
    frames = {n: types.SimpleNamespace(name=n, depth=np.zeros((2, 2)), t_capture=1.0) for n in ("overhead", "wrist")}
    grip = types.SimpleNamespace(width=0.039, is_grasped=True)
    p = Params.from_config(cfg["runtime_v2"]["perception"])
    geo = Perception(su, p, thr)
    EST.GeometryPerception.attach(geo, EST.EstimatedGeometryFinder.from_spec(SPEC, geo, su))
    cl2 = CL2.make_perception(Perception(su, p, thr), SPEC, su, RULES)
    a, b = geo.update(frames, grip, 1.0, tip), cl2.update(frames, grip, 1.0, tip)
    assert sorted(a.cubes) == sorted(b.cubes)
    for c in a.cubes:
        assert np.array_equal(a.cubes[c].pos, b.cubes[c].pos) and a.cubes[c].status == b.cubes[c].status
    assert b.geometry.objects["red"].near_hand is True
    assert b.heightmap.band_max(tip[:2], tip[:2], 0.01) < 0.01                 # 持っている物は地図から外した
    assert b.heightmap.band_max((0.45 - 0.085, 0.25), (0.45 - 0.085, 0.25), 0.005) == pytest.approx(0.06, abs=0.005)


# ---------------------------------------------------------------- 6. 入口・ほかの条件の動きが同じ
@pytest.fixture(scope="module")
def ladder():
    return _load(ROOT / "scripts" / "api" / "ladder.py", "api_ladder_for_cl2")


def test_existing_arm_plans_are_unchanged(ladder):
    ref = json.loads((ROOT / "tests" / "api" / "fixtures" / "ladder_plans_main_7629a01.json").read_text(encoding="utf-8"))
    for key, v in ref.items():
        if key.startswith("_"):
            continue
        arm, target, finder = key.split("|")
        assert json.loads(json.dumps(ladder.arm_plan(arm, target, finder), ensure_ascii=False)) == v["plan"], key
        assert ladder.format_plan(ladder.arm_plan(arm, target, finder)) == v["text"], key
    assert list(ladder.FILES) == ref["_FILES"]
    assert {k: v for k, v in ladder.ARMS.items() if k not in ("CL2", "Lf")} == ref["_ARMS"]   # Lf は後から足した条件


def test_cl2_plan(ladder):
    p = ladder.arm_plan("CL2")
    assert p["skill"]["name"] == "script_cl2" and p["finder"] == "estimated" and p["monitors"] is None
    assert "executor_script" in p["executor"] and p["replan"] is None and "decompose_haiku" in p["planner"]
    assert set(ladder.CL2_FILES + ladder.EST_FILES) <= {str(x.relative_to(ROOT)).replace("\\", "/") for x in ROOT.rglob("*")
                                                        if x.is_file() and ".git" not in x.parts}


def test_cl2_refuses_other_finders_and_step_reset(ladder, capsys):
    base = ["run", "--arm", "CL2", "--experiment", "X", "--condition", "y", "--trials", "193900:1", "--model", "Rreal",
            "--dry-run"]
    assert ladder.main(base + ["--finder", "color"]) == 3
    assert ladder.main(base + ["--step-reset", "home"]) == 3


def test_cl2_dry_run_prints_the_plan():
    cond = "pytest_api_dry_CL2"
    out = ROOT / "outputs" / "v2eval" / "APIDRY" / cond
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "api" / "ladder.py"), "run", "--arm", "CL2", "--experiment",
                        "APIDRY", "--condition", cond, "--trials", "193900:2", "--model", "Rreal", "--exec-interval", "6",
                        "--grip-debounce", "5", "--dry-run"], capture_output=True, text=True, cwd=ROOT, timeout=600,
                       encoding="utf-8", errors="replace", env=dict(os.environ, PYTHONUTF8="1"))
    text = r.stdout + r.stderr
    assert r.returncode in (0, 3), text
    assert "[api] 腕 CL2" in text and "executor_script" in text and "[dry-run]" in text and "seed 193900" in text, text
    assert not out.exists()


# ---------------------------------------------------------------- 7. 既知の寸法を読まない・G1（実行系は真値を読まない）
CL2_RUNTIME = ("src/recovla/api/skills_script_self.py", "src/recovla/api/executor_script.py", "src/recovla/api/heightmap.py")
NOT_ALLOWED = ("cube_size", "box_outer", "box_wall", "box_floor", "box_height", "box_nominal_xy", "retreat_pose",
               "approach_height", "carry_z", "release_z", "min_grip_gap", "min_grip_gap_m", "lift_m", "grasp_z",
               "slot_occupied_radius", "box_slots")


def known_reads(src: str) -> list:
    """属性・添字・文字列（getattr の名前）で読む「使ってはいけない」名前と、frames. の読み → [(行, 名前)]。"""
    out = []
    for node in ast.walk(ast.parse(src)):
        name = None
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == "frames":
                name = "frames." + node.attr
            elif node.attr in NOT_ALLOWED:
                name = node.attr
        elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and node.slice.value in NOT_ALLOWED:
            name = node.slice.value
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in NOT_ALLOWED:
            name = node.value
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] + ([node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            if any(m and (m.endswith("frames") or m.startswith("recovla.sim")) for m in mods):
                name = "import:" + ",".join(m for m in mods if m)
        if name:
            out.append((getattr(node, "lineno", 0), name))
    return sorted(set(out))


def test_cl2_runtime_reads_no_known_sizes_or_fixed_heights():
    for f in CL2_RUNTIME + CL2_DEPS:
        assert known_reads((ROOT / f).read_text(encoding="utf-8")) == [], f
    bad = "x = setup.cube_size\ny = e['carry_z']\nz = getattr(r, 'release_z')\nw = frames.BOX_SLOTS\n"
    assert {n for _, n in known_reads(bad)} == {"cube_size", "carry_z", "release_z", "frames.BOX_SLOTS"}


# CL2 が丸ごと使う api の依存（CL2_RUNTIME のほか）。skills_script は条件 CL と共有なので、CL2 が import する名前と、そこから
# 呼ぶ同じファイルの定義だけを見る（ScriptRules・holding などの CL の決まりは既知の高さを読む）
CL2_DEPS = ("src/recovla/api/skills.py", "src/recovla/api/estimated.py", "src/recovla/api/estimated_place.py",
            "src/recovla/api/geometry.py", "src/recovla/api/geometry_place.py", "src/recovla/api/relations.py")
SHARED_SCRIPT = "src/recovla/api/skills_script.py"


def _reachable_defs(src: str, roots) -> dict:
    """ファイルの最上位の定義のうち、roots から名前で辿れるもの → {名前: ノード}。"""
    tree = ast.parse(src)
    defs = {}
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            defs[n.name] = n
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    defs[t.id] = n
    def names(node):
        """node の中で使う名前（型の注釈は呼ばないので除く）。"""
        skip = set()
        for x in ast.walk(node):
            for a in ([x.returns] if isinstance(x, ast.FunctionDef) else []) + (
                    [x.annotation] if isinstance(x, (ast.arg, ast.AnnAssign)) else []):
                skip |= {id(y) for y in ast.walk(a)} if a is not None else set()
        return [x.id for x in ast.walk(node) if isinstance(x, ast.Name) and id(x) not in skip]

    seen, todo = {}, [r for r in roots if r in defs]
    while todo:
        k = todo.pop()
        if k in seen:
            continue
        seen[k] = defs[k]
        todo += [n for n in names(defs[k]) if n in defs and n not in seen]
    return seen


def test_cl2_shared_script_parts_read_no_known_sizes():
    cl2_src = (ROOT / "src/recovla/api/skills_script_self.py").read_text(encoding="utf-8")
    names = [a.name for n in ast.walk(ast.parse(cl2_src)) if isinstance(n, ast.ImportFrom)
             and n.module == "recovla.api.skills_script" for a in n.names]
    assert names, "skills_script_self が skills_script から何も import していない（このテストを見直す）"
    src = (ROOT / SHARED_SCRIPT).read_text(encoding="utf-8")
    used = _reachable_defs(src, names)
    assert set(names) <= set(used) and "ScriptRules" not in used and "holding" not in used, sorted(used)
    for name, node in used.items():
        assert known_reads(ast.unparse(node)) == [], name
    assert known_reads(src) != []                                               # CL の決まりの方には既知の高さがある（検査が効いている）


def test_cl2_files_pass_the_g1_strict_scan_and_main():
    g = _load(ROOT / "scripts" / "check_real_boundary.py", "check_g1_boundary_cl2")
    files = g.runtime_v2_files(None, g.API_DIR)
    for f in CL2_RUNTIME:
        assert f in files
        assert not g.scan_runtime((ROOT / f).read_text(encoding="utf-8"), f, g.API_ALLOWED_IMPORTS, g.API_DENY), f
    import shutil
    git = shutil.which("git") or (str(g.GIT) if pathlib.Path(g.GIT).is_file() else None)   # PATH に無ければ .tools/git
    if git is None or subprocess.run([git, "-C", str(ROOT), "rev-parse", "--verify", "--quiet", "g1-baseline^{commit}"],
                                     capture_output=True).returncode != 0:
        pytest.skip("git かタグ g1-baseline がない")
    g.GIT = git
    assert g.main([]) == 0


# ---------------------------------------------------------------- 8. 試しの試行で見つかった不具合の確かめ
def test_points_below_the_table_and_depth_edges_are_not_mapped():
    rng = np.random.default_rng(6)
    P = table_points(rng)
    floor = np.column_stack([rng.uniform(0.75, 0.9, 500), rng.uniform(0.0, 0.2, 500), np.full(500, -0.7)])   # 机の外の床
    fly = np.array([[0.40, 0.10, 0.30]] * 4)                                   # 縁の空飛ぶ画素（宙の点）
    Q = np.vstack([P, floor, fly])
    edge = np.r_[np.zeros(len(P) + len(floor), bool), np.ones(len(fly), bool)]
    hm = HM.HeightMap(FRAME, noise=0.0015)
    hm.update([{"pw": Q, "h": Q[:, 2].copy(), "edge": edge}], 0.0)
    assert hm.band_max((0.82, 0.1), (0.82, 0.1), 0.02) is None                # 床は地図に入らない
    assert hm.band_max((0.40, 0.10), (0.40, 0.10), 0.01) < 0.005               # 縁の画素は入らない


def test_depth_edges_mark_steps_but_not_flat_or_slanted_surfaces():
    z = np.full((40, 40), 0.80, np.float32)
    z += np.linspace(0.0, 0.004, 40, dtype=np.float32)[None, :]                 # 斜めの面（画素あたり 0.1 mm）
    z[10:20, 10:20] = 0.50                                                      # 手前の物
    e = HM.depth_edges(z, 5 * 0.0015)
    assert e[10, 15] and e[9, 15] and not e[15, 15] and not e[30, 30]


def test_robot_box_points_are_not_mapped_with_a_margin_of_cells():
    rng = np.random.default_rng(7)
    P = table_points(rng)
    arm = rng.uniform([-0.03, -0.03, 0.0], [0.03, 0.03, 0.4], (2000, 3)) + [0.45, 0.10, 0.10]     # 腕の柱
    leak = np.array([[0.45 + 0.03 + 0.004, 0.10, 0.30]] * 3)                  # 箱の外 4 mm（較正の誤差の漏れ）
    Q = np.vstack([P, arm, leak])
    R = np.eye(3)
    boxes = [(np.array([0.45, 0.10, 0.30]), R, np.array([0.03, 0.03, 0.2]))]
    hm = HM.HeightMap(FRAME, noise=0.0015, self_k=2.0)
    hm.update([{"pw": Q, "h": Q[:, 2].copy(), "robot_boxes": boxes}], 0.0)
    assert 2.0 * hm.cell > 0.004
    assert hm.band_max((0.45, 0.10), (0.45, 0.10), 0.045) < 0.005             # 腕も漏れも地図に無い
    yaw = math.radians(30)
    Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    hit = HM.near_boxes(np.array([[0.0, 0.0, 0.0], Rz @ [0.05, 0.0, 0.0], Rz @ [0.0, 0.05, 0.0]]),
                        [(np.zeros(3), Rz, np.array([0.04, 0.01, 0.01]))], 0.012)
    assert hit.tolist() == [True, True, False]                                 # 回した箱の向きで見る


def test_robot_boxes_cover_the_hand_from_the_robot_model():
    import mujoco
    from recovla.harness.setup import nominal_setup                           # テストの道具（既知の情報の公称値）
    from recovla.runtime import robot_model
    m = robot_model.load(nominal_setup(config.load(*config.V2_LAYERS)).robot_xml)
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    boxes = HM.robot_boxes(m, d)
    assert 10 <= len(boxes) <= 30                                              # 当たり判定の geom だけ（見た目の細かい geom は外す）
    hand = d.xpos[m.body("hand").id]
    assert HM.near_boxes(hand[None, :], boxes, 0.0)[0]
    assert not HM.near_boxes(np.array([[hand[0], hand[1], -0.5]]), boxes, 0.01)[0]


def test_held_column_removes_the_object_top_but_keeps_the_floor_below():
    rng = np.random.default_rng(8)
    tip = np.array([0.45, 0.25, 0.10])
    top = S.cuboid(rng, tip + [0.012, 0.012, -0.012], np.eye(3), (0.04, 0.04, 0.04))   # 指先からずれて持った立方体
    P = np.vstack([table_points(rng), top])
    hm = HM.HeightMap(FRAME, noise=0.0015)
    hm.update([{"pw": P, "h": P[:, 2].copy()}], 0.0, exclude=[(tip, 0.036 + 0.008, float(tip[2]) - 0.036)])
    assert hm.band_max(tip[:2], tip[:2], 0.03) < 0.005                         # 立方体の上面も残らない（球なら角が漏れた）
    assert hm.band_max(tip[:2], tip[:2], 0.03) is not None                     # 下の机は見えている


def test_free_slot_moves_off_the_wall_toward_the_box_center():
    b = CL2.CL2Brain(RULES, 0.02, lambda w: PROF)
    b.clocks.obj_h, b.obj_r = 0.04, 0.0346
    import dataclasses
    o = obs(0.0, np.array([0.45, 0.25, 0.20]), cube(CUBE), 0.039, True)
    o = dataclasses.replace(o, hmap=K.heightmap_of(W({"red": CUBE}), cell=0.0025))   # 幅 5 mm の縁が升に入る細かさ
    assert o.hmap.band_max(K.BOX_XY + [K.INNER + 0.0025, 0], K.BOX_XY + [K.INNER + 0.0025, 0], 0.001) == pytest.approx(K.RIM)
    s = np.array([K.BOX_XY[0] + K.INNER - 0.01, K.BOX_XY[1] + 0.02, K.RIM])    # 壁から 1 cm（立方体が縁に載る）
    p = b._free_slot(o, s)
    reach = b.obj_r + o.hmap.cell + RULES.unc_k * 0.001
    assert o.hmap.band_max(p[:2], p[:2], reach) <= 0.01 + RULES.clear_frac * 0.04 + RULES.map_unc_k * o.hmap.sigma
    assert np.hypot(*(p[:2] - K.BOX_XY)) < np.hypot(*(s[:2] - K.BOX_XY))      # 箱の中心へ寄った
    inside = np.array([K.BOX_XY[0] + 0.02, K.BOX_XY[1] + 0.02, K.RIM])        # 十分内側ならそのまま
    assert np.allclose(b._free_slot(o, inside), inside)


def test_cells_above_a_robot_part_are_cleared_but_the_table_below_is_kept():
    rng = np.random.default_rng(9)
    P = table_points(rng)
    stale = np.array([[0.45, 0.10, 0.50]] * 3)                                # 前のフレームに漏れた腕の点（いまは腕に隠れて見えない）
    hm = HM.HeightMap(FRAME, noise=0.0015)
    hm.update([{"pw": np.vstack([P, stale]), "h": np.r_[P[:, 2], stale[:, 2]]}], 0.0)
    assert hm.band_max((0.45, 0.10), (0.45, 0.10), 0.005) == pytest.approx(0.50)
    arm = (np.array([0.45, 0.10, 0.40]), np.eye(3), np.array([0.03, 0.03, 0.05]))   # 下の端 0.35
    Pv = P[np.linalg.norm(P[:, :2] - [0.45, 0.10], axis=1) > 0.05]            # 腕の下は俯瞰から隠れる
    hm.update([{"pw": Pv, "h": Pv[:, 2].copy(), "robot_boxes": [arm]}], 0.1)
    assert hm.band_max((0.45, 0.10), (0.45, 0.10), 0.005) is None or hm.band_max((0.45, 0.10), (0.45, 0.10), 0.005) < 0.005
    assert hm.band_max((0.45, 0.10), (0.45, 0.10), 0.02) < 0.005             # 腕の下の机は前の値のまま


# ---------------------------------------------------------------- 9. 速い地図・推定の道具は、素直な参照の実装と同じ値
_SIGNS = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], float)


def _ref_in_box(pw, c, R, e, margin):
    """near_boxes の参照の判定: 世界の軸の外接箱（|R| e ＋ margin）の中、かつ箱の座標で e ＋ margin の中。"""
    return np.all(np.abs(pw - c) <= np.abs(R) @ e + margin, axis=1) & np.all(np.abs((pw - c) @ R) <= e + margin, axis=1)


def _ref_update(cells, cell, clouds, frame, self_k, below):
    """HeightMap.update の参照の実装（升の辞書・全部の点と全部の箱を素直に比べる）。cells を書き換える。"""
    shadows, qs = [], []
    margin = self_k * cell
    for P in clouds:
        keep = P["h"] >= below
        pw, h = P["pw"][keep], P["h"][keep]
        boxes = P.get("robot_boxes") or []
        if boxes:
            hit = np.zeros(len(pw), bool)
            for c, R, e in boxes:
                hit |= _ref_in_box(pw, c, R, e, margin)
                w = frame.to_table(c + (_SIGNS * e) @ R.T)
                shadows.append((w[:, :2].min(0) - margin, w[:, :2].max(0) + margin, float(w[:, 2].min()) - margin))
            pw, h = pw[~hit], h[~hit]
        q = frame.to_table(pw)
        q[:, 2] = h
        for lo, hi, zb in shadows:
            q = q[~(np.all((q[:, :2] >= lo) & (q[:, :2] <= hi), axis=1) & (q[:, 2] >= zb))]
        qs.append(q)
    q = np.vstack(qs)
    groups = {}
    for k, z in zip(map(tuple, np.floor(q[:, :2] / cell).astype(np.int64).tolist()), q[:, 2].tolist()):
        groups.setdefault(k, []).append(z)
    cells.update({k: sorted(v)[-2] for k, v in groups.items() if len(v) >= 2})
    for k in list(cells):
        x, y = (k[0] + 0.5) * cell, (k[1] + 0.5) * cell
        if any(lo[0] <= x <= hi[0] and lo[1] <= y <= hi[1] and cells[k] >= zb for lo, hi, zb in shadows):
            del cells[k]


def _rot(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def test_grid_map_matches_the_plain_dict_map_with_robot_boxes():
    rng = np.random.default_rng(21)
    frame = G.TableFrame.from_plane([0.02, -0.01, 1.0], 0.01)
    hm = HM.HeightMap(frame, cell=0.008, noise=0.0015, self_k=2.0)
    ref = {}
    for t in range(4):
        clouds = []
        for name in ("overhead", "wrist"):
            P = np.vstack([table_points(rng, step=0.006), S.cuboid(rng, np.r_[rng.uniform([0.3, -0.1], [0.6, 0.4]), 0.02],
                                                                    np.eye(3), (0.04, 0.04, 0.04)),
                           rng.uniform([0.25, -0.2, -0.05], [0.7, 0.5, 0.4], (300, 3))])
            P[:, 2] += 0.01
            boxes = [(np.r_[rng.uniform([0.3, -0.1], [0.6, 0.4]), rng.uniform(0.0, 0.3)], _rot(rng),
                      rng.uniform(0.01, 0.06, 3)) for _ in range(int(rng.integers(0, 6)))]
            clouds.append({"pw": P, "h": frame.to_table(P)[:, 2], "name": name, "robot_boxes": boxes})
        hm.update(clouds, float(t))
        _ref_update(ref, hm.cell, clouds, frame, hm.self_k, -hm.below_k * hm.sigma)
        assert hm.cells == ref, t
    xy = (np.array(list(ref), float) + 0.5) * hm.cell
    h = np.array(list(ref.values()))
    for _ in range(200):                                                       # シード範囲の最大・低い方の百分位も辞書の版と同じ
        a, b, r = rng.uniform([0.25, -0.2], [0.7, 0.5]), rng.uniform([0.25, -0.2], [0.7, 0.5]), float(rng.uniform(0, 0.1))
        reach = r + 0.5 * math.sqrt(2.0) * hm.cell
        s = np.clip((xy - a) @ (b - a) / ((b - a) @ (b - a)), 0.0, 1.0)
        m = np.linalg.norm(xy - (a + s[:, None] * (b - a)), axis=1) <= reach
        assert hm.band_max(a, b, r) == (float(h[m].max()) if m.any() else None)
        m = np.linalg.norm(xy - a, axis=1) <= r
        assert hm.band_low(a, r) == (float(np.percentile(h[m], 20.0)) if m.any() else None)


def test_near_boxes_and_coarse_candidates_keep_every_point_inside():
    rng = np.random.default_rng(22)
    pw = rng.uniform(-0.3, 0.3, (20000, 3))
    boxes = [(rng.uniform(-0.2, 0.2, 3), _rot(rng), rng.uniform(0.005, 0.08, 3)) for _ in range(12)]
    ref = np.zeros(len(pw), bool)
    for c, R, e in boxes:
        ref |= _ref_in_box(pw, c, R, e, 0.01)
    assert np.array_equal(HM.near_boxes(pw, boxes, 0.01), ref) and 100 < ref.sum() < len(pw)


def test_clean_points_and_voxel_down_match_the_plain_versions():
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(23)
    pts = np.vstack([S.cuboid(rng, np.array([0.4, 0.1, 0.02]), np.eye(3), (0.04, 0.04, 0.04)),
                     rng.uniform([0.3, 0.0, 0.0], [0.5, 0.2, 0.1], (40, 3))])
    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=9)
    r = 1.5 * float(np.median(d[:, -1]))
    want = np.asarray(tree.query_ball_point(pts, r, return_length=True)) - 1 >= 4
    got = G.clean_points(pts)
    assert np.array_equal(got, want) and 0 < (~got).sum() < 100
    v = 0.004
    key = np.floor(pts / v).astype(np.int64)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    out = np.zeros((len(cnt), 3))
    np.add.at(out, inv.reshape(-1), pts)
    assert np.array_equal(G.voxel_down(pts, v), out / cnt[:, None])
