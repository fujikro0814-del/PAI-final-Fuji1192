"""別のタスク「カップを皿に置く」の確かめ: 場面（sim/scene の cup_plate・harness/world_cup_plate）、成功の真値（goal_truth の
all_on）、タスクの書式、知覚（api/geometry_place・api/estimated_place・api/relations.in_place）、条件 CL2（掴む高さ・皿の置き場所・
放す高さ・完了・偽の世界で一通り）、計画役（planner/decompose_task）、入口（ladder.py --task）、既知の寸法を読まないこと・
G1（実行系は真値を読まない）。CPU（MuJoCo のモデルは読むが描画はしない）。"""
import ast
import dataclasses
import importlib.util
import json
import math
import os
import pathlib
import subprocess
import sys
import types

import mujoco
import numpy as np
import pytest

from recovla.api import executor_script as XC
from recovla.api import geometry_place as GP
from recovla.api import heightmap as HM
from recovla.api import perception as AP
from recovla.api import relations as REL
from recovla.api import skills_script_self as CL2
from recovla.api import task_spec as TS
from recovla.api.estimated_place import PlaceGeometryFinder, finder_class
from recovla.api.harness import goal_truth as GT
from recovla.common import config
from recovla.sim import scene as S

from . import kin_world as K
from . import synth_points as SP
from .test_script_policy import CL2_RUNTIME, PROF, RULES, known_reads

ROOT = pathlib.Path(__file__).resolve().parents[2]
CFG = config.load()
FMT = CFG["convert"]["instruction"]
SPEC = TS.load_task("configs/tasks/cup_plate.yaml")
TIDY = TS.load_task()
SC = S.cup_plate_config()
PALM = (HM.Part(0.11, 0.0325),)                                  # 手のひら（指先の中心より 3.25 cm 上。ロボットのモデルの値）


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------- 1. 場面
def test_scene_loads_and_matches_the_config():
    m = S.build_model("cup_plate")
    assert m.body("cup").jntnum == 1 and m.body("plate").jntnum == 0
    assert abs(float(m.body_subtreemass[m.body("cup").id]) - 0.06) < 1e-9
    S.check_cup_plate_geometry(m, SC)
    bad = dict(SC, cup=dict(SC["cup"], height=0.09))
    with pytest.raises(RuntimeError):
        S.check_cup_plate_geometry(m, bad)
    for v in ("3cube", "g0"):                                     # 既存の場面は変わらず読める
        S.build_model(v)


def test_cup_and_plate_colors_pass_the_perception_thresholds():
    """カップは赤の閾値、皿は箱の黄の切り出し（runtime/perception.Perception.points と同じ式）に入る。"""
    from recovla.runtime import cue as C
    thr = C.Thresholds.from_dict(config.color_detect(config.load(*config.V2_LAYERS)))
    red = np.array(SC["cup"]["rgba"][:3]) * 255
    others = np.max(np.delete(red, C.CHANNEL["red"]))
    assert red[C.CHANNEL["red"]] >= thr.min_value and red[C.CHANNEL["red"]] - others >= thr.min_margin
    for c in ("green", "blue"):
        assert not red[C.CHANNEL[c]] - np.max(np.delete(red, C.CHANNEL[c])) >= thr.min_margin
    y = np.array(SC["plate"]["rgba"][:3]) * 255
    assert y[0] >= 120 and y[1] >= 90 and y[2] <= 0.6 * min(y[0], y[1])
    box = S.build_model("3cube")
    assert np.allclose(box.geom_rgba[box.geom("box_bottom").id], SC["plate"]["rgba"])   # 箱と同じ黄


def test_layout_is_decided_by_the_seed_and_never_overlaps():
    lay = SC["layout"]
    need = SC["plate"]["radius"] + SC["cup"]["radius"] + lay["min_gap_m"]
    for seed in range(193900, 193960):
        a, b = S.sample_cup_plate_layout(seed, SC), S.sample_cup_plate_layout(seed, SC)
        assert a == b
        assert np.hypot(a.cup[0] - a.plate[0], a.cup[1] - a.plate[1]) >= need
        for key, (x, y, _) in (("plate_region", a.plate), ("cup_region", a.cup)):
            r = lay[key]
            assert r["x"][0] <= x <= r["x"][1] and r["y"][0] <= y <= r["y"][1]
        assert S.CupPlateLayout.from_json(a.to_json()) == a
    assert S.sample_cup_plate_layout(1, SC) != S.sample_cup_plate_layout(2, SC)
    old = S.sample_layout(193900, "empty", start="home")         # 既存の配置は変わらない（同じ乱数列の名前でも別の関数）
    assert old == S.sample_layout(193900, "empty", start="home") and old.kind == "empty"


def _settle(m, d, n=1500):
    for _ in range(n):
        mujoco.mj_step(m, d)


def _truth_of(m, d):
    b, pb = m.body("cup").id, m.body("plate").id
    R = d.xmat[pb].reshape(3, 3)
    place = {"center": d.xpos[pb] + R[:, 2] * SC["plate"]["thickness"], "normal": R[:, 2], "radius": SC["plate"]["radius"]}
    return GT.on_surface_truth(d.xpos[b], d.xquat[b], place, SC["success"])


def test_truth_on_the_plate_off_the_plate_and_fallen():
    m = S.build_model("cup_plate")
    lay = S.CupPlateLayout(0, "home", (0.45, -0.10, 0.3), (0.46, 0.24, 0.0))
    cases = {}
    for name, pos, quat in (("on", (0.47, 0.25, 0.012 + 0.002), (1, 0, 0, 0)),
                            ("table", (0.45, -0.10, 0.0005), (1, 0, 0, 0)),
                            ("edge", (0.46 + 0.085, 0.24, 0.012 + 0.002), (1, 0, 0, 0)),
                            ("fallen", (0.47, 0.25, 0.012 + 0.031), (math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0))):
        d = mujoco.MjData(m)
        S.apply_cup_plate_layout(m, d, lay, SC)
        qa = m.jnt_qposadr[m.body("cup").jntadr[0]]
        d.qpos[qa:qa + 3], d.qpos[qa + 3:qa + 7] = pos, quat
        mujoco.mj_forward(m, d)
        _settle(m, d)
        cases[name] = _truth_of(m, d)
    assert cases["on"]["on"] and cases["on"]["touch"] and cases["on"]["upright"]
    assert not cases["table"]["on"] and not cases["table"]["inside"]
    assert not cases["fallen"]["on"] and not cases["fallen"]["upright"]
    assert not cases["edge"]["on"]                                  # 縁にかかる（底の中心が半径 − 余白の外）か傾いた
    spec_in = {"cup": True}
    assert GT.goal_met_truth(SPEC, spec_in) and not GT.goal_met_truth(SPEC, {"cup": False})


def test_world_rig_resets_the_layout_and_reports_truth():
    from recovla.harness import real_matched as V
    from recovla.harness.world_cup_plate import make_cup_plate_eval_world
    cfg = V.load_config("v3_real", generation=False)
    w, suite = make_cup_plate_eval_world(cfg, spec=SPEC)
    lay = w.sample_layout(193901)
    w.reset(lay)
    bottom, quat, _ = w.object_pose("cup")
    assert np.allclose(bottom[:2], lay.cup[:2], atol=1e-3) and abs(bottom[2]) < 1e-3
    pl = w.place_geometry()
    assert np.allclose(pl["center"][:2], lay.plate[:2]) and pl["top_z"] == pytest.approx(SC["plate"]["thickness"])
    assert w.in_place_truth() == {"cup": False} and w.box is None
    su = w.setup_info(cfg)                                          # VLA の条件の既知の値: 箱の欄に皿（試行の位置は渡さない）
    assert su.box_height == pytest.approx(SC["plate"]["thickness"]) and su.cube_size == pytest.approx(0.06)
    assert su.box_floor + su.cube_size == pytest.approx(SC["plate"]["thickness"] + SC["cup"]["height"])   # 置いた物の上面
    assert w.cue_height() == pytest.approx(0.5 * SC["cup"]["height"])
    assert np.allclose(su.box_nominal_xy, [np.mean(SC["layout"]["plate_region"]["x"]), np.mean(SC["layout"]["plate_region"]["y"])])
    assert 0.5 * su.box_outer - su.box_wall == pytest.approx(SC["plate"]["radius"] / math.sqrt(2))
    tr = w.truth("red")                                             # 生成プログラムが読む真値（物の中心・皿の上面の中心）
    assert tr.target == 0 and np.allclose(tr.target_pos, w.object_center("cup")) and np.allclose(tr.box, pl["center"])
    assert w.truth("cup").target == 0 and w.cubes_in_box() == ()
    suite.close()


def test_place_task_trial_records_final_in_place_without_touching_the_cube_records():
    """run_task_trial は place_task の世界を run_place_task_trial へ渡す。記録に final_in_place・goal_met を足し、評価の
    成功の欄（all_three_in_box）に goal_met を写す。描画しない偽の実行器（何もしない）で 0.3 s。"""
    from recovla.harness import real_matched as V
    from recovla.harness.task_loop import PLACE_TRUTH_KEYS, run_task_trial
    from recovla.harness.world_cup_plate import make_cup_plate_eval_world
    cfg = V.load_config("v3_real", generation=False)
    w, suite = make_cup_plate_eval_world(cfg, spec=SPEC)

    class Idle:
        def __init__(self, io, setup):
            self.io, self.setup, self.finished, self.j = io, setup, False, -1
            self.prt = types.SimpleNamespace(motion=types.SimpleNamespace(x_cmd=np.zeros(3)), closed=False, trace=lambda: {})

        def start(self, text, seed):
            self.text = text

        def tick(self):                                           # 今の指令を保つ（1 ステップで 2 つ。実行系と同じ口）
            q = np.asarray(self.io.sense(cameras=False).joints.q_d, float)
            self.io.command_joints(q)
            self.io.command_joints(q)

        def record(self):
            return {"detected": None, "plan": {"steps": []}, "steps": [], "returns": [], "stopped": None, "startup": None,
                    "judge_rows": []}
    lay = S.sample_layout(193900, "empty", start="home")
    meta, arrays, rlog = run_task_trial(w, suite, Idle, lay, "カップを皿に置いて", 193900, cfg=cfg, time_limit_s=0.3)
    assert meta["task"] == "cup_plate" and meta["layout"]["kind"] == "cup_plate"
    assert meta["final_in_place"] == {"cup": False} and meta["goal_met"] is False
    assert meta["all_three_in_box"] is False and meta["final_in_box"] == {"cup": False}
    assert meta["final_on_detail"]["cup"]["upright"] and meta["vla_known"]["cube_size"] == pytest.approx(0.06)
    assert meta["vla_known"]["cue_plane_z"] == pytest.approx(0.04)
    assert meta["audit"]["g1"]["violations"] == 0 and meta["audit"]["g2"]["world_stops"] == 0
    assert set(arrays) == set(PLACE_TRUTH_KEYS) and arrays["obj_pos"].shape[1:] == (1, 3)
    assert GT.in_place_of_meta(SPEC, meta) == {"cup": False}
    suite.close()


# ---------------------------------------------------------------- 2. タスクの書式
def test_cup_plate_task_spec():
    assert SPEC.goal == TS.GoalSpec("all_on", ("cup",), "plate")
    assert SPEC.object("cup").shape == "cup" and SPEC.object("cup").size_m == ()
    assert SPEC.place("plate").kind == "surface" and TS.finder_kind(SPEC.place("plate").finder) == ("plate", "v1")
    st = TS.StepSpec("pick_place", "cup", "plate")
    assert TS.cue_of(SPEC, st) == "red" and TS.instruction_of(SPEC, st, FMT) == "put the red cube in the box"
    assert TS.step_of_instruction(SPEC, "put the red cube in the box", FMT) == st
    assert TS.policy_format(SPEC, "fixed", FMT) == FMT
    assert TS.policy_format(SPEC, "task", FMT).format(color="red") == "put the red cup on the plate"
    assert TS.policy_format(TIDY, "task", FMT) == FMT                # policy_instruction の無いタスクは決まった形
    from recovla.runtime import cue as C
    assert C.color_of_instruction("put the red cup on the plate") == "red"   # 方策の目標位置の入力の色が読める


def test_task_spec_rejects_a_goal_on_the_wrong_place_kind():
    d = {"name": "x", "objects": [{"name": "cup", "shape": "cup", "finder": "color:red"}],
         "places": [{"name": "plate", "kind": "surface", "finder": "plate:v1"}],
         "goal": {"kind": "all_in", "objects": ["cup"], "place": "plate"}}
    with pytest.raises(ValueError):
        TS.parse_task(d)
    d["goal"]["kind"] = "all_on"
    TS.parse_task(d)
    d["places"][0]["kind"] = "shelf"
    with pytest.raises(ValueError):
        TS.parse_task(d)
    d["places"][0]["kind"] = "surface"
    d["policy_instruction"] = "put the cup on the plate"
    with pytest.raises(ValueError):                                 # {color} の無い書式は断る
        TS.parse_task(d)


def test_goal_met_estimate_uses_in_place_for_all_on():
    region = K.plate_region()
    place = AP.PlaceState(K.PLATE_XY, 0.0, True, None, region)
    on = AP.ObjState(np.array([0.46, 0.24, 0.052]), 0.0, "seen", {"plate": True}, None)
    est = AP.WorldEstimate(0.0, {"cup": on}, {"plate": place})
    assert TS.goal_met(SPEC, est)
    est2 = AP.WorldEstimate(0.0, {"cup": dataclasses.replace(on, in_place={"plate": False})}, {"plate": place})
    assert not TS.goal_met(SPEC, est2)


# ---------------------------------------------------------------- 3. 知覚
def _cut(P, h=0.0045):
    return P[P[:, 2] > h]


@pytest.mark.parametrize("noise, cams", [(0.0015, SP.CAMS), (0.0015, SP.CAMS2), (0.0025, SP.CAMS2)])
def test_fit_cylinder_recovers_radius_height_and_top(noise, cams):
    rng = np.random.default_rng(3)
    P = _cut(SP.cup(rng, (0.45, -0.05, 0.0), cams=cams, noise=noise))
    g = GP.fit_cylinder(P, noise, rng, support_h=0.0, assume_supported=True)
    assert g.radius == pytest.approx(0.03, abs=0.003)
    assert g.height == pytest.approx(0.08, abs=0.003) and g.top_h == pytest.approx(0.08, abs=0.003)
    assert np.allclose(g.center, [0.45, -0.05, 0.04], atol=0.003)
    assert g.open_top and g.bottom_h == pytest.approx(0.0)
    assert np.allclose(g.extent, [2 * g.radius, 2 * g.radius, g.height])


def test_fit_cylinder_on_the_plate_extends_to_the_plate_and_a_fallen_cup_is_not_open():
    rng = np.random.default_rng(4)
    P = _cut(SP.cup(rng, (0.47, 0.25, 0.012), cams=SP.CAMS2))
    g = GP.fit_cylinder(P, 0.0015, rng, support_h=0.012, assume_supported=True)
    assert g.bottom_h == pytest.approx(0.012, abs=0.002) and g.height == pytest.approx(0.08, abs=0.003)
    P = _cut(SP.cup(rng, (0.45, -0.05, 0.03), R=SP.rot(0.3, math.pi / 2, 0.0), cams=SP.CAMS2))
    assert not GP.fit_cylinder(P, 0.0015, rng, assume_supported=True).open_top
    held = SP.cup(rng, (0.45, 0.0, 0.10), cams=SP.CAMS2)              # 宙の物は延ばさない
    g = GP.fit_cylinder(held, 0.0015, rng, support_h=0.0, assume_supported=False)
    assert not g.supported and g.bottom_h > 0.05
    low = SP.cup(rng, (0.45, 0.0, 0.03), cams=SP.CAMS2)               # 持ち上げた直後（隙間 3 cm < 高さの半分）でも、持って
    assert not GP.fit_cylinder(low, 0.0015, rng, assume_supported=False, held=True).supported   # いれば延ばさない


@pytest.mark.parametrize("roll_deg, pitch_deg", [(0, 0), (5, 0), (0, -4), (3, 3)])
def test_fit_plate_recovers_center_radius_top_and_tilt(roll_deg, pitch_deg):
    rng = np.random.default_rng(5)
    P, T = SP.plate(rng, (0.46, 0.24), roll=math.radians(roll_deg), pitch=math.radians(pitch_deg), cams=SP.CAMS2)
    p = GP.fit_plate(_cut(P), 0.0015, rng)
    assert np.linalg.norm(p.center - T["center"]) < 0.003 and p.radius == pytest.approx(0.09, abs=0.003)
    assert math.degrees(math.acos(min(1.0, float(p.R[:, 2] @ T["normal"])))) < 0.6
    assert p.top_h == pytest.approx(T["center"][2], abs=0.002)
    assert p.depth(T["center"] + [0, 0, 0.03]) > 0 and p.depth(T["center"] + [0.12, 0, 0.01]) < 0
    assert p.over_xy(T["center"]) and not p.over_xy(T["center"] + [0.1, 0, 0])


def test_fit_plate_with_a_hidden_part_of_the_rim():
    rng = np.random.default_rng(6)
    P, T = SP.plate(rng, (0.46, 0.24), cams=SP.CAMS2, hole=(0.46 + 0.07, 0.24, 0.012), hole_r=0.04, drop=0.3)
    p = GP.fit_plate(_cut(P), 0.0015, rng)
    assert np.linalg.norm(p.center - T["center"]) < 0.004 and p.radius == pytest.approx(0.09, abs=0.004)
    assert p.n_edges < 36


def _clouds(cup_bottom, roll=0.0, R=np.eye(3)):
    rng = np.random.default_rng(1)
    Pp, T = SP.plate(rng, (0.46, 0.24), roll=roll, cams=SP.CAMS2)
    Pc = SP.cup(rng, cup_bottom, R=R, cams=SP.CAMS2)
    pw = np.vstack([Pp, Pc])
    col = np.r_[np.full(len(Pp), 3), np.zeros(len(Pc), int)]
    return [{"pw": pw, "h": pw[:, 2].copy(), "col": col, "name": "overhead"}], T


SU = types.SimpleNamespace(table_normal=np.array([0.0, 0.0, 1.0]), table_z=0.0)


@pytest.mark.parametrize("where, bottom, on", [("table", (0.45, -0.05, 0.0), False), ("plate", (0.47, 0.25, 0.012), True),
                                               ("plate_edge_out", (0.46 + 0.10, 0.24, 0.0), False)])
def test_place_finder_in_place_on_the_plate(where, bottom, on):
    f = PlaceGeometryFinder.from_spec(SPEC, None, SU)
    cl, T = _clouds(bottom)
    est = f.estimate_clouds(cl, 0.0, types.SimpleNamespace(is_grasped=False, width=0.08), None)
    o, p = est.objects["cup"], est.places["plate"]
    assert p.ok and isinstance(p.region, GP.PlateGeom) and REL.place_kind(p) == "surface"
    assert o.in_place == {"plate": on}, where
    assert REL.width_min(o) == pytest.approx(0.06, abs=0.006) and REL.height(o) == pytest.approx(0.08, abs=0.005)


def test_place_finder_held_cup_is_near_the_hand_and_not_on_the_plate():
    f = PlaceGeometryFinder.from_spec(SPEC, None, SU)
    b = np.array([0.46, 0.24, 0.05])                                 # 皿の真上 3.8 cm に持っている
    cl, _ = _clouds(b)
    est = f.estimate_clouds(cl, 0.0, types.SimpleNamespace(is_grasped=True, width=0.06), b + [0, 0, 0.055])
    o = est.objects["cup"]
    assert o.near_hand and o.in_place == {"plate": False}
    est = f.estimate_clouds([], 0.1, types.SimpleNamespace(is_grasped=True, width=0.06), b + [0, 0, 0.065])
    assert est.objects["cup"].status == "occluded" and est.objects["cup"].near_hand  # 隠れても指先と動く


def test_place_finder_fallen_cup_on_the_plate_is_not_in_place():
    f = PlaceGeometryFinder.from_spec(SPEC, None, SU)
    cl, _ = _clouds((0.47, 0.25, 0.012 + 0.03), R=SP.rot(0.0, math.pi / 2, 0.0))
    est = f.estimate_clouds(cl, 0.0, types.SimpleNamespace(is_grasped=False, width=0.08), None)
    assert est.objects["cup"].in_place == {"plate": False} and f.upright["cup"] is False


def test_finder_class_keeps_the_cube_task_on_the_old_finder():
    from recovla.api.estimated import EstimatedGeometryFinder
    assert finder_class(TIDY) is EstimatedGeometryFinder and finder_class(SPEC) is PlaceGeometryFinder
    with pytest.raises(ValueError):
        EstimatedGeometryFinder.from_spec(SPEC)                     # 箱を前提とする口は皿のタスクを断る
    assert isinstance(AP.make_finder(TIDY, kind="estimated"), EstimatedGeometryFinder)


def test_in_place_for_a_container_is_inside():
    box = AP.PlaceState(K.BOX_XY, 0.0, True, None, K.box_region())
    for p in ((0.45, 0.25, 0.03), (0.45 + 0.09, 0.25, 0.03), (0.30, 0.0, 0.02)):
        o = AP.ObjState(np.array(p), 0.0, "seen", {}, None, 0.0, np.full(3, 0.04), np.eye(3), K.SIGMA ** 2 * np.eye(3))
        assert REL.in_place(o, box) == REL.inside(o, box)
    plate = AP.PlaceState(K.PLATE_XY, 0.0, True, None, K.plate_region())
    o = AP.ObjState(np.array([0.46, 0.24, 0.052]), 0.0, "seen", {}, None, 0.0, np.array([0.06, 0.06, 0.08]), np.eye(3),
                    K.SIGMA ** 2 * np.eye(3))
    assert REL.in_place(o, plate) and not REL.in_place(o, plate, upright=False)
    assert not REL.in_place(dataclasses.replace(o, pos=np.array([0.46, 0.24, 0.08])), plate)        # 宙
    assert not REL.in_place(dataclasses.replace(o, pos=np.array([0.46 + 0.095, 0.24, 0.052])), plate)  # 底の中心が円の外


# ---------------------------------------------------------------- 4. CL2
def _obs(target, hand=(0.45, -0.05, 0.25), place=None, hmap=None, width=0.08, closed=False, t=0.0):
    hand = np.asarray(hand, float)
    return CL2.CL2Obs(t=t, target=target, others={}, place=place, place_name="plate", closed=closed, width=width, width_t=t,
                      hand=hand, tip=hand + [0, 0, -0.1034], x_cmd=hand, table_z=0.0, est_t=t, hmap=hmap,
                      home=np.array([0.33, 0.15, 0.30]), ws_z=(0.0, 0.5))


def _obj(pos, ext=(0.06, 0.06, 0.08), on=False):
    return AP.ObjState(np.asarray(pos, float), 0.0, "seen", {"plate": on, "box": on}, None, 0.0, np.asarray(ext, float),
                       np.eye(3), K.SIGMA ** 2 * np.eye(3))


def test_grasp_height_keeps_the_palm_above_a_tall_object_and_is_unchanged_for_a_cube():
    old = CL2.CL2Brain(RULES, 0.02, lambda w: PROF)
    new = CL2.CL2Brain(RULES, 0.02, lambda w: PROF, palm_fn=lambda: PALM)
    hm = K.plate_heightmap_of(types.SimpleNamespace(cubes={}, held=None, t=0.0))
    cube = _obs(_obj((0.45, -0.05, 0.02), (0.04, 0.04, 0.04)), hmap=hm)
    assert new.grasp_hand_z(cube) == old.grasp_hand_z(cube) and new.grasp_tip_h(cube) == pytest.approx(0.02)
    for z in (0.02, 0.0215, 0.025):                                    # 推定の揺れ・斜めの立方体でも
        c = _obs(_obj((0.45, -0.05, z), (0.04, 0.04, 0.04 + (z - 0.02))), hmap=hm)
        assert new.grasp_tip_h(c) == old.grasp_tip_h(c)
    cup = _obs(_obj((0.45, -0.05, 0.04)), hmap=hm)
    tip = new.grasp_tip_h(cup)
    assert tip > 0.04 and tip + PALM[0].low >= 0.08 + RULES.map_unc_k * hm.sigma - 1e-9   # 手のひらが縁＋余白より上
    assert tip == pytest.approx(0.08 + RULES.map_unc_k * hm.sigma - PALM[0].low)
    assert old.grasp_tip_h(cup) == pytest.approx(0.04)                 # 手のひらの決まりなしなら中心


def test_cl2_skill_palm_profile_from_the_robot_model():
    from recovla.harness.setup import nominal_setup
    from recovla.runtime import robot_model
    m = robot_model.load(nominal_setup(config.load(*config.V2_LAYERS)).robot_xml)
    palm = HM.gripper_profile(m, 0.1034, 0.08, bodies=("hand",))
    full = HM.gripper_profile(m, 0.1034, 0.08)
    assert min(p.low for p in palm) == pytest.approx(0.0325, abs=0.003)
    assert min(p.low for p in full) < 0 < min(p.low for p in palm)    # 指は指先より下、手のひらは上


def test_surface_spot_and_release_height_on_the_plate():
    b = CL2.CL2Brain(RULES, 0.02, lambda w: PROF, palm_fn=lambda: PALM)
    b.clocks.obj_h, b.obj_r, b.grasp_off = 0.08, 0.5 * math.sqrt(0.06 ** 2 * 2 + 0.08 ** 2), 0.0185
    b.obj_rh = 0.5 * math.hypot(0.06, 0.06)
    place = AP.PlaceState(K.PLATE_XY, 0.0, True, None, K.plate_region())
    w = types.SimpleNamespace(cubes={}, held=None, t=0.0)
    hm = K.plate_heightmap_of(w, cell=0.005)
    o = _obs(_obj((0.45, -0.05, 0.10)), place=place, hmap=hm, closed=True, width=0.059)
    s = b._surface_spot(o)
    assert np.allclose(s, [K.PLATE_XY[0], K.PLATE_XY[1], K.PLATE_TOP])          # 空いていれば上面の中心
    rt = b.release_tip_h(o, s)                                                   # 持った物の下端が上面＋余裕
    assert rt == pytest.approx(K.PLATE_TOP + 0.04 + 0.0185 + RULES.clear_frac * 0.08 + RULES.map_unc_k * hm.sigma)
    # 中心の近くに別の小さな物（高さ 5 cm）がある: シード範囲が上面だけになる所へずらす（皿の上面の円の中）
    hm2 = K.plate_heightmap_of(w, cell=0.005, extra=[(K.PLATE_XY + [0.03, 0.0], 0.01, K.PLATE_TOP + 0.05)])
    s2 = b._surface_spot(dataclasses.replace(o, hmap=hm2))
    reach = b.obj_rh + hm2.cell + RULES.unc_k * K.SIGMA
    assert np.hypot(*(s2[:2] - K.PLATE_XY)) > 0
    assert hm2.band_max(s2[:2], s2[:2], reach) <= K.PLATE_TOP + RULES.clear_frac * 0.08 + RULES.map_unc_k * hm2.sigma
    assert np.hypot(*(s2[:2] - K.PLATE_XY)) <= K.PLATE_R - reach + 1e-9
    # 箱（container）のときは箱の置き場所の決まり
    box = AP.PlaceState(K.BOX_XY, 0.0, True, None, K.box_region())
    ob = _obs(_obj((0.45, -0.05, 0.10), (0.04, 0.04, 0.04)), place=box,
              hmap=K.heightmap_of(types.SimpleNamespace(cubes={}, held=None, t=0.0)))
    b2 = CL2.CL2Brain(RULES, 0.02, lambda w: PROF)
    b2.clocks.obj_h, b2.obj_r = 0.04, 0.0346
    assert REL.place_kind(box) == "container" and b2._slot(ob) is not None


def test_step_done_when_the_cup_rests_on_the_plate():
    place = AP.PlaceState(K.PLATE_XY, 0.0, True, None, K.plate_region())

    def seq(on, hand_z):
        b = CL2.CL2Brain(RULES, 0.02, lambda w: PROF, palm_fn=lambda: PALM)
        b.obj_w = 0.06
        w = types.SimpleNamespace(cubes={"cup": np.array([0.47, 0.25, 0.052])}, held=None, t=0.0)
        hm = K.plate_heightmap_of(w)
        out = []
        for i in range(40):
            t = round(0.02 * i, 3)
            o = _obs(_obj((0.47, 0.25, 0.052), on=on), hand=(0.47, 0.25, hand_z), place=place, hmap=hm, t=t)
            out.append(b.step(o).phase)
        return out, b
    ph, b = seq(True, 0.30)
    assert ph[-1] == CL2.CL2Phase.done and b.step_done
    ph, _ = seq(True, 0.092 + 0.1034)                                   # 手が縁のすぐ上: 離れるまで済まない
    assert CL2.CL2Phase.done not in ph and ph[-1] == CL2.CL2Phase.clear
    ph, _ = seq(False, 0.30)
    assert CL2.CL2Phase.done not in ph


def test_cl2_puts_the_cup_on_the_plate_in_the_kinematic_world():
    """偽の世界（運動学だけ・地図つき）で、CL2Skill と CL2TaskRuntime がカップを皿の上に置き、始めの姿勢へ戻る。運ぶ途中で
    1 回落とすと、落下を記録して掴み直す。掴む高さは中心より上（手のひらが縁より上）。"""
    w = K.KinPlateWorld({"cup": (0.45, -0.08, 0.04)})
    w.with_map = True
    sk = CL2.CL2Skill(w, K.setup(), RULES, SPEC, FMT, motion=K.KinMotion(w), profile=PROF, palm=PALM)
    ex = XC.CL2TaskRuntime(w, K.setup(), sk, None, {"step_timeout_s": 30.0, "retry": 1},
                           lambda text, table, box: {"steps": list(table), "reply": "x"}, FMT, SPEC, RULES)
    st = {"dropped": None}

    def drop(world):
        if st["dropped"] is None and world.held == "cup" and sk.last_phase == CL2.CL2Phase.carry and world.hand[2] > 0.17:
            world.drop()
            st["dropped"] = world.t
    ex.start("カップを皿に置いて", 5)
    K.run(ex, w, 120, every=drop, skill=sk, until=lambda: ex.finished)
    rec = ex.record()
    assert ex.finished and rec["stopped"] is None and rec["ended"]["at_home"], rec
    assert rec["detected"] == {"table": ["red"], "box": []} and [s["end"] for s in rec["steps"]] == ["self"]
    p = w.cubes["cup"]
    assert np.hypot(*(p[:2] - K.PLATE_XY)) < K.PLATE_R - K.CUP_R and p[2] == pytest.approx(K.PLATE_TOP + 0.04)
    assert st["dropped"] is not None and len(rec["cl2"]["drops"]) == 1 and rec["cl2"]["drops"][0]["in_place"] is False
    grasps = sk.trace()["script"]["grasp"]
    assert grasps and all(g["width"] == pytest.approx(0.06) for g in grasps)
    assert w.grip_off[2] < -0.01                                         # 指先は物の中心より上で掴んだ


def test_cl2_executor_stop_message_names_the_place():
    est = AP.WorldEstimate(0.0, {"cup": _obj((0.45, -0.05, 0.04))}, {"plate": AP.PlaceState(np.zeros(2), 0.0, False, None, None)})
    from .test_script_policy import FakeIO, FakeSkill
    clock = types.SimpleNamespace(t=0.0)
    sk = FakeSkill(clock, est, None)
    ex = XC.CL2TaskRuntime(FakeIO(clock), None, sk, None, {"step_timeout_s": 30.0, "retry": 1},
                           lambda *a: {"steps": []}, FMT, SPEC, RULES)
    ex.start("カップを皿に置いて", 1)
    while not ex.finished and clock.t < 10:
        ex.tick()
    assert "皿らしい物が見つからない" in ex.record()["stopped"]["reply"]


# ---------------------------------------------------------------- 5. 計画役
class _FakeClient:
    def __init__(self, raw):
        self.raw, self.calls = raw, []
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        return types.SimpleNamespace(model=kw["model"], content=[types.SimpleNamespace(type="text", text=self.raw)],
                                     stop_reason="end_turn", usage=types.SimpleNamespace(input_tokens=1, output_tokens=1))


def test_task_planner_maps_colors_to_names_and_back(tmp_path, monkeypatch):
    from recovla.api import planner as APL
    from recovla.planner import decompose_haiku as D4
    from recovla.planner import decompose_task as DT
    monkeypatch.setattr(D4, "CACHE", tmp_path)
    fn = DT.make_decompose(SPEC)
    cli = _FakeClient(json.dumps({"steps": ["cup"], "reply": "カップを皿に置きます"}))
    out = fn("カップを皿に置いて", ["red", "green"], [], cli=cli)
    assert out["steps"] == ["red"] and out["llm_steps"] == ["cup"] and out["valid"] and out["task"] == "cup_plate"
    kw = cli.calls[0]
    assert kw["system"] == APL.decompose_system(SPEC) and "皿の上に置きます" in kw["system"] and "箱" not in kw["system"]
    assert json.loads(kw["messages"][0]["content"]) == {"instruction": "カップを皿に置いて", "table_objects": ["cup"],
                                                        "in_place_objects": []}
    again = fn("カップを皿に置いて", ["red"], [], cli=_FakeClient("壊れた"))   # キャッシュから（同じ入力）
    assert again["from_cache"] and again["steps"] == ["red"]
    bad = fn("カップを皿に置いて", [], ["red"], cli=_FakeClient(json.dumps({"steps": ["cup"], "reply": "x"})))
    assert bad["steps"] == [] and not bad["valid"] and "すでに皿にあります" in bad["reply"]
    assert DT.cache_key(SPEC, "m", "s", {}, "u") != DT.cache_key(TIDY, "m", "s", {}, "u")


def test_container_planner_text_is_unchanged():
    from recovla.api import planner as APL
    s = APL.decompose_system(TIDY)
    assert "1 個ずつ箱へ入れます" in s and "箱へ入れる物の名前を、入れる順に並べた列" in s


# ---------------------------------------------------------------- 6. 入口
@pytest.fixture(scope="module")
def ladder():
    return _load(ROOT / "scripts" / "api" / "ladder.py", "api_ladder_for_cup_plate")


def test_ladder_task_plan(ladder):
    assert ladder.task_path("cup_plate") == "configs/tasks/cup_plate.yaml" and ladder.is_default_task("tidy3_cubes")
    p = ladder.arm_plan("CL2", task="cup_plate")
    assert p["task"] == "cup_plate" and "decompose_task" in p["planner"] and "PlaceGeometryFinder" in p["place_finder"]
    assert p["instruction"]["mode"] == "fixed" and p["vla_known"] is None
    u = ladder.arm_plan("U0", task="cup_plate", vla_text="task")
    assert u["instruction"]["example"] == "put the red cup on the plate" and u["vla_known"] and u["place_finder"] is None
    assert "decompose_haiku" in ladder.arm_plan("L2v2", task="cup_plate", task_planner="s4")["planner"]
    assert "task_file" not in ladder.arm_plan("CL2") and ladder.arm_plan("CL2") == ladder.arm_plan("CL2", task="tidy3_cubes")
    assert "タスクの書式: configs/tasks/cup_plate.yaml" in ladder.format_plan(u)
    assert set(ladder.TASK_FILES) <= {str(x.relative_to(ROOT)).replace("\\", "/") for x in ROOT.rglob("*")
                                      if x.is_file() and ".git" not in x.parts}


def test_ladder_refuses_cl_color_and_unknown_task_and_models(ladder):
    base = ["run", "--experiment", "X", "--condition", "y", "--trials", "193900:1", "--model", "Rreal", "--dry-run"]
    assert ladder.main(base + ["--arm", "CL", "--task", "cup_plate"]) == 3
    assert ladder.main(base + ["--arm", "L2v2", "--task", "cup_plate", "--finder", "color"]) == 3
    assert ladder.main(base + ["--arm", "U0", "--task", "nothing_here"]) == 3
    assert ladder.main(base + ["--arm", "U0", "--model-ckpt", "Rreal=x"]) == 3
    b2 = [x if x != "Rreal" else "Rcp" for x in base]
    assert ladder.main(b2 + ["--arm", "U0"]) == 3                       # 足していない名前は断る


def _dry(args, cond):
    out = ROOT / "outputs" / "v2eval" / "APIDRY" / cond
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "api" / "ladder.py"), "run", "--experiment", "APIDRY",
                        "--condition", cond, "--trials", "193900:2", "--exec-interval", "6", "--dry-run"] + args,
                       capture_output=True, text=True, cwd=ROOT, timeout=600, encoding="utf-8", errors="replace",
                       env=dict(os.environ, PYTHONUTF8="1"))
    return r, r.stdout + r.stderr, out


def test_cup_plate_cl2_dry_run_prints_the_plan():
    r, text, out = _dry(["--arm", "CL2", "--task", "cup_plate", "--model", "Rreal"], "pytest_api_dry_cup_CL2")
    assert r.returncode in (0, 3), text
    assert "タスク cup_plate" in text and "PlaceGeometryFinder" in text and "decompose_task" in text, text
    assert "指示文「カップを皿に置いて」" in text and "[dry-run]" in text and "seed 193900" in text, text
    assert not out.exists()


def test_cup_plate_vla_dry_run_with_a_registered_checkpoint():
    r, text, out = _dry(["--arm", "U0", "--task", "cup_plate", "--vla-text", "task", "--model", "Rcp",
                         "--model-ckpt", "Rcp=outputs/real/cup_plate/train/x/pretrained_model"], "pytest_api_dry_cup_U0")
    assert r.returncode in (0, 3), text
    assert "put the red cup on the plate" in text and "保存点 Rcp=" in text and "[dry-run]" in text, text
    assert not out.exists()


# ---------------------------------------------------------------- 7. 既知の寸法を読まない・G1（実行系は真値を読まない）
CUP_RUNTIME = ("src/recovla/api/geometry_place.py", "src/recovla/api/estimated_place.py")
SCENE_NAMES = ("cup_plate.yaml", "scene_cup_plate", "plate_region", "cup_region", "radius_margin_m", "contact_tol_m",
               "max_tilt_deg", "vla_setup", "thickness")


def test_cup_plate_runtime_reads_no_known_sizes_or_scene_values():
    for f in CUP_RUNTIME + CL2_RUNTIME + ("src/recovla/api/relations.py",):
        src = (ROOT / f).read_text(encoding="utf-8")
        assert known_reads(src) == [], f
        strs = {n.value for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        attrs = {n.attr for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Attribute)}
        assert not any(k in s for s in strs for k in SCENE_NAMES), f
        assert not ({"size_m", "sc", "scene_cfg"} & attrs), f            # 書式の size_m・場面の設定を読まない


def test_cup_plate_runtime_files_pass_the_g1_strict_scan():
    g = _load(ROOT / "scripts" / "check_real_boundary.py", "check_g1_boundary_cup")
    files = g.runtime_v2_files(None, g.API_DIR)
    for f in CUP_RUNTIME:
        assert f in files
        assert not g.scan_runtime((ROOT / f).read_text(encoding="utf-8"), f, g.API_ALLOWED_IMPORTS, g.API_DENY), f


def test_plate_estimate_is_not_replaced_by_a_hidden_fragment():
    """腕に隠れて皿の点が縁の小さな切れ端だけになった時刻の当てはめで、皿の推定を替えない（試しの試行で起きた形）。"""
    f = PlaceGeometryFinder.from_spec(SPEC, None, SU)
    cl, T = _clouds((0.45, -0.05, 0.0))
    est = f.estimate_clouds(cl, 0.0, types.SimpleNamespace(is_grasped=False, width=0.08), None)
    r0 = est.places["plate"].region.radius
    rng = np.random.default_rng(2)
    Pp, _ = SP.plate(rng, (0.46, 0.24), cams=SP.CAMS2)
    frag = Pp[np.hypot(Pp[:, 0] - 0.46, Pp[:, 1] - 0.165) < 0.016]  # 手前の縁のそばの小さな切れ端だけ（試しの試行の形）
    assert len(frag) > 60
    g = GP.fit_plate(_cut(frag), 0.0015, rng)
    assert g is not None and g.radius < 0.85 * r0 and g.n_edges >= 27  # 小さいが扇は埋まる（扇の数だけでは見分けられない）
    pw = np.vstack([frag, SP.cup(rng, (0.45, -0.05, 0.0), cams=SP.CAMS2)])
    col = np.r_[np.full(len(frag), 3), np.zeros(len(pw) - len(frag), int)]
    est = f.estimate_clouds([{"pw": pw, "h": pw[:, 2].copy(), "col": col, "name": "overhead"}], 0.1,
                            types.SimpleNamespace(is_grasped=False, width=0.08), None)
    assert est.places["plate"].region.radius == pytest.approx(r0)


def test_held_tall_cup_is_kept_out_of_the_map_down_to_its_bottom(monkeypatch):
    """持っている高いカップ（指先から 7.5 cm 下まで出る）の下の部分が地図に残らない（指先の真上の柱を、持っている物の推定の
    下端まで外す。推定の中心が遅延して球の外へ出ても残らないように）。皿の上面は残る。"""
    from recovla.harness.setup import nominal_setup
    from recovla.runtime import cue as C
    from recovla.runtime.perception import Params, Perception
    cfg = config.load(*config.V2_LAYERS)
    su = nominal_setup(cfg)
    thr = C.Thresholds.from_dict(config.color_detect(cfg))
    rng = np.random.default_rng(11)
    tip = np.array([0.46, 0.24, 0.125])
    Pc = SP.cup(rng, tip - [0, 0, 0.075], cams=SP.CAMS2)              # 縁の近くを持って皿の上 3.8 cm に（底 0.050）
    Pp, _ = SP.plate(rng, (0.46, 0.24), cams=SP.CAMS2)
    X, Y = np.meshgrid(np.arange(0.25, 0.70, 0.004), np.arange(-0.2, 0.5, 0.004))
    tab = np.column_stack([X.ravel(), Y.ravel(), rng.normal(0, 0.0015, X.size)])
    tab = tab[np.hypot(tab[:, 0] - 0.46, tab[:, 1] - 0.24) > 0.09]
    pw = np.vstack([tab, Pp, Pc])
    col = np.r_[np.full(len(tab), -1), np.full(len(Pp), 3), np.zeros(len(Pc), int)]
    P = {"pw": pw, "h": pw[:, 2].copy(), "col": col, "plane": (np.array([0.0, 0.0, 1.0]), np.zeros(3))}
    empty = {"pw": np.zeros((0, 3)), "h": np.zeros(0), "col": np.zeros(0, int), "plane": P["plane"]}
    monkeypatch.setattr(Perception, "points", lambda self, fr, w, margin=None: P if fr.name == "overhead" else empty)
    frames_ = {n: types.SimpleNamespace(name=n, depth=np.zeros((2, 2)), t_capture=1.0) for n in ("overhead", "wrist")}
    grip = types.SimpleNamespace(width=0.059, is_grasped=True)
    per = CL2.make_perception(Perception(su, Params.from_config(cfg["runtime_v2"]["perception"]), thr), SPEC, su, RULES)
    assert isinstance(per.geom, PlaceGeometryFinder)
    wm = per.update(frames_, grip, 1.0, tip)
    o = wm.geometry.objects["cup"]
    assert o.near_hand and o.in_place == {"plate": False}
    top = wm.heightmap.band_max(tip[:2], tip[:2], 0.035)                # カップの壁（半径 3 cm）まで
    assert top == pytest.approx(0.012, abs=0.004)                     # カップの底の部分は外れ、皿の上面だけが残る
