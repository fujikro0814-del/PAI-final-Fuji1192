"""task API の書式（api/task_spec）・計画役の口（api/planner）・目標の判定（api/harness/goal_truth）の確かめ。
実行系の COLORS・指示文・計画役の入力と完全に一致することを固定する。CPU だけ。outputs/・チェックポイント・API キーは使わない。"""
import importlib.util
import json
import pathlib
import sys
import types

import numpy as np
import pytest

from recovla.api import perception as AP
from recovla.api import planner as AP_PL
from recovla.api import task_spec as TS
from recovla.api.harness import goal_truth as GT
from recovla.common import config
from recovla.common.seeds import COLORS as SEED_COLORS
from recovla.planner import decompose as D
from recovla.planner import decompose_haiku as D4
from recovla.planner import replan_event as RL
from recovla.planner import replan_base as R4
from recovla.runtime import cue as C
from recovla.runtime.executor import TaskRuntime
from recovla.runtime.perception import COLORS as RT_COLORS

ROOT = pathlib.Path(__file__).resolve().parents[2]
FMT = config.load()["convert"]["instruction"]


@pytest.fixture(scope="module")
def spec():
    return TS.load_task("configs/tasks/tidy3_cubes.yaml")


def test_tidy3_objects_are_the_current_colors(spec):
    assert spec.name == "tidy3_cubes"
    assert spec.object_names == tuple(RT_COLORS) == tuple(SEED_COLORS) == tuple(D.COLORS)
    assert spec.goal.kind == "all_in" and spec.goal.objects == tuple(RT_COLORS) and spec.goal.place == "box"
    assert spec.step_vocab == ("pick_place",) and spec.place_names == ("box",)
    assert all(o.shape == "cube" and o.size_m == (0.04, 0.04, 0.04) for o in spec.objects)
    assert config.load()["scene"]["cube_size"] == 0.04
    assert [o.words[0] for o in spec.objects] == [R4.JA[c] for c in RT_COLORS]          # 赤・緑・青（組み直しの報告と同じ）


@pytest.mark.parametrize("color", RT_COLORS)
def test_instruction_is_exactly_the_current_sentence(spec, color):
    """方策への指示文は、実行器（runtime/executor.TaskRuntime.instruction、fmt は configs の convert.instruction）と同じ文字列。"""
    step = TS.StepSpec("pick_place", color, "box")
    now = TaskRuntime.instruction(types.SimpleNamespace(fmt=FMT), color)
    assert TS.instruction_of(spec, step, FMT) == now == f"put the {color} cube in the box"
    assert TS.cue_of(spec, step) == C.color_of_instruction(now) == color                 # 目標位置の入力の色（PolicyRuntime が読む色）
    assert TS.step_of_instruction(spec, now, FMT) == step
    assert TS.step_of_cue(spec, color) == step                                           # 計画役の手順（色）→ 手順


def test_instruction_text_is_the_96_default(spec):
    spec96 = importlib.util.spec_from_file_location("s4_resume_for_api_spec", ROOT / "scripts" / "eval_resumable.py")
    m = importlib.util.module_from_spec(spec96)
    sys.modules["s4_resume_for_api_spec"] = m
    spec96.loader.exec_module(m)
    a = m.build_parser().parse_args(["task", "--experiment", "x", "--condition", "y", "--trials", "1:1"])
    assert spec.instructions == (a.text,) == ("全部片付けて",)


def test_planner_message_has_the_same_values_as_decompose(spec):
    """計画役の口（api/planner）の入力は、decompose（decompose_haiku も同じ関数）と同じ並び・同じ値。"""
    assert D4.user_message is D.user_message
    for table, box in ((["blue", "red"], ["green"]), (list(RT_COLORS), []), ([], ["red", "green", "blue"])):
        now = json.loads(D.user_message("全部片付けて", table, box))
        mine = json.loads(AP_PL.decompose_message(spec, "全部片付けて", table, box))
        assert list(mine) == ["instruction", "table_objects", "in_place_objects"]
        assert [now["instruction"], now["table_colors"], now["box_colors"]] == list(mine.values())
        for steps in (["red"], ["red", "red"], ["green"], ["purple"], []):
            assert list(D.check(steps, table, box).values()) == list(AP_PL.check_steps(spec, steps, table, box).values())
    enum = AP_PL.decompose_schema(spec)["properties"]["steps"]["items"]["enum"]
    assert enum == D.SCHEMA["properties"]["steps"]["items"]["enum"]
    sys_ = AP_PL.decompose_system(spec)
    assert "赤" in sys_ and '"blue"（青）' in sys_ and "red, green, blue の順" in sys_ and "箱" in sys_


def test_planner_replan_context_matches_replan_l2(spec):
    per = {"table": ["blue", "red"], "in_box": ["green"], "in_hand": [], "lost": []}
    now = RL.make_context("全部片付けて", ["red", "green", "blue"], [{"color": "red", "result": "failed", "attempts": 1}],
                          per, 0, "red", 0, "dropped")
    mine = AP_PL.replan_context(spec, "全部片付けて", ["red", "green", "blue"], [{"obj": "red", "result": "failed", "attempts": 1}],
                                {"table": per["table"], "in_place": per["in_box"], "in_hand": [], "lost": []}, 0, "red", 0,
                                "dropped")
    assert mine["perception"]["table"] == now["perception"]["table"]
    assert mine["perception"]["in_place"] == now["perception"]["in_box"]
    assert mine["failed"] == {"step": 0, "obj": "red", "event": "dropped"} and now["failed"]["event"] == "dropped"
    assert [s["obj"] for s in mine["steps"]] == [s["color"] for s in now["steps"]]
    with pytest.raises(ValueError):
        AP_PL.replan_context(spec, "x", [], [], {}, 0, "red", 0, "other")


def _est(in_box: dict, lost=()):
    objs = {c: AP.ObjState(np.zeros(3), 0.0, "lost" if c in lost else "seen", {"box": v}, None) for c, v in in_box.items()}
    return AP.WorldEstimate(0.0, objs, {"box": AP.PlaceState(np.zeros(2), 0.0, True, 0.08)})


def test_goal_estimate_and_truth_are_separate(spec):
    assert TS.goal_met(spec, _est({"red": True, "green": True, "blue": True}))
    assert not TS.goal_met(spec, _est({"red": True, "green": True, "blue": False}))
    assert not TS.goal_met(spec, _est({"red": True, "green": True, "blue": True}, lost=("blue",)))
    assert not TS.goal_met(spec, None)
    meta = {"final_in_box": {"red": True, "green": True, "blue": True}, "all_three_in_box": True}
    assert GT.goal_met_truth(spec, GT.in_place_of_meta(spec, meta)) == meta["all_three_in_box"]
    meta = {"final_in_box": {"red": True, "green": False, "blue": True}, "all_three_in_box": False}
    assert GT.goal_met_truth(spec, GT.in_place_of_meta(spec, meta)) == meta["all_three_in_box"]
    box = np.array([0.45, 0.25, 0.0])
    pos = {"red": box + [0.04, 0.04, 0.03], "green": box + [0.0, 0.0, 0.03], "blue": np.array([0.4, -0.1, 0.02])}
    assert GT.in_place_truth(spec, pos, box) == {"red": True, "green": True, "blue": False}


@pytest.mark.parametrize("change, why", [
    (lambda d: d["objects"].append(dict(d["objects"][0])), "重複"),
    (lambda d: d["objects"][0].update(finder="sonar:red"), "finder"),
    (lambda d: d["goal"].update(kind="stack"), "目標の種類"),
    (lambda d: d["goal"].update(place="shelf"), "shelf"),
    (lambda d: d.update(step_vocab=["pick_place", "pour"]), "手順の語彙"),
])
def test_parse_rejects_bad_tasks(change, why):
    import yaml
    d = yaml.safe_load((ROOT / TS.DEFAULT_TASK).read_text(encoding="utf-8"))
    change(d)
    with pytest.raises((ValueError, KeyError), match=why):
        TS.parse_task(d)


def test_open_vocab_finder_is_only_a_port(spec):
    import dataclasses
    ov = dataclasses.replace(spec, objects=spec.objects[:2] + (dataclasses.replace(spec.objects[2], finder="open_vocab:青いコップ"),))
    with pytest.raises(NotImplementedError, match="口だけ"):
        AP.make_finder(ov)
    with pytest.raises(ValueError):
        TS.cue_of(ov, TS.StepSpec("pick_place", "blue", "box"))
