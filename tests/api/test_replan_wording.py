"""組み直しの前の版（L2v2）の LLM に渡すタスクの言葉（planner/replan_event の wording、scripts/api/ladder.replan_wording）。
既定のタスクでは SYSTEM・キャッシュの鍵・定型の報告が変わらず、カップを皿にでは立方体・箱・入れるが残らない。"""
import hashlib
import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from recovla.api import task_spec as TS  # noqa: E402
from recovla.planner import replan_event as RL  # noqa: E402

# 評価に使った版（コミット 7629a01）の replan_event.SYSTEM の SHA-256
SYSTEM_SHA_APIL = "b114b995df52cec2c925bb534672df66c7424d41ea03f88eed470ba805ebd16a"


def _ladder():
    spec = importlib.util.spec_from_file_location("api_ladder_for_wording", ROOT / "scripts" / "api" / "ladder.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["api_ladder_for_wording"] = m
    spec.loader.exec_module(m)
    return m


def _ctx(event="dropped", in_box=()):
    return RL.make_context("x", ["red"], [{"color": "red", "result": "failed", "attempts": 1}],
                           {"table": [] if in_box else ["red"], "in_box": list(in_box), "in_hand": [], "lost": []},
                           0, "red", 0, event)


def test_default_wording_keeps_the_apil_prompt():
    assert hashlib.sha256(RL.SYSTEM.encode()).hexdigest() == SYSTEM_SHA_APIL
    assert RL.system_for(None) == RL.SYSTEM
    msg = RL.R4.user_message(_ctx())
    assert RL.cache_key(RL.DEFAULT_MODEL, msg) == RL.cache_key(RL.DEFAULT_MODEL, msg, None)
    assert RL.build_params(RL.DEFAULT_MODEL, msg)["system"] == RL.SYSTEM
    for a in RL.R4.ACTIONS:
        assert RL.template_report(a, _ctx(), "red", ["red"]) == RL.template_report(a, _ctx(), "red", ["red"], wording=None)
    step = RL.make_replan_step(RL.DEFAULT_MODEL)
    assert step.wording is None


def test_cup_plate_wording_has_no_cube_or_box():
    lad = _ladder()
    spec = TS.load_task(ROOT / "configs" / "tasks" / "cup_plate.yaml")
    w = lad.replan_wording(spec, False)
    assert w == {"obj": "カップ", "place": "皿", "colors": ["red"], "goal_kind": "all_on", "objs": "赤のカップ"}
    assert lad.replan_wording(TS.load_task(), True) is None
    s = RL.system_for(w)
    for word in ("立方体", "箱", "入れ"):
        assert word not in s, word
    assert "赤のカップ" in s and "皿の上" in s and "perception.in_box" in s
    msg = RL.R4.user_message(_ctx())
    assert RL.cache_key(RL.DEFAULT_MODEL, msg, w) != RL.cache_key(RL.DEFAULT_MODEL, msg)
    for ev, box in (("dropped", ()), ("timeout", ("red",))):
        for a in RL.R4.ACTIONS:
            rep = RL.template_report(a, _ctx(ev, box), "red", ["red"], wording=w)
            for word in ("立方体", "箱", "入れ"):
                assert word not in rep, (a, ev, rep)
            assert not RL.RM.check_report(rep), (a, ev, rep)


def test_wording_reaches_the_request_and_the_record():
    w = RL.make_wording("カップ", "皿", ["red"], "all_on")

    class Fake:
        def __init__(self):
            self.messages = self
            self.system = None

        def create(self, **kw):
            self.system = kw["system"]
            raise RuntimeError("偽の API")

    cli = Fake()
    out = RL.replan_step(text="x", plan_order=["red"], steps=[{"color": "red", "result": "failed", "attempts": 1}],
                         perception={"table": ["red"], "in_box": [], "in_hand": [], "lost": []}, failed_step=0,
                         failed_color="red", replans_done=0, event="dropped", cli=cli, use_cache=False, wording=w)
    assert cli.system == RL.system_for(w)
    assert out["decision"]["action"] == "stop" and out["wording"] == w
    assert "カップ" in out["report"] and "箱" not in out["report"]
