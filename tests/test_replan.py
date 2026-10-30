"""LLM による手順の組み直しの最初の版（条件 U4）の確かめ: 組み直しの計画役（src/recovla/planner/replan_base.py）、
実行器の包み（src/recovla/runtime/executor_replan.py）、入口（scripts/eval_replan_task.py）。CPU だけ。ネットワーク・GPU・シミュレーションは使わない（LLM は偽のクライアント、実行器は偽の io・知覚・判定）。

使い方: .venv\\Scripts\\python.exe -m pytest -q tests\\test_replan.py -p no:cacheprovider
読むもの: replan_base.py・executor_replan.py・scripts/eval_replan_task.py・scripts/paper_pdf.py（構文木だけ）・scripts/check_real_boundary.py（importlib）。
書くもの: pytest の一時フォルダだけ。
"""
import ast
import importlib.util
import itertools
import json
import pathlib
import sys
import types

import numpy as np
import pytest

from recovla.planner import replan_base as R4
from recovla.runtime import executor_replan as U4

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def raw(action="reorder", color="none", order=("green", "blue", "red"), reason="赤を後に回す", report=None):
    report = "赤の立方体を箱に入れられませんでした。先に緑と青を入れて、赤は最後にもう一度試します。" if report is None else report
    return json.dumps({"action": action, "color": color, "order": list(order), "reason": reason, "report": report},
                      ensure_ascii=False)


class FakeClient:
    """messages.create を模擬する。応答（文字列、例外、("refusal", 文字列)）を順に返し、受けた引数を残す。"""

    def __init__(self, *responses):
        self.responses = list(responses) or [raw()]
        self.calls = []
        self.messages = types.SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw)
        r = self.responses[min(len(self.calls), len(self.responses)) - 1]
        if isinstance(r, BaseException):
            raise r
        stop, text = ("end_turn", r) if isinstance(r, str) else r
        return types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text=text)], model=kw["model"],
                                     stop_reason=stop, usage=types.SimpleNamespace(input_tokens=900, output_tokens=60))


class NoCallClient:
    def __init__(self):
        self.messages = types.SimpleNamespace(create=self._create)

    def _create(self, **kw):
        raise AssertionError("呼ばないはずの API を呼んだ")


@pytest.fixture(autouse=True)
def _tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(R4, "CACHE", tmp_path / "llm_cache")


def ctx(failed="red", table=("red", "green", "blue"), box=(), steps=None, plan=("red", "green", "blue"), replans=0,
        in_hand=(), lost=()):
    steps = [{"color": failed, "result": "timeout", "attempts": 2}] if steps is None else steps
    return R4.make_context("全部片付けて", list(plan), steps, {"table": list(table), "in_box": list(box), "in_hand": list(in_hand),
                                                         "lost": list(lost)}, len(steps) - 1, failed, replans)


# ---------------------------------------------------------------- API の作法
def test_call_follows_decompose_s4_manner():
    cli = FakeClient()
    out = R4.replan(ctx(), cli=cli)
    kw = cli.calls[0]
    assert kw["model"] == "claude-haiku-5-5"
    assert "temperature" not in kw and "extra_body" not in kw                       # 温度は送らない（Haiku 5.5 は 400）
    assert kw["thinking"] == {"type": "disabled"}
    assert kw["output_config"]["format"] == {"type": "json_schema", "schema": R4.SCHEMA}
    assert kw["system"] == R4.SYSTEM
    body = json.loads(kw["messages"][0]["content"])
    assert body["candidates"] == ["red", "green", "blue"] and body["failed"] == {"step": 0, "color": "red"}
    assert out["accepted"] and out["decision"] == {"action": "reorder", "color": "none", "order": ["green", "blue", "red"]}
    assert out["report_source"] == "llm" and not out["fallback"]
    assert out["llm"]["calls"] == 1 and out["llm"]["temperature"] is None


def test_schema_is_strict_and_closed():
    s = R4.SCHEMA
    assert s["additionalProperties"] is False and set(s["required"]) == set(s["properties"])
    assert s["properties"]["action"]["enum"] == list(R4.ACTIONS)


# ---------------------------------------------------------------- 弾く規則と安全な手
@pytest.mark.parametrize("c, resp, why", [
    (ctx(box=("green",), table=("red", "blue")), raw("next", "green", []), "next:in_box:green"),
    (ctx(table=("red", "green")), raw("next", "blue", []), "next:not_on_table:blue"),
    (ctx(steps=[{"color": "red", "result": "timeout", "attempts": 2}, {"color": "red", "result": "timeout", "attempts": 2}],
         plan=("red", "red", "green", "blue")), raw("next", "red", []), "next:runs_limit:red:2"),
    (ctx(), raw("reorder", "none", ["green", "green", "blue"]), "reorder:duplicates"),
    (ctx(), raw("reorder", "none", ["green", "blue"]), "reorder:not_permutation_of_candidates"),
    (ctx(), raw("reorder", "none", []), "reorder:empty"),
    (ctx(), raw("skip", "green", []), "skip:not_failed_color:green:red"),
    (ctx(in_hand=("green",), table=("red", "blue")), raw("next", "green", []), "next:not_on_table:green"),
    (ctx(), raw("dance", "none", []), "unknown_action:dance"),
])
def test_disallowed_output_falls_back_to_stop(c, resp, why):
    out = R4.replan(c, cli=FakeClient(resp))
    assert not out["accepted"] and out["fallback"]
    assert out["decision"] == {"action": "stop", "color": "none", "order": []}
    assert any(r.startswith(why) for r in out["rejected"]), out["rejected"]
    assert out["report"] == R4.template_report("stop", c) and out["report_source"] == "template"
    assert out["llm_output"] is not None                                            # 弾いた応答も記録に残す


def test_third_run_of_same_color_is_never_a_candidate():
    c = ctx(steps=[{"color": "red", "result": "timeout", "attempts": 2}, {"color": "green", "result": "success", "attempts": 1},
                   {"color": "red", "result": "timeout", "attempts": 2}], plan=("red", "green", "red", "blue"),
            table=("red", "blue"), box=("green",))
    assert R4.candidates(c) == ["blue"]
    assert R4.check({"action": "next", "color": "red", "order": []}, c) == ["next:runs_limit:red:2"]
    assert R4.check({"action": "reorder", "color": "none", "order": ["blue", "red"]}, c)


def test_allowed_actions_and_next_order():
    c = ctx()
    for resp, order in ((raw("next", "green", []), ["green", "red", "blue"]),
                        (raw("next", "red", []), ["red", "green", "blue"]),
                        (raw("reorder", "none", ["blue", "green", "red"]), ["blue", "green", "red"]),
                        (raw("skip", "red", []), ["green", "blue"]),
                        (raw("finish", "none", []), []), (raw("stop", "none", []), [])):
        out = R4.replan_step("全部片付けて", ["red", "green", "blue"], c["steps"], c["perception"], 0, "red", 0,
                             cli=FakeClient(resp), use_cache=False)          # 同じ入力に違う応答を返すのでキャッシュは使わない
        assert out["accepted"], out["rejected"]
        assert out["next_order"] == order
        assert out["plan_change"] == (out["decision"]["action"] != "stop")
        assert out["context"] == c


def test_replan_limit_does_not_call_the_api():
    out = R4.replan(ctx(replans=R4.MAX_REPLANS), cli=NoCallClient())
    assert out["decision"]["action"] == "stop" and out["fallback"] and out["rejected"] == [f"replan_limit:{R4.MAX_REPLANS}"]
    assert out["llm"] == {"called": False, "calls": 0}


# ---------------------------------------------------------------- キャッシュと回し直し
def test_cache_is_used_and_key_has_model_version_and_input():
    c = ctx()
    out1 = R4.replan(c, cli=FakeClient())
    out2 = R4.replan(c, cli=NoCallClient())
    assert out2["llm"]["from_cache"] and out2["decision"] == out1["decision"] and out2["report"] == out1["report"]
    msg = R4.user_message(c)
    key = R4.cache_key("claude-haiku-5-5", msg)
    assert out1["llm"]["cache"] == key
    assert key != R4.cache_key("claude-haiku-4-5", msg)
    assert key != R4.cache_key("claude-haiku-5-5", R4.user_message(ctx(box=("green",), table=("red", "blue"))))
    rec = json.loads((R4.CACHE / f"{key}.json").read_text(encoding="utf-8"))
    assert rec["request"]["temperature"] is None and rec["request"]["thinking"] == {"type": "disabled"}
    assert rec["request"]["prompt_version"] == R4.PROMPT_VERSION and rec["kind"] == "replan_s4"
    # 版を変えると鍵が変わる
    old = R4.PROMPT_VERSION
    try:
        R4.PROMPT_VERSION = old + 1
        assert R4.cache_key("claude-haiku-5-5", msg) != key
    finally:
        R4.PROMPT_VERSION = old
    # 組み直しの鍵は手順を分ける計画役（decompose_haiku）の鍵と混ざらない
    from recovla.planner import decompose_haiku as D4
    assert key != D4.cache_key("claude-haiku-5-5", msg)


class _ApiError(Exception):
    pass


@pytest.mark.parametrize("first", [_ApiError("overloaded"), "{これは JSON ではない", ("refusal", raw())])
def test_one_retry_with_same_input_then_success(first):
    cli = FakeClient(first, raw())
    out = R4.replan(ctx(), cli=cli)
    assert len(cli.calls) == 2 and cli.calls[0]["messages"] == cli.calls[1]["messages"]
    assert out["accepted"] and out["llm"]["retried"] and out["llm"]["calls"] == 2 and len(out["llm"]["errors"]) == 1


@pytest.mark.parametrize("bad", [_ApiError("down"), "{壊れた", ("refusal", raw())])
def test_two_failures_fall_back_and_do_not_cache(bad):
    c = ctx()
    cli = FakeClient(bad, bad, raw())
    out = R4.replan(c, cli=cli)
    assert len(cli.calls) == 2                                                      # 回し直しは 1 回だけ
    assert out["decision"]["action"] == "stop" and out["fallback"] and out["rejected"][0] == "llm_failed"
    assert not (R4.CACHE / f"{out['llm']['cache']}.json").exists()                  # 壊れた応答を次に使い回さない


def test_missing_key_falls_back_without_raising(monkeypatch):
    def no_key():
        raise RuntimeError("ANTHROPIC_API_KEY が設定されていない")
    monkeypatch.setattr(R4, "client", no_key)
    out = R4.replan(ctx())
    assert out["decision"]["action"] == "stop" and out["fallback"] and "RuntimeError" in out["llm"]["errors"][0]


# ---------------------------------------------------------------- 日本語の報告
def test_forbidden_list_matches_60_paper():
    tree = ast.parse((ROOT / "scripts" / "paper_pdf.py").read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "FORBIDDEN" for t in n.targets))
    assert ast.literal_eval(node.value) == R4.FORBIDDEN


@pytest.mark.parametrize("text, why", [
    ("", "report:empty"),
    ("赤を入れられませんでした。緑を入れます。青も入れます。", "report:sentences:3"),
    ("赤" * (R4.REPORT_MAX_CHARS + 1), "report:too_long"),
    ("Could not place the red cube.", "report:not_japanese"),
    ("赤の誘発が起きました。", "report:forbidden:誘発"),
    ("E7 の 2 番目で止まりました。", "report:forbidden:E7"),
    ("R1v3 が赤をつかめませんでした。", "report:forbidden:R1v3"),
    ("red を入れられませんでした。", "report:forbidden:red"),
    ("赤の立方体を 2 手でつかめませんでした。", "report:forbidden:2 手"),
    ("SmolVLA が赤を入れられませんでした。", "report:forbidden:SmolVLA"),
])
def test_report_check_rejects(text, why):
    assert any(r.startswith(why) for r in R4.check_report(text)), R4.check_report(text)


def test_bad_llm_report_is_replaced_but_action_kept():
    out = R4.replan(ctx(), cli=FakeClient(raw(report="E7 の red は誘発で失敗。G1 は守った。次は blue。")))
    assert out["accepted"] and out["decision"]["action"] == "reorder"
    assert out["report_source"] == "template" and out["report_rejected"]
    assert out["report"] == R4.template_report("reorder", ctx(), "none", ["green", "blue", "red"])


def test_every_template_report_passes_the_check():
    n = 0
    for failed in ("red", "green", "blue"):
        others = [c for c in ("red", "green", "blue") if c != failed]
        for k in range(len(others) + 1):
            for box in itertools.combinations(others, k):
                table = [c for c in ("red", "green", "blue") if c not in box]
                c = ctx(failed=failed, table=table, box=box)
                for a in R4.ACTIONS:
                    col = failed if a in ("next", "skip") else None
                    order = list(R4.candidates(c)) if a == "reorder" else None
                    t = R4.template_report(a, c, col, order)
                    assert R4.check_report(t) == [], (a, failed, box, t)
                    n += 1
    assert n > 0


# ---------------------------------------------------------------- 実行器の包み（偽の io・知覚・判定）
class Clock:
    t = 0.0


class FakeMotion:
    def __init__(self, clock):
        self.clock, self.x_cmd, self.v = clock, np.array([0.45, 0.1, 0.12]), np.zeros(3)

    def set_velocity(self, v):
        self.v = np.asarray(v, float)

    def hand_pose(self, q):
        return self.x_cmd.copy(), None


class FakeWM:
    def __init__(self, colors):
        self.cubes = {c: types.SimpleNamespace(status="seen", in_box=False) for c in colors}
        self.box = {"xy": [0.5, -0.2], "yaw": 0.0, "ok": True}


class Forbidden:
    """実行系が触れてはいけない物（真値・世界）。触れたら失敗する。"""

    def __getattr__(self, k):
        raise AssertionError(f"実行系が真値・世界に触れた: {k}")


class FakePRT:
    def __init__(self, clock, colors=("red", "green", "blue")):
        self.clock = clock
        self.motion = FakeMotion(clock)
        self.wm, self.ready, self.stop_reason, self.closed = FakeWM(colors), True, None, False
        self.external, self.tasks = True, []
        self.startup = {"ok": True}

    def start(self, task, seed):
        self.tasks.append(task)

    def set_task(self, task, seed):
        self.tasks.append(task)

    def reset_chunks(self):
        pass

    def tick(self):
        self.motion.x_cmd = self.motion.x_cmd + self.motion.v * 0.002
        self.clock.t += 0.002

    @property
    def truth(self):
        raise AssertionError("実行系が真値に触れた: truth")


class Pending:
    def __init__(self, value):
        self.value = value

    def ready(self, t):
        return True

    def result(self, t):
        return self.value


class FakeIO:
    def __init__(self, clock):
        self.clock, self.n_compute = clock, []

    def now(self):
        return self.clock.t

    def sense(self, cameras=True):
        return types.SimpleNamespace(cameras={"overhead": None, "wrist": None} if cameras else {},
                                     joints=types.SimpleNamespace(q=np.zeros(7)), gripper=types.SimpleNamespace(width=0.08))

    def compute(self, kind, fn):
        self.n_compute.append(kind)
        return Pending(fn())

    def gripper_move(self, w, speed):
        pass

    @property
    def world(self):
        raise AssertionError("実行系が世界に触れた: world")


class FakeJudge:
    """色 doable は 3 回目の判定で完了（知覚の箱の中へ）。それ以外は完了しない。"""

    def __init__(self, wm, doable):
        self.p = {"gripper_open_m": 0.07, "retreat_tol_m": 0.05, "box_min_pixels": 56}
        self.wm, self.doable, self.color, self.k = wm, set(doable), None, 0
        self.last = {}

    def reset(self, color):
        self.color, self.k = color, 0

    def update(self, sensor, wm, hand):
        self.k += 1
        done = self.color in self.doable and self.k >= 3
        if done:
            self.wm.cubes[self.color].in_box = True
        self.last = {"ok": done, "held_s": 0.0, "box_pixels": 0, "wrist_in_pixels": 0, "depth_ok": True,
                     "retreat_dist_m": 0.0, "gripper_open": True}
        return done


PLANNER_CFG = {"step_timeout_s": 0.4, "retry": 1,
               "return_to_retreat": {"enabled": True, "rise_z": 0.3, "trigger_s": 4.0, "wait_s": 2.0, "timeout_s": 3.0}}
MP = {"gain": 4.0, "xy_max": 0.3, "z_max": 0.3, "z_tol": 0.005, "tol": 0.01}


def make_rt(doable, replan, max_replans=2, ret=True):
    clock = Clock()
    io, prt = FakeIO(clock), FakePRT(clock)
    judge = FakeJudge(prt.wm, doable)
    setup = types.SimpleNamespace(retreat_pose=np.array([0.4, 0.0, 0.3]), gripper_speed=0.1, truth=Forbidden())
    cfg = json.loads(json.dumps(PLANNER_CFG))
    cfg["return_to_retreat"]["enabled"] = ret

    def decompose(text, table, box):
        return {"steps": [c for c in ("red", "green", "blue") if c in table], "reply": "赤・緑・青の順に入れます"}
    ex = U4.ReplanTaskRuntime(io, setup, prt, judge, cfg, MP, decompose, "pick up the {color} cube", replan,
                              max_replans=max_replans)
    return ex, io, prt


def run(ex, limit=200000):
    ex.start("全部片付けて", 191400)
    for _ in range(limit):
        if ex.finished:
            break
        ex.tick()
    assert ex.finished
    return ex.record()


def recording(fn):
    calls = []

    def f(**kw):
        calls.append(kw)
        return fn(**kw)
    f.calls = calls
    return f


@pytest.mark.parametrize("ret", [True, False])
def test_executor_reorders_after_failure_then_falls_back(ret):
    cli = FakeClient(raw())                                   # 2 回目も同じ応答 → 候補が空なので弾かれる（LLM は呼ぶ。finish・stop は選べる）
    rp = recording(lambda **kw: R4.replan_step(**kw, cli=cli))
    ex, io, prt = make_rt({"green", "blue"}, rp, ret=ret)
    rec = run(ex)
    assert [s["color"] for s in rec["steps"]] == ["red", "green", "blue", "red"]
    assert rec["plan"]["steps_initial"] == ["red", "green", "blue"] and rec["plan"]["steps"] == ["red", "green", "blue", "red"]
    assert [ex.step_result(s) for s in rec["steps"]] == ["timeout", "success", "success", "timeout"]
    r1, r2 = rec["replans"]
    assert r1["action"] == "reorder" and r1["intervention_kind"] == "plan_change" and r1["applied"]["order"] == ["green", "blue", "red"]
    assert r2["action"] == "stop" and r2["out"]["fallback"] and r2["intervention_kind"] is None
    assert rec["stopped"]["reply"] == "4 番目の手順（red）を 2 回試して終えられなかったので、止めました"   # 組み直しのない実行器と同じ文
    assert rec["stopped"]["report"] == r2["report"] and rec["stopped"]["by"] == "replan_fallback"
    assert rec["interventions"] == {"plan_change": 1, "replan_requests": 2, "llm_calls": 2}
    assert len(cli.calls) == 2 and io.n_compute.count("llm") == 3                 # 計画 1 回 + 組み直し 2 回（2 回目は弾いた）
    assert ("replan" in [r["kind"] for r in rec["returns"]]) == ret
    # 渡したのは色の名前と記録だけ（真値・世界の物は渡さない）
    kw = rp.calls[0]
    assert set(kw) == {"text", "plan_order", "steps", "perception", "failed_step", "failed_color", "replans_done"}
    assert kw["perception"] == {"table": ["blue", "green", "red"], "in_box": [], "in_hand": [], "lost": []}
    assert rp.calls[1]["perception"]["in_box"] == ["blue", "green"] and rp.calls[1]["replans_done"] == 1
    json.dumps(rec["replans"], ensure_ascii=False)                                   # 記録は JSON に書ける


def test_executor_skip_and_finish():
    ex, _, _ = make_rt({"green", "blue"}, lambda **kw: R4.replan_step(**kw, cli=FakeClient(raw("skip", "red", [])), use_cache=False))
    rec = run(ex)
    assert [s["color"] for s in rec["steps"]] == ["red", "green", "blue"] and rec["steps"][0]["skipped"]
    assert rec["stopped"] is None and rec["ended"] is None and rec["interventions"]["plan_change"] == 1
    ex, _, _ = make_rt({"green", "blue"}, lambda **kw: R4.replan_step(**kw, cli=FakeClient(
        raw("finish", "none", [], report="赤を箱に入れられませんでした。箱にはまだ何も入っていないので、ここで終えます。")),
        use_cache=False))
    rec = run(ex)
    assert rec["ended"]["kind"] == "finish" and rec["stopped"] is None and len(rec["steps"]) == 1
    assert rec["interventions"]["plan_change"] == 1


def test_executor_without_replans_left_stops_like_the_current_executor():
    rp = recording(lambda **kw: pytest.fail("立て直しの上限なのに呼んだ"))
    ex, io, _ = make_rt({"green", "blue"}, rp, max_replans=0)
    rec = run(ex)
    assert rec["stopped"] == {"step": 0, "t": rec["stopped"]["t"],
                              "reply": "1 番目の手順（red）を 2 回試して終えられなかったので、止めました"}
    assert rec["replans"] == [] and io.n_compute.count("llm") == 1


def test_install_swaps_class_like_diag_e7():
    from recovla.runtime.executor import TaskRuntime
    clock = Clock()
    io, prt = FakeIO(clock), FakePRT(clock)
    tr = TaskRuntime(io, types.SimpleNamespace(retreat_pose=np.zeros(3), gripper_speed=0.1), prt, FakeJudge(prt.wm, ()),
                     PLANNER_CFG, MP, lambda *a: {"steps": []}, "{color}")
    ex = U4.install(tr, R4.replan_step, max_replans=3)
    assert ex is tr and isinstance(ex, U4.ReplanTaskRuntime) and ex.max_replans == 3 and ex.replan is R4.replan_step


# ---------------------------------------------------------------- G1
def _g1():
    return _load(ROOT / "scripts" / "check_real_boundary.py", "check_g1_boundary_u4")


def test_g1_executor_u4_is_in_the_strict_scan_and_clean():
    g = _g1()
    assert "src/recovla/runtime/executor_replan.py" in g.runtime_v2_files(None)
    src = (ROOT / "src" / "recovla" / "runtime" / "executor_replan.py").read_text(encoding="utf-8")
    assert g.scan_runtime(src, "src/recovla/runtime/executor_replan.py") == {}
    assert g.scan(src) == {}


def test_g1_replanner_reads_no_truth():
    g = _g1()
    src = (ROOT / "src" / "recovla" / "planner" / "replan_base.py").read_text(encoding="utf-8")
    assert g.scan(src) == {}
    mods = set()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Import):
            mods |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            mods.add(n.module)
    assert not [m for m in mods if m.startswith(("recovla.sim", "recovla.eval", "recovla.expert", "recovla.record",
                                                  "recovla.harness"))]


def test_g1_reachability_from_runtime_with_replanner():
    from recovla.harness.audit import reachable_forbidden
    world = types.SimpleNamespace(data=object())
    ex, io, prt = make_rt({"green", "blue"}, R4.replan_step)
    ex.start("全部片付けて", 191400)
    res = reachable_forbidden(ex, {id(world), id(world.data)}, (), ())
    assert res["violations"] == [] and not res["truncated"]
    ex.planted = world                                                              # 置けば見つかる（検査が効いている）
    assert reachable_forbidden(ex, {id(world), id(world.data)}, (), ())["violations"]


# ---------------------------------------------------------------- 入口（eval_replan_task.py）
@pytest.fixture(scope="module")
def s98():
    return _load(ROOT / "scripts" / "eval_replan_task.py", "s4_u4_entry")


def test_band_check(s98):
    assert s98.band_check(range(191400, 191440)) == ""
    assert s98.band_check(range(191400, 191500)) == ""                             # 評価用のシード 100 個
    assert s98.band_check(range(191490, 191501))                                   # 許した範囲の外にはみ出す
    assert s98.band_check(range(191500, 191502))                                   # 許した範囲の外
    assert s98.band_check(range(190300, 190340))                                   # 別の評価のシードは拒む
    assert s98.band_check(range(44400, 44402))                                     # 動作確認用は既定では拒む
    assert s98.band_check(range(44400, 44402), allow_smoke=True) == ""
    g = json.loads((ROOT / "configs" / "s4_gates.json").read_text(encoding="utf-8"))
    assert next(x for x in g["bands"]["allocations"] if x["id"] == s98.ALLOC)["range"] == [191400, 191499]


@pytest.mark.parametrize("args", [["--arm", "U4", "--trials", "191500:2"], ["--arm", "U0", "--trials", "160000:2"],
                                  ["--arm", "U4", "--trials", "191400:2", "--exec-interval", "10"]])
def test_run_refuses_before_loading_anything(s98, args, capsys):
    base = ["--experiment", "S4U4T", "--condition", "X", "--model", "R1v3"]
    assert s98.cmd_run(args[:2] + base + args[2:]) == 3


def test_run_u4_needs_api_key_unless_dry_run(s98, monkeypatch):
    from recovla.planner import decompose as D
    monkeypatch.setattr(D, "_api_key", lambda: None)
    assert s98.cmd_run(["--arm", "U4", "--experiment", "S4U4T", "--condition", "U4", "--model", "R1v3",
                        "--trials", "191400:1"]) == 3


def test_summary_helpers(s98):
    meta = {"run": 0, "seed": 191400, "truth_success_t": {"green": 40.0, "blue": 75.0},
            "final_in_box": {"red": False, "green": True, "blue": True},
            "plan": {"from_cache": False, "usage": {"input_tokens": 800, "output_tokens": 30}},
            "replans": [{"n": 0, "t_request": 70.0, "failed_color": "red", "action": "reorder", "applied": {"kind": "continue"},
                         "report": "赤を入れられませんでした。", "perception": {"in_box": ["green"]},
                         "out": {"fallback": False, "rejected": [], "report_source": "llm", "report_rejected": [],
                                 "llm": {"calls": 2, "usage": [{"input_tokens": 900, "output_tokens": 60}] * 2}}}]}
    assert s98.truth_box_at(meta, 70.0) == ["green"] and s98.truth_box_at(meta, 80.0) == ["green", "blue"]
    u = s98.llm_usage(meta)
    assert u == {"decompose_calls": 1, "replan_calls": 2, "input_tokens": 2600, "output_tokens": 150}
    row = s98.grading_rows(meta)[0]
    assert row["truth_box_at_request"] == ["green"] and row["perceived_box"] == ["green"] and row["grade"] is None


# ---------------------------------------------------------------- 報告の検査・壊れたキャッシュ・空の並び
def test_report_check_latin_next_to_japanese():
    assert "report:forbidden:red" in R4.check_report("redの立方体を入れられませんでした。")
    assert "report:forbidden:LLM" in R4.check_report("LLMが次の手を選びました。")
    assert "report:forbidden:Claude" in R4.check_report("赤はClaudeが後に回しました。")
    assert R4.check_report("赤の立方体を入れられませんでした。") == []


def test_broken_cache_is_recorded_and_api_is_called_again():
    c = ctx()
    key = R4.cache_key("claude-haiku-5-5", R4.user_message(c))
    R4.CACHE.mkdir(parents=True, exist_ok=True)
    (R4.CACHE / f"{key}.json").write_text("{書きかけ", encoding="utf-8")
    cli = FakeClient()
    out = R4.replan(c, cli=cli)
    assert len(cli.calls) == 1 and out["accepted"] and not out["llm"]["from_cache"]
    assert out["llm"]["errors"][0].startswith("cache_unreadable:")
    assert json.loads((R4.CACHE / f"{key}.json").read_text(encoding="utf-8"))["key"] == key     # 読めた応答で書き直す


def test_empty_order_falls_back_to_stop(monkeypatch):
    monkeypatch.setattr(R4, "apply", lambda decision, ctx: [])
    c = ctx()
    out = R4.replan_step("全部片付けて", c["plan"], c["steps"], c["perception"], 0, "red", 0, cli=FakeClient(), use_cache=False)
    assert out["decision"]["action"] == "stop" and "empty_order" in out["rejected"] and out["fallback"]
    assert out["report"] == R4.template_report("stop", c)
    # 実行器の守り: 続ける手なのに並びが空なら empty_order として止まる（報告なし）
    ex, _, _ = make_rt({"green", "blue"}, lambda **kw: {"decision": {"action": "next", "color": "green", "order": []},
                                                         "next_order": [], "report": "緑を先に入れます。", "fallback": False})
    rec = run(ex)
    r = rec["replans"][0]
    assert r["rejected"] == ["empty_order"] and r["applied"] == {"kind": "stop", "fallback": True}
    assert rec["stopped"]["report"] == "" and rec["stopped"]["by"] == "replan_fallback" and r["intervention_kind"] is None


def test_allow_smoke_rejects_x2_band(s98):
    assert s98.band_check(range(44400, 44404), allow_smoke=True) == ""
    assert "X2" in s98.band_check(range(44404, 44406), allow_smoke=True)
    assert "X2" in s98.band_check(range(44420, 44426), allow_smoke=True)
    assert s98.band_check(range(44424, 44427), allow_smoke=True) == ""


def meta_of(rec, run_i=0, truth=None, in_box=None, t_end=150.0, timed_out=False):
    """実行器の記録から harness/task_loop.py の meta と同じ欄を作る（真値の欄は与える）。"""
    in_box = in_box or {"red": False, "green": True, "blue": True}
    return {"run": run_i, "seed": 191400 + run_i, "text": "全部片付けて", "plan": rec["plan"], "steps": rec["steps"],
            "returns": rec["returns"], "stopped": rec["stopped"], "detected": rec["detected"],
            "truth_success_t": truth or {"green": 20.0, "blue": 40.0}, "final_in_box": in_box,
            "all_three_in_box": all(in_box.values()), "t_end": t_end, "timed_out": timed_out, "replans": rec["replans"],
            "u4": {"interventions": rec["interventions"], "ended": rec["ended"]},
            "audit": {"g1": {"violations": 0}}}


@pytest.fixture(scope="module")
def m56():
    return _load(ROOT / "scripts" / "count_interventions.py", "intervention_s3_t")


def test_56_counts_u4_records(m56):
    rec = run(make_rt({"green", "blue"}, lambda **kw: R4.replan_step(**kw, cli=FakeClient(raw()), use_cache=False))[0])
    assert "replan" in [r["kind"] for r in rec["returns"]]                     # 組み直しで足した戻す動きの種類
    r = m56.count_run(meta_of(rec))
    assert r["interventions"]["replan"] == 1                                    # meta["replans"] から数える
    assert r["interventions"]["retry"] == 2                                     # 赤の 2 回（1 番目と 4 番目の手順）
    assert r["interventions"]["scripted_return"] == 0 and r["stopped"]
    # 戻す動き（replan）が、続ける手を通した組み直しより多い記録は止める
    bad = meta_of(rec)
    bad["returns"] = bad["returns"] + [dict(x) for x in bad["returns"] if x["kind"] == "replan"]
    with pytest.raises(ValueError):
        m56.count_run(bad)
    # 組み直しのない記録（replans なし）は 0
    s3 = meta_of(rec)
    s3.pop("replans")
    s3["returns"] = [x for x in s3["returns"] if x["kind"] != "replan"]
    assert m56.count_run(s3)["interventions"]["replan"] == 0


def _write_runs(d, metas, lat=True):
    d.mkdir(parents=True)
    for i, m in enumerate(metas):
        (d / f"run_{i:04d}.json").write_text(json.dumps(m, ensure_ascii=False, default=float), encoding="utf-8")
        if lat:
            n = (1 if m.get("plan") else 0) + len(m.get("replans") or [])
            (d / f"run_{i:04d}_runtime.json").write_text(json.dumps({"latency": {"llm": [1.0] * n}}), encoding="utf-8")


def _u4_metas():
    out = []
    for i, resp in enumerate([raw(), raw("skip", "red", []), raw("finish", "none", []), raw("stop", "none", [])]):
        rec = run(make_rt({"green", "blue"}, lambda **kw: R4.replan_step(**kw, cli=FakeClient(resp), use_cache=False))[0])
        out.append(meta_of(rec, i, timed_out=(i == 3), t_end=200.0 if i == 3 else 150.0))
    rec = run(make_rt({"red", "green", "blue"}, lambda **kw: R4.replan_step(**kw, cli=FakeClient(), use_cache=False))[0])   # 失敗なし（組み直しなし）。本物の API に届かないよう偽のクライアントを渡す
    out.append(meta_of(rec, 4, truth={"red": 10.0, "green": 20.0, "blue": 40.0}, in_box={"red": True, "green": True, "blue": True}))
    return out


def test_cross_check_98_and_56_agree(s98, tmp_path):
    d = tmp_path / "U4"
    _write_runs(d, _u4_metas())
    s = s98.summarize_condition(d, (0.10, 0.50))
    cc = s["cross_check"]
    assert cc["all_match"], {k: v for k, v in cc["items"].items() if not v["match"]}
    assert "llm_compute" in cc["items"]
    assert s["interventions"]["plan_change"] == 3 and s["interventions"]["retry"] >= 4
    assert s["interventions"]["scripted_return"] == 0 and s["interventions"]["replan_return"] == 2
    assert s["deviations_timed_out"]["n"] == 1 and s["deviations_timed_out"]["runs"][0]["run"] == 3
    assert s["success_at_k"]["0"] == 1 and s["all_three_true"] == 1


def test_cross_check_mismatch_stops_summary(s98, tmp_path, monkeypatch):
    metas = _u4_metas()
    metas[2]["replans"][0]["intervention_kind"] = None                          # finish の試行: 介入の数え方（count_interventions.py）だけが変わる記録
    _write_runs(tmp_path / "outputs" / "v2eval" / "S4U4T" / "U4", metas)
    monkeypatch.setattr(s98, "ROOT", tmp_path)
    monkeypatch.setattr(s98, "OUTD", tmp_path / "outputs" / "s4" / "u4")
    assert s98.main(["summary", "--experiment", "S4U4T", "--conditions", "U4"]) == 1
    summ = json.loads((tmp_path / "outputs" / "s4" / "u4" / "summary_S4U4T.json").read_text(encoding="utf-8"))
    assert not summ["conditions"]["U4"]["cross_check"]["items"]["plan_change"]["match"]
