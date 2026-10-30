"""指示を手順に分ける LLM（計画役、src/recovla/planner/decompose_haiku.py、Haiku 5.5）と、評価の本体 eval_resumable.py の
計画役の選び方・30 秒の採点の分母の確かめ。
CPU だけ。ネットワーク・GPU・シミュレーションは使わない（LLM は偽のクライアント）。

使い方: .venv\\Scripts\\python.exe -m pytest -q tests\\test_decompose_haiku.py -p no:cacheprovider
読むもの: decompose.py・decompose_haiku.py・scripts/eval_resumable.py（importlib）。書くもの: pytest の一時フォルダだけ。
"""
import importlib.util
import json
import pathlib
import sys
import types

import pytest

from recovla.planner import decompose as D
from recovla.planner import decompose_haiku as D4

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = json.dumps({"steps": ["blue", "red"], "reply": "青と赤を入れます"}, ensure_ascii=False)


class _FakeClient:
    """messages.create を模擬する。受けた引数を残し、与えた JSON の文字列を返す。"""

    def __init__(self, raw=RAW):
        self.raw = raw
        self.calls = []
        self.messages = types.SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw)
        return types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text=self.raw)], model=kw["model"],
                                     stop_reason="end_turn", usage=types.SimpleNamespace(input_tokens=821, output_tokens=37))


class _NoCallClient:
    """呼ばれたら失敗する（キャッシュが効いていることの確かめ）。"""

    def __init__(self):
        self.messages = types.SimpleNamespace(create=self._create)

    def _create(self, **kw):
        raise AssertionError("キャッシュがあるのに API を呼んだ")


@pytest.fixture(autouse=True)
def _tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "CACHE", tmp_path / "llm_cache")
    monkeypatch.setattr(D4, "CACHE", tmp_path / "llm_cache")       # 同じ置き場所にして、鍵で混ざらないことも見る


def test_s4_sends_no_temperature_and_disabled_thinking():
    cli = _FakeClient()
    out = D4.decompose("青と赤", ["red", "green", "blue"], [], cli=cli)
    assert out["valid"] and out["steps"] == ["blue", "red"]
    kw = cli.calls[0]
    assert kw["model"] == "claude-haiku-5-5"
    assert "temperature" not in kw and "extra_body" not in kw                    # 温度は送らない（Haiku 5.5 は 400）
    assert kw["thinking"] == {"type": "disabled"}
    assert kw["output_config"]["format"] == {"type": "json_schema", "schema": D.SCHEMA}
    assert kw["system"] == D.SYSTEM
    assert json.loads(kw["messages"][0]["content"])["table_colors"] == ["red", "green", "blue"]


def test_cache_key_contains_model_temperature_none_and_thinking():
    msg = D.user_message("全部片付けて", ["red", "green", "blue"], [])
    k55 = D4.cache_key("claude-haiku-5-5", msg)
    assert k55 != D4.cache_key("claude-haiku-4-5", msg)                           # モデルが鍵に入る
    # 前の版の decompose（Haiku 4.5、温度 0）の鍵とは別（同じ置き場所でも混ざらない）
    D.decompose("全部片付けて", ["red", "green", "blue"], [], cli=_FakeClient())
    legacy_keys = {p.stem for p in D.CACHE.glob("*.json")}
    assert k55 not in legacy_keys and D4.cache_key("claude-haiku-4-5", msg) not in legacy_keys
    out = D4.decompose("全部片付けて", ["red", "green", "blue"], [], cli=_FakeClient())
    assert out["cache"] == k55
    rec = json.loads((D4.CACHE / f"{k55}.json").read_text(encoding="utf-8"))
    assert rec["request"]["temperature"] is None and rec["request"]["thinking"] == {"type": "disabled"}
    assert rec["planner_variant"] == "s4_haiku55" and rec["model"] == "claude-haiku-5-5"


def test_cache_is_used_on_second_call():
    a = D4.decompose("青と赤", ["red", "green", "blue"], [], cli=_FakeClient())
    b = D4.decompose("青と赤", ["red", "green", "blue"], [], cli=_NoCallClient())
    assert a["from_cache"] is False and b["from_cache"] is True
    assert a["steps"] == b["steps"] and a["cache"] == b["cache"]
    # use_cache=False なら呼ぶ
    cli = _FakeClient()
    D4.decompose("青と赤", ["red", "green", "blue"], [], cli=cli, use_cache=False)
    assert len(cli.calls) == 1


def test_return_shape_matches_legacy_plus_marks():
    legacy = D.decompose("青と赤", ["red", "green", "blue"], [], cli=_FakeClient())
    s4 = D4.decompose("青と赤", ["red", "green", "blue"], [], cli=_FakeClient())
    assert set(legacy) <= set(s4)                                                  # 前の版の欄は全部ある
    assert set(s4) - set(legacy) == {"planner_variant", "model"}
    for k in ("steps", "reply", "llm_reply", "llm_steps", "valid", "checks", "usage", "stop_reason"):
        assert s4[k] == legacy[k], k
    assert s4["planner_variant"] == "s4_haiku55"
    # 検査に落ちたときも同じ扱い（手順を空にして尋ねる側に倒す）
    bad = json.dumps({"steps": ["green"], "reply": "緑を入れます"}, ensure_ascii=False)
    lo = D.decompose("緑", ["red"], ["green"], cli=_FakeClient(bad))
    so = D4.decompose("緑", ["red"], ["green"], cli=_FakeClient(bad))
    assert so["valid"] is False and so["steps"] == [] and so["reply"] == lo["reply"] and so["checks"] == lo["checks"]


# ---------------------------------------------------------------- eval_resumable.py の計画役・既定値・採点
@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("s4_resume_planner_t", ROOT / "scripts" / "eval_resumable.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["s4_resume_planner_t"] = m
    spec.loader.exec_module(m)
    return m


def test_planner_choice(mod):
    fn, info = mod.planner_of("s4")
    assert fn is D4.decompose and info["variant"] == "s4_haiku55" and info["model"] == "claude-haiku-5-5"
    fn, info = mod.planner_of("legacy")
    assert fn is D.decompose and info["name"] == "legacy"
    with pytest.raises(SystemExit):
        mod.planner_of("nope")


def test_task_defaults_and_quiet_window(mod):
    ap = mod.build_parser()
    a = ap.parse_args(["task", "--experiment", "S4T", "--condition", "E", "--model", "R1v3", "--trials", "44404:1"])
    assert a.step_timeout_s == 30.0 and a.task_time_limit_s == 200.0 and a.planner == "s4"
    assert mod.quiet_window_of(a) is None                                          # 既定は窓なし
    a = ap.parse_args(["run", "--experiment", "S4T", "--condition", "C", "--model", "R1v3", "--trials", "natural:44404:1"])
    assert a.time_limit_s == 60.0 and mod.quiet_window_of(a) is None
    a = ap.parse_args(["run", "--experiment", "S4T", "--condition", "C", "--model", "R1v3", "--trials", "natural:44404:1",
                       "--quiet-window", "01:45-02:45"])
    assert mod.quiet_window_of(a) == "01:45-02:45"
    a = ap.parse_args(["run", "--experiment", "S4T", "--condition", "C", "--model", "R1v3", "--trials", "natural:44404:1",
                       "--quiet-window", "01:45-02:45", "--ignore-quiet"])
    assert mod.quiet_window_of(a) is None
    a = ap.parse_args(["task", "--experiment", "S4T", "--condition", "E", "--model", "R1v3", "--trials", "44404:1",
                       "--planner", "legacy"])
    assert a.planner == "legacy" and mod.make_spec(a)["planner"] == "legacy"
    lim = mod.limits_of(a, {"planner": {"retry": 1, "step_timeout_s": 30.0}})
    assert lim["step_timeout_s"] == 30.0 and lim["task_time_limit_s"] == 200.0 and lim["retry"] == 1
    a.world_per_trial = True
    rj = json.loads(mod.run_json_text(a, "task", [{"all_three": True}], 1.0, 6, lim))
    assert rj["planner"]["name"] == "legacy"


def _trial(d, i, success, t_success, induce):
    m = {"trial": i, "success": success, "t_success": t_success, "time_limits": {"time_limit_s": 60.0}, "induce": induce}
    (d / f"trial_{i:04d}.json").write_text(json.dumps(m), encoding="utf-8")


def test_score_induced_denominator_is_established_by_T(mod, tmp_path):
    d = tmp_path / "P1"
    d.mkdir()
    P1 = "P1"
    _trial(d, 0, True, 20.0, {"kind": P1, "established": True, "t_established": 5.0})     # 30・60 とも分母・成功
    _trial(d, 1, True, 50.0, {"kind": P1, "established": True, "t_established": 10.0})    # 分母は両方、成功は 60 だけ
    _trial(d, 2, True, 55.0, {"kind": P1, "established": True, "t_established": 40.0})    # 30 の分母に入らない
    _trial(d, 3, False, None, {"kind": P1, "established": False, "t_established": None})  # どちらの分母にも入らない
    _trial(d, 4, False, None, {"kind": P1, "established": True, "t_established": 12.0})   # 分母だけ
    r = mod.score_condition(d, [30.0, 60.0])
    assert r["at"]["30"]["n"] == 3 and r["at"]["30"]["successes"] == 1
    assert r["at"]["60"]["n"] == 4 and r["at"]["60"]["successes"] == 3
    assert r["at"]["30"]["n_trials"] == 5 and r["n_induced"] == 5
    # 意図的な失敗なしの試行は全試行が分母
    n = tmp_path / "nat"
    n.mkdir()
    _trial(n, 0, True, 20.0, {"kind": None, "established": False})
    _trial(n, 1, True, 45.0, {"kind": None, "established": False})
    _trial(n, 2, False, None, {"kind": None, "established": False})
    r = mod.score_condition(n, [30.0, 60.0])
    assert (r["at"]["30"]["successes"], r["at"]["30"]["n"]) == (1, 3)
    assert (r["at"]["60"]["successes"], r["at"]["60"]["n"]) == (2, 3)
    assert "n_induced" not in r
