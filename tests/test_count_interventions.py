"""介入の回数の数え方（scripts/count_interventions.py）の検査。合成の記録だけで回す（CPU、シミュレーションなし）。"""
import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _script():
    spec = importlib.util.spec_from_file_location("intervention_s3", ROOT / "scripts" / "count_interventions.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M = _script()


def step(j, color, attempts, judged=True, t_judge=10.0):
    """executor.py の steps[] と同じ欄。attempts は各試みの t_begin の列。"""
    return {"step": j, "color": color, "attempts": [{"attempt": i, "t_begin": t, "t_judge": None} for i, t in enumerate(attempts)],
            "t_start": attempts[0], "judged_complete": judged, "t_judge": t_judge if judged else None, "t_end": 0.0}


def ret(j, attempt, kind="retry", t_begin=30.0):
    return {"step": j, "attempt": attempt, "kind": kind, "t_begin": t_begin, "t_arrive": t_begin + 3, "t_end": t_begin + 3,
            "arrived": True, "opened": False, "judged": False}


def meta(steps, returns=(), truth=None, in_box=None, stopped=None, t_end=60.0, plan=None, run=0):
    """harness/task_loop.py の meta と同じ欄（使うものだけ）。"""
    in_box = in_box if in_box is not None else {"red": True, "green": True, "blue": True}
    if plan is None:
        plan = {"steps": ["red", "green", "blue"][:len(steps)] or ["red", "green", "blue"], "from_cache": True,
                "usage": {"input_tokens": 700, "output_tokens": 30}}
    return {"run": run, "seed": 145000 + run, "plan": plan, "steps": list(steps), "returns": list(returns),
            "stopped": stopped, "truth_success_t": truth or {}, "final_in_box": in_box,
            "all_three_in_box": all(in_box.values()), "t_end": t_end, "timed_out": False}


def clean(run=0, t_end=60.0):
    return meta([step(0, "red", [2.0]), step(1, "green", [20.0]), step(2, "blue", [40.0])],
                truth={"red": 15.0, "green": 35.0, "blue": 55.0}, run=run, t_end=t_end)


# ------------------------------------------------------------------ count_run

def test_no_intervention():
    r = M.count_run(clean())
    assert r["total"] == 0 and r["success"] and r["judged_all_done"]
    assert r["interventions"] == {"scripted_return": 0, "retry": 0, "replan": 0, "judge_override": 0}
    assert r["llm_calls"] == 1                     # latency が無ければ plan があれば 1
    assert r["t_all_three"] == 55.0 and r["n_in_box"] == 3


def test_retry_genuine_and_judge_only():
    # 緑: 30 s に時間切れ。真の成功 35 s（時間切れの後）→ 本当のやり直し
    m = meta([step(0, "red", [2.0]), step(1, "green", [20.0, 53.0]), step(2, "blue", [70.0])],
             [ret(1, 0, t_begin=50.0)], truth={"red": 15.0, "green": 60.0, "blue": 85.0})
    r = M.count_run(m)
    assert r["interventions"]["retry"] == 1 and r["retry_genuine"] == 1 and r["retry_judge_only"] == 0
    # 真の成功が時間切れの時刻ちょうど → 判定だけのやり直し（境界は含む）
    m2 = meta([step(0, "red", [2.0]), step(1, "green", [20.0, 53.0]), step(2, "blue", [70.0])],
              [ret(1, 0, t_begin=50.0)], truth={"red": 15.0, "green": 50.0, "blue": 85.0})
    r2 = M.count_run(m2)
    assert r2["retry_judge_only"] == 1 and r2["retry_genuine"] == 0 and r2["total"] == 1


def test_retry_without_return_uses_next_attempt():
    # 置いた後・時間切れの後の戻す動きが無効（returns なし）でも attempts から数え、時間切れの時刻は次の試みの t_begin
    m = meta([step(0, "red", [2.0, 32.0])], truth={"red": 31.0}, in_box={"red": True, "green": False, "blue": False})
    r = M.count_run(m)
    assert r["interventions"]["retry"] == 1 and r["retries"][0]["t_fire"] == 32.0 and r["retry_judge_only"] == 1
    assert not r["success"]


def test_retry_count_mismatch_raises():
    m = meta([step(0, "red", [2.0])], [ret(0, 0)])           # 戻す動き（retry）があるのに、やり直しの試みがない
    with pytest.raises(ValueError):
        M.count_run(m)


def test_unknown_return_kind_raises():
    m = meta([step(0, "red", [2.0])], [ret(0, 0, kind="goal")])
    with pytest.raises(ValueError):
        M.count_run(m)


def test_placed_return_is_scripted_not_retry():
    m = meta([step(0, "red", [2.0]), step(1, "green", [20.0]), step(2, "blue", [40.0])], [ret(0, 0, kind="placed", t_begin=12.0)],
             truth={"red": 10.0, "green": 35.0, "blue": 55.0})
    r = M.count_run(m)
    assert r["interventions"]["scripted_return"] == 1 and r["interventions"]["retry"] == 0 and r["total"] == 1


def test_judge_override_and_replan():
    s = [step(0, "red", [2.0]), step(1, "green", [20.0]), step(2, "blue", [40.0])]
    s[1]["t_judge"] = None                                    # 判定の時刻なしで完了 → 判定の上書き
    plans = [{"steps": ["red", "green", "blue"]}, {"steps": ["green", "blue"]}]
    r = M.count_run(meta(s, plan=plans, truth={"red": 1.0, "green": 2.0, "blue": 3.0}))
    assert r["interventions"]["judge_override"] == 1 and r["interventions"]["replan"] == 1 and r["total"] == 2


def test_llm_calls_from_latency():
    r = M.count_run(clean(), latency={"llm": [1.1, 1.3], "policy": [0.3]})
    assert r["llm_calls"] == 2 and r["llm_latency_s"] == [1.1, 1.3]
    assert r["total"] == 0                                   # LLM の呼び出しは介入に数えない


def test_stopped_but_all_in_box():
    m = meta([step(0, "red", [2.0]), step(1, "green", [20.0, 53.0], judged=False)], [ret(1, 0, t_begin=50.0)],
             truth={"red": 15.0, "green": 40.0, "blue": 45.0}, stopped={"step": 1, "t": 80.0, "reply": "x"})
    r = M.count_run(m)
    assert r["stopped"] and r["success"] and not r["judged_all_done"] and r["stopped_step"] == 1


# ------------------------------------------------------------------ まとめ

def _rows():
    a = M.count_run(clean(0, t_end=60.0))                                                  # 0 回で成功
    b = M.count_run(meta([step(0, "red", [2.0]), step(1, "green", [20.0, 53.0]), step(2, "blue", [70.0])],
                         [ret(1, 0, t_begin=50.0)], truth={"red": 15.0, "green": 50.0, "blue": 85.0}, run=1, t_end=90.0))  # 1 回（判定だけ）
    c = M.count_run(meta([step(0, "red", [2.0, 33.0]), step(1, "green", [50.0, 83.0]), step(2, "blue", [100.0])],
                         [ret(0, 0, t_begin=30.0), ret(1, 0, t_begin=80.0)],
                         truth={"red": 45.0, "green": 95.0, "blue": 120.0}, run=2, t_end=125.0))  # 2 回で成功
    d = M.count_run(meta([step(0, "red", [2.0]), step(1, "green", [20.0, 53.0], judged=False)], [ret(1, 0, t_begin=50.0)],
                         truth={"red": 15.0}, in_box={"red": True, "green": False, "blue": False},
                         stopped={"step": 1, "t": 85.0, "reply": "x"}, run=3, t_end=85.0))  # 1 回で失敗
    return [a, b, c, d]


def test_success_at_k_cumulative():
    rows = _rows()
    sk = M.success_at_k(rows)
    assert [sk[str(k)]["k"] for k in range(4)] == [1, 2, 3, 3]          # 少なくとも k=3 まで
    assert sk["0"]["n"] == 4
    res = M.aggregate(rows)
    g = res["success_at_k_genuine_only"]
    assert [g[str(k)]["k"] for k in range(3)] == [2, 2, 3]              # 判定だけのやり直しを数えないと b は 0 回
    assert res["interventions_total"] == 4 and res["interventions_per_run"] == 1.0
    assert res["distribution"] == {"0": 1, "1": 2, "2": 1}
    assert res["retry_breakdown"]["judge_only"] == 1 and res["retry_breakdown"]["genuine"] == 3


def test_time_metrics():
    res = M.aggregate(_rows())
    t = res["time"]
    assert t["sim_s_total"] == 360.0 and t["sim_s_mean"] == 90.0
    assert t["s_per_success"] == 120.0 and t["success_per_hour"] == 30.0     # 3 本 ÷ 0.1 時間
    assert t["cubes_in_box_mean"] == pytest.approx(2.5)
    assert t["success_by_time"]["60"] == 1 and t["success_by_time"]["90"] == 2 and t["success_by_time"]["120"] == 3
    assert res["stopped"]["n"] == 1 and res["stopped"]["all_three_in_box"] == 0


def test_baseline_check():
    res = M.aggregate(_rows())
    base = {"source": "x", "interventions_per_run": 1.0, "llm_calls_per_run": 1.0, "success_at_k": {"0": 1, "1": 2, "2": 3}, "n": 4}
    assert M.baseline_check(res, base)["all_match"]
    base["success_at_k"]["1"] = 5
    bc = M.baseline_check(res, base)
    assert not bc["all_match"] and not bc["items"]["success_at_1"]["match"]


def test_phrases_use_values_and_no_forbidden_words():
    res = M.aggregate(_rows())
    p = M.phrases(res)
    assert "4 回の作業" in p["success_at_k"] and "1 回、" in p["success_at_k"]
    assert M.forbidden_hits(p) == []
    assert M.forbidden_hits({"x": "\u53f0\u672c\u306e\u52d5\u304d"})    # 一覧が実際に効いている（使わない語の見本をエスケープで書く）


def test_main_end_to_end(tmp_path):
    d = tmp_path / "E7"
    d.mkdir()
    for i, m in enumerate([clean(0), clean(1)]):
        (d / f"run_{i:04d}.json").write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
        (d / f"run_{i:04d}_runtime.json").write_text(json.dumps({"latency": {"llm": [1.0]}}), encoding="utf-8")
    (d / "run.json").write_text("{}", encoding="utf-8")                  # 集計の run.json は読まない
    out = tmp_path / "out.json"
    assert M.main(["--dir", str(d), "--out", str(out)]) == 0
    r = json.loads(out.read_text(encoding="utf-8"))
    assert r["n"] == 2 and r["llm"]["calls_total"] == 2 and r["success_at_k"]["0"]["k"] == 2
    assert r["phrases_forbidden_hits"] == []
