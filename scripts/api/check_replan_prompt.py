"""LLM への指示文の直し（docs/api/r1_protocol.md 5 節の 1）の回帰の確かめ。以前の評価（実験 CLC）の L2v2 の記録にある
組み直しの入力を、以前の指示文（planner/replan_event）と直した指示文（planner/replan_improved）で LLM（claude-sonnet-5-5）に聞き直し、
選ぶ手が変わるかを数える。

    .venv\\Scripts\\python.exe scripts\\api\\check_replan_prompt.py [--n-ok 15] [--dry-run]

読むもの: outputs/api/failure_causes_CLC.json（「候補が残るのに LLM が止めた」の 20 本）、outputs/v2eval/CLC*/L2v2/run_NNNN.json の
  replans[].out.context（その組み直しで LLM に渡した入力）。
聞き直す組:
- flagged: 20 本それぞれの、candidates が空でないのに stop を選んだ組み直し（以前の指示文で止めた所）
- ok: 3 個とも入った試行の、続ける手（next・reorder・skip）を選んだ組み直しから、シードの小さい順に --n-ok 個（正しかった手が
  直した指示文で止める手に変わらないか）
どちらも、以前の指示文と直した指示文で 1 回ずつ聞き直す。キャッシュは outputs/api/r1_prompt_cache/ だけを使う
（評価のキャッシュ outputs/llm_cache/ は読まない。直した指示文は KIND が違うので鍵も違う）。
直した指示文の入力の場面の要約は、記録にある 4 つの欄だけ（stacked_in_box などの新しい欄は空）。回数の欄（tries・runs）は
記録の steps から作る。書くもの: outputs/api/check_replan_prompt.json。
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
MODEL = "claude-sonnet-5-5"
CONTINUE = ("next", "reorder", "skip", "skip_notify")


def records(root: pathlib.Path, exps) -> dict:
    out = {}
    for e in exps:
        for p in sorted((root / e / "L2v2").glob("run_[0-9][0-9][0-9][0-9].json")):
            m = json.loads(p.read_text(encoding="utf-8"))
            out[int(m["seed"])] = m
    return out


def pick(fc: dict, recs: dict, n_ok: int) -> list:
    flagged = [r["seed"] for r in fc["conds"]["L2v2"]["failed_trials"] if "llm_stop_with_candidates" in r["overlays"]]
    rows = []
    for s in flagged:
        for r in recs[s].get("replans") or []:
            o = r.get("out") or {}
            if r.get("action") == "stop" and o.get("candidates"):
                rows.append({"group": "flagged", "seed": s, "n": r["n"], "ctx": o["context"], "rec_action": r["action"],
                             "candidates": o["candidates"]})
    ok = []
    for s in sorted(recs):
        m = recs[s]
        if not m.get("all_three_in_box"):
            continue
        for r in m.get("replans") or []:
            o = r.get("out") or {}
            if r.get("action") in ("next", "reorder", "skip") and not o.get("fallback"):
                ok.append({"group": "ok", "seed": s, "n": r["n"], "ctx": o["context"], "rec_action": r["action"],
                           "candidates": o.get("candidates")})
    return rows + ok[:n_ok]


def ask(row: dict, cache: pathlib.Path) -> dict:
    from recovla.planner import replan_event as RL
    from recovla.planner import replan_improved as RR
    c = row["ctx"]
    f = c["failed"]
    ev = f.get("event", "timeout")
    old_ctx = RL.make_context(c["instruction"], c["plan"], c["steps"], c["perception"], f["step"], f["color"], c["replans_done"], ev)
    old = RL.finish_step(old_ctx, RL.replan(old_ctx, model=MODEL, cache_dir=cache / "old"))
    new_ctx = RR.make_context(c["instruction"], c["plan"], c["steps"], c["perception"], f["step"], f["color"], c["replans_done"], ev)
    new = RR.finish_step(new_ctx, RR.replan(new_ctx, model=MODEL, cache_dir=cache / "new"))

    def brief(o):
        return {"action": o["decision"]["action"], "color": o["decision"]["color"], "order": o["decision"]["order"],
                "accepted": o["accepted"], "rejected": o["rejected"], "next_order": o["next_order"], "reason": o.get("reason"),
                "from_cache": (o.get("llm") or {}).get("from_cache"), "usage": (o.get("llm") or {}).get("usage")}
    return {"old": brief(old), "new": brief(new), "new_candidates": RR.candidates(new_ctx), "old_candidates": old["candidates"]}


def summarize(rows: list) -> dict:
    out = {}
    for g in ("flagged", "ok"):
        rs = [r for r in rows if r["group"] == g]
        old_c = sum(1 for r in rs if r["old"]["action"] in CONTINUE and r["old"]["accepted"])
        new_c = sum(1 for r in rs if r["new"]["action"] in CONTINUE and r["new"]["accepted"])
        out[g] = {"n": len(rs), "recorded_continue": sum(1 for r in rs if r["rec_action"] in CONTINUE),
                  "old_prompt_continue": old_c, "new_prompt_continue": new_c,
                  "new_rejected": sum(1 for r in rs if not r["new"]["accepted"]),
                  "changed_vs_recorded": sum(1 for r in rs if r["new"]["action"] != r["rec_action"]),
                  "flip_continue_to_stop": sum(1 for r in rs if r["rec_action"] in CONTINUE and r["new"]["action"] in ("stop", "finish")),
                  "new_actions": {a: sum(1 for r in rs if r["new"]["action"] == a) for a in sorted({r["new"]["action"] for r in rs})},
                  "old_actions": {a: sum(1 for r in rs if r["old"]["action"] == a) for a in sorted({r["old"]["action"] for r in rs})}}
    tok = [u for r in rows for k in ("old", "new") for u in (r[k]["usage"] or [])]
    out["tokens"] = {"input": sum(u.get("input_tokens", 0) for u in tok), "output": sum(u.get("output_tokens", 0) for u in tok),
                     "calls": len(tok)}
    return out


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:                                    # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fc", default=str(ROOT / "outputs" / "api" / "failure_causes_CLC.json"))
    ap.add_argument("--root", default=str(ROOT / "outputs" / "v2eval"))
    ap.add_argument("--exps", default="CLC,CLC_B")
    ap.add_argument("--n-ok", type=int, default=15)
    ap.add_argument("--cache", default=str(ROOT / "outputs" / "api" / "r1_prompt_cache"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "api" / "r1_prompt_check.json"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    fc = json.loads(pathlib.Path(a.fc).read_text(encoding="utf-8"))
    recs = records(pathlib.Path(a.root), a.exps.split(","))
    rows = pick(fc, recs, a.n_ok)
    print(f"聞き直す組: flagged {sum(r['group'] == 'flagged' for r in rows)}、ok {sum(r['group'] == 'ok' for r in rows)}")
    if a.dry_run:
        from recovla.planner import replan_improved as RR
        r = rows[0]
        c, f = r["ctx"], r["ctx"]["failed"]
        print(RR.user_message(RR.make_context(c["instruction"], c["plan"], c["steps"], c["perception"], f["step"], f["color"],
                                              c["replans_done"], f.get("event", "timeout"))))
        return 0
    cache = pathlib.Path(a.cache)
    out_rows = []
    for r in rows:
        res = ask(r, cache)
        out_rows.append({k: v for k, v in r.items() if k != "ctx"} | res | {"ctx": r["ctx"]})
        print(f"{r['group']:7s} {r['seed']} n{r['n']} 記録 {r['rec_action']:8s} 前の指示文 {res['old']['action']:11s} "
              f"直した指示文 {res['new']['action']}{'' if res['new']['accepted'] else '(弾いた)'}", flush=True)
    res = {"note": "過去の判定済みの記録の入力を聞き直した（事後の確かめ。評価の種は回していない）", "model": MODEL,
           "summary": summarize(out_rows), "rows": out_rows}
    p = pathlib.Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res["summary"], ensure_ascii=False, indent=1))
    print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
