"""条件の一覧（JSON の計画）を、10 本のブロック（block）ずつ交互に、3 枠で回す運用の道具。試行の中身は各条件のコマンド
（評価の入口 scripts/eval_resumable.py の包み）が決め、この道具は呼ぶ順と並行の数だけを決める。

    .venv\\Scripts\\python.exe scripts\\real\\rotate_jobs.py <計画.json> [--dry-run]

計画: {"name": 名, "experiment": 実験名, "lanes": 3, "block": 10, "max_lead_blocks": 2,
       "conditions": {条件名: {"n": 試行の数, "argv": [スクリプトのパス（作業場所から）, 引数…]}}}
  argv には --max-new を入れない（この道具が --max-new <block> を足す）。
決まり: 同じ条件を同時に 2 つ回さない。どの条件も、いちばん遅延した条件より max_lead_blocks ブロックを超えて先に進まない。
  ブロックの呼び出しは、終了コード 0 か「max_new で止まった」なら済み。それ以外は同じブロックを後で回し直し、3 回続けて
  済まなければ止める。
止める: outputs\\s4\\STOP（評価の入口も見る）か outputs\\real\\rotate\\<名>.STOP。続きは同じコマンドで。
書くもの: outputs\\real\\rotate\\<名>.json（呼び出しの記録）・<名>_logs\\。
"""
import argparse
import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUTD = ROOT / "outputs" / "real" / "rotate"
MAX_FAIL, STAGGER_S = 3, 90.0


def n_done(exp: str, cond: str) -> int:
    d = ROOT / "outputs" / "v2eval" / exp / cond
    if not d.is_dir():
        return 0
    return sum(1 for _ in d.glob("trial_[0-9][0-9][0-9][0-9].json")) + sum(1 for _ in d.glob("run_[0-9][0-9][0-9][0-9].json"))


def progress(exp: str, cond: str) -> dict:
    try:
        return json.loads((ROOT / "outputs" / "v2eval" / exp / cond / "progress.json").read_text(encoding="utf-8"))
    except Exception:                                    # noqa: BLE001
        return {}


def pick(plan: dict, running: set, fails: dict) -> str:
    exp, conds, block = plan["experiment"], plan["conditions"], int(plan.get("block", 10))
    done = {c: n_done(exp, c) for c in conds}
    left = [c for c in conds if done[c] < int(conds[c]["n"])]
    if not left:
        return ""
    low = min(done[c] for c in left)
    lead = int(plan.get("max_lead_blocks", 2)) * block
    cand = [c for c in left if c not in running and fails.get(c, 0) < MAX_FAIL and done[c] < low + lead]
    cand.sort(key=lambda c: (done[c], list(conds).index(c)))
    return cand[0] if cand else ""


def argv_of(plan: dict, cond: str) -> list:
    a = list(plan["conditions"][cond]["argv"])
    return [sys.executable, str(ROOT / a[0])] + a[1:] + ["--max-new", str(int(plan.get("block", 10)))]


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:                                    # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("plan")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    plan = json.loads(pathlib.Path(a.plan).read_text(encoding="utf-8"))
    name, exp, conds = plan["name"], plan["experiment"], plan["conditions"]
    if a.dry_run:
        for c in conds:
            print(f"{c}（済み {n_done(exp, c)}/{conds[c]['n']}）: " + " ".join(argv_of(plan, c)[1:]))
        return 0
    OUTD.mkdir(parents=True, exist_ok=True)
    logs = OUTD / f"{name}_logs"
    logs.mkdir(exist_ok=True)
    log_p = OUTD / f"{name}.json"
    hist = json.loads(log_p.read_text(encoding="utf-8")) if log_p.is_file() else []
    stops = (ROOT / "outputs" / "s4" / "STOP", OUTD / f"{name}.STOP")
    lanes = int(plan.get("lanes", 3))
    live, fails, last = {}, {}, 0.0

    def save():
        log_p.write_text(json.dumps(hist, ensure_ascii=False, indent=1), encoding="utf-8")

    while True:
        for c, (p, rec, f) in list(live.items()):
            rc = p.poll()
            if rc is None:
                continue
            f.close()
            pr = progress(exp, c)
            reason = str(pr.get("stop_reason") or "")
            ok = rc == 0 or (rc == 1 and reason.startswith("max_new"))
            rec.update(end=time.strftime("%Y-%m-%d %H:%M:%S"), rc=rc, status=pr.get("status"), stop_reason=reason,
                       done_after=n_done(exp, c), ok=ok)
            fails[c] = 0 if ok else fails.get(c, 0) + 1
            print(f"[rotate] {rec['end']} {c} 終わり rc={rc} {pr.get('status')} {reason} 済み {rec['done_after']}", flush=True)
            del live[c]
            save()
        stop = next((s for s in stops if s.is_file()), None)
        if all(n_done(exp, c) >= int(conds[c]["n"]) for c in conds) and not live:
            print(f"[rotate] {name}: 全条件そろった", flush=True)
            return 0
        if any(v >= MAX_FAIL for v in fails.values()) and not live:
            print(f"[rotate] 同じ条件の塊が {MAX_FAIL} 回続けて済まなかった: {fails}。止める", flush=True)
            return 2
        if stop and not live:
            print(f"[rotate] 止める合図 {stop}", flush=True)
            return 1
        if not stop and len(live) < lanes and time.time() - last >= STAGGER_S:
            c = pick(plan, set(live), fails)
            if c:
                k = sum(1 for h in hist if h["cond"] == c)
                rec = {"cond": c, "start": time.strftime("%Y-%m-%d %H:%M:%S"), "done_before": n_done(exp, c), "call": k}
                f = open(logs / f"{c}_{k:03d}.log", "w", encoding="utf-8")
                p = subprocess.Popen(argv_of(plan, c), cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
                rec["pid"] = p.pid
                hist.append(rec)
                live[c] = (p, rec, f)
                last = time.time()
                print(f"[rotate] {rec['start']} {c} 始め（済み {rec['done_before']}、pid {p.pid}）", flush=True)
                save()
        time.sleep(5)


if __name__ == "__main__":
    raise SystemExit(main())
