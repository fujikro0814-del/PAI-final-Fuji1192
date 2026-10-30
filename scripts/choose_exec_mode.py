"""実行のしかた（同期か非同期か・行動列から実行する行数）と安全フィルタの有無を、決まった組で回して選ぶ。

scripts/check_slow_approach.py・95_s3_round2.py が、試行の並列実行と集計（run_args・run_pool・e3_stats・ARMS）の部品として読む。

    .venv\\Scripts\\python.exe scripts\\choose_exec_mode.py run-a [--parallel 3]   # 段 A: 失敗なしの 99 試行 × 5 組（R1v2 の 2 万ステップ）
    .venv\\Scripts\\python.exe scripts\\choose_exec_mode.py decide-a               # 実行のしかたの候補と、安全フィルタの仮の判定
    .venv\\Scripts\\python.exe scripts\\choose_exec_mode.py run-b [--parallel 3]   # 段 B: P1（掴み損ね）50 試行、R・N × {N10, 候補}
    .venv\\Scripts\\python.exe scripts\\choose_exec_mode.py decide-b               # R と N の立て直しの差が保たれるかを見て確定

組の名前: N10＝naive・10 行、S10＝sync・10 行、N6＝naive・6 行。_sf＝安全フィルタあり、_nosf＝なし。
試行は scripts/eval_trials.py run で回し、監査（gate.require）を通らない走行の数字は出さない。決まりは回す前に固定した。
結果: docs/results/s3_round0_a.json・s3_round0_b.json。
"""
import argparse
import importlib.util
import json
import subprocess
import sys
import time

from recovla.common import config

OUT = config.path(config.load()["paths"]["outputs"]) / "v2eval"
RES = config.ROOT / "docs" / "results"
EXP_A, EXP_B = "V3R0A", "V3R0B"
TRIALS_A, TRIALS_B = "natural:199433:33", "induced:199010:50"
MODEL_A = "R1v2_20000"
MODELS_B = {"R": "R1v2_20000", "N": "N1v2"}                 # N1v2 はチェックポイントの選択で選んだ 3 万ステップ
ARMS = {"N10": ("naive", 10), "S10": ("sync", 10), "N6": ("naive", 6)}
COND_A = ["N10_sf", "N10_nosf", "S10_sf", "S10_nosf", "N6_sf"]
PASS_POINTS = 5            # 候補 − N10（フィルタあり）≥ +5（99 中）
TIE_POINTS = 3             # N6 と S10 がどちらも通り、差が 3 以内なら N6
SF_DROP_POINTS = 3.0       # 安全フィルタ: あり の成功が なし より 3 ポイント以上低ければ外す
E3_DIFF_DROP = 2           # 段 B: 「R だけ立て直し − N だけ立て直し」が N10 より 2 以上小さくならない
R_REC_DROP = 0.10          # 段 B: R の立て直し／実際に失敗した数が N10 より 10 ポイント以上下がらない


def _mod(name, file):
    spec = importlib.util.spec_from_file_location(name, config.ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def run_args(exp, cond, model, trials, arm, safety, induce=None) -> list:
    mode, s = ARMS[arm]
    cmd = [sys.executable, str(config.ROOT / "scripts" / "eval_trials.py"), "run", "--experiment", exp, "--condition", cond,
           "--model", model, "--trials", trials, "--mode", mode, "--exec-interval", str(s)]
    if not safety:
        cmd.append("--no-safety")
    if induce:
        cmd += ["--induce", induce]
    return cmd


def run_pool(exp, jobs: dict, parallel: int) -> None:
    """jobs = {条件: コマンド}。済んだ条件（run.json あり）は飛ばす。同時に parallel 本まで。"""
    logs = OUT / f"{exp}_logs"
    logs.mkdir(parents=True, exist_ok=True)
    todo = [(c, cmd) for c, cmd in jobs.items() if not (OUT / exp / c / "run.json").is_file()]
    running = []
    while todo or running:
        while todo and len(running) < parallel:
            c, cmd = todo.pop(0)
            f = open(logs / f"{c}.log", "ab")
            running.append((c, subprocess.Popen(cmd, cwd=config.ROOT, stdout=f, stderr=subprocess.STDOUT), f))
            print(f"[{time.strftime('%H:%M:%S')}] start {exp}/{c}", flush=True)
        time.sleep(10)
        for item in list(running):
            c, p, f = item
            if p.poll() is not None:
                f.close()
                running.remove(item)
                print(f"[{time.strftime('%H:%M:%S')}] end {exp}/{c} exit {p.returncode}", flush=True)
                if p.returncode != 0:
                    raise SystemExit(f"{exp}/{c} exited with {p.returncode}（ログ {logs / (c + '.log')}）")


def collect(exp, cond) -> dict:
    from recovla.eval import report as R
    return {(r["seed"], r["target"]): r for r in R.collect([OUT / exp / cond])}


def cmd_run_a(a) -> None:
    jobs = {}
    for c in COND_A:
        arm, sf = c.split("_")
        jobs[c] = run_args(EXP_A, c, MODEL_A, TRIALS_A, arm, sf == "sf")
    run_pool(EXP_A, jobs, a.parallel)


def cmd_decide_a(a) -> None:
    from recovla.eval import gate
    e50 = _mod("e50", "compare_subtask.py")
    gsum = gate.require([OUT / EXP_A / c for c in COND_A], "段階 3 の 0 周目 段 A（0126・0127）")
    rows = {c: collect(EXP_A, c) for c in COND_A}
    succ = {c: sum(bool(r["success"]) for r in rows[c].values()) for c in COND_A}
    n = {c: len(rows[c]) for c in COND_A}
    if len(set(n.values())) != 1 or n["N10_sf"] != 99:
        raise SystemExit(f"試行の数が揃っていない: {n}")
    pairs = {c: e50.paired_binary(rows[c], rows["N10_sf"], e50.SUCC) for c in ("S10_sf", "N6_sf", "N10_nosf", "S10_nosf")}
    diff = {c: succ[c] - succ["N10_sf"] for c in ("S10_sf", "N6_sf")}
    passed = [c for c in ("S10_sf", "N6_sf") if diff[c] >= PASS_POINTS]
    if len(passed) == 2 and abs(diff["N6_sf"] - diff["S10_sf"]) <= TIE_POINTS:
        cand = "N6"
    elif passed:
        cand = max(passed, key=lambda c: diff[c]).split("_")[0]
    else:
        cand = "N10"
    # 安全フィルタの仮の判定: 選んだ方式のあり・なしの対で読む（naive の方式は N10 の対）
    sf_pair = "S10" if cand == "S10" else "N10"
    on, off = rows[f"{sf_pair}_sf"], rows[f"{sf_pair}_nosf"]
    sfp = e50.paired_binary(on, off, e50.SUCC)
    contact = e50.paired_binary(on, off, e50.CONTACT)
    drop = (sfp["y_rate"] - sfp["x_rate"]) * 100
    res = {"what": "段階 3 の 0 周目 段 A（0126 の 4・0127）", "experiment": EXP_A, "trials": TRIALS_A, "model": MODEL_A,
           "successes": succ, "n": n, "diff_vs_N10_sf": diff, "paired_vs_N10_sf": pairs,
           "rule_l1": f"候補 − N10_sf ≥ +{PASS_POINTS}。N6 と S10 がどちらも通り差が {TIE_POINTS} 以内なら N6",
           "passed": passed, "candidate": cand,
           "rule_l2": f"0107 の 5: あり の成功が なし より {SF_DROP_POINTS} ポイント以上低ければ外す（{sf_pair} の対）",
           "safety_pair": sf_pair, "safety_success": sfp, "safety_contact": contact, "safety_drop_points": drop,
           "safety_keep_tentative": drop < SF_DROP_POINTS,
           "next": "段 B（E3 の守り）" if cand != "N10" else "候補なし: N10 のまま（段 B は回さない）",
           "g_audit": gsum, "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "s3_round0_a.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("successes", "diff_vs_N10_sf", "candidate", "safety_drop_points",
                                          "safety_keep_tentative", "next")}, ensure_ascii=False, indent=1, default=float))


def _decision_a() -> dict:
    p = RES / "s3_round0_a.json"
    if not p.is_file():
        raise SystemExit(f"{p} がない（decide-a を先に）")
    return json.loads(p.read_text(encoding="utf-8"))


def cmd_run_b(a) -> None:
    da = _decision_a()
    if da["candidate"] == "N10":
        raise SystemExit("段 A で候補がない（N10 のまま）。段 B は回さない")
    sf = bool(da["safety_keep_tentative"])
    jobs = {}
    for who, model in MODELS_B.items():
        for arm in ("N10", da["candidate"]):
            jobs[f"{who}_{arm}"] = run_args(EXP_B, f"{who}_{arm}", model, TRIALS_B, arm, sf, induce="P1")
    run_pool(EXP_B, jobs, a.parallel)


def e3_stats(R, N) -> dict:
    keys = sorted(set(R) & set(N))
    both = [k for k in keys if R[k]["induce_established"] and N[k]["induce_established"]]
    r_only = sum(bool(R[k]["recovered"]) and not N[k]["recovered"] for k in both)
    n_only = sum(bool(N[k]["recovered"]) and not R[k]["recovered"] for k in both)
    est = [k for k in R if R[k]["induce_established"]]
    rec = sum(bool(R[k]["recovered"]) for k in est)
    return {"pairs_both_established": len(both), "r_only": r_only, "n_only": n_only, "diff": r_only - n_only,
            "r_established": len(est), "r_recovered": rec, "r_recovery_rate": rec / len(est) if est else None}


def cmd_decide_b(a) -> None:
    from recovla.eval import gate
    da = _decision_a()
    cand = da["candidate"]
    conds = [f"{w}_{arm}" for w in MODELS_B for arm in ("N10", cand)]
    gsum = gate.require([OUT / EXP_B / c for c in conds], "段階 3 の 0 周目 段 B（0126・0127）")
    rows = {c: collect(EXP_B, c) for c in conds}
    st = {arm: e3_stats(rows[f"R_{arm}"], rows[f"N_{arm}"]) for arm in ("N10", cand)}
    ok_diff = st[cand]["diff"] > st["N10"]["diff"] - E3_DIFF_DROP
    rb, rc = st["N10"]["r_recovery_rate"], st[cand]["r_recovery_rate"]
    ok_rec = rb is None or (rc is not None and rc > rb - R_REC_DROP)
    final = cand if (ok_diff and ok_rec) else "N10"
    res = {"what": "段階 3 の 0 周目 段 B（E3 の守り、0126 の 4・0127）", "experiment": EXP_B, "trials": TRIALS_B,
           "models": MODELS_B, "safety": bool(da["safety_keep_tentative"]), "candidate": cand, "stats": st,
           "rule": f"候補の diff が N10 より {E3_DIFF_DROP} 以上小さくならず、R の復帰／成立が {int(R_REC_DROP * 100)} ポイント以上下がらない",
           "ok_diff": ok_diff, "ok_recovery": ok_rec, "final_arm": final, "final_mode": ARMS[final][0],
           "final_exec_interval": ARMS[final][1], "safety_keep_tentative": da["safety_keep_tentative"],
           "g_audit": gsum, "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    (RES / "s3_round0_b.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("stats", "ok_diff", "ok_recovery", "final_arm")}, ensure_ascii=False, indent=1,
                     default=float))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run-a", "run-b"):
        p = sub.add_parser(name)
        p.add_argument("--parallel", type=int, default=3)
    sub.add_parser("decide-a")
    sub.add_parser("decide-b")
    a = ap.parse_args(argv)
    {"run-a": cmd_run_a, "decide-a": cmd_decide_a, "run-b": cmd_run_b, "decide-b": cmd_decide_b}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
