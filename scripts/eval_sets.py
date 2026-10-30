"""センサのモデルを通す実行系（scripts/eval_trials.py）で、1 つのサブタスクの評価の組を回して比べる。

scripts/paper_pdf.py が、評価の組の表（STAGES）の部品として読む。

    .venv\\Scripts\\python.exe scripts\\eval_sets.py plan --stage s3 [--sets A,B,...]     # 実行の順番（outputs/v2eval/<実験>_queue.json）
    .venv\\Scripts\\python.exe scripts\\eval_sets.py run --stage s3 [--parallel 3]        # 順に eval_trials.py を回す（済んだ条件は飛ばす）
    .venv\\Scripts\\python.exe scripts\\eval_sets.py report --stage s3                    # 比較ごとの表と検定 → outputs/results/v2_e_report_<stage>.json
    .venv\\Scripts\\python.exe scripts\\eval_sets.py safety-decide --stage s3             # 安全フィルタを既定に残すかの判定

--stage の s1・s2・s3 はモデルと実行系の版（STAGES）。各組は、失敗なし（nat）と P1（掴み損ね）・P2（運ぶ途中の落下）・
P3（置き損ね）を同じシードで回す。A・B は R（立て直しデータあり）・N（成功データだけ）、C・D は同期の実行、E は RTC、
H は安全フィルタなし。E1〜E8 は compare_subtask.py と同じ事前に決めた比較。E9 は監査（G1 実機の境界・G2 時刻の扱い・
G3 指令の上限）の違反の数（基準 0）、E10 は知覚の精度（机の上の立方体の位置の誤差、箱の中かの判定と真値の一致率）。
"""
import argparse
import importlib.util
import json
import pathlib
import subprocess
import sys
import time

import numpy as np

from recovla.common import config

CFG = config.load()
OUT = config.path(CFG["paths"]["outputs"])
RES = OUT / "results"
PARTS = ("nat", "P1", "P2", "P3")
STAGES = {
    "s1": {"experiment": "V2S1", "base": 120000, "final": "F",
           "sets": {"A": ("R1", "naive", True), "B": ("N1", "naive", True), "C": ("R1", "sync", True),
                    "D": ("N1", "sync", True), "E": ("R1", "rtc", True), "F": ("R2", "naive", True),
                    "G": ("R1plus", "naive", True), "H": ("R2", "naive", False)}},
    "s2": {"experiment": "V2S2", "base": 130000, "final": "A",
           "sets": {"A": ("R1v2", "naive", True), "B": ("N1v2", "naive", True), "C": ("R1v2", "sync", True),
                    "D": ("N1v2", "sync", True), "E": ("R1v2", "rtc", True), "H": ("R1v2", "naive", False)}},
    # s3: 4 つ目の値は行動列から実行する行数（naive・RTC は 6 行、同期は 10 行）
    "s3": {"experiment": "V3S3", "base": 140000, "final": "A",
           "sets": {"A": ("R1v3", "naive", True, 6), "B": ("N1v3", "naive", True, 6), "C": ("R1v3", "sync", True, 10),
                    "D": ("N1v3", "sync", True, 10), "E": ("R1v3", "rtc", True, 6), "H": ("R1v3", "naive", False, 6)}},
    "dev": {"experiment": "V2DEVE", "base": 59800, "final": "F",
            "sets": {"F": ("R2", "naive", True), "H": ("R2", "naive", False)}},
}


def trials_of(stage: str, part: str) -> str:
    b = STAGES[stage]["base"]
    if stage == "dev":
        return {"nat": f"natural:{b}:2", "P1": f"induced:{b + 100}:3", "P2": f"induced:{b + 200}:3", "P3": f"induced:{b + 300}:3"}[part]
    return {"nat": f"natural:{b}:33", "P1": f"induced:{b + 1000}:50", "P2": f"induced:{b + 2000}:50",
            "P3": f"induced:{b + 3000}:50"}[part]


def _interval_args(entry) -> list:
    """組の 4 つ目の値（行動列から実行する行数）。s1・s2 の組にはない（configs の値＝10 行）。"""
    return ["--exec-interval", str(entry[3])] if len(entry) > 3 else []


def _e50():
    spec = importlib.util.spec_from_file_location("e50", config.ROOT / "scripts" / "compare_subtask.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def cmd_plan(a) -> list:
    st = STAGES[a.stage]
    main_safety = a.main_safety == "on"
    q = []
    for s in (a.sets.split(",") if a.sets else st["sets"]):
        model, mode, safety = st["sets"][s][:3]
        if s == "H" and not main_safety:
            continue                                    # 既定がフィルタなしなら H は既定と同じ
        safety = safety and main_safety
        for part in PARTS:
            args = ["run", "--experiment", st["experiment"], "--condition", f"{s}_{part}", "--model", model,
                    "--mode", mode, "--trials", trials_of(a.stage, part)] + _interval_args(st["sets"][s])
            if part != "nat":
                args += ["--induce", part]
            if not safety:
                args += ["--no-safety"]
            q.append({"name": f"{s}_{part}", "args": args})
    path = OUT / "v2eval" / f"{st['experiment']}_queue.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"stage": a.stage, "main_safety": a.main_safety, "queue": q}, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    for x in q:
        print(x["name"], " ".join(x["args"]))
    return q


def cmd_run(a) -> None:
    st = STAGES[a.stage]
    q = json.loads((OUT / "v2eval" / f"{st['experiment']}_queue.json").read_text(encoding="utf-8"))["queue"]
    logs = OUT / "v2eval" / f"{st['experiment']}_logs"
    logs.mkdir(parents=True, exist_ok=True)
    todo = [x for x in q if not (OUT / "v2eval" / st["experiment"] / x["name"] / "run.json").is_file()]
    running = []
    while todo or running:
        while todo and len(running) < a.parallel:
            x = todo.pop(0)
            d = OUT / "v2eval" / st["experiment"] / x["name"]
            if d.exists() and any(d.glob("trial_*.json")):          # 途中で落ちた条件は消してやり直す
                import shutil
                shutil.rmtree(d)
            f = open(logs / f"{x['name']}.log", "w", encoding="utf-8")
            p = subprocess.Popen([sys.executable, str(config.ROOT / "scripts" / "eval_trials.py"), *x["args"]],
                                 stdout=f, stderr=subprocess.STDOUT, cwd=config.ROOT)
            running.append((x, p, f))
            print(time.strftime("%H:%M:%S"), "start", x["name"], flush=True)
        time.sleep(10)
        for item in list(running):
            x, p, f = item
            if p.poll() is not None:
                f.close()
                running.remove(item)
                print(time.strftime("%H:%M:%S"), "done", x["name"], "exit", p.returncode, flush=True)


def _audits(d: pathlib.Path) -> dict:
    out = {"trials": 0, "g1": 0, "g2_stops": 0, "g2_early": 0, "g3": 0, "g3_by_item": {}}
    for p in sorted(d.glob("trial_*.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        au = m["audit"]
        out["trials"] += 1
        out["g1"] += int(au["g1"]["violations"])
        out["g2_stops"] += int(au["g2"]["world_stops"])
        out["g2_early"] += int(au["g2"]["early_use"])
        out["g3"] += int(au["g3"]["total_violations"])
        for k, v in au["g3"]["violations"].items():
            out["g3_by_item"][k] = out["g3_by_item"].get(k, 0) + int(v)
    return out


def _perception_accuracy(d: pathlib.Path) -> dict:
    """E10: 実行系の知覚の記録（runtime_NNNN.json の perception）と、真値のフレーム（trial_NNNN.npz）を時刻で合わせる。"""
    from recovla.common.seeds import COLORS
    from recovla.sim import frames
    errs, agree = [], []
    box = np.asarray(CFG["scene"]["box"]["pos"], float)
    for p in sorted(d.glob("trial_*.json")):
        i = p.stem.split("_")[1]
        rp = d / f"runtime_{i}.json"
        if not rp.is_file():
            continue
        z = np.load(p.with_suffix(".npz"))
        t = z["sim_time"]
        for e in json.loads(rp.read_text(encoding="utf-8"))["runtime"].get("perception", []):
            k = int(np.argmin(np.abs(t - e["t_obs"])))
            for c, v in e["cubes"].items():
                tp = z["cube_pos"][k, COLORS.index(c)]
                on_table = tp[2] < 0.03 and not frames.in_box(tp, box)
                if on_table and v["status"] in ("seen", "held"):
                    errs.append(float(np.linalg.norm(np.asarray(v["pos"]) - tp)))
                if v["status"] != "in_hand":
                    agree.append(bool(v["in_box"]) == bool(frames.in_box(tp, box)))
    return {"n": len(errs), "median_m": float(np.median(errs)) if errs else None,
            "p95_m": float(np.percentile(errs, 95)) if errs else None,
            "in_box_agreement": float(np.mean(agree)) if agree else None, "n_in_box": len(agree)}


def cmd_report(a) -> None:
    from recovla.eval import report as R
    e50 = _e50()
    st = STAGES[a.stage]
    base = OUT / "v2eval" / st["experiment"]
    have = [s for s in st["sets"] if (base / f"{s}_nat").is_dir()]
    from recovla.eval import gate
    gate.require([base / f"{s}_{p}" for s in have for p in PARTS if (base / f"{s}_{p}").is_dir()],
                 f"最終評価の集計（{st['experiment']}）")

    def rows(s, part):
        d = base / f"{s}_{part}"
        return {(r["seed"], r["target"]): r for r in R.collect([d])} if d.is_dir() else {}
    S = {s: {p: rows(s, p) for p in PARTS} for s in have}
    fin = st["final"]
    res = {"stage": a.stage, "sets": {s: {p: e50.summary(S[s][p]) for p in PARTS} for s in have},
           "final_set": fin, "final_model": st["sets"][fin][0], "primary": {}, "secondary": {}, "new": {}}
    if fin in S:
        res["primary"]["E1"] = {k: res["sets"][fin]["nat"][k] for k in ("n", "successes", "success_wilson")}
        res["primary"]["E2"] = {k: res["sets"][fin]["P1"][k] for k in ("established", "recovered", "recovery_rate", "recovery_wilson")}
    pairs = {"E3_main": ("A", "B"), "E3_sync": ("C", "D"), "E4_naive_vs_rtc": ("A", "E"), "E4_sync_vs_naive": ("C", "A"),
             "E8": ("F", "G"), "E5": (fin, "H")}
    for name, (x, y) in pairs.items():
        if x in S and y in S:
            res["secondary"][name] = {"x": x, "y": y, **e50.compare(S[x], S[y])}
    sec = res["secondary"]
    if "E3_main" in sec and "E3_sync" in sec:
        ps = [sec["E3_main"]["P1_recovery"]["mcnemar_exact_p"], sec["E3_sync"]["P1_recovery"]["mcnemar_exact_p"]]
        adj = e50._holm(ps)
        res["primary"]["E3"] = {"main_A_vs_B": {**sec["E3_main"]["P1_recovery"], "holm_p": adj[0]},
                                "sync_C_vs_D": {**sec["E3_sync"]["P1_recovery"], "holm_p": adj[1]}}
    for e, key in (("E4", "E4_naive_vs_rtc"), ("E8", "E8")):
        if key in sec:
            res["primary"][e] = sec[key]["P1_recovery"]
    if "E5" in sec:
        res["primary"]["E5"] = sec["E5"]["nat_contact"]
    aud = {f"{s}_{p}": _audits(base / f"{s}_{p}") for s in have for p in PARTS if (base / f"{s}_{p}").is_dir()}
    tot = {k: sum(v[k] for v in aud.values()) for k in ("trials", "g1", "g2_stops", "g2_early", "g3")}
    res["new"]["E9"] = {"criterion": "全試行で違反 0", "total": tot, "met": tot["g1"] == 0 and tot["g2_stops"] == 0
                        and tot["g2_early"] == 0 and tot["g3"] == 0, "by_condition": aud}
    res["new"]["E10"] = {s: _perception_accuracy(base / f"{s}_nat") for s in have}
    res["written"] = time.strftime("%Y-%m-%d %H:%M:%S")
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"v2_e_report_{a.stage}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps({"primary": res["primary"], "E9": res["new"]["E9"]["total"], "E10": res["new"]["E10"]},
                     ensure_ascii=False, indent=1, default=float))


def cmd_safety_decide(a) -> None:
    """安全フィルタを既定に残すかの判定（決まりは回す前に固定した）。検証用のシードの失敗なしの試行を、最終モデルで
    フィルタあり・なしの同じシードの対で回し、あり の成功が なし より 3 ポイント以上低ければ既定から外す。
    接触の減り方は記録だけ。結果は outputs/results/v2_safety_main_<stage>.json。"""
    from recovla.eval import report as R
    e50 = _e50()
    fin = STAGES[a.stage]["sets"][STAGES[a.stage]["final"]]
    model = fin[0]
    exp = a.experiment or f"V2SF_{a.stage}"
    procs = []
    for cond, extra in (("on", []), ("off", ["--no-safety"])):
        d = OUT / "v2eval" / exp / cond
        if not (d / "run.json").is_file():
            cmd = [sys.executable, str(config.ROOT / "scripts" / "eval_trials.py"), "run", "--experiment", exp,
                   "--condition", cond, "--model", model, "--mode", fin[1], "--trials", a.trials, *_interval_args(fin), *extra]
            p = subprocess.Popen(cmd, cwd=config.ROOT)
            if not a.parallel:
                p.wait()
            procs.append(p)
    if any(p.wait() != 0 for p in procs):
        raise SystemExit("安全フィルタの判定の評価が落ちた")
    from recovla.eval import gate
    gsum = gate.require([OUT / "v2eval" / exp / c for c in ("on", "off")], f"安全フィルタの判定（{exp}）")
    X = {(r["seed"], r["target"]): r for r in R.collect([OUT / "v2eval" / exp / "on"])}
    Y = {(r["seed"], r["target"]): r for r in R.collect([OUT / "v2eval" / exp / "off"])}
    succ = e50.paired_binary(X, Y, e50.SUCC)
    contact = e50.paired_binary(X, Y, e50.CONTACT)
    drop = (succ["y_rate"] - succ["x_rate"]) * 100
    res = {"stage": a.stage, "model": model, "trials": a.trials, "rule": "あり の成功が なし より 3 ポイント以上低ければ主系から外す（0107 の 5）",
           "success": succ, "contact": contact, "drop_points": drop, "keep_in_main": drop < 3.0,
           "experiment": exp, "g_audit": gsum, "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"v2_safety_main_{a.stage}{'' if exp == f'V2SF_{a.stage}' else '_' + exp}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1, default=float))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--stage", required=True, choices=sorted(STAGES))
    p.add_argument("--sets", default=None)
    p.add_argument("--main-safety", default="on", choices=("on", "off"))
    p = sub.add_parser("run")
    p.add_argument("--stage", required=True, choices=sorted(STAGES))
    p.add_argument("--parallel", type=int, default=3)
    p = sub.add_parser("report")
    p.add_argument("--stage", required=True, choices=sorted(STAGES))
    p = sub.add_parser("safety-decide")
    p.add_argument("--stage", required=True, choices=sorted(STAGES))
    p.add_argument("--trials", default="natural:199800:33")
    p.add_argument("--experiment", default=None, help="出力の実験名（既定 V2SF_<段階>。回し直しで前の判定を上書きしない）")
    p.add_argument("--parallel", action="store_true", help="あり・なしを同時に回す（値は変わらない）")
    a = ap.parse_args(argv)
    {"plan": cmd_plan, "run": cmd_run, "report": cmd_report, "safety-decide": cmd_safety_decide}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
