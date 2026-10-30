"""以前の評価（1 つのサブタスク）の条件の組を作り、同じシードの対で比べて集計する。

scripts/eval_sets.py が、集計（summary・compare・paired_binary・_holm）の部品として読む。

    .venv\\Scripts\\python.exe scripts\\compare_subtask.py plan --stage pre [--sets A,B,...]   # eval_subtask_direct.py run の引数を出す
    .venv\\Scripts\\python.exe scripts\\compare_subtask.py final-band-check                     # 最終評価用のシードが未使用か
    .venv\\Scripts\\python.exe scripts\\compare_subtask.py decide-final --stage pre            # 最終モデル（R2 か R1+）を決める
    .venv\\Scripts\\python.exe scripts\\compare_subtask.py report --stage pre                  # 比較ごとの表・検定・図

各組は、わざと起こす失敗なし（nat）と P1（掴み損ね）・P2（運ぶ途中の落下）・P3（置き損ね）を同じシードで回す。
E1〜E8 は事前に決めた比較の名前。成否は McNemar の正確検定、差の信頼区間は Newcombe 法、連続量は Wilcoxon、
複数の比較は Holm で補正する。比べる組・検定・最終モデルの決め方は、結果を見る前に固定した。
"""
import argparse
import json
import math
import pathlib
import time

import numpy as np

from recovla.common import config

CFG = config.load()
OUT = config.path(CFG["paths"]["outputs"])
RES = OUT / "results"
CKPT = {
    "R1": "outputs/train/train_R1_20260926-153959_20260926-153959/checkpoints/030000/pretrained_model",
    "N1": "outputs/train/train_N1_20260926-191928_20260926-191928/checkpoints/030000/pretrained_model",
    "R2": "outputs/train/train_R2_20260927-145256_20260927-145256/checkpoints/010000/pretrained_model",
    "R1plus": "outputs/train/train_R1plus_20260927-160208_20260927-160208/checkpoints/010000/pretrained_model",
}
MAIN = ["--mode", "naive", "--s", "10", "--d", "4", "--safety", "on"]
SETS = {
    "A": ("R1", MAIN),
    "B": ("N1", MAIN),
    "C": ("R1", ["--mode", "sync", "--s", "10", "--safety", "on"]),
    "D": ("N1", ["--mode", "sync", "--s", "10", "--safety", "on"]),
    "E": ("R1", ["--mode", "rtc", "--s", "10", "--d", "4", "--horizon", "40", "--schedule", "EXP", "--safety", "on"]),
    "F": ("R2", MAIN),
    "G": ("R1plus", MAIN),
    "H": (None, ["--mode", "naive", "--s", "10", "--d", "4", "--safety", "off"]),     # モデルは最終モデル
}
PARTS = ("nat", "P1", "P2", "P3")
STAGES = {
    "pre": {"experiment": "EPRE", "nat": "natural:199000:10", "P1": "induced:199100:30", "P2": "induced:199200:30",
            "P3": "induced:199300:30"},
    "final": {"experiment": "EFINAL", "nat": "natural:110000:33", "P1": "induced:111000:50",
              "P2": "induced:112000:50", "P3": "induced:113000:50"},
}
FINAL_BAND = (110000, 189999)
NAT_WITHIN = 3
REC_SAME = 3


def _final_model(stage):
    p = RES / f"e_final_model_{stage}.json"
    if not p.is_file():
        raise SystemExit(f"{p} がない（decide-final を先に）")
    return json.loads(p.read_text(encoding="utf-8"))["chosen"]


def runs(stage, sets):
    st = STAGES[stage]
    out = []
    for s in sets:
        model, args = SETS[s]
        if model is None:
            model = _final_model("pre")                  # 予備評価で決めた最終モデル
        for part in PARTS:
            a = ["--experiment", st["experiment"], "--condition", f"{s}_{part}", "--checkpoint", CKPT[model],
                 "--model", model, *args, "--trials", st[part], "--videos", "2"]
            if part != "nat":
                a += ["--induce", part]
            out.append((f"{s}_{part}", a))
    return out


def cmd_plan(a) -> None:
    for name, args in runs(a.stage, a.sets.split(",")):
        print(name, json.dumps(args))


def cmd_final_band_check(a) -> None:
    """最終評価用のシード（110000〜189999）を使った記録が outputs/ の評価・生成にないことを確かめる。"""
    hits = []
    for p in (OUT / "eval").rglob("trial_*.json"):
        try:
            s = json.loads(p.read_text(encoding="utf-8")).get("seed")
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(s, int) and FINAL_BAND[0] <= s <= FINAL_BAND[1]:
            hits.append(str(p.relative_to(config.ROOT)))
    for p in (OUT / "gen").rglob("r2_generation.jsonl"):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip() and FINAL_BAND[0] <= json.loads(line).get("seed", -1) <= FINAL_BAND[1]:
                hits.append(str(p.relative_to(config.ROOT)))
    n = sum(1 for _ in (OUT / "eval").rglob("trial_*.json"))
    res = {"band": FINAL_BAND, "trial_files_scanned": n, "hits": hits, "unused": not hits,
           "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    (RES / "final_band_check.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "hits"}, ensure_ascii=False), len(hits))


# ------------------------------------------------------------------ 集計

def _rows(stage, cond):
    from recovla.eval import report
    d = OUT / "eval" / STAGES[stage]["experiment"] / cond
    return {(r["seed"], r["target"]): r for r in report.collect([d])} if d.is_dir() else {}


def _mcnemar(b, c):
    from scipy.stats import binomtest
    return float(binomtest(min(b, c), b + c, 0.5).pvalue) if b + c else 1.0


def _wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [c - h, c + h]


def _newcombe(a, b, c, d):
    """対応のある割合の差 p1 − p2 の 95% 信頼区間（Newcombe 1998 の方法 10）。a 両方成功、b 1 だけ、c 2 だけ、d 両方失敗。"""
    n = a + b + c + d
    if n == 0:
        return [None, None]
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = _wilson(a + b, n)
    l2, u2 = _wilson(a + c, n)
    den = (a + b) * (c + d) * (a + c) * (b + d)
    phi = (a * d - b * c) / math.sqrt(den) if den > 0 else 0.0
    D = p1 - p2
    lo = D - math.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    hi = D + math.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    return [lo, hi]


def paired_binary(x, y, f, keep=lambda r: True):
    """同じシードの対（x と y に共通で、keep を両方で満たす）で f の真偽を比べる。"""
    keys = sorted(k for k in set(x) & set(y) if keep(x[k]) and keep(y[k]))
    a = sum(f(x[k]) and f(y[k]) for k in keys)
    b = sum(f(x[k]) and not f(y[k]) for k in keys)
    c = sum(f(y[k]) and not f(x[k]) for k in keys)
    d = len(keys) - a - b - c
    return {"pairs": len(keys), "x_rate": (a + b) / len(keys) if keys else None, "y_rate": (a + c) / len(keys) if keys else None,
            "x_only": b, "y_only": c, "both": a, "neither": d, "mcnemar_exact_p": _mcnemar(b, c),
            "diff_95ci_newcombe": _newcombe(a, b, c, d)}


def paired_cont(x, y, key):
    from scipy.stats import wilcoxon
    keys = sorted(k for k in set(x) & set(y) if _ok(x[k].get(key)) and _ok(y[k].get(key)))
    if len(keys) < 2:
        return {"pairs": len(keys)}
    dx = np.array([x[k][key] for k in keys])
    dy = np.array([y[k][key] for k in keys])
    diff = dx - dy
    p = float(wilcoxon(dx, dy).pvalue) if np.any(diff != 0) else 1.0
    return {"pairs": len(keys), "x_median": float(np.median(dx)), "y_median": float(np.median(dy)),
            "median_diff": float(np.median(diff)), "wilcoxon_p": p}


def _ok(v):
    return v is not None and not (isinstance(v, float) and math.isnan(v))


def summary(rows):
    rs = list(rows.values())
    est = [r for r in rs if r["induce_established"]]
    rec = sum(bool(r["success"]) for r in est)
    return {"n": len(rs), "successes": sum(bool(r["success"]) for r in rs),
            "success_wilson": _wilson(sum(bool(r["success"]) for r in rs), len(rs)),
            "established": len(est), "establish_rate": len(est) / len(rs) if rs else None,
            "recovered": rec, "recovery_rate": rec / len(est) if est else None, "recovery_wilson": _wilson(rec, len(est)),
            "contact_trials": sum(int(r["contacts_n"] or 0) > 0 for r in rs),
            "seam_jump_median": _median([r["seam_jump_mean"] for r in rs]),
            "jerk_rms_median": _median([r["jerk_rms"] for r in rs]),
            "reaction_time_median_s": _median([r["reaction_time_s"] for r in est])}


def _median(v):
    v = [x for x in v if _ok(x)]
    return float(np.median(v)) if v else None


def _set_rows(stage, s):
    return {p: _rows(stage, f"{s}_{p}") for p in PARTS}


SUCC = lambda r: bool(r["success"])          # noqa: E731
EST = lambda r: bool(r["induce_established"])  # noqa: E731
CONTACT = lambda r: int(r["contacts_n"] or 0) > 0   # noqa: E731


def compare(X, Y):
    """2 つの組の比較の一式（主要な検定もここから取る）。"""
    out = {"nat_success": paired_binary(X["nat"], Y["nat"], SUCC),
           "nat_contact": paired_binary(X["nat"], Y["nat"], CONTACT)}
    for p in ("P1", "P2", "P3"):
        out[f"{p}_recovery"] = paired_binary(X[p], Y[p], SUCC, keep=EST)
        out[f"{p}_establish"] = paired_binary(X[p], Y[p], EST)
    pooled_x = {(p,) + k: r for p in ("P1", "P2", "P3") for k, r in X[p].items()}
    pooled_y = {(p,) + k: r for p in ("P1", "P2", "P3") for k, r in Y[p].items()}
    out["P123_recovery"] = paired_binary(pooled_x, pooled_y, SUCC, keep=EST)
    for key in ("seam_jump_mean", "jerk_rms"):
        allx = {**{("nat",) + k: r for k, r in X["nat"].items()}, **pooled_x}
        ally = {**{("nat",) + k: r for k, r in Y["nat"].items()}, **pooled_y}
        out[f"{key}_all"] = paired_cont(allx, ally, key)
    out["P1_reaction_time"] = paired_cont({k: r for k, r in X["P1"].items() if EST(r)},
                                          {k: r for k, r in Y["P1"].items() if EST(r)}, "reaction_time_s")
    return out


def _holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, m, run = [None] * len(ps), len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i]))
        adj[i] = run
    return adj


def cmd_decide_final(a) -> None:
    F, G = _set_rows(a.stage, "F"), _set_rows(a.stage, "G")
    st = {}
    for name, S in (("R2", F), ("R1plus", G)):
        est = [r for p in ("P1", "P2", "P3") for r in S[p].values() if EST(r)]
        st[name] = {"nat_n": len(S["nat"]), "nat_success": sum(SUCC(r) for r in S["nat"].values()),
                    "recovered_total": sum(SUCC(r) for r in est), "established_total": len(est)}
    best = max(v["nat_success"] for v in st.values())
    step_a = [m for m in ("R2", "R1plus") if st[m]["nat_success"] >= best - NAT_WITHIN]
    best_r = max(st[m]["recovered_total"] for m in step_a)
    step_b = [m for m in step_a if st[m]["recovered_total"] >= best_r - REC_SAME]
    chosen = "R2" if "R2" in step_b else step_b[0]
    res = {"rule": "0083 の 5・0084 の 2（自然で 3 以内 → 立ち直りの合計、差 3 以内は同じ → 同じなら R2）",
           "stats": st, "step_a": step_a, "step_b_same": step_b, "chosen": chosen,
           "note": "以後の結果によらず変えない。R1+ が選ばれても E1・E2・E5 はそれで回す",
           "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"e_final_model_{a.stage}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


def _failure_forms(stage, s):
    """失敗を起こさない試行での失敗の形の内訳（最初に検出したもの。なければ時間切れ）。"""
    import collections
    from recovla.eval import failure_detect as FD
    d = OUT / "eval" / STAGES[stage]["experiment"] / f"{s}_nat"
    kinds = collections.Counter()
    for p in sorted(d.glob("trial_*.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        if m["success"]:
            continue
        ev = FD.detect_trial(np.load(p.with_suffix(".npz")))
        kinds[ev[0].kind if ev else "timeout"] += 1
    return dict(kinds)


def _blocked(stage, on, off):
    """E5: 安全フィルタありで失敗・なしで成功した試行のうち、フィルタが 5% 以上のフレームで働いたもの。"""
    d_on = OUT / "eval" / STAGES[stage]["experiment"] / f"{on}_nat"
    off_rows = _rows(stage, f"{off}_nat")
    out = []
    for p in sorted(d_on.glob("trial_*.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        k = (m["seed"], m["steps"][0]["target"])
        share = float(np.mean(np.load(p.with_suffix(".npz"))["safety_active"]))
        if not m["success"] and k in off_rows and off_rows[k]["success"] and share >= 0.05:
            out.append({"seed": k[0], "target": k[1], "safety_active_share": share})
    return out


def cmd_report(a) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    stage = a.stage
    have = [s for s in SETS if (OUT / "eval" / STAGES[stage]["experiment"] / f"{s}_nat").is_dir()]
    S = {s: _set_rows(stage, s) for s in have}
    final = _final_model("pre") if (RES / "e_final_model_pre.json").is_file() else None   # 予備評価で決めた最終モデル
    fin_set = {"R2": "F", "R1plus": "G"}.get(final)
    res = {"stage": stage, "sets": {s: {p: summary(S[s][p]) for p in PARTS} for s in have}, "final_model": final,
           "primary": {}, "secondary": {}}
    if fin_set in S:
        res["primary"]["E1"] = {"metric": "最終モデルの自然の成功率（Wilson）", **{k: res["sets"][fin_set]["nat"][k]
                                for k in ("n", "successes", "success_wilson")}}
        res["primary"]["E2"] = {"metric": "最終モデルの P1 の立ち直りの率（Wilson）", **{k: res["sets"][fin_set]["P1"][k]
                                for k in ("established", "recovered", "recovery_rate", "recovery_wilson")},
                                "note": "P1 の反応時間: R2 の P1 は閉じた直後の低い姿勢をほとんど覆っていない（0078・0079）"}
    pairs = {"E3_main": ("A", "B"), "E3_sync": ("C", "D"), "E4_naive_vs_rtc": ("A", "E"), "E4_sync_vs_naive": ("C", "A"),
             "E4_sync_vs_rtc": ("C", "E"), "E8": ("F", "G")}
    if fin_set in S and "H" in S:
        pairs["E5"] = (fin_set, "H")
    for name, (x, y) in pairs.items():
        if x in S and y in S:
            res["secondary"][name] = {"x": x, "y": y, **compare(S[x], S[y])}
    sec = res["secondary"]
    if "E3_main" in sec and "E3_sync" in sec:
        ps = [sec["E3_main"]["P1_recovery"]["mcnemar_exact_p"], sec["E3_sync"]["P1_recovery"]["mcnemar_exact_p"]]
        adj = _holm(ps)
        res["primary"]["E3"] = {"metric": "P1 の立ち直り（両方で成立した対の正確な McNemar）、Holm",
                                "main_A_vs_B": {**sec["E3_main"]["P1_recovery"], "holm_p": adj[0]},
                                "sync_C_vs_D": {**sec["E3_sync"]["P1_recovery"], "holm_p": adj[1]}}
    for e, key, metric in (("E4", "E4_naive_vs_rtc", "naive（A）対 rtc（E）の P1 の立ち直り（McNemar）"),
                           ("E8", "E8", "R2（F）対 R1+（G）の P1 の立ち直り（McNemar）")):
        if key in sec:
            res["primary"][e] = {"metric": metric, **sec[key]["P1_recovery"]}
    if "E5" in sec:
        res["primary"]["E5"] = {"metric": "フィルタあり対なしの自然の接触（McNemar）", **sec["E5"]["nat_contact"]}
        res["secondary"]["E5"]["blocked_by_filter"] = _blocked(stage, pairs["E5"][0], "H")
    res["secondary"]["E4_failure_forms"] = {s: _failure_forms(stage, s) for s in ("C", "A", "E") if s in S}
    # E4 の図: 横に行動列の境界での速度の跳び、縦に P1 の立て直しの率（白抜きは P1〜P3 の合計）
    if all(s in S for s in ("C", "A", "E")):
        fig, ax = plt.subplots(figsize=(6.5, 5))
        for s, lab, col in (("C", "sync", "C0"), ("A", "naive d=4", "C3"), ("E", "rtc d=4", "C2")):
            allr = [r for p in PARTS for r in S[s][p].values()]
            seam = [r["seam_jump_mean"] for r in allr if _ok(r["seam_jump_mean"])]
            q = np.percentile(seam, [25, 50, 75])
            p1 = res["sets"][s]["P1"]
            est = [r for p in ("P1", "P2", "P3") for r in S[s][p].values() if EST(r)]
            rec = sum(SUCC(r) for r in est)
            lo, hi = p1["recovery_wilson"]
            ax.errorbar(q[1], p1["recovery_rate"], xerr=[[q[1] - q[0]], [q[2] - q[1]]],
                        yerr=[[max(0.0, p1["recovery_rate"] - lo)], [max(0.0, hi - p1["recovery_rate"])]], fmt="o",
                        color=col, capsize=4,
                        label=f"{lab}: P1 {p1['recovered']}/{p1['established']}")
            ax.plot(q[1], rec / len(est), "o", mfc="none", color=col, ms=9)
        ax.set_xlabel("seam jump per trial, median and IQR [m/s]")
        ax.set_ylabel("P1 recovery rate (filled, Wilson 95%) / P1-P3 pooled (hollow)")
        ax.set_title(f"E4 ({stage}): smoothness vs recovery, R1 with safety filter")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(RES / f"e4_tradeoff_{stage}.png", dpi=120)
        plt.close(fig)
    res["written"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (RES / f"e_report_{stage}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps(res["primary"], ensure_ascii=False, indent=1, default=float))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("plan")
    s.add_argument("--stage", choices=list(STAGES), required=True)
    s.add_argument("--sets", default="A,B,C,D,E,F,G")
    sub.add_parser("final-band-check")
    s = sub.add_parser("decide-final")
    s.add_argument("--stage", choices=list(STAGES), default="pre")
    s = sub.add_parser("report")
    s.add_argument("--stage", choices=list(STAGES), required=True)
    a = ap.parse_args(argv)
    {"plan": cmd_plan, "final-band-check": cmd_final_band_check, "decide-final": cmd_decide_final,
     "report": cmd_report}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
