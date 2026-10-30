"""結果の表と図（docs/interfaces/results.md）。1 試行の指標は metrics.trial_metrics、統計は stats。

    rows = collect(run_dirs)                           # 試行の記録（trial_NNNN.json）を読み、1 試行 1 行
    write_all(out_dir, rows, experiment, pairs=[("U0", "N0")])   # pairs: 対にして比べる条件の名前の組

出力（outputs/results/<実験>/）: trials.csv・summary.csv・paired.csv・continuous.csv・tables.md・fig_*.png。
CSV は UTF-8（BOM なし）、真偽は true・false、欠けた値は空欄。条件の色はどの図でも同じ（COLOR_OF）。
"""
import csv
import math
import pathlib

import numpy as np

from recovla.common import config
from recovla.eval import metrics, stats

_CFG = config.load()
STAGES = list(metrics.STAGES)
CONTINUOUS = ("seam_jump_mean", "jerk_rms", "reaction_time_s", "recovery_time_s", "t_success_s")
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return "" if math.isnan(v) else repr(v)
    return str(v)


def write_csv(path, rows: list, columns: list = None) -> None:
    columns = columns or (list(rows[0].keys()) if rows else [])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for r in rows:
            w.writerow([_fmt(r.get(c)) for c in columns])


def collect(run_dirs) -> list:
    rows = []
    for d in run_dirs:
        for p in sorted(pathlib.Path(d).glob("trial_*.json")):
            rows.append(metrics.trial_metrics(metrics.load_trial(p), _CFG["eval"]))
    return rows


def _rate(k, n):
    return (k / n) if n else float("nan")


def _median(x):
    x = [v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.median(x)) if x else float("nan")


def summary(rows: list) -> list:
    out = []
    for cond in sorted({r["condition"] for r in rows}, key=lambda c: [r["condition"] for r in rows].index(c)):
        rs = [r for r in rows if r["condition"] == cond]
        n = len(rs)
        s = sum(bool(r["success"]) for r in rs)
        lo, hi = stats.wilson_interval(s, n)
        ind = [r for r in rs if r["induce"] not in ("none", None, "")]
        est = [r for r in ind if r["induce_established"]]
        rec = sum(bool(r["recovered"]) for r in est)
        rlo, rhi = stats.wilson_interval(rec, len(est))
        row = {"experiment": rs[0]["experiment"], "condition": cond, "n": n, "successes": s,
               "success_rate": _rate(s, n), "success_lo": lo, "success_hi": hi,
               "error_rate": _rate(sum(bool(r["error"]) for r in rs), n),
               "collateral_rate": _rate(sum(bool(r["collateral"]) for r in rs), n),
               "induced_n": len(ind), "established_n": len(est), "establish_rate": _rate(len(est), len(ind)),
               "recovered_n": rec, "recovery_rate": _rate(rec, len(est)), "recovery_lo": rlo, "recovery_hi": rhi,
               "reaction_time_median_s": _median([r["reaction_time_s"] for r in rs]),
               "recovery_time_median_s": _median([r["recovery_time_s"] for r in rs]),
               "seam_jump_median": _median([r["seam_jump_mean"] for r in rs]),
               "nonseam_jump_median": _median([r["nonseam_jump_mean"] for r in rs]),
               "jerk_rms_median": _median([r["jerk_rms"] for r in rs]),
               "contact_rate": _rate(sum(int(r["contacts_n"] or 0) > 0 for r in rs), n)}
        for i, st in enumerate(STAGES):
            row[f"stage_{st}"] = _rate(sum(STAGES.index(r["stage_reached"]) >= i for r in rs
                                           if r["stage_reached"] in STAGES), n)
        out.append(row)
    return out


def establish_by_layout_kind(rows: list) -> list:
    """条件 × 意図的な失敗 × 配置の種類ごとの意図的な失敗の実際に失敗した割合。"""
    out = []
    keys = sorted({(r["condition"], r["induce"], r["layout_kind"]) for r in rows
                   if r["induce"] not in ("none", None, "")}, key=str)
    for cond, ind, lk in keys:
        rs = [r for r in rows if r["condition"] == cond and r["induce"] == ind and r["layout_kind"] == lk]
        est = sum(bool(r["induce_established"]) for r in rs)
        out.append({"condition": cond, "induce": ind, "layout_kind": lk, "induced_n": len(rs), "established_n": est,
                    "establish_rate": _rate(est, len(rs))})
    return out


def paired(rows: list, pairs: list) -> tuple:
    pb, pc = [], []
    for a, b in pairs:
        ra = [r for r in rows if r["condition"] == a]
        rb = [r for r in rows if r["condition"] == b]
        prs = stats.pair_by_seed(ra, rb)
        exp = (ra or rb)[0]["experiment"] if (ra or rb) else ""
        for metric in ("success", "recovered"):
            use = prs if metric == "success" else [(x, y) for x, y in prs
                                                   if x["induce_established"] and y["induce_established"]]
            n11 = sum(bool(x[metric]) and bool(y[metric]) for x, y in use)
            n10 = sum(bool(x[metric]) and not bool(y[metric]) for x, y in use)
            n01 = sum(not bool(x[metric]) and bool(y[metric]) for x, y in use)
            n00 = len(use) - n11 - n10 - n01
            diff, lo, hi = stats.paired_diff_ci(n11, n10, n01, n00) if use else (float("nan"),) * 3
            pb.append({"experiment": exp, "metric": metric, "condition_a": a, "condition_b": b, "n_pairs": len(use),
                       "n11": n11, "n10": n10, "n01": n01, "n00": n00,
                       "rate_a": _rate(n11 + n10, len(use)), "rate_b": _rate(n11 + n01, len(use)),
                       "diff": diff, "diff_lo": lo, "diff_hi": hi, "mcnemar_p": stats.mcnemar_exact(n10, n01)})
        for metric in CONTINUOUS:
            av = np.array([float("nan") if x[metric] is None else float(x[metric]) for x, _ in prs])
            bv = np.array([float("nan") if y[metric] is None else float(y[metric]) for _, y in prs])
            stat, p, npair = stats.wilcoxon_paired(av, bv)
            ok = ~(np.isnan(av) | np.isnan(bv))
            pc.append({"experiment": exp, "metric": metric, "condition_a": a, "condition_b": b, "n_pairs": int(npair),
                       "median_a": _median(av[ok].tolist()), "median_b": _median(bv[ok].tolist()),
                       "median_diff": _median((av[ok] - bv[ok]).tolist()), "wilcoxon_stat": stat, "wilcoxon_p": p})
    return pb, pc


def _md_table(rows: list, cols: list) -> str:
    if not rows:
        return "（なし）\n"
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c)
            cells.append(f"{v:.3f}" if isinstance(v, float) and not math.isnan(v) else _fmt(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def figures(out_dir, rows: list, summ: list) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conds = [s["condition"] for s in summ]
    color = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(conds)}
    made = []

    def save(fig, name):
        p = pathlib.Path(out_dir) / name
        fig.tight_layout()
        fig.savefig(p, dpi=120)
        plt.close(fig)
        made.append(p.name)

    fig, ax = plt.subplots(figsize=(6, 0.5 + 0.45 * len(summ)))
    for y, s in enumerate(summ):
        ax.barh(y, s["success_rate"], color=color[s["condition"]])
        if s["success_lo"] is not None:
            # 0 回・全回のとき、部分の端と率の差が丸めで負になる（-1e-17 など）ので 0 で切る
            ax.errorbar(s["success_rate"], y, xerr=[[max(0.0, s["success_rate"] - s["success_lo"])],
                                                    [max(0.0, s["success_hi"] - s["success_rate"])]],
                        color="k", capsize=3)
    ax.set_yticks(range(len(summ)), conds)
    ax.set_xlim(0, 1)
    ax.set_xlabel("success rate (Wilson 95%)")
    save(fig, "fig_success.png")

    fig, ax = plt.subplots(figsize=(6, 3))
    x = np.arange(len(summ))
    ax.bar(x - 0.2, [s["establish_rate"] for s in summ], 0.4, label="established", color="#bbbbbb")
    ax.bar(x + 0.2, [s["recovery_rate"] for s in summ], 0.4, label="recovered", color=[color[c] for c in conds])
    ax.set_xticks(x, conds, rotation=20)
    ax.set_ylim(0, 1)
    ax.legend()
    save(fig, "fig_recovery.png")

    fig, ax = plt.subplots(figsize=(6, 3))
    data, labels = [], []
    for c in conds:
        rs = [r for r in rows if r["condition"] == c]
        for key, tag in (("seam_jump_mean", "seam"), ("nonseam_jump_mean", "non-seam")):
            v = [r[key] for r in rs if r[key] is not None and not math.isnan(r[key])]
            data.append(v or [float("nan")])
            labels.append(f"{c}\n{tag}")
    ax.boxplot(data, tick_labels=labels)
    ax.set_ylabel("jump [m/s]")
    ax.tick_params(axis="x", labelsize=7)
    save(fig, "fig_seam.png")

    fig, ax = plt.subplots(figsize=(6, 3))
    drawn = False
    for c in conds:
        v = [r["reaction_time_s"] for r in rows if r["condition"] == c and r["reaction_time_s"] is not None
             and not math.isnan(r["reaction_time_s"])]
        if v:
            ax.hist(v, bins=15, alpha=0.5, color=color[c], label=c)
            drawn = True
    ax.set_xlabel("reaction time [s]")
    if drawn:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "no established induction", ha="center", va="center", transform=ax.transAxes)
    save(fig, "fig_reaction.png")

    fig, ax = plt.subplots(figsize=(6, 3))
    for c in conds:
        s = next(s for s in summ if s["condition"] == c)
        ax.plot(range(len(STAGES)), [s[f"stage_{st}"] for st in STAGES], marker="o", color=color[c], label=c)
    ax.set_xticks(range(len(STAGES)), ["approach", "grasp", "carry", "place", "retreat", "done"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("reached (share)")
    ax.legend()
    save(fig, "fig_stage.png")
    return made


def write_all(out_dir, rows: list, experiment: str, pairs: list = (), figs: bool = True) -> dict:
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "trials.csv", rows)
    summ = summary(rows)
    write_csv(out / "summary.csv", summ)
    pb, pc = paired(rows, list(pairs))
    write_csv(out / "paired.csv", pb, ["experiment", "metric", "condition_a", "condition_b", "n_pairs", "n11", "n10",
                                       "n01", "n00", "rate_a", "rate_b", "diff", "diff_lo", "diff_hi", "mcnemar_p"])
    write_csv(out / "continuous.csv", pc, ["experiment", "metric", "condition_a", "condition_b", "n_pairs", "median_a",
                                           "median_b", "median_diff", "wilcoxon_stat", "wilcoxon_p"])
    warn = [s for s in summ if s["induced_n"] and s["establish_rate"] < float(_CFG["eval"]["induction_min_rate"])]
    md = [f"# {experiment}\n",
          "## 条件ごと\n", _md_table(summ, ["condition", "n", "successes", "success_rate", "success_lo", "success_hi",
                                           "error_rate", "collateral_rate", "established_n", "establish_rate",
                                           "recovered_n", "recovery_rate", "recovery_lo", "recovery_hi", "contact_rate"])]
    for s in warn:
        md.append(f"\n**警告**: {s['condition']} の誘発の成立率 {s['establish_rate']:.2f} が "
                  f"{_CFG['eval']['induction_min_rate']} 未満（手順書 Step G の 3）\n")
    by_kind = establish_by_layout_kind(rows)
    if by_kind:
        md += ["\n## 配置の種類ごとの誘発の成立率（決裁 0059。不成立の試行は差し替えない）\n",
               _md_table(by_kind, ["condition", "induce", "layout_kind", "induced_n", "established_n", "establish_rate"])]
    md += ["\n## 対の比較（成否・復帰。復帰は両方の条件で誘発が成立した種だけ）\n",
           _md_table(pb, ["metric", "condition_a", "condition_b", "n_pairs", "rate_a", "rate_b",
                                                   "diff", "diff_lo", "diff_hi", "mcnemar_p"]),
           "\n## 対の比較（連続量）\n", _md_table(pc, ["metric", "condition_a", "condition_b", "n_pairs", "median_a",
                                                 "median_b", "median_diff", "wilcoxon_p"])]
    (out / "tables.md").write_text("".join(md), encoding="utf-8")
    made = figures(out, rows, summ) if figs and rows else []
    return {"trials": len(rows), "conditions": [s["condition"] for s in summ], "figures": made,
            "establish_warnings": [s["condition"] for s in warn]}
