"""比較 E の判定の後の記述: 箱にすでに立方体がある場面で、放した位置とその立方体の距離と、結果（箱に入った／その上に載った）を数える。"""
import collections
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "api"))
import failure_causes as F  # noqa: E402

V = ROOT / "outputs" / "v2eval"
ARMS = {"Ld": ["R1E", "R1E_B"] + [f"R2LD{i:02d}" for i in range(1, 13)],
        "Ldc": [f"R2LP{i}" for i in range(1, 7)] + [f"R2LC{i:02d}" for i in range(1, 13)]}
res = {}
for arm, exps in ARMS.items():
    eps, n_trials, steps_with_occ = [], 0, 0
    for e in exps:
        for js in sorted((V / e / arm).glob("run_[0-9][0-9][0-9][0-9].json")):
            try:
                r = F.trial(js, arm)
            except AttributeError:
                continue
            n_trials += 1
            eps += [x for x in r["episodes"] if x["kind"] in ("placed", "misplaced", "slip_drop") and x["rel_occupied_release"] is not None]
    rows = []
    for x in eps:
        out = "in_box" if x["kind"] == "placed" else ("stacked" if x["land"] == "stacked_in_box" else x["land"] or x["kind"])
        rows.append((x["rel_occupied_release"], out, x["n_in_box"], x["prior_fail"]))
    d = np.array([r[0] for r in rows])
    bins = [0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 1]
    print(f"\n== {arm}: trials {n_trials}, releases with an occupant in the box {len(rows)}")
    print("  outcome", dict(collections.Counter(r[1] for r in rows).most_common()))
    for lo, hi in zip(bins, bins[1:]):
        sel = [r for r in rows if lo <= r[0] < hi]
        if sel:
            st = sum(1 for r in sel if r[1] == "stacked")
            print(f"  dist {lo:.2f}-{hi:.2f} m: n {len(sel):4d}  stacked {st:3d} ({100 * st / len(sel):5.1f}%)  in_box {sum(1 for r in sel if r[1] == 'in_box')}")
    for k in (1, 2):
        sel = [r for r in rows if r[2] == k]
        st = sum(1 for r in sel if r[1] == "stacked")
        if sel:
            print(f"  n_in_box {k}: n {len(sel)}, stacked {st} ({100 * st / len(sel):.1f}%), median dist {np.median([r[0] for r in sel]):.3f}")
    res[arm] = rows
(ROOT / "outputs" / "api" / "r2l_release_posthoc.json").write_text(json.dumps(res), encoding="utf-8")
