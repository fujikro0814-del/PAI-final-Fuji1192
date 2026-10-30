"""判定の後の記述（感度）: 箱の中の立方体の上に載った立方体（stacked_in_box）を「箱に入った」と数えたら、3 個とも箱へ入った数はどう変わるか。"""
import json
import pathlib
import sys

from scipy.stats import binomtest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "api"))
import failure_causes as F  # noqa: E402

V = ROOT / "outputs" / "v2eval"
ARMS = {"Ld": ["R1E", "R1E_B"] + [f"R2LD{i:02d}" for i in range(1, 13)],
        "Ldc": [f"R2LP{i}" for i in range(1, 7)] + [f"R2LC{i:02d}" for i in range(1, 13)]}
ok = {}
for arm, exps in ARMS.items():
    strict, loose, flipped = {}, {}, []
    for e in exps:
        for js in sorted((V / e / arm).glob("run_[0-9][0-9][0-9][0-9].json")):
            m = json.loads(js.read_text(encoding="utf-8"))
            s = m["seed"]
            strict[s] = loose[s] = bool(m.get("all_three_in_box"))
            if strict[s]:
                continue
            try:
                r = F.trial(js, arm)
            except AttributeError:
                continue
            if r["missing"] and all(r["final_state"][c] == "stacked_in_box" for c in r["missing"]):
                loose[s] = True
                flipped.append(s)
    ok[arm] = (strict, loose)
    print(f"{arm}: strict {sum(strict.values())}/{len(strict)}  stacked-counts-as-in {sum(loose.values())}/{len(loose)}  flipped {len(flipped)}")
seeds = sorted(set(ok["Ld"][0]) & set(ok["Ldc"][0]))
for k, name in ((0, "strict"), (1, "stacked counts")):
    a, b_ = ok["Ld"][k], ok["Ldc"][k]
    b = sum(1 for s in seeds if a[s] and not b_[s])
    c = sum(1 for s in seeds if b_[s] and not a[s])
    print(f"{name}: Ld {sum(a[s] for s in seeds)}  Ldc {sum(b_[s] for s in seeds)}  b {b} c {c}  p {binomtest(min(b, c), b + c, 0.5).pvalue:.3f}")
