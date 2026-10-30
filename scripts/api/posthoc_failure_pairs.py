"""比較 E の判定の後の記述: Ld・Ldc の失敗をシードごとに組にして、片方だけ失敗した組の中身（失敗の分類・立方体の向き）を数える。"""
import collections
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "api"))
import failure_causes as F  # noqa: E402

V = ROOT / "outputs" / "v2eval"
LD = ["R1E", "R1E_B"] + [f"R2LD{i:02d}" for i in range(1, 13)]
LDC = [f"R2LP{i}" for i in range(1, 7)] + [f"R2LC{i:02d}" for i in range(1, 13)]


def rows(exps, cond):
    out = {}
    for e in exps:
        for js in sorted((V / e / cond).glob("run_[0-9][0-9][0-9][0-9].json")):
            m = json.loads(js.read_text(encoding="utf-8"))
            if m.get("all_three_in_box"):
                out[m["seed"]] = {"success": True}
                continue
            try:
                r = F.trial(js, cond)
            except AttributeError:                       # 計画が無い試行（LLM の計画が出なかった）
                r = {"seed": m["seed"], "success": False, "root": "plan_none", "episodes": [], "steps": [], "root_step": None,
                     "final_state": {}, "term": "plan_none"}
            r["cat"] = "plan_none" if r["root"] == "plan_none" else F.category(r)
            ep = [e for e in r["episodes"] if e["step"] == r.get("root_step") and e["kind"] in F.FAIL_KINDS]
            r["yaw_max"] = max([abs(e["yaw"]) for e in ep], default=None)
            r["first_fail_kind"] = ep[0]["kind"] if ep else None
            out[m["seed"]] = r
    return out


ld, ldc = rows(LD, "Ld"), rows(LDC, "Ldc")
seeds = sorted(set(ld) & set(ldc))
print("seeds", len(seeds), "Ld ok", sum(ld[s]["success"] for s in seeds), "Ldc ok", sum(ldc[s]["success"] for s in seeds))
groups = {"b (Ld ok, Ldc fail)": [(s, ldc[s]) for s in seeds if ld[s]["success"] and not ldc[s]["success"]],
          "c (Ld fail, Ldc ok)": [(s, ld[s]) for s in seeds if not ld[s]["success"] and ldc[s]["success"]],
          "both fail (Ld)": [(s, ld[s]) for s in seeds if not ld[s]["success"] and not ldc[s]["success"]],
          "both fail (Ldc)": [(s, ldc[s]) for s in seeds if not ld[s]["success"] and not ldc[s]["success"]]}
res = {}
for g, xs in groups.items():
    cats = collections.Counter(r["cat"] for _, r in xs)
    yaws = [r["yaw_max"] for _, r in xs if r["yaw_max"] is not None]
    hi = sum(1 for y in yaws if y >= 25)
    print(f"\n== {g}: {len(xs)}")
    print("  cat", dict(cats.most_common()))
    print("  first fail kind", dict(collections.Counter(r["first_fail_kind"] for _, r in xs).most_common()))
    print(f"  |yaw| at failed grasps: n {len(yaws)}, >=25deg {hi}, sorted {sorted(round(y) for y in yaws)}")
    res[g] = [{"seed": s, "cat": r["cat"], "yaw_max": r["yaw_max"], "first_fail_kind": r["first_fail_kind"], "term": r.get("term")}
              for s, r in xs]
(ROOT / "outputs" / "api" / "r2l_pairs_posthoc.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
