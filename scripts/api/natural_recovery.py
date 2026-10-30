"""評価の記録から、運用中に起きた失敗と、それを誰が立て直したかを数える（事後の集計。比較の判定には使わない）。

    .venv\\Scripts\\python.exe scripts\\api\\natural_recovery.py            # 立方体の評価（実験 CLC・CLC_B）の N0・U0・L2v2

わざと起こした失敗ではなく、運用中に方策が起こした失敗から立て直せたかを数える。
各試行の真値の時系列（run_NNNN.npz: グリッパの開閉・立方体の位置・箱の中か・実行中のサブタスク）と、実行器の試みの区切り
（run_NNNN.json の steps[].attempts[].t_begin）を使う。

- 把持: サブタスクの実行中にグリッパが閉じていた間。目標はそのサブタスクの色
- 掴み損ね: 指先が目標から NEAR_M 以内で閉じたのに、どの立方体も LIFT_M 以上上がらなかった
  （目標から離れた空中で閉じたものは把持の試みとして数えない）
- 別の立方体: 目標でない立方体が上がった
- 落下・置き損ね: 目標は上がったが、放してから SETTLE_S の間に箱に入らなかった
- 試み（step, attempt）ごとに、目標が箱に入る前の最初の失敗だけを数える
- 立て直し: self＝同じ試みの中で目標が箱に入った（方策が自力で立て直した。実行器のやり直しなし）、retry＝同じサブタスクの
  後の試み（実行器が待機位置に戻してやり直した後）、replan＝同じ色の後のサブタスク（LLM の組み直しの後）、none＝入らなかった
信頼区間は Wilson 法（1 つの試行に失敗が複数ありうるので目安）。
"""
import argparse
import json
import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
COLORS = ("red", "green", "blue")
LIFT_M = 0.015          # 持ち上がったとみなす高さ
NEAR_M = 0.06           # 把持の試みとみなす、閉じた時点の指先と目標の距離
SETTLE_S = 2.0          # 放した後、箱へ落ちきるまで待つ時間
CONDS = ("N0", "U0", "L2v2")


def wilson(k, n, z=1.959964):
    if n == 0:
        return None
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round((c - h) * 100, 1), round((c + h) * 100, 1)]


def closed_intervals(closed):
    d = np.diff(closed.astype(np.int8), prepend=0, append=0)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def trial(js: pathlib.Path) -> dict:
    m = json.loads(js.read_text(encoding="utf-8"))
    z = np.load(js.with_suffix(".npz"))
    t, closed, pos, inbox, ps, tip = (z["sim_time"], z["gripper_closed"], z["cube_pos"], z["cube_in_box"], z["plan_step"],
                                      z["fingertip"])
    steps = {s["step"]: s for s in m["steps"]}
    eps, seen = [], set()
    for a, b in closed_intervals(closed):
        k = int(ps[a])
        if k < 0 or k not in steps or b >= len(t):
            continue
        st = steps[k]
        c = COLORS.index(st["color"])
        if inbox[a, c]:
            continue                                      # 目標はもう箱の中
        ta = float(t[a])
        att = max([x for x in st["attempts"] if x["t_begin"] <= ta + 1e-9], key=lambda x: x["t_begin"], default=None)
        if att is None or (k, att["attempt"]) in seen:
            continue
        rise = pos[a:b, :, 2].max(axis=0) - pos[a, :, 2]
        if rise[c] >= LIFT_M:
            w = np.flatnonzero((t >= t[b - 1]) & (t <= t[b - 1] + SETTLE_S))
            if inbox[b - 1, c] or inbox[w, c].any():
                continue                                  # 普通に箱へ入れた
            kind = "dropped_or_misplaced"
        elif (rise >= LIFT_M).any():
            kind = "wrong_cube"
        elif np.linalg.norm(tip[a] - pos[a, c]) < NEAR_M:
            kind = "miss"
        else:
            continue                                      # 目標から離れた空中で閉じた（把持の試みではない）
        seen.add((k, att["attempt"]))
        later = [x["t_begin"] for x in st["attempts"] if x["t_begin"] > att["t_begin"] + 1e-9]
        t_st_end = float(st.get("t_end", m["t_end"]))      # 時間切れの試行の最後の手順は t_end が無い
        t_att_end = min(later) if later else t_st_end
        t_in = [float(t[i]) for i in np.flatnonzero(inbox[:, c]) if t[i] > ta]
        t_first = t_in[0] if t_in else None
        if t_first is not None and t_first <= t_att_end + 0.5:
            who = "self"
        elif t_first is not None and t_first <= t_st_end + 0.5:
            who = "retry"
        elif t_first is not None:
            who = "replan" if any(s["color"] == st["color"] and s["step"] > k for s in m["steps"]) else "other"
        else:
            who = "none"
        eps.append({"step": k, "attempt": att["attempt"], "color": st["color"], "kind": kind, "t": round(ta, 2), "who": who,
                    "dt_self_s": round(t_first - ta, 1) if who == "self" else None})
    return {"seed": m["seed"], "success": bool(m.get("all_three_in_box")), "episodes": eps}


def summarize(rows: list) -> dict:
    eps = [e for r in rows for e in r["episodes"]]
    out = {"trials": len(rows), "episodes": len(eps), "by_kind": {}}
    for kind in ("miss", "dropped_or_misplaced", "wrong_cube"):
        es = [e for e in eps if e["kind"] == kind]
        who = {w: sum(1 for e in es if e["who"] == w) for w in ("self", "retry", "replan", "other", "none")}
        out["by_kind"][kind] = {"n": len(es), "who": who, "self_pct": round(100 * who["self"] / len(es), 1) if es else None,
                                "self_ci95": wilson(who["self"], len(es)),
                                "recovered": len(es) - who["none"], "recovered_ci95": wilson(len(es) - who["none"], len(es))}
    dts = [e["dt_self_s"] for e in eps if e["dt_self_s"] is not None]
    out["self_dt_median_s"] = float(np.median(dts)) if dts else None
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(ROOT / "outputs" / "v2eval"))
    ap.add_argument("--exps", default="CLC,CLC_B")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "api" / "natural_recovery_CLC.json"))
    a = ap.parse_args(argv)
    res = {"note": "事後の集計（記述）。判定には使わない", "params": {"LIFT_M": LIFT_M, "NEAR_M": NEAR_M, "SETTLE_S": SETTLE_S},
           "exps": a.exps.split(","), "conds": {}}
    for cond in CONDS:
        rows = []
        for exp in a.exps.split(","):
            d = pathlib.Path(a.root) / exp / cond
            rows += [trial(js) for js in sorted(d.glob("run_[0-9][0-9][0-9][0-9].json")) if js.with_suffix(".npz").is_file()]
        if not rows:
            sys.exit(f"{cond} の記録が見つからない（{a.root}）")
        res["conds"][cond] = summarize(rows)
        res["conds"][cond]["per_seed"] = {r["seed"]: r["episodes"] for r in rows}
        s = res["conds"][cond]
        mi = s["by_kind"]["miss"]
        print(f"{cond}: 試行 {s['trials']}  把持失敗 {mi['n']}  自分で立て直し {mi['who']['self']}（{mi['self_pct']}%、{mi['self_ci95']}）"
              f"  やり直し {mi['who']['retry']}  再計画 {mi['who']['replan']}  入らず {mi['who']['none']}")
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
