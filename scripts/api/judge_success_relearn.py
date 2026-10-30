"""自分の成功も足したモデルと、行動列の候補から選ぶ条件の判定（事前に決めた比較、手順書 docs/api/r2_protocol.md 7 節）と、
成功の内訳（記述）。数え・統計・二重集計は scripts/api/judge_failure_relearn.py（A）と judge_failure_relearn_b.py（B）を使い、条件・シード・比べる組だけを替える。

    .venv\\Scripts\\python.exe scripts\\api\\judge_success_relearn.py check                 # 評価（実験 R2E・R2E_B）。判定は 1 回だけ
    .venv\\Scripts\\python.exe scripts\\api\\judge_success_relearn.py entry --exp R2SMOKE   # 試しの実行の確かめ（本数・無効・チェックポイントの道・HEAD だけ）
    .venv\\Scripts\\python.exe scripts\\api\\judge_success_relearn.py check-c               # H-R2C（U2a → Uc、6.1・7 節）。1 回だけ
    .venv\\Scripts\\python.exe scripts\\api\\judge_success_relearn.py entry --exp R2CSMOKE --plan configs\\rotate\\plan_r2csmoke_a.json
    .venv\\Scripts\\python.exe scripts\\api\\judge_success_relearn.py check-d               # H-R2D（Ud → Udc、6.2・7 節）。1 回だけ
    .venv\\Scripts\\python.exe scripts\\api\\judge_success_relearn.py check-l               # H-R2L（Ld → Ldc、600 配置、6.3・7 節）。1 回だけ
    .venv\\Scripts\\python.exe scripts\\api\\judge_success_relearn.py check-p               # H-R2P（Ld → Lp、600 配置、6.4・7 節）。1 回だけ
    .venv\\Scripts\\python.exe scripts\\api\\judge_success_relearn.py check-r               # H-R2R（Lp → Lp ＋ RTC の 3 設定、6.5・7 節）。1 回だけ
    .venv\\Scripts\\python.exe scripts\\api\\judge_success_relearn.py decompose --exps R1E,R1E_B --conds N0,U0,Ud,Lf,Ld,Ln --after outputs/api/r1_judge_R1E.json
        # 成功の内訳（記述、判定に使わない）。--after の判定が complete で書かれた後だけ読む（判定より先に数えない）

条件（どれも LLM の組み直しなし）: U0（Rreal）・Ud（Rdag）・U2a（R2a＝Rdag に自分の成功も足して学び直したモデル）・
  Uc（R2a ＋ 行動列の候補 8 本から、学習した成功の見込みが最も高いものを選ぶ）・Udc（Rdag ＋ 同じく候補から選ぶ）。
主な比較（1 つ、補正なし）: H-R2「3 個とも箱へ」Ud → U2a。同じシードの対、McNemar の正確検定（両側、α 0.05）、差の部分は
  Newcombe の方法 10、各条件は Wilson。p < 0.05 かつ向きが改善なら「上がった」、それ以外は「確かめられなかった」。
  H-R2C（U2a → Uc）・H-R2D（Ud → Udc）も同じ方法で、それぞれ 1 回だけ数える。
参考（記述、判定に使わない）: 条件ごとの成功の内訳（掴み損ねなしで 3 個とも／掴み損ねの後に 3 個とも）、自力の立て直し率
  （r1_protocol.md 7.1 節の数え）、掴み損ねがあった試行の割合。比べる組の立て直し率の差は、judge_failure_relearn.py と同じ並べ替え検定の
  p と部分を記述として書く。
内訳の定義: 掴み損ね＝r1_protocol.md 7.1 節の数え（natural_recovery.py の kind "miss"）。落とした・置き損ねは数えない（B と同じ
  範囲）。「掴み損ねなしで 3 個とも」＝3 個とも かつ 掴み損ね 0 回、「掴み損ねの後に 3 個とも」＝3 個とも かつ 1 回以上。
有効性: judge_failure_relearn.py と同じ（各条件のシードがちょうど 1 本ずつ、G_AUDIT の met、HEAD が 1 つ、実験ごとに環境が 1 つ、条件・モデル・
  チェックポイントの道・世界・開閉の絞り・手順の切り替えが条件と同じ、1 台で無効が 10 本を超えたら判定しない、A と B が一致しなければ判定しない）。
R2a のチェックポイントの道は、実行の計画（configs/rotate/plan_r2e_a.json の U2a の --model-ckpt）から読む。
書くもの: outputs/api/r2_judge_R2E.json・.md（check-c は r2_judge_R2CE、check-d は r2_judge_R2DE）、内訳は
  outputs/api/r2_decompose_<実験>.json・.md。
"""
import argparse
import importlib.util
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUTD = ROOT / "outputs" / "api"
EXPERIMENT = "R2E"
PLAN_A = ROOT / "configs" / "rotate" / "plan_r2e_a.json"


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


J = _load(ROOT / "scripts" / "api" / "judge_failure_relearn.py", "api_r1_judge_for_r2")
B = _load(ROOT / "scripts" / "api" / "judge_failure_relearn_b.py", "api_r1_judge_b_for_r2")
PLAN = _load(ROOT / "scripts" / "api" / "plan_success_relearn.py", "api_r2_plan_for_judge")
CONDS = dict(PLAN.CONDS)
FAMILY = {"H-R2": ("Ud", "U2a", "all_three")}
SELF_DESC = ("Ud", "U2a")                # 立て直しの割合の差（記述だけ）
SPLIT = dict(PLAN.MAIN)
R1_CONDS = dict(J.CONDS)                 # 内訳を judge_failure_relearn.py の評価の記録に使うとき
# H-R2C（r2_protocol.md 6.1・7 節）: Uc（実験 R2CE*）対 U2a（R2E・R2E_B の記録をそのまま読む）
FAMILY_C = {"H-R2C": ("U2a", "Uc", "all_three")}
SPLIT_C = dict(PLAN.MAIN_C)
PLAN_C = ROOT / "configs" / "rotate" / "plan_r2ce.json"
# H-R2D（r2_protocol.md 6.2・7 節）: Udc（実験 R2DE*、Rdag ＋ 見込みの重み scorer_run4）対 Ud（R2E・R2E_B の記録をそのまま読む）
FAMILY_D = {"H-R2D": ("Ud", "Udc", "all_three")}
SPLIT_D = dict(PLAN.MAIN_D)
PLAN_D = ROOT / "configs" / "rotate" / "plan_r2de.json"
# H-R2L（r2_protocol.md 6.3・7 節）: Ldc（R2LP*・R2LC*）対 Ld（シード 171400〜171599 は R1E・R1E_B の記録をそのまま、171900〜172299 は R2LD*）
FAMILY_L = {"H-R2L": ("Ld", "Ldc", "all_three")}
SPLIT_L_OLD = {"R1E": (171400, 100), "R1E_B": (171500, 100)}      # R1 の評価の Ld（outputs/v2eval に写しを置いて読むだけ）
PLAN_L = ROOT / "configs" / "rotate" / "plan_r2lp1.json"
# H-R2P（r2_protocol.md 6.4・7 節）: Lp（R2PP*・R2PN*、Ld の VLA を置く位置の入力つきの Rplace に替えたもの）対 Ld（H-R2L と同じ記録）
FAMILY_P = {"H-R2P": ("Ld", "Lp", "all_three")}
PLAN_P = ROOT / "configs" / "rotate" / "plan_r2pp1.json"
# 段階 R（docs/api/r2_protocol.md 6.5・7 節）: Lp ＋ RTC の 3 設定（R2RA*・R2RB*・R2RC*）対 Lp（R2PP1〜6 の記録、表 1 と同じ 200 通り）
FAMILY_R = {f"H-R2R-{c}": ("Lp", c, "all_three") for c in PLAN.RTC_CANDS}
ACTION_DT = 0.1


def rplace_ckpt_from_plan(plan: pathlib.Path = PLAN_P) -> str:
    if not plan.is_file():
        raise SystemExit(f"{plan} が無い（先に scripts/api/plan_success_relearn.py --stage-p）")
    argv = json.loads(plan.read_text(encoding="utf-8"))["conditions"]["Lp"]["argv"]
    v = argv[argv.index("--model-ckpt") + 1]
    if not v.startswith("Rplace="):
        raise SystemExit(f"{plan} の --model-ckpt が Rplace=… でない: {v}")
    return v.split("=", 1)[1]


def r2a_ckpt_from_plan(plan: pathlib.Path = PLAN_A) -> str:
    if not plan.is_file():
        raise SystemExit(f"{plan} が無い（先に scripts/api/plan_success_relearn.py）")
    conds = json.loads(plan.read_text(encoding="utf-8"))["conditions"]
    argv = (conds.get("U2a") or conds["Uc"])["argv"]
    v = argv[argv.index("--model-ckpt") + 1]
    if not v.startswith("R2a="):
        raise SystemExit(f"{plan} の --model-ckpt が R2a=… でない: {v}")
    return v.split("=", 1)[1]


def scorer_sha_from_plan(plan: pathlib.Path = PLAN_C) -> str:
    if not plan.is_file():
        raise SystemExit(f"{plan} が無い（先に scripts/api/plan_success_relearn.py --stage-c か --stage-d）")
    conds = json.loads(plan.read_text(encoding="utf-8"))["conditions"]
    argv = next(conds[c]["argv"] for c in ("Uc", "Udc", "Ldc") if c in conds)
    return argv[argv.index("--select-scorer-sha256") + 1]


def use_r2_conditions(r2a: str) -> None:
    """judge_failure_relearn.py（J）の条件とチェックポイントの照合を、この判定の条件に差し替える。"""
    J.CONDS = dict(CONDS, **PLAN.CONDS_C, **PLAN.CONDS_D, **PLAN.CONDS_L, **PLAN.CONDS_P, **PLAN.CONDS_R)
    J.CKPT_OF = {"Rreal": "None", "Rdag": PLAN.RDAG, "R2a": r2a}
    if PLAN_P.is_file():
        J.CKPT_OF["Rplace"] = rplace_ckpt_from_plan(PLAN_P)


def place_check(dirs) -> tuple:
    """Lp の全試行に "place_cue"（手順・やり直しの始めに決めた置く位置）の記録があるか。決め方の内訳（記述）も返す。"""
    import collections
    probs, src, n = [], collections.Counter(), 0
    for exp, d in dirs:
        for js in sorted(pathlib.Path(d).glob("run_[0-9][0-9][0-9][0-9].json")):
            if not js.with_suffix(".npz").is_file():
                continue
            m = json.loads(js.read_text(encoding="utf-8"))
            rows = m.get("place_cue") or []
            if not rows:
                probs.append(f"Lp: {exp} 種 {m.get('seed')} に置く場所の手がかりの記録が無い")
            n += 1
            src.update(r.get("source") for r in rows)
    return probs, {"trials": n, "sources": dict(src)}


def rtc_check(dirs, cond: str) -> tuple:
    """段階 R: 設定 cond の全試行の "u4"."runtime" が計画の RTC の設定と同じで、方策に RTCConfig が入っていたか。"""
    want = PLAN.RTC_CANDS[cond]
    probs, n = [], 0
    for exp, d in dirs:
        for js in sorted(pathlib.Path(d).glob("run_[0-9][0-9][0-9][0-9].json")):
            if not js.with_suffix(".npz").is_file():
                continue
            m = json.loads(js.read_text(encoding="utf-8"))
            rt = (m.get("u4") or {}).get("runtime") or {}
            rc = rt.get("rtc_config") or {}
            ok = (rt.get("mode") == "rtc" and rt.get("horizon") == want["horizon"] and rt.get("schedule") == want["schedule"]
                  and rt.get("max_guidance_weight") == want["max_guidance_weight"] and rc.get("enabled") is True
                  and rc.get("execution_horizon") == want["horizon"] and rc.get("prefix_attention_schedule") == want["schedule"]
                  and rc.get("max_guidance_weight") == want["max_guidance_weight"])
            if not ok:
                probs.append(f"{cond}: {exp} シード {m.get('seed')} の RTC の記録が計画と違う {rt}")
            n += 1
    return probs, {"trials": n, "want": want}


def seam_jumps(js: pathlib.Path) -> tuple:
    """1 試行の行動列のつなぎ目の段差: 方策の行動が続けて使われた 2 つの区切り（どちらも保持でない）で、使う行動列が替わった所の
    手先の速さの指令の差のノルム（行動の変位 ÷ 0.1 s、m/s）。(つなぎ目の段差の列, 行動列の中の続く区切りの差の列)。"""
    import numpy as np
    rt = json.loads(js.with_name(js.stem + "_runtime.json").read_text(encoding="utf-8"))["runtime"]["actions"]
    seam, within = [], []
    for p_, q_ in zip(rt, rt[1:]):
        if q_["k"] != p_["k"] + 1 or p_["held"] or q_["held"]:
            continue
        v = float(np.linalg.norm(np.subtract(q_["a"][:3], p_["a"][:3]))) / ACTION_DT
        (seam if q_["chunk"] != p_["chunk"] else within).append(v)
    return seam, within


def seam_table(dirs) -> dict:
    """{シード: 試行のつなぎ目の段差の中央値}（つなぎ目が 1 つも無い試行は入れない）と、全体の中央値（記述）。"""
    import numpy as np
    per, alls, allw = {}, [], []
    for exp, d in dirs:
        for js in sorted(pathlib.Path(d).glob("run_[0-9][0-9][0-9][0-9].json")):
            if not js.with_suffix(".npz").is_file():
                continue
            s, w = seam_jumps(js)
            if s:
                per[int(json.loads(js.read_text(encoding="utf-8"))["seed"])] = float(np.median(s))
            alls += s
            allw += w
    return {"per_seed": per, "median_all": float(np.median(alls)) if alls else None,
            "median_within": float(np.median(allw)) if allw else None, "n_seams": len(alls)}


def regrip_free_self(dirs) -> dict:
    """{シード: (自力の立て直し, 掴み損ね)}。掴み損ねから、持ち上げた後の握り直し（直前の閉じている間に目標が 1.5 cm 以上上がり、
    開いてから 1 s 未満でまた閉じ、そのとき目標が置いた高さ＋1.5 cm より上）を除いたもの（README の数え方の点検と同じ）。"""
    import numpy as np
    NR = J.NR
    out = {}
    for exp, d in dirs:
        for js in sorted(pathlib.Path(d).glob("run_[0-9][0-9][0-9][0-9].json")):
            if not js.with_suffix(".npz").is_file():
                continue
            z = np.load(js.with_suffix(".npz"))
            t_, pos, cl = z["sim_time"], z["cube_pos"], z["gripper_closed"]
            ivs = NR.closed_intervals(cl)
            rest = pos[0, :, 2]
            r = NR.trial(js)
            k = n = 0
            for e in r["episodes"]:
                if e["kind"] != "miss":
                    continue
                a = int(np.argmin(np.abs(t_ - e["t"])))
                c = NR.COLORS.index(e["color"])
                prev = [(x, y) for x, y in ivs if y <= a]
                if prev:
                    x, y = prev[-1]
                    if (pos[x:y, c, 2].max() - pos[x, c, 2] >= NR.LIFT_M and t_[a] - t_[y - 1] < 1.0
                            and pos[a, c, 2] > rest[c] + NR.LIFT_M):
                        continue
                n += 1
                k += e["who"] == "self"
            out[int(r["seed"])] = (k, n)
    return out


def after_miss(dirs) -> dict:
    """掴み損ねの後の動き（記述、6.5 節の原因の指標）: 掴み損ね（握り直しを除く）ごとに、開いてから次に閉じるまでの時間と、
    開いてから 1 s の方策の手先の速さの指令の平均の向き（箱へ向かう＝前の運ぶ向き、目標へ向かう、その他。水平の向きの余弦 0.5 以上）。"""
    import numpy as np
    FC = _load(ROOT / "scripts" / "api" / "failure_causes.py", "api_failure_causes_for_r2")
    NR = J.NR
    box = np.array(FC.BOX_XY, float)
    gaps, dirs_n = [], {"to_box": 0, "to_target": 0, "other": 0, "still": 0}
    no_reclose = n = 0
    for exp, d in dirs:
        for js in sorted(pathlib.Path(d).glob("run_[0-9][0-9][0-9][0-9].json")):
            if not js.with_suffix(".npz").is_file():
                continue
            z = np.load(js.with_suffix(".npz"))
            t_, pos, cl, tip = z["sim_time"], z["cube_pos"], z["gripper_closed"], z["fingertip"]
            rest = pos[0, :, 2]
            ivs = NR.closed_intervals(cl)
            acts = json.loads(js.with_name(js.stem + "_runtime.json").read_text(encoding="utf-8"))["runtime"]["actions"]
            for e in NR.trial(js)["episodes"]:
                if e["kind"] != "miss":
                    continue
                a = int(np.argmin(np.abs(t_ - e["t"])))
                c = NR.COLORS.index(e["color"])
                prev = [(x, y) for x, y in ivs if y <= a]
                if prev:
                    x, y = prev[-1]
                    if (pos[x:y, c, 2].max() - pos[x, c, 2] >= NR.LIFT_M and t_[a] - t_[y - 1] < 1.0
                            and pos[a, c, 2] > rest[c] + NR.LIFT_M):
                        continue
                b = next((y for x, y in ivs if x == a), None)
                if b is None or b >= len(t_):
                    continue
                n += 1
                t_open = float(t_[b])
                nxt = [x for x, y in ivs if x > b]
                if nxt:
                    gaps.append(float(t_[nxt[0]]) - t_open)
                else:
                    no_reclose += 1
                vs = [np.asarray(r_["a"][:2], float) / ACTION_DT for r_ in acts
                      if t_open <= r_["t"] < t_open + 1.0 and not r_["held"]]
                if not vs or np.linalg.norm(np.mean(vs, axis=0)) < 0.02:
                    dirs_n["still"] += 1
                    continue
                v = np.mean(vs, axis=0)
                h = tip[b, :2]
                cos = lambda u: float(np.dot(v, u) / (np.linalg.norm(v) * np.linalg.norm(u) + 1e-12))
                if cos(box - h) >= 0.5:
                    dirs_n["to_box"] += 1
                elif cos(pos[b, c, :2] - h) >= 0.5:
                    dirs_n["to_target"] += 1
                else:
                    dirs_n["other"] += 1
    return {"n_miss": n, "reclose_gap_s_median": float(np.median(gaps)) if gaps else None, "no_reclose": no_reclose,
            "direction_1s": dirs_n}


def failure_kinds(dirs, cond: str) -> dict:
    """失敗した試行の分類（failure_causes.category、記述）。"""
    import collections
    FC = _load(ROOT / "scripts" / "api" / "failure_causes.py", "api_failure_causes_for_r2")
    cnt = collections.Counter()
    for exp, d in dirs:
        for js in sorted(pathlib.Path(d).glob("run_[0-9][0-9][0-9][0-9].json")):
            if not js.with_suffix(".npz").is_file():
                continue
            r = FC.trial(js, cond)
            if not r["success"]:
                cnt[FC.category(r)] += 1
    return dict(cnt)


def wilcoxon_row(base: dict, arm: dict) -> dict:
    """シードを組にしたつなぎ目の段差（試行の中央値）の Wilcoxon の符号順位検定（両側、ゼロの差は除く）。"""
    import numpy as np
    from scipy.stats import wilcoxon
    ss = sorted(set(base) & set(arm))
    x = np.array([base[s] for s in ss])
    y = np.array([arm[s] for s in ss])
    p = float(wilcoxon(y, x, zero_method="wilcox", alternative="two-sided").pvalue) if len(ss) > 0 else float("nan")
    return {"n_pairs": len(ss), "base_median": float(np.median(x)), "arm_median": float(np.median(y)),
            "diff_median": float(np.median(y - x)), "p": p}


def holm_plain(ps: dict) -> dict:
    order = sorted(ps, key=lambda k: (ps[k], k))
    out, run = {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * ps[k]))
        out[k] = run
    return out


def sheet_r(res: dict) -> str:
    L = ["# 段階 R の評価の判定（最終モデル Lp に RTC を足した 3 設定 対 Lp）", "",
         f"- RTC の実験 {'・'.join(res['experiments_r'])}、Lp は {'・'.join(res['experiments_lp'])}、状態 {res['status']}、"
         f"二重集計（3 個とも）{'一致' if res.get('agree') else '—'}",
         "- 評価の決まり: docs/api/r2_protocol.md 6.5・7 節（族 2 つ、それぞれ Holm、α 0.05、表 1 と同じ 200 通り）", ""]
    if res["status"] != "complete":
        return "\n".join(L + ["## 足りないもの・問題", ""] + [f"- {p}" for p in res["problems"]]) + "\n"
    L += ["## 族 1: 3 個とも箱へ（McNemar の正確検定・両側、Holm）", "",
          "| 条件 | RTC の設定 | Lp | RTC | 差 | 差の 95% 信頼区間 | p | Holm 後の p | 結果 |", "|---|---|---|---|---|---|---|---|---|"]
    for h, r in res["a"]["primary"].items():
        ci = r["diff_ci_newcombe"]
        L.append(f"| {r['arm']} | {res['settings'][r['arm']]} | {r['base_success']}/{r['n_seeds']} | {r['arm_success']}/{r['n_seeds']}"
                 f"（b {r['b']}・c {r['c']}） | {100 * r['diff']:+.1f} pt | [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}] pt | "
                 f"{r['p']:.4g} | {r['p_holm']:.4g} | {res['task_result'][h]} |")
    L += ["", "## 族 2: つなぎ目の段差（試行の中央値、m/s。Wilcoxon の符号順位検定・両側、Holm）", "",
          "| 条件 | 組 | Lp の中央値 | RTC の中央値 | 差の中央値 | p | Holm 後の p | 結果 |", "|---|---|---|---|---|---|---|---|"]
    for c, r in res["seam"].items():
        L.append(f"| {c} | {r['n_pairs']} | {r['base_median']:.4f} | {r['arm_median']:.4f} | {r['diff_median']:+.4f} | "
                 f"{r['p']:.4g} | {r['p_holm']:.4g} | {r['result']} |")
    L += ["", "## 記述（判定に使わない）", ""]
    for c, s in res["self_desc"].items():
        f = res["self_desc_regrip_free"][c]
        L.append(f"- 自力の立て直し {c} 対 Lp: 定義どおり {s['arm_k']}/{s['arm_n']} 対 {s['base_k']}/{s['base_n']}（並べ替えの p {s['p']:.4g}）、"
                 f"握り直しを除く {f['arm_k']}/{f['arm_n']} 対 {f['base_k']}/{f['base_n']}（p {f['p']:.4g}）")
    for c, x in res["after_miss"].items():
        L.append(f"- 掴み損ねの後 {c}: {x}")
    for c, x in res["failure_kinds"].items():
        L.append(f"- 失敗の分類 {c}: {x}")
    L.append(f"- 段差の全体の中央値（つなぎ目・行動列の中）: {res['seam_all']}")
    return "\n".join(L) + "\n"


def cmd_check_r(a) -> int:
    """段階 R を 1 回だけ数える（6.5・7 節）。Lp は段階 P の R2PP1〜6、RTC は R2RA*・R2RB*・R2RC*。"""
    root = pathlib.Path(a.root)
    use_r2_conditions(r2a_ckpt_from_plan(PLAN_A))
    seeds = list(range(171400, 171600))
    split_lp = dict(PLAN.MAIN_PP)
    split_c = {c: {e: s for e, (cc, s) in PLAN.EXPS_R.items() if cc == c} for c in PLAN.RTC_CANDS}
    problems = []
    dirs = {"Lp": [(e, root / e / "Lp") for e in split_lp]}
    for c, sp in split_c.items():
        dirs[c] = [(e, root / e / c) for e in sp]
        if sorted(s for s0, n in sp.values() for s in range(s0, s0 + n)) != seeds:
            problems.append(f"{c} のシードが Lp のシードと同じでない")
    for c, ds in dirs.items():
        problems += [f"{c}: 記録が無い {d}" for _, d in ds if not d.is_dir()]
    res = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "experiments_lp": list(split_lp), "experiments_r": list(PLAN.EXPS_R),
           "settings": {c: v for c, v in PLAN.RTC_CANDS.items()}, "n_seeds": len(seeds), "status": "incomplete", "problems": problems}
    code = 3
    if not problems:
        raw = {c: J.read_condition(ds) for c, ds in dirs.items()}
        problems += J.entry_check({"Lp": raw["Lp"]}, seeds, split_lp, False)                    # HEAD は段階 P の固定のもの
        for c in PLAN.RTC_CANDS:
            problems += J.entry_check({c: raw[c]}, seeds, split_c[c], False)
            problems += rtc_check(dirs[c], c)[0]
        heads_r = sorted({h for c in PLAN.RTC_CANDS for h in raw[c]["heads"]})
        if len(heads_r) != 1:
            problems.append(f"RTC の評価の HEAD が 1 つでない: {heads_r}")
        res["problems"] = problems
        if not problems:
            data = {c: raw[c] for c in ["Lp", *PLAN.RTC_CANDS]}
            ja = J.judge(data, seeds, FAMILY_R)
            exps = list(split_lp) + list(PLAN.EXPS_R)
            jb = B.judge_b(root, exps, tuple(data), FAMILY_R, seeds)
            diff = J.compare({"counts": J.summary_counts(data, seeds), "rows": J.key_numbers(ja)}, jb)
            task_result = {h: ("上がった" if r["p_holm"] < J.ALPHA and r["diff"] > 0 else
                               "下がった" if r["p_holm"] < J.ALPHA and r["diff"] < 0 else "確かめられなかった")
                           for h, r in ja["primary"].items()}
            st = {c: seam_table(ds) for c, ds in dirs.items()}
            seam = {c: wilcoxon_row(st["Lp"]["per_seed"], st[c]["per_seed"]) for c in PLAN.RTC_CANDS}
            hp = holm_plain({c: r["p"] for c, r in seam.items()})
            for c, r in seam.items():
                r["p_holm"] = hp[c]
                r["result"] = ("小さくなった" if hp[c] < J.ALPHA and r["diff_median"] < 0 else
                               "大きくなった" if hp[c] < J.ALPHA and r["diff_median"] > 0 else "確かめられなかった")
            rf = {c: regrip_free_self(ds) for c, ds in dirs.items()}
            res.update(status="complete" if not diff else "disagree", a=ja, task_result=task_result, seam=seam,
                       seam_all={c: {"seam": s["median_all"], "within": s["median_within"], "n": s["n_seams"]} for c, s in st.items()},
                       self_desc={c: dict(J.self_rate_row(data["Lp"]["self"], data[c]["self"], seeds), base="Lp", arm=c)
                                  for c in PLAN.RTC_CANDS},
                       self_desc_regrip_free={c: dict(J.self_rate_row(rf["Lp"], rf[c], seeds), base="Lp", arm=c)
                                              for c in PLAN.RTC_CANDS},
                       after_miss={c: after_miss(ds) for c, ds in dirs.items()},
                       failure_kinds={c: failure_kinds(ds, c) for c, ds in dirs.items()},
                       counts=J.summary_counts(data, seeds), agree=not diff, diffs=diff[:50],
                       heads={c: d["heads"] for c, d in raw.items()})
            code = 0 if not diff else 1
    OUTD.mkdir(parents=True, exist_ok=True)
    stem = "r2_judge_R2R"
    (OUTD / f"{stem}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    txt = sheet_r(res)
    (OUTD / f"{stem}.md").write_text(txt, encoding="utf-8")
    print(txt)
    return code


def select_check(dirs, sha: str, cond: str = "Uc") -> tuple:
    """候補から選ぶ条件（Uc・Udc）の試行の "select"（重みの SHA-256・K・control never）を照らし、選びの記述（推論の数・
    候補 0 以外を選んだ数）を返す。"""
    probs, n_inf, n_mov, n = [], 0, 0, 0
    for exp, d in dirs:
        for js in sorted(pathlib.Path(d).glob("run_[0-9][0-9][0-9][0-9].json")):
            if not js.with_suffix(".npz").is_file():
                continue
            m = json.loads(js.read_text(encoding="utf-8"))
            s = m.get("select") or {}
            if s.get("scorer_sha256") != sha or s.get("k") != 8 or s.get("control") != "never":
                probs.append(f"{cond}: {exp} 種 {m.get('seed')} の選びが計画と違う（{s.get('scorer_sha256')}, k {s.get('k')}）")
            n, n_inf, n_mov = n + 1, n_inf + int(s.get("n_infer") or 0), n_mov + int(s.get("n_moved") or 0)
    return probs, {"trials": n, "n_infer": n_inf, "n_moved": n_mov, "moved_frac": n_mov / n_inf if n_inf else None}


# ---------------------------------------------------------------- 成功の内訳（記述）
def decompose_a(data: dict, seeds: list) -> dict:
    """data は J.read_condition の {条件: …}。条件ごとの本数・3 個とも・内訳・立て直し。"""
    out = {}
    for c, d in data.items():
        ok = [s for s in seeds if d["success"][s]]
        miss = [s for s in seeds if d["self"][s][1] > 0]
        out[c] = {"n": len(seeds), "all_three": len(ok),
                  "all_three_no_miss": sum(1 for s in ok if d["self"][s][1] == 0),
                  "all_three_after_miss": sum(1 for s in ok if d["self"][s][1] > 0),
                  "trials_with_miss": len(miss),
                  "self_k": sum(d["self"][s][0] for s in seeds), "miss_n": sum(d["self"][s][1] for s in seeds)}
    return out


def decompose_b(root: pathlib.Path, exps, conds, seeds) -> dict:
    out = {}
    for c in conds:
        D = B.read_b(pathlib.Path(root), exps, c)
        ok = [s for s in seeds if D[s]["ok"]]
        out[c] = {"n": len(seeds), "all_three": len(ok),
                  "all_three_no_miss": sum(1 for s in ok if D[s]["n"] == 0),
                  "all_three_after_miss": sum(1 for s in ok if D[s]["n"] > 0),
                  "trials_with_miss": sum(1 for s in seeds if D[s]["n"] > 0),
                  "self_k": sum(D[s]["k"] for s in seeds), "miss_n": sum(D[s]["n"] for s in seeds)}
    return out


def decompose_sheet(title: str, dec: dict) -> list:
    L = [f"## {title}", "", "| 条件 | 3 個とも | うち掴み損ねなし | うち掴み損ねの後 | 掴み損ねがあった試行 | 自分で立て直した |",
         "|---|---|---|---|---|---|"]
    for c, v in dec.items():
        rate = f"{v['self_k']}/{v['miss_n']}" + (f"（{100 * v['self_k'] / v['miss_n']:.1f}%）" if v["miss_n"] else "")
        L.append(f"| {c} | {v['all_three']}/{v['n']} | {v['all_three_no_miss']} | {v['all_three_after_miss']} | "
                 f"{v['trials_with_miss']} | {rate} |")
    return L + [""]


# ---------------------------------------------------------------- 判定
def sheet(res: dict) -> str:
    L = [f"# R2 の段階 A の評価の判定（{'練習・判定に使わない' if res['practice'] else '本番'}）", "",
         f"- 実験 {'・'.join(res['experiments'])}、状態 {res['status']}、二重集計 {'一致' if res.get('agree') else '—'}",
         "- 手順書: docs/api/r2_protocol.md 7 節（1 本、McNemar の正確検定 両側、α 0.05）", ""]
    if res["status"] != "complete":
        return "\n".join(L + ["## 足りないもの・問題", ""] + [f"- {p}" for p in res["problems"]]) + "\n"
    L += ["| 項目 | 比べる組 | 基準 | 腕 | 差（腕 − 基準） | 差の 95% 区間 | p | 結果 |", "|---|---|---|---|---|---|---|---|"]
    for h, r in res["a"]["primary"].items():
        ci = r["diff_ci_newcombe"]
        L.append(f"| {h} | {r['arm']} 対 {r['base']} | {r['base_success']}/{r['n_seeds']} | {r['arm_success']}/{r['n_seeds']}"
                 f"（b {r['b']}・c {r['c']}） | {100 * r['diff']:+.1f} pt | [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}] pt | "
                 f"{r['p']:.4g} | {r['result']} |")
    L += [""] + decompose_sheet("副次（記述、判定に使わない）: 成功の内訳", res["decompose"])
    s = res.get("self_desc")
    if s:
        L += [f"- 立て直しの割合 {s['arm']} 対 {s['base']}（記述）: {s['arm_k']}/{s['arm_n']} 対 {s['base_k']}/{s['base_n']}、"
              f"差 {100 * s['diff']:+.1f} pt、並べ替えの p {s['p']:.4g}、区間 [{100 * s['diff_ci_boot'][0]:+.1f}, "
              f"{100 * s['diff_ci_boot'][1]:+.1f}] pt", ""]
    return "\n".join(L) + "\n"


def cmd_check(a) -> int:
    root = pathlib.Path(a.root)
    use_r2_conditions(r2a_ckpt_from_plan())
    practice = bool(a.practice)
    if practice:
        exps = [e for e in a.practice.split(",") if e]
        split, stem = {e: None for e in exps}, f"r2_judge_practice_{exps[0]}"
    else:
        split, stem = dict(SPLIT), f"r2_judge_{EXPERIMENT}"
    conds = tuple(CONDS)
    data, problems = {}, []
    for c in conds:
        dirs = [(e, root / e / c) for e in split]
        miss = [str(d) for _, d in dirs if not d.is_dir()]
        if miss:
            problems += [f"{c}: 記録が無い {m}" for m in miss]
            continue
        data[c] = J.read_condition(dirs)
    if practice:
        seeds = sorted(set.intersection(*(set(d["success"]) for d in data.values()))) if data else []
        if not seeds:
            problems.append("練習: 共通の種が無い")
    else:
        seeds = sorted(s for s0, n in split.values() for s in range(s0, s0 + n))
    if not problems:
        problems += J.entry_check(data, seeds, split, practice)
    res = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "experiments": list(split), "practice": practice,
           "conditions": list(conds), "n_seeds": len(seeds), "status": "complete" if not problems else "incomplete",
           "problems": problems}
    code = 3
    if not problems:
        ja = J.judge(data, seeds, FAMILY)
        jb = B.judge_b(root, list(split), conds, FAMILY, seeds)
        dec = decompose_a(data, seeds)
        diff = J.compare({"counts": J.summary_counts(data, seeds), "rows": J.key_numbers(ja)}, jb)
        diff += J.compare(dec, decompose_b(root, list(split), conds, seeds), ".decompose")
        b0, a0 = SELF_DESC
        sd = dict(J.self_rate_row(data[b0]["self"], data[a0]["self"], seeds), base=b0, arm=a0)
        res.update(a=ja, decompose=dec, self_desc=sd, counts=J.summary_counts(data, seeds),
                   invalid_by_exp=J.invalid_by_exp(data, split), agree=not diff, diffs=diff[:50])
        code = 0 if not diff else 1
        if diff:
            res["status"] = "disagree"
    OUTD.mkdir(parents=True, exist_ok=True)
    (OUTD / f"{stem}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    txt = sheet(res)
    (OUTD / f"{stem}.md").write_text(txt, encoding="utf-8")
    print(txt)
    if res.get("diffs"):
        print("[r2-judge] A と B が一致しない: " + "; ".join(res["diffs"][:10]), file=sys.stderr)
    return code


def sheet_c(res: dict) -> str:
    L = ["# R2 の段階 C の評価の判定（本番、H-R2C）", "",
         f"- Uc の実験 {'・'.join(res['experiments_c'])}、U2a・Ud は {'・'.join(res['experiments_a'])} の記録、状態 {res['status']}、"
         f"二重集計 {'一致' if res.get('agree') else '—'}",
         "- 手順書: docs/api/r2_protocol.md 6.1・7 節（1 本、McNemar の正確検定 両側、α 0.05）", ""]
    if res["status"] != "complete":
        return "\n".join(L + ["## 足りないもの・問題", ""] + [f"- {p}" for p in res["problems"]]) + "\n"
    L += ["| 項目 | 比べる組 | 基準 | 腕 | 差（腕 − 基準） | 差の 95% 区間 | p | 結果 |", "|---|---|---|---|---|---|---|---|"]
    for h, r in res["a"]["primary"].items():
        ci = r["diff_ci_newcombe"]
        L.append(f"| {h} | {r['arm']} 対 {r['base']} | {r['base_success']}/{r['n_seeds']} | {r['arm_success']}/{r['n_seeds']}"
                 f"（b {r['b']}・c {r['c']}） | {100 * r['diff']:+.1f} pt | [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}] pt | "
                 f"{r['p']:.4g} | {r['result']} |")
    L += [""] + decompose_sheet("副次（記述、判定に使わない）: 成功の内訳", res["decompose"])
    d = res["desc_vs_Ud"]
    L += [f"- Uc 対 Ud（記述）: {d['arm_success']}/{d['n_seeds']} 対 {d['base_success']}/{d['n_seeds']}、差 {100 * d['diff']:+.1f} pt、"
          f"McNemar の p {d['p']:.4g}",
          f"- 立て直しの割合 Uc 対 U2a（記述）: {res['self_desc']['arm_k']}/{res['self_desc']['arm_n']} 対 "
          f"{res['self_desc']['base_k']}/{res['self_desc']['base_n']}、並べ替えの p {res['self_desc']['p']:.4g}",
          f"- 選び（記述）: 推論 {res['select']['n_infer']} 回のうち、候補 0（選ばない腕と同じ乱数）以外を選んだのは "
          f"{res['select']['n_moved']} 回", ""]
    return "\n".join(L) + "\n"


def cmd_check_c(a) -> int:
    """H-R2C を 1 回だけ数える。U2a・Ud は R2E・R2E_B の記録（6 節）、Uc は R2CE*（6.1 節）。"""
    root = pathlib.Path(a.root)
    use_r2_conditions(r2a_ckpt_from_plan(PLAN_C))
    sha = scorer_sha_from_plan()
    seeds = sorted(s for s0, n in SPLIT.values() for s in range(s0, s0 + n))
    seeds_c = sorted(s for s0, n in SPLIT_C.values() for s in range(s0, s0 + n))
    problems = [] if seeds == seeds_c else ["Uc の帯が U2a の帯と同じでない"]
    dirs = {"U2a": [(e, root / e / "U2a") for e in SPLIT], "Ud": [(e, root / e / "Ud") for e in SPLIT],
            "Uc": [(e, root / e / "Uc") for e in SPLIT_C]}
    for c, ds in dirs.items():
        problems += [f"{c}: 記録が無い {d}" for _, d in ds if not d.is_dir()]
    res = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "experiments_a": list(SPLIT), "experiments_c": list(SPLIT_C),
           "scorer_sha256": sha, "n_seeds": len(seeds), "status": "incomplete", "problems": problems}
    code = 3
    if not problems:
        data = {c: J.read_condition(ds) for c, ds in dirs.items()}
        problems += J.entry_check({c: data[c] for c in ("U2a", "Ud")}, seeds, SPLIT, False)
        problems += J.entry_check({"Uc": data["Uc"]}, seeds, SPLIT_C, False)
        sp, sel = select_check(dirs["Uc"], sha)
        problems += sp
        res["problems"] = problems
        if not problems:
            ja = J.judge(data, seeds, FAMILY_C)
            exps = list(SPLIT) + list(SPLIT_C)
            jb = B.judge_b(root, exps, ("U2a", "Ud", "Uc"), FAMILY_C, seeds)
            dec = decompose_a(data, seeds)
            diff = J.compare({"counts": J.summary_counts(data, seeds), "rows": J.key_numbers(ja)}, jb)
            diff += J.compare(dec, decompose_b(root, exps, ("U2a", "Ud", "Uc"), seeds), ".decompose")
            res.update(status="complete" if not diff else "disagree", a=ja, decompose=dec, select=sel,
                       desc_vs_Ud=dict(J.success_row(data["Ud"]["success"], data["Uc"]["success"], seeds), base="Ud", arm="Uc"),
                       self_desc=dict(J.self_rate_row(data["U2a"]["self"], data["Uc"]["self"], seeds), base="U2a", arm="Uc"),
                       counts=J.summary_counts(data, seeds), agree=not diff, diffs=diff[:50],
                       heads={c: d["heads"] for c, d in data.items()})
            code = 0 if not diff else 1
    OUTD.mkdir(parents=True, exist_ok=True)
    stem = "r2_judge_R2CE"
    (OUTD / f"{stem}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    txt = sheet_c(res)
    (OUTD / f"{stem}.md").write_text(txt, encoding="utf-8")
    print(txt)
    return code


def sheet_d(res: dict) -> str:
    L = ["# R2 の段階 C の評価の判定（本番、H-R2D: R1 の土台 ＋ 選び）", "",
         f"- Udc の実験 {'・'.join(res['experiments_d'])}、Ud は {'・'.join(res['experiments_a'])} の記録、状態 {res['status']}、"
         f"二重集計 {'一致' if res.get('agree') else '—'}",
         "- 手順書: docs/api/r2_protocol.md 6.2・7 節（1 本、McNemar の正確検定 両側、α 0.05）", ""]
    if res["status"] != "complete":
        return "\n".join(L + ["## 足りないもの・問題", ""] + [f"- {p}" for p in res["problems"]]) + "\n"
    L += ["| 項目 | 比べる組 | 基準 | 腕 | 差（腕 − 基準） | 差の 95% 区間 | p | 結果 |", "|---|---|---|---|---|---|---|---|"]
    for h, r in res["a"]["primary"].items():
        ci = r["diff_ci_newcombe"]
        L.append(f"| {h} | {r['arm']} 対 {r['base']} | {r['base_success']}/{r['n_seeds']} | {r['arm_success']}/{r['n_seeds']}"
                 f"（b {r['b']}・c {r['c']}） | {100 * r['diff']:+.1f} pt | [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}] pt | "
                 f"{r['p']:.4g} | {r['result']} |")
    L += [""] + decompose_sheet("副次（記述、判定に使わない）: 成功の内訳", res["decompose"])
    s = res["self_desc"]
    L += [f"- 立て直しの割合 Udc 対 Ud（記述）: {s['arm_k']}/{s['arm_n']} 対 {s['base_k']}/{s['base_n']}、並べ替えの p {s['p']:.4g}",
          f"- 選び（記述）: 推論 {res['select']['n_infer']} 回のうち、候補 0（選ばない腕と同じ乱数）以外を選んだのは "
          f"{res['select']['n_moved']} 回", ""]
    return "\n".join(L) + "\n"


def cmd_check_d(a) -> int:
    """H-R2D を 1 回だけ数える。Ud は R2E・R2E_B の記録（6 節）、Udc は R2DE*（6.2 節）。"""
    root = pathlib.Path(a.root)
    use_r2_conditions(r2a_ckpt_from_plan(PLAN_A))
    sha = scorer_sha_from_plan(PLAN_D)
    seeds = sorted(s for s0, n in SPLIT.values() for s in range(s0, s0 + n))
    seeds_d = sorted(s for s0, n in SPLIT_D.values() for s in range(s0, s0 + n))
    problems = [] if seeds == seeds_d else ["Udc の帯が Ud の帯と同じでない"]
    dirs = {"Ud": [(e, root / e / "Ud") for e in SPLIT], "Udc": [(e, root / e / "Udc") for e in SPLIT_D]}
    for c, ds in dirs.items():
        problems += [f"{c}: 記録が無い {d}" for _, d in ds if not d.is_dir()]
    res = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "experiments_a": list(SPLIT), "experiments_d": list(SPLIT_D),
           "scorer_sha256": sha, "n_seeds": len(seeds), "status": "incomplete", "problems": problems}
    code = 3
    if not problems:
        data = {c: J.read_condition(ds) for c, ds in dirs.items()}
        problems += J.entry_check({"Ud": data["Ud"]}, seeds, SPLIT, False)
        problems += J.entry_check({"Udc": data["Udc"]}, seeds, SPLIT_D, False)
        sp, sel = select_check(dirs["Udc"], sha, "Udc")
        problems += sp
        res["problems"] = problems
        if not problems:
            ja = J.judge(data, seeds, FAMILY_D)
            exps = list(SPLIT) + list(SPLIT_D)
            jb = B.judge_b(root, exps, ("Ud", "Udc"), FAMILY_D, seeds)
            dec = decompose_a(data, seeds)
            diff = J.compare({"counts": J.summary_counts(data, seeds), "rows": J.key_numbers(ja)}, jb)
            diff += J.compare(dec, decompose_b(root, exps, ("Ud", "Udc"), seeds), ".decompose")
            res.update(status="complete" if not diff else "disagree", a=ja, decompose=dec, select=sel,
                       self_desc=dict(J.self_rate_row(data["Ud"]["self"], data["Udc"]["self"], seeds), base="Ud", arm="Udc"),
                       counts=J.summary_counts(data, seeds), agree=not diff, diffs=diff[:50],
                       heads={c: d["heads"] for c, d in data.items()})
            code = 0 if not diff else 1
    OUTD.mkdir(parents=True, exist_ok=True)
    stem = "r2_judge_R2DE"
    (OUTD / f"{stem}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    txt = sheet_d(res)
    (OUTD / f"{stem}.md").write_text(txt, encoding="utf-8")
    print(txt)
    return code


def sheet_l(res: dict) -> str:
    L = ["# R2 の段階 L の評価の判定（本番、H-R2L: R1 の仕組み全体 Ld ＋ 選び）", "",
         f"- Ldc の実験 {'・'.join(res['experiments_ldc'])}、Ld は {'・'.join(res['experiments_ld'])}、状態 {res['status']}、"
         f"二重集計 {'一致' if res.get('agree') else '—'}",
         "- 手順書: docs/api/r2_protocol.md 6.3・7 節（1 本、McNemar の正確検定 両側、α 0.05、600 配置）", ""]
    if res["status"] != "complete":
        return "\n".join(L + ["## 足りないもの・問題", ""] + [f"- {p}" for p in res["problems"]]) + "\n"
    L += ["| 項目 | 比べる組 | 基準 | 腕 | 差（腕 − 基準） | 差の 95% 区間 | p | 結果 |", "|---|---|---|---|---|---|---|---|"]
    for h, r in res["a"]["primary"].items():
        ci = r["diff_ci_newcombe"]
        L.append(f"| {h} | {r['arm']} 対 {r['base']} | {r['base_success']}/{r['n_seeds']} | {r['arm_success']}/{r['n_seeds']}"
                 f"（b {r['b']}・c {r['c']}） | {100 * r['diff']:+.1f} pt | [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}] pt | "
                 f"{r['p']:.4g} | {r['result']} |")
    L += ["", "## 帯ごと（記述。動画の成功率の階段の 5 本目は R1 の帯の数を使う、6.3 節）", "",
          "| 帯 | 配置 | Ld | Ldc | b・c |", "|---|---|---|---|---|"]
    for name, r in res["bands"].items():
        L.append(f"| {name} | {r['n']} | {r['ld']} | {r['ldc']} | {r['b']}・{r['c']} |")
    L += [""] + decompose_sheet("副次（記述、判定に使わない）: 成功の内訳", res["decompose"])
    s = res["self_desc"]
    L += [f"- 立て直しの割合 Ldc 対 Ld（記述）: {s['arm_k']}/{s['arm_n']} 対 {s['base_k']}/{s['base_n']}、並べ替えの p {s['p']:.4g}",
          f"- 選び（記述）: 推論 {res['select']['n_infer']} 回のうち、候補 0（選ばない腕と同じ乱数）以外を選んだのは "
          f"{res['select']['n_moved']} 回", ""]
    return "\n".join(L) + "\n"


def sheet_p(res: dict) -> str:
    L = ["# R2 の段階 P の評価の判定（本番、H-R2P: Ld の重みを置く場所の手がかりつきの Rplace に替える）", "",
         f"- Lp の実験 {'・'.join(res['experiments_lp'])}、Ld は {'・'.join(res['experiments_ld'])}、状態 {res['status']}、"
         f"二重集計 {'一致' if res.get('agree') else '—'}",
         "- 手順書: docs/api/r2_protocol.md 6.4・7 節（1 本、McNemar の正確検定 両側、α 0.05、600 配置）", ""]
    if res["status"] != "complete":
        return "\n".join(L + ["## 足りないもの・問題", ""] + [f"- {p}" for p in res["problems"]]) + "\n"
    L += ["| 項目 | 比べる組 | 基準 | 腕 | 差（腕 − 基準） | 差の 95% 区間 | p | 結果 |", "|---|---|---|---|---|---|---|---|"]
    for h, r in res["a"]["primary"].items():
        ci = r["diff_ci_newcombe"]
        L.append(f"| {h} | {r['arm']} 対 {r['base']} | {r['base_success']}/{r['n_seeds']} | {r['arm_success']}/{r['n_seeds']}"
                 f"（b {r['b']}・c {r['c']}） | {100 * r['diff']:+.1f} pt | [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}] pt | "
                 f"{r['p']:.4g} | {r['result']} |")
    L += ["", "## 帯ごと（記述。動画の成功率の階段の 5 本目は R1 の帯の数を使う、6.4 節）", "",
          "| 帯 | 配置 | Ld | Lp | b・c |", "|---|---|---|---|---|"]
    for name, r in res["bands"].items():
        L.append(f"| {name} | {r['n']} | {r['ld']} | {r['lp']} | {r['b']}・{r['c']} |")
    L += [""] + decompose_sheet("副次（記述、判定に使わない）: 成功の内訳", res["decompose"])
    s = res["self_desc"]
    L += [f"- 立て直しの割合 Lp 対 Ld（記述）: {s['arm_k']}/{s['arm_n']} 対 {s['base_k']}/{s['base_n']}、並べ替えの p {s['p']:.4g}",
          f"- 置く場所の手がかりの決め方（記述）: {res['place']['sources']}（{res['place']['trials']} 試行）", ""]
    return "\n".join(L) + "\n"


def cmd_check_p(a) -> int:
    """H-R2P を 1 回だけ数える。Ld は H-R2L と同じ記録（シード 171400〜171599 は R1E・R1E_B、171900〜172299 は R2LD*）、Lp は R2PP*・R2PN*（6.4 節）。"""
    root = pathlib.Path(a.root)
    use_r2_conditions(r2a_ckpt_from_plan(PLAN_A))
    split_lp = {**PLAN.MAIN_PP, **PLAN.MAIN_PN}
    seeds_old = sorted(s for s0, n in SPLIT_L_OLD.values() for s in range(s0, s0 + n))
    seeds_new = sorted(s for s0, n in PLAN.MAIN_LD.values() for s in range(s0, s0 + n))
    seeds = sorted(seeds_old + seeds_new)
    problems = []
    if sorted(s for s0, n in split_lp.values() for s in range(s0, s0 + n)) != seeds:
        problems.append("Lp の帯が Ld の帯と同じでない")
    dirs = {"Ld_old": [(e, root / e / "Ld") for e in SPLIT_L_OLD], "Ld_new": [(e, root / e / "Ld") for e in PLAN.MAIN_LD],
            "Lp": [(e, root / e / "Lp") for e in split_lp]}
    for c, ds in dirs.items():
        problems += [f"{c}: 記録が無い {d}" for _, d in ds if not d.is_dir()]
    res = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "experiments_ld": list(SPLIT_L_OLD) + list(PLAN.MAIN_LD),
           "experiments_lp": list(split_lp), "rplace_ckpt": J.CKPT_OF.get("Rplace"), "n_seeds": len(seeds),
           "status": "incomplete", "problems": problems}
    code = 3
    if not problems:
        raw = {c: J.read_condition(ds) for c, ds in dirs.items()}
        problems += J.entry_check({"Ld": raw["Ld_old"]}, seeds_old, SPLIT_L_OLD, False)       # HEAD は R1 の固定のもの
        problems += J.entry_check({"Ld": raw["Ld_new"]}, seeds_new, dict(PLAN.MAIN_LD), False)  # コミットは Ldc の評価の固定のもの
        problems += J.entry_check({"Lp": raw["Lp"]}, seeds, split_lp, False)
        if len(set(raw["Lp"]["heads"])) != 1:
            problems.append(f"Lp の HEAD が 1 つでない: {sorted(set(raw['Lp']['heads']))}")
        pp, place = place_check(dirs["Lp"])
        problems += pp
        res["problems"] = problems
        if not problems:
            ld = {k: {**raw["Ld_old"][k], **raw["Ld_new"][k]} for k in ("self", "success", "invalid")}
            data = {"Ld": ld, "Lp": raw["Lp"]}
            ja = J.judge(data, seeds, FAMILY_P)
            exps = list(SPLIT_L_OLD) + list(PLAN.MAIN_LD) + list(split_lp)
            jb = B.judge_b(root, exps, ("Ld", "Lp"), FAMILY_P, seeds)
            dec = decompose_a(data, seeds)
            diff = J.compare({"counts": J.summary_counts(data, seeds), "rows": J.key_numbers(ja)}, jb)
            diff += J.compare(dec, decompose_b(root, exps, ("Ld", "Lp"), seeds), ".decompose")
            bands = {}
            for name, ss in (("R1 の帯 171400〜171599", seeds_old), ("新しい帯 171900〜172299", seeds_new)):
                S0, S1 = data["Ld"]["success"], data["Lp"]["success"]
                bands[name] = {"n": len(ss), "ld": sum(S0[s] for s in ss), "lp": sum(S1[s] for s in ss),
                               "b": sum(1 for s in ss if S0[s] and not S1[s]), "c": sum(1 for s in ss if S1[s] and not S0[s])}
            res.update(status="complete" if not diff else "disagree", a=ja, decompose=dec, place=place, bands=bands,
                       self_desc=dict(J.self_rate_row(data["Ld"]["self"], data["Lp"]["self"], seeds), base="Ld", arm="Lp"),
                       counts=J.summary_counts(data, seeds), agree=not diff, diffs=diff[:50],
                       heads={c: d["heads"] for c, d in raw.items()})
            code = 0 if not diff else 1
    OUTD.mkdir(parents=True, exist_ok=True)
    stem = "r2_judge_R2P"
    (OUTD / f"{stem}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    txt = sheet_p(res)
    (OUTD / f"{stem}.md").write_text(txt, encoding="utf-8")
    print(txt)
    return code


def cmd_check_l(a) -> int:
    """H-R2L を 1 回だけ数える。Ld はシード 171400〜171599 が R1E・R1E_B の記録（r1_protocol.md 6 節）、171900〜172299 が R2LD*。Ldc は R2LP*・R2LC*（6.3 節）。"""
    root = pathlib.Path(a.root)
    use_r2_conditions(r2a_ckpt_from_plan(PLAN_A))
    sha = scorer_sha_from_plan(PLAN_L)
    split_ldc = {**PLAN.MAIN_LP, **PLAN.MAIN_LC}
    seeds_old = sorted(s for s0, n in SPLIT_L_OLD.values() for s in range(s0, s0 + n))
    seeds_new = sorted(s for s0, n in PLAN.MAIN_LD.values() for s in range(s0, s0 + n))
    seeds = sorted(seeds_old + seeds_new)
    problems = []
    if sorted(s for s0, n in split_ldc.values() for s in range(s0, s0 + n)) != seeds:
        problems.append("Ldc の帯が Ld の帯と同じでない")
    dirs = {"Ld_old": [(e, root / e / "Ld") for e in SPLIT_L_OLD], "Ld_new": [(e, root / e / "Ld") for e in PLAN.MAIN_LD],
            "Ldc": [(e, root / e / "Ldc") for e in split_ldc]}
    for c, ds in dirs.items():
        problems += [f"{c}: 記録が無い {d}" for _, d in ds if not d.is_dir()]
    res = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "experiments_ld": list(SPLIT_L_OLD) + list(PLAN.MAIN_LD),
           "experiments_ldc": list(split_ldc), "scorer_sha256": sha, "n_seeds": len(seeds), "status": "incomplete",
           "problems": problems}
    code = 3
    if not problems:
        raw = {c: J.read_condition(ds) for c, ds in dirs.items()}
        problems += J.entry_check({"Ld": raw["Ld_old"]}, seeds_old, SPLIT_L_OLD, False)       # HEAD は R1 の固定のもの
        problems += J.entry_check({"Ld": raw["Ld_new"]}, seeds_new, dict(PLAN.MAIN_LD), False)
        problems += J.entry_check({"Ldc": raw["Ldc"]}, seeds, split_ldc, False)
        new_heads = set(raw["Ld_new"]["heads"]) | set(raw["Ldc"]["heads"])
        if len(new_heads) != 1:
            problems.append(f"新しい帯の Ld と Ldc の HEAD が 1 つでない: {sorted(new_heads)}")
        sp, sel = select_check(dirs["Ldc"], sha, "Ldc")
        problems += sp
        res["problems"] = problems
        if not problems:
            ld = {k: {**raw["Ld_old"][k], **raw["Ld_new"][k]} for k in ("self", "success", "invalid")}
            data = {"Ld": ld, "Ldc": raw["Ldc"]}
            ja = J.judge(data, seeds, FAMILY_L)
            exps = list(SPLIT_L_OLD) + list(PLAN.MAIN_LD) + list(split_ldc)
            jb = B.judge_b(root, exps, ("Ld", "Ldc"), FAMILY_L, seeds)
            dec = decompose_a(data, seeds)
            diff = J.compare({"counts": J.summary_counts(data, seeds), "rows": J.key_numbers(ja)}, jb)
            diff += J.compare(dec, decompose_b(root, exps, ("Ld", "Ldc"), seeds), ".decompose")
            bands = {}
            for name, ss in (("R1 の帯 171400〜171599", seeds_old), ("新しい帯 171900〜172299", seeds_new)):
                S0, S1 = data["Ld"]["success"], data["Ldc"]["success"]
                bands[name] = {"n": len(ss), "ld": sum(S0[s] for s in ss), "ldc": sum(S1[s] for s in ss),
                               "b": sum(1 for s in ss if S0[s] and not S1[s]), "c": sum(1 for s in ss if S1[s] and not S0[s])}
            res.update(status="complete" if not diff else "disagree", a=ja, decompose=dec, select=sel, bands=bands,
                       self_desc=dict(J.self_rate_row(data["Ld"]["self"], data["Ldc"]["self"], seeds), base="Ld", arm="Ldc"),
                       counts=J.summary_counts(data, seeds), agree=not diff, diffs=diff[:50],
                       heads={c: d["heads"] for c, d in raw.items()})
            code = 0 if not diff else 1
    OUTD.mkdir(parents=True, exist_ok=True)
    stem = "r2_judge_R2L"
    (OUTD / f"{stem}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    txt = sheet_l(res)
    (OUTD / f"{stem}.md").write_text(txt, encoding="utf-8")
    print(txt)
    return code


def cmd_entry(a) -> int:
    """試しの実行の確かめ: 条件ごとの本数・無効・チェックポイントの道・HEAD と、入口の点検の問題だけ（3 個とも・立て直しは出さない）。"""
    root = pathlib.Path(a.root)
    stage_d = a.exp.startswith(("R2D", "R2L", "R2P", "R2R"))          # Udc・Ld・Ldc・Lp は R2a を読まない（R2a の計画は読む道の照合だけ）
    use_r2_conditions(r2a_ckpt_from_plan(PLAN_A if stage_d else pathlib.Path(a.plan)) if a.plan else "<未定>")
    want = (tuple(PLAN.CONDS_R) if a.exp.startswith("R2RSMOKE")
            else tuple(c for c, x in PLAN.UNIT_R.items() if a.exp.startswith(f"R2R{x}")) if a.exp.startswith("R2R")
            else tuple(PLAN.CONDS_P) if a.exp.startswith(("R2PSMOKE", "R2PP", "R2PN"))
            else tuple(PLAN.CONDS_L) if a.exp.startswith("R2LSMOKE") else ("Ldc",) if a.exp.startswith(("R2LP", "R2LC"))
            else ("Ld",) if a.exp.startswith("R2LD") else tuple(PLAN.CONDS_D) if stage_d
            else tuple(PLAN.CONDS_C) if a.exp.startswith("R2C") else tuple(CONDS))
    conds = tuple(c for c in want if (root / a.exp / c).is_dir())
    data = {c: J.read_condition([(a.exp, root / a.exp / c)]) for c in conds}
    probs = J.entry_check(data, [], {a.exp: None}, True)
    for c, d in data.items():
        print(f"[entry] {a.exp}/{c}: 試行 {len(d['success'])}、無効 {sum(1 for v in d['invalid'].values() if v)}、"
              f"保存点 {d['ckpts']}、HEAD {d['heads']}")
    for sc in ("Uc", "Udc", "Ldc"):
        if sc in data:
            sp, sel = select_check([(a.exp, root / a.exp / sc)], scorer_sha_from_plan(pathlib.Path(a.plan)), sc)
            probs += sp
            print(f"[entry] {a.exp}/{sc}: 選びの推論 {sel['n_infer']} 回（候補 0 以外 {sel['n_moved']} 回）")
    for rc in (c for c in PLAN.RTC_CANDS if c in data):
        rp, rinfo = rtc_check([(a.exp, root / a.exp / rc)], rc)
        probs += rp
        rpp, _ = place_check([(a.exp, root / a.exp / rc)])
        probs += [x.replace("Lp:", f"{rc}:") for x in rpp]
        st = seam_table([(a.exp, root / a.exp / rc)])
        print(f"[entry] {a.exp}/{rc}: RTC {rinfo['want']}、つなぎ目 {st['n_seams']} 個、段差の中央値 {st['median_all']}"
              f"（行動列の中 {st['median_within']}）")
    if "Lp" in data:
        pp, place = place_check([(a.exp, root / a.exp / "Lp")])
        probs += pp
        print(f"[entry] {a.exp}/Lp: 置く場所の手がかりの決め方 {place['sources']}")
    for m in (c for c in want if c not in conds):
        print(f"[entry] {a.exp}/{m}: 記録が無い")
    for p in probs:
        print(f"[entry] 問題: {p}")
    return 0 if not probs and len(conds) == len(want) else 1


def cmd_decompose(a) -> int:
    """記録（judge_failure_relearn.py の評価など）の成功の内訳。--after の判定の記録が complete のときだけ数える（判定より先に数えない）。"""
    after = pathlib.Path(a.after)
    if not after.is_file() or json.loads(after.read_text(encoding="utf-8")).get("status") != "complete":
        print(f"{after} が complete でない（判定の後だけ内訳を数える）", file=sys.stderr)
        return 3
    root = pathlib.Path(a.root)
    exps = [e for e in a.exps.split(",") if e]
    conds = [c for c in a.conds.split(",") if c]
    data = {c: J.read_condition([(e, root / e / c) for e in exps]) for c in conds}
    dups = {c: d["dup"][:5] for c, d in data.items() if d["dup"]}
    if dups:
        print(f"同じ種が 2 本以上ある（実験の並びを確かめる）: {dups}", file=sys.stderr)
        return 3
    seeds = sorted(set.intersection(*(set(d["success"]) for d in data.values())))
    dec = decompose_a(data, seeds)
    diff = J.compare(dec, decompose_b(root, exps, conds, seeds))
    res = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "experiments": exps, "conditions": conds, "n_seeds": len(seeds),
           "after": str(after), "decompose": dec, "agree": not diff, "diffs": diff[:50],
           "note": "記述（判定に使わない）。掴み損ね＝R1 の手順書 7.1 節の把持失敗。落とした・置き損ねは数えない"}
    stem = f"r2_decompose_{exps[0]}"
    OUTD.mkdir(parents=True, exist_ok=True)
    (OUTD / f"{stem}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    txt = "\n".join([f"# 成功の内訳（記述、判定に使わない）: {'・'.join(exps)}", "",
                     f"- {len(seeds)} 配置、二重集計 {'一致' if not diff else '不一致'}", ""]
                    + decompose_sheet("内訳", dec)) + "\n"
    (OUTD / f"{stem}.md").write_text(txt, encoding="utf-8")
    print(txt)
    return 0 if not diff else 1


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:                                    # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check", help="入口の点検 → A・B の二重集計 → 判定の 1 枚")
    p.add_argument("--root", default=str(ROOT / "outputs" / "v2eval"))
    p.add_argument("--practice", default="", help="練習の実験の並び（例 R2SMOKE,R2SMOKE_B）。判定に使わない")
    pc = sub.add_parser("check-c", help="段階 C の H-R2C（Uc 対 U2a）。Uc がそろった後に 1 回だけ")
    pc.add_argument("--root", default=str(ROOT / "outputs" / "v2eval"))
    pd = sub.add_parser("check-d", help="段階 C の H-R2D（Udc 対 Ud）。Udc がそろった後に 1 回だけ")
    pd.add_argument("--root", default=str(ROOT / "outputs" / "v2eval"))
    pl = sub.add_parser("check-l", help="段階 L の H-R2L（Ldc 対 Ld、600 配置）。Ld・Ldc がそろった後に 1 回だけ")
    pl.add_argument("--root", default=str(ROOT / "outputs" / "v2eval"))
    pp = sub.add_parser("check-p", help="段階 P の H-R2P（Lp 対 Ld、600 配置）。Lp がそろった後に 1 回だけ")
    pp.add_argument("--root", default=str(ROOT / "outputs" / "v2eval"))
    pr = sub.add_parser("check-r", help="段階 R（Lp ＋ RTC の 3 設定 対 Lp、200 通り）。全部そろった後に 1 回だけ")
    pr.add_argument("--root", default=str(ROOT / "outputs" / "v2eval"))
    q = sub.add_parser("entry", help="入口の点検だけ（smoke の確かめ）")
    q.add_argument("--root", default=str(ROOT / "outputs" / "v2eval"))
    q.add_argument("--exp", required=True)
    q.add_argument("--plan", default=str(PLAN_A), help="R2a の保存点の道を読む計画（smoke は plan_r2smoke_a.json）")
    d = sub.add_parser("decompose", help="成功の内訳（記述）。判定の後だけ")
    d.add_argument("--root", default=str(ROOT / "outputs" / "v2eval"))
    d.add_argument("--exps", required=True)
    d.add_argument("--conds", required=True)
    d.add_argument("--after", required=True, help="先に書かれているべき判定の記録（例 outputs/api/r1_judge_R1E.json）")
    a = ap.parse_args(argv)
    return {"check": cmd_check, "check-c": cmd_check_c, "check-d": cmd_check_d, "check-l": cmd_check_l, "check-p": cmd_check_p,
            "check-r": cmd_check_r, "entry": cmd_entry,
            "decompose": cmd_decompose}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
