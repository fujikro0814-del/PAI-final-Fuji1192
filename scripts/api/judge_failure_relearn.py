"""自分の失敗から学び直す評価の判定（事前に決めた比較、手順書 docs/api/r1_protocol.md 7 節）。
自力の立て直し率の比較 2 つ（H2・H-R1a）と、3 個とも箱に入った数の比較 1 つ（H-R1b）を Holm 法（3 つ）でまとめる。
同じ判定を別の書き方で数える scripts/api/judge_failure_relearn_b.py（B）と、このファイル（A）の結果を照らす。

    .venv\\Scripts\\python.exe scripts\\api\\judge_failure_relearn.py check                          # 評価（実験 R1E・R1E_B）
    .venv\\Scripts\\python.exe scripts\\api\\judge_failure_relearn.py check --dry-run-clc --root <outputs\\v2eval>
        # 予行: 以前の評価の記録（実験 CLC）の U0 と N0 だけで道具を確かめる（判定に使わない）
    .venv\\Scripts\\python.exe scripts\\api\\judge_failure_relearn.py check --practice R1SMOKE,R1SMOKE_B --practice-disturb R1DSMOKE,R1DSMOKE_B
        # 練習: 試しの実行（smoke）の記録で A・B の一致を見る（共通のシードだけ。判定に使わない）
    .venv\\Scripts\\python.exe scripts\\api\\judge_failure_relearn.py entry --exp R1SMOKE [--disturb]
        # 試しの実行の確かめ: 入口の点検だけ（本数・無効・チェックポイントの道・HEAD。3 個とも・立て直しの数は出さない）

数え方（scripts/api/natural_recovery.py の trial を使う）: 掴み損ね＝指先が目標から 6 cm 以内で閉じ、どれも 1.5 cm 以上
  上がらなかった（試みごとに最初の 1 回）。自力で立て直した＝同じ試みの中で目標が箱に入った（who "self"）。
  シードごとに (立て直した数 k, 掴み損ねの数 n)。条件の割合は全シードの Σk / Σn（まとめた割合）。
H2（N0 → U0、立て直しデータを足す）・H-R1a（U0 → Ud、自分の失敗から学び直す）: 統計量 = 比べる側の割合 − 基準の割合。
  配置（シード）ごとの組の並べ替え検定: シードごとに 2 条件の (k, n) を確率 1/2 で入れ替える。100,000 回、乱数
  default_rng(20261020)、10,000 回ずつ rng.random((回数, シードの数)) < 0.5 を入れ替えの印にする。
  両側 p = (1 + #{|T*| ≥ |T| − 1e-12}) / (1 + 100,000)。
  差の部分: シードを単位にしたブートストラップ（10,000 回、default_rng(20261021)、rng.integers(0, シードの数, (10,000, シードの数))
  を一度に引く。分母が 0 の回は除く）の 2.5・97.5 百分位。各条件の割合の Wilson 95% 信頼区間（シードの重なりは見ない目安）。
H-R1b（Lf → Ld、LLM の組み直しありで学び直す）: 同じシードの対、McNemar の正確検定（両側、recovla.eval.stats.mcnemar_exact）、
  差の部分は Newcombe の方法 10（recovla.eval.stats.paired_diff_ci、φ の補正つき）、各条件は Wilson。
判定: Holm（3 つ、α 0.05）の補正後の p < 0.05 かつ向きが改善なら「上がった」、それ以外は「確かめられなかった」。
参考（記述、判定に使わない）: 立て直しデータ × LLM の組み直しの 2×2（N0・U0・Ln・Lf）の 3 個とも箱へ。飽和した
  ロジスティック回帰の交互作用 bRP（各組の割合の logit で閉じた形、Wald の部分）と、配置を単位にしたブートストラップの部分
  （10,000 回、default_rng(20261022)、4 条件を同じシードで一緒に引く）。割合の尺度の交互作用 (Lf − Ln) − (U0 − N0) も書く。
有効性: 各条件のシードがちょうど 1 本ずつ（実験ごとに受け持ちのシード）、G_AUDIT.json の met、全試行の HEAD が 1 つ、実験ごとに
  環境（ドライバー・torch・OS のビルド）が 1 つ、条件・モデル・チェックポイントの道・知覚の口・世界 v3_real・開閉の絞り 5・手順の切り替え
  home・外乱が条件と同じ。試行ごとの無効＝audit なし・audit の g1 の違反・g2 の世界の止まりか早い使用・g3 の違反。どちらかの
  PC（実験）で無効な試行が全条件の和で 10 本を超えたら判定しない。10 本以下なら無効な試行も除かずに数え、件数を書く。
  A と B の数え・p・部分が一致しなければ判定しない。
外乱の評価（実験 R1D・R1D_B の Lf・Ld、記述）: 3 個とも・押した・落ちた・無効（3 個とも・無効は B とも照らす）。判定に使わない。
書くもの: outputs/api/r1_judge_<実験>.json と .md（予行は r1_judge_dryrun_CLC、練習は r1_judge_practice_<実験>）。
"""
import argparse
import importlib.util
import json
import math
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from recovla.eval.stats import mcnemar_exact, paired_diff_ci, wilson_interval  # noqa: E402

OUTD = ROOT / "outputs" / "api"
EXPERIMENT = "R1E"
BAND = (171400, 200)
SPLIT = {"R1E": (171400, 100), "R1E_B": (171500, 100)}
# 条件: (--arm の値, モデル)
CONDS = {"N0": ("U0", "Nreal"), "U0": ("U0", "Rreal"), "Ud": ("U0", "Rdag"), "Lf": ("Lf", "Rreal"), "Ld": ("Lf", "Rdag"),
         "Ln": ("Lf", "Nreal")}
FAMILY = {"H2": ("N0", "U0", "self"), "H-R1a": ("U0", "Ud", "self"), "H-R1b": ("Lf", "Ld", "all_three")}   # 基準, 比べる側, 指標
# 参考の 2×2（記述、判定に使わない。r1_protocol.md 6 節）: (立て直しデータ R, LLM の組み直し P) → 条件
GRID = {(0, 0): "N0", (1, 0): "U0", (0, 1): "Ln", (1, 1): "Lf"}
GRID_BOOT_SEED = 20261022             # 2×2 のブートストラップの乱数（判定の部分の 20261021 とは別の列）
ALPHA = 0.05
N_PERM, PERM_SEED, CHUNK = 100_000, 20261020, 10_000
N_BOOT, BOOT_SEED = 10_000, 20261021
MAX_INVALID = 10


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


NR = _load(ROOT / "scripts" / "api" / "natural_recovery.py", "api_natural_recovery_for_r1")
PLAN = _load(ROOT / "scripts" / "api" / "plan_failure_relearn.py", "api_r1_plan_for_judge")
# 条件ごとに記録で照らす引数（入口の点検）: チェックポイントの道（Rreal は記録しない）・知覚の口（--arm U0 は None）
CKPT_OF = {"Rreal": "None", "Nreal": PLAN.NREAL, "Rdag": PLAN.RDAG}
FINDER_OF = {"U0": "None", "Lf": "estimated"}
GRIP, RESET, WORLD = "5", "home", "v3_real"
ENV_KEYS = ("driver", "torch", "torch_cuda", "os_build")
# 外乱の評価（記述だけ、判定に使わない。r1_protocol.md 6 節）
DSPLIT = dict(PLAN.DIST)
DCONDS = tuple(PLAN.DIST_CONDS)


# ---------------------------------------------------------------- 統計（純関数）
def pooled(k, n) -> float:
    return float(np.sum(k)) / float(np.sum(n)) if np.sum(n) > 0 else math.nan


def perm_test(kb, nb, ka, na, n_perm: int = N_PERM, seed: int = PERM_SEED, chunk: int = CHUNK) -> dict:
    """シードごとの組の並べ替え検定（両側）。kb・nb は基準、ka・na は比べる側のシードごとの (立て直した数, 掴み損ねの数)。"""
    kb, nb, ka, na = (np.asarray(x, float) for x in (kb, nb, ka, na))
    t_obs = pooled(ka, na) - pooled(kb, nb)
    if not math.isfinite(t_obs):                         # どちらかに掴み損ねが 1 つも無い（試しの実行など）: 検定しない、p = 1
        return {"stat": t_obs, "p": 1.0, "n_perm": n_perm, "seed": seed, "hits": None}
    rng = np.random.default_rng(seed)
    hits, done = 0, 0
    nsum = float(nb.sum() + na.sum())
    while done < n_perm:
        m = min(chunk, n_perm - done)
        sw = rng.random((m, len(kb))) < 0.5
        k_arm = np.where(sw, kb, ka).sum(axis=1)
        n_arm = np.where(sw, nb, na).sum(axis=1)
        k_base = (kb.sum() + ka.sum()) - k_arm
        n_base = nsum - n_arm
        with np.errstate(divide="ignore", invalid="ignore"):
            t = k_arm / n_arm - k_base / n_base
        hits += int(np.sum(np.abs(t) >= abs(t_obs) - 1e-12))
        done += m
    return {"stat": t_obs, "p": (1 + hits) / (1 + n_perm), "n_perm": n_perm, "seed": seed, "hits": hits}


def boot_ci(kb, nb, ka, na, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """シードを単位にしたブートストラップの差（比べる側 − 基準）の 95% 信頼区間（百分位）。"""
    kb, nb, ka, na = (np.asarray(x, float) for x in (kb, nb, ka, na))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(kb), (n_boot, len(kb)))
    with np.errstate(divide="ignore", invalid="ignore"):
        d = ka[idx].sum(1) / na[idx].sum(1) - kb[idx].sum(1) / nb[idx].sum(1)
    d = d[np.isfinite(d)]
    if not d.size:                                       # 使える回が無い（掴み損ねが無い）: 部分は NaN
        return {"lo": math.nan, "hi": math.nan, "n_boot": n_boot, "n_used": 0, "seed": seed}
    return {"lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)), "n_boot": n_boot, "n_used": int(d.size),
            "seed": seed}


def holm(ps: dict) -> dict:
    """{名前: p} → 補正後の p（(m−i+1)p(i) の累積の最大、上限 1）と逐次の棄却。"""
    order = sorted(ps, key=lambda k: (ps[k], k))
    m, out, run, going = len(order), {}, 0.0, True
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * ps[k]))
        going = going and ps[k] <= ALPHA / (m - i) + 1e-15
        out[k] = {"rank": i + 1, "p": ps[k], "p_holm": run, "rejected": going}
    return {"order": order, "m": m, "by": out}


def wilson(k, n) -> list:
    lo, hi = wilson_interval(int(k), int(n))
    return None if lo is None else [lo, hi]


def self_rate_row(base: dict, arm: dict, seeds: list, **kw) -> dict:
    """base・arm は {シード: (k, n)}。"""
    kb, nb = [base[s][0] for s in seeds], [base[s][1] for s in seeds]
    ka, na = [arm[s][0] for s in seeds], [arm[s][1] for s in seeds]
    pt = perm_test(kb, nb, ka, na, **{k: v for k, v in kw.items() if k in ("n_perm", "seed", "chunk")})
    bc = boot_ci(kb, nb, ka, na, **{k: v for k, v in kw.items() if k in ("n_boot",)})
    return {"metric": "self_recovery", "n_seeds": len(seeds), "base_k": int(sum(kb)), "base_n": int(sum(nb)),
            "arm_k": int(sum(ka)), "arm_n": int(sum(na)), "base_rate": pooled(kb, nb), "arm_rate": pooled(ka, na),
            "base_wilson": wilson(sum(kb), sum(nb)), "arm_wilson": wilson(sum(ka), sum(na)),
            "diff": pt["stat"], "p": pt["p"], "perm": pt, "diff_ci_boot": [bc["lo"], bc["hi"]], "boot": bc}


def success_row(base: dict, arm: dict, seeds: list) -> dict:
    """base・arm は {シード: 3 個とも}。"""
    n11 = sum(1 for s in seeds if base[s] and arm[s])
    b = sum(1 for s in seeds if base[s] and not arm[s])
    c = sum(1 for s in seeds if arm[s] and not base[s])
    n00 = len(seeds) - n11 - b - c
    d, lo, hi = paired_diff_ci(n11, c, b, n00)
    return {"metric": "all_three", "n_seeds": len(seeds), "n11": n11, "b": b, "c": c, "n00": n00,
            "base_success": n11 + b, "arm_success": n11 + c, "base_wilson": wilson(n11 + b, len(seeds)),
            "arm_wilson": wilson(n11 + c, len(seeds)), "diff": d, "diff_ci_newcombe": [lo, hi], "p": mcnemar_exact(b, c)}


def judge(data: dict, seeds: list, family=FAMILY) -> dict:
    rows = {}
    for h, (b, a, metric) in family.items():
        if b not in data or a not in data:
            continue
        if metric == "self":
            rows[h] = dict(self_rate_row(data[b]["self"], data[a]["self"], seeds), base=b, arm=a)
        else:
            rows[h] = dict(success_row(data[b]["success"], data[a]["success"], seeds), base=b, arm=a)
    hl = holm({h: r["p"] for h, r in rows.items()})
    for h, r in rows.items():
        r["p_holm"] = hl["by"][h]["p_holm"]
        r["result"] = "上がった" if (r["p_holm"] < ALPHA and r["diff"] > 0) else "確かめられなかった"
    return {"primary": rows, "holm": {"order": hl["order"], "m": hl["m"], "family": list(family)}}


def _logit(k, n) -> float:
    return math.log(k / (n - k)) if 0 < k < n else math.nan


def grid_coefs(k: dict, n: int) -> dict:
    """2×2 の飽和したロジスティック回帰 logit p = b0 + bR·R + bP·P + bRP·R·P の最尤の係数（各組の割合の logit で閉じた形）。
    k は {(R, P): 3 個とも の数}、n は組あたりのシードの数。0 か n の組があれば係数は NaN。差の交互作用（割合の尺度）も返す。"""
    L = {c: _logit(k[c], n) for c in k}
    if any(v != v for v in L.values()):                  # 0 か n の組が 1 つでもあれば、logit の係数は全部 NaN
        L = {c: math.nan for c in k}
    p = {c: k[c] / n for c in k}
    return {"b0": L[(0, 0)], "bR": L[(1, 0)] - L[(0, 0)], "bP": L[(0, 1)] - L[(0, 0)],
            "bRP": L[(1, 1)] - L[(1, 0)] - L[(0, 1)] + L[(0, 0)],
            "diff_R_noP": p[(1, 0)] - p[(0, 0)], "diff_R_withP": p[(1, 1)] - p[(0, 1)],
            "interaction_pp": (p[(1, 1)] - p[(0, 1)]) - (p[(1, 0)] - p[(0, 0)])}


def grid_2x2(data: dict, seeds: list, n_boot: int = N_BOOT, seed: int = GRID_BOOT_SEED) -> dict:
    """立て直しデータ × LLM の組み直しの 2×2（N0・U0・Ln・Lf）の、3 個とも箱に入った数の記述。交互作用の Wald の部分
    （各組の数から）と、配置（シード）を単位にしたブートストラップの部分（4 条件を同じシードで一緒に引く。係数が NaN の回は除く）。"""
    if not all(c in data for c in GRID.values()):
        return {"status": "missing", "missing": [c for c in GRID.values() if c not in data]}
    S = len(seeds)
    Y = np.array([[bool(data[GRID[g]]["success"][s]) for g in GRID] for s in seeds], float)   # (シード, 4)
    k = {g: int(Y[:, i].sum()) for i, g in enumerate(GRID)}
    est = grid_coefs(k, S)
    se = (math.sqrt(sum(1.0 / k[g] + 1.0 / (S - k[g]) for g in GRID)) if all(0 < k[g] < S for g in GRID) else math.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, S, (n_boot, S))
    ks = Y[idx].sum(axis=1)                                                                      # (回, 4)
    bs = [grid_coefs({g: int(ks[b, i]) for i, g in enumerate(GRID)}, S) for b in range(n_boot)]
    out = {"status": "complete", "n_seeds": S, "cells": {GRID[g]: k[g] for g in GRID}, "estimate": est,
           "bRP_wald_ci": [est["bRP"] - 1.959963984540054 * se, est["bRP"] + 1.959963984540054 * se] if se == se else None,
           "boot": {"n_boot": n_boot, "seed": seed}}
    for key in ("bR", "bP", "bRP", "interaction_pp", "diff_R_noP", "diff_R_withP"):
        v = np.array([b[key] for b in bs], float)
        v = v[np.isfinite(v)]
        out["boot"][key] = {"ci": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v.size else None,
                            "n_used": int(v.size)}
    return out


# ---------------------------------------------------------------- 記録の読み
def invalid_reason(m: dict):
    au = m.get("audit")
    if not au:
        return "no_audit"
    if (au.get("g1") or {}).get("violations"):
        return "g1"
    g2 = au.get("g2") or {}
    if g2.get("world_stops") or g2.get("early_use"):
        return "g2"
    if (au.get("g3") or {}).get("total_violations"):
        return "g3"
    return None


def read_condition(dirs) -> dict:
    """dirs＝[(実験, フォルダ)]。シードごとの自力で立て直した (k, n)・3 個とも・無効の理由、実験ごとのシード。"""
    keys = ("models", "arms", "heads", "worlds", "grips", "resets", "finders", "ckpts", "disturbs")
    out = {"self": {}, "success": {}, "invalid": {}, "dup": [], "by_exp": {}, "g_audit": [], "envs": {}, "disturb": {}}
    for k in keys:
        out[k] = set()
    for exp, d in dirs:
        seeds, envs = [], set()
        for js in sorted(pathlib.Path(d).glob("run_[0-9][0-9][0-9][0-9].json")):
            if not js.with_suffix(".npz").is_file():
                continue
            m = json.loads(js.read_text(encoding="utf-8"))
            s = int(m["seed"])
            if s in out["success"]:
                out["dup"].append(s)
            r = NR.trial(js)
            miss = [e for e in r["episodes"] if e["kind"] == "miss"]
            out["self"][s] = (sum(1 for e in miss if e["who"] == "self"), len(miss))
            out["success"][s] = bool(m.get("all_three_in_box"))
            out["invalid"][s] = invalid_reason(m)
            u4, w = m.get("u4") or {}, m.get("world") or {}
            out["models"].add(str(m.get("model")))
            out["arms"].add(str(u4.get("arm")))
            out["heads"].add(str((m.get("env") or {}).get("git_head")))
            out["worlds"].add(str(w.get("world")))
            out["grips"].add(str((w.get("grip_debounce") or {}).get("open_steps")))
            out["resets"].add(str(u4.get("step_reset")))
            out["finders"].add(str(u4.get("finder")))
            out["ckpts"].add(str(u4.get("model_ckpt")))
            out["disturbs"].add(str(u4.get("disturb") or "none"))
            dist = m.get("disturb") or {}
            out["disturb"][s] = {"pushed": bool(dist.get("pushed")), "dropped": bool(dist.get("dropped"))}
            envs.add(json.dumps({k: (m.get("env") or {}).get(k) for k in ENV_KEYS}, sort_keys=True))
            seeds.append(s)
        ga = pathlib.Path(d) / "G_AUDIT.json"
        out["g_audit"].append(json.loads(ga.read_text(encoding="utf-8")).get("met") if ga.is_file() else None)
        out["by_exp"][exp] = sorted(seeds)
        out["envs"][exp] = sorted(envs)
    for k in keys:
        out[k] = sorted(out[k])
    return out


def invalid_by_exp(data: dict, exps) -> dict:
    """{実験: 無効な試行の数（その PC の全条件の和）}。"""
    return {e: sum(1 for d in data.values() for s in d["by_exp"].get(e, []) if d["invalid"].get(s)) for e in exps}


def entry_check(data: dict, seeds: list, split: dict, practice: bool, disturb: str = "none") -> list:
    """入口の点検。practice（練習・試しの実行）は評価のシードの範囲・受け持ちのシード・HEAD の照合を外し、ほかは同じ。
    無効（audit の g1〜g3 の違反・世界の止まり・早い使用・audit なし）は、PC（実験）ごとに全条件の和で MAX_INVALID を超えたら問題。"""
    probs = []
    for c, d in data.items():
        arm, model = CONDS[c]
        if d["dup"]:
            probs.append(f"{c}: 同じ種が 2 本以上 {d['dup'][:5]}")
        if not practice:
            if set(d["success"]) != set(seeds):
                probs.append(f"{c}: 種が帯と合わない（{len(d['success'])} 本）")
            for exp, (s0, n) in split.items():
                if d["by_exp"].get(exp) != list(range(s0, s0 + n)):
                    probs.append(f"{c}: 実験 {exp} の種が受け持ち {s0}〜{s0 + n - 1} と合わない")
        for exp in split:
            if d["by_exp"].get(exp) and len(d["envs"].get(exp, [])) != 1:
                probs.append(f"{c}: 実験 {exp} の環境（ドライバー・torch・OS のビルド）が 1 つでない {d['envs'].get(exp)}")
        if not all(x is True for x in d["g_audit"]):
            probs.append(f"{c}: G_AUDIT.json の met が真でない {d['g_audit']}")
        for key, want, what in (("arms", [arm], "腕"), ("models", [model], "方策"), ("ckpts", [CKPT_OF[model]], "保存点の道"),
                                ("finders", [FINDER_OF[arm]], "知覚の口"), ("worlds", [WORLD], "世界"),
                                ("grips", [GRIP], "開閉の絞り"), ("resets", [RESET], "手順の切り替えの直し"),
                                ("disturbs", [disturb], "外乱")):
            if d[key] != want:
                probs.append(f"{c}: {what}が {want} だけでない {d[key][:3]}")
    for exp, n in invalid_by_exp(data, split).items():
        if n > MAX_INVALID:
            probs.append(f"実験 {exp}（1 台）で無効な試行が {n} 本（全条件の和、{MAX_INVALID} を超える）")
    heads = sorted({h for d in data.values() for h in d["heads"]})
    if not practice and len(heads) != 1:
        probs.append(f"試行の HEAD が 1 つでない: {heads}")
    return probs


def summary_counts(data: dict, seeds: list) -> dict:
    """A と B で照らす数え（条件ごと）。"""
    return {c: {"self_k": sum(d["self"][s][0] for s in seeds), "miss_n": sum(d["self"][s][1] for s in seeds),
                "all_three": sum(d["success"][s] for s in seeds),
                "invalid": sum(1 for s in seeds if d["invalid"].get(s))} for c, d in data.items()}


def disturb_summary(root: pathlib.Path, split: dict = None, practice: bool = False) -> dict:
    """外乱の評価（R1D・R1D_B の Lf・Ld、記述だけ）。数え（3 個とも・無効）は B とも照らす。押した・落ちたは A だけ。"""
    split = dict(DSPLIT if split is None else split)
    dirs = {c: [(e, root / e / c) for e in split] for c in DCONDS}
    if not all(d.is_dir() for c in DCONDS for _, d in dirs[c]):
        return {"status": "missing", "missing": [str(d) for c in DCONDS for _, d in dirs[c] if not d.is_dir()]}
    data = {c: read_condition(dirs[c]) for c in DCONDS}
    if practice:
        seeds = sorted(set.intersection(*(set(d["success"]) for d in data.values())))
    else:
        seeds = sorted(s for s0, n in split.values() for s in range(s0, s0 + n))
    probs = entry_check(data, seeds, split, practice, disturb="push")
    if probs:
        return {"status": "incomplete", "problems": probs}
    counts = {c: {"n": len(seeds), "all_three": sum(d["success"][s] for s in seeds),
                  "invalid": sum(1 for s in seeds if d["invalid"].get(s))} for c, d in data.items()}
    events = {c: {"pushed": sum(d["disturb"][s]["pushed"] for s in seeds), "dropped": sum(d["disturb"][s]["dropped"] for s in seeds)}
              for c, d in data.items()}
    return {"status": "complete", "experiments": list(split), "seeds": seeds, "counts": counts, "events": events,
            "heads": sorted({h for d in data.values() for h in d["heads"]})}


def compare(a, b, path="") -> list:
    """件数・真偽・文字列は完全一致、実数は相対 1e-9（NaN どうしは一致）。"""
    if isinstance(a, dict) and isinstance(b, dict):
        out = [f"{path}: キーが違う {sorted(set(a) ^ set(b))}"] if set(a) != set(b) else []
        return out + [x for k in sorted(set(a) & set(b)) for x in compare(a[k], b[k], f"{path}.{k}")]
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [f"{path}: 長さが違う"]
        return [x for i, (u, v) in enumerate(zip(a, b)) for x in compare(u, v, f"{path}[{i}]")]
    if isinstance(a, float) or isinstance(b, float):
        if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
            return [] if a == b else [f"{path}: {a!r} / {b!r}"]
        if math.isnan(float(a)) and math.isnan(float(b)):
            return []
        return [] if math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-15) else [f"{path}: {a!r} / {b!r}"]
    return [] if (a == b and type(a) is type(b)) else [f"{path}: {a!r} / {b!r}"]


def key_numbers(j: dict) -> dict:
    """A・B で照らす判定の数（行ごと）。"""
    keys = ("n_seeds", "base_k", "base_n", "arm_k", "arm_n", "n11", "b", "c", "n00", "base_success", "arm_success", "diff", "p",
            "p_holm", "diff_ci_boot", "diff_ci_newcombe", "result")
    return {h: {k: r[k] for k in keys if k in r} for h, r in j["primary"].items()}


def sheet(res: dict) -> str:
    L = [f"# R1 の評価の判定（{'予行・判定に使わない' if res['practice'] else '本番'}）", "",
         f"- 実験 {'・'.join(res['experiments'])}、状態 {res['status']}、二重集計 {'一致' if res.get('agree') else '—'}",
         "- 手順書: docs/api/r1_protocol.md 7 節（Holm、族は 3 つ、α 0.05）", ""]
    if res["status"] != "complete":
        return "\n".join(L + ["## 足りないもの・問題", ""] + [f"- {p}" for p in res["problems"]]) + "\n"
    L += ["| 項目 | 比べる組 | 指標 | 基準 | 腕 | 差（腕 − 基準） | 差の 95% 区間 | p | Holm 後の p | 結果 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for h, r in res["a"]["primary"].items():
        if r["metric"] == "self_recovery":
            b, a = f"{r['base_k']}/{r['base_n']}（{100 * r['base_rate']:.1f}%）", f"{r['arm_k']}/{r['arm_n']}（{100 * r['arm_rate']:.1f}%）"
            ci = r["diff_ci_boot"]
        else:
            b, a = f"{r['base_success']}/{r['n_seeds']}", f"{r['arm_success']}/{r['n_seeds']}（b {r['b']}・c {r['c']}）"
            ci = r["diff_ci_newcombe"]
        L.append(f"| {h} | {r['arm']} 対 {r['base']} | {r['metric']} | {b} | {a} | {100 * r['diff']:+.1f} pt | "
                 f"[{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}] pt | {r['p']:.4g} | {r['p_holm']:.4g} | {r['result']} |")
    L += ["", "数え（A）: " + "; ".join(f"{c} 立て直し {v['self_k']}/{v['miss_n']}・3 個とも {v['all_three']}・無効 {v['invalid']}"
                                     for c, v in res["counts"].items()), ""]
    g = res.get("grid_2x2") or {}
    if g.get("status") == "complete":
        e, b = g["estimate"], g["boot"]

        def ci(key, scale=1.0):
            c = b[key]["ci"]
            return "—" if c is None else f"[{scale * c[0]:+.2f}, {scale * c[1]:+.2f}]"
        L += ["## 副次（記述、判定に使わない）: 回復デモ × 再計画の 2×2（3 個とも収納）", "",
              "組: " + "、".join(f"{c} {v}/{g['n_seeds']}" for c, v in g["cells"].items()), "",
              f"- ロジスティック回帰（飽和）の交互作用 bRP = {e['bRP']:+.3f}（Wald {g['bRP_wald_ci']}、種のブートストラップ {ci('bRP')}）",
              f"- 主効果 bR = {e['bR']:+.3f} {ci('bR')}、bP = {e['bP']:+.3f} {ci('bP')}（logit）",
              f"- 割合の尺度の交互作用（(Lf − Ln) − (U0 − N0)）= {100 * e['interaction_pp']:+.1f} pt {ci('interaction_pp', 100)} pt", ""]
    ds = res.get("disturb") or {}
    L += ["## 副次（記述、判定に使わない）: 外乱の評価（運ぶ途中で横から押して落とす、Lf・Ld）", ""]
    if ds.get("status") == "complete":
        L += [f"- 実験 {'・'.join(ds['experiments'])}、{len(ds['seeds'])} 配置"]
        L += [f"- {c}: 3 個とも {v['all_three']}/{v['n']}、押した {ds['events'][c]['pushed']}・落ちた {ds['events'][c]['dropped']}・"
              f"無効 {v['invalid']}" for c, v in ds["counts"].items()]
    else:
        L += [f"- {ds.get('status', '—')}: " + "; ".join((ds.get("problems") or ds.get("missing") or [])[:5])]
    return "\n".join(L) + "\n"


def cmd_check(a) -> int:
    root = pathlib.Path(a.root)
    dsplit = None
    if a.dry_run_clc:
        practice = True
        split = {"CLC": (171200, 100), "CLC_B": (171300, 100)}
        conds, family, stem = ("N0", "U0"), {"H2": FAMILY["H2"]}, "r1_judge_dryrun_CLC"
    elif a.practice:
        # 練習（試しの実行の記録。判定に使わない）: 実験の並び（例 R1SMOKE,R1SMOKE_B）の、どの実験にもある条件を読み、
        # 全部の条件にあるシードだけで数える。外乱は --practice-disturb の実験の並び（例 R1DSMOKE,R1DSMOKE_B）
        practice = True
        exps = [e for e in a.practice.split(",") if e]
        split = {e: None for e in exps}
        conds = tuple(c for c in CONDS if all((root / e / c).is_dir() for e in exps))
        family, stem = FAMILY, f"r1_judge_practice_{exps[0]}"
        if a.practice_disturb:
            dsplit = {e: None for e in a.practice_disturb.split(",") if e}
    else:
        practice = False
        split, conds, family, stem = dict(SPLIT), tuple(CONDS), FAMILY, f"r1_judge_{EXPERIMENT}"
    data, problems = {}, []
    for c in conds:
        dirs = [(e, root / e / c) for e in split]
        miss = [str(d) for _, d in dirs if not d.is_dir()]
        if miss:
            problems += [f"{c}: 記録が無い {m}" for m in miss]
            continue
        data[c] = read_condition(dirs)
    if a.practice:
        seeds = sorted(set.intersection(*(set(d["success"]) for d in data.values()))) if data else []
        if not seeds:
            problems.append("練習: 共通の種が無い")
    else:
        seeds = sorted(s for s0, n in split.values() for s in range(s0, s0 + n))
    if not problems:
        problems += entry_check(data, seeds, split, practice)
        if a.dry_run_clc:   # 予行の CLC の記録は条件・モデル・チェックポイントの名前の付け方が違うので、その照合だけ外す
            problems = [p for p in problems if not any(w in p for w in ("腕が", "方策が", "保存点の道", "知覚の口"))]
    res = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "experiments": list(split), "practice": practice,
           "conditions": list(conds), "n_seeds": len(seeds), "status": "complete" if not problems else "incomplete",
           "problems": problems}
    code = 3
    if not problems:
        B = _load(ROOT / "scripts" / "api" / "judge_failure_relearn_b.py", "api_r1_judge_b")
        ja = judge(data, seeds, family)
        jb = B.judge_b(root, list(split), conds, family, seeds)
        ga = grid_2x2(data, seeds)
        mine = {"counts": summary_counts(data, seeds), "rows": key_numbers(ja)}
        if ga["status"] == "complete":                       # 2×2 の数と係数も照らす（部分は A だけ）
            mine["grid"] = {"cells": ga["cells"], "estimate": ga["estimate"]}
        diff = compare(mine, jb)
        ds = None
        if not a.dry_run_clc and (not practice or dsplit):
            ds = disturb_summary(root, dsplit, practice)
            if ds["status"] == "complete":                    # 外乱の数え（3 個とも・無効）も B と照らす
                diff += compare(ds["counts"], B.disturb_counts_b(root, ds["experiments"], DCONDS, ds["seeds"]), ".disturb")
                main_heads = sorted({h for d in data.values() for h in d["heads"]})
                if not practice and ds["heads"] != main_heads:
                    ds = {"status": "incomplete", "problems": [f"外乱の評価の HEAD {ds['heads']} が本体 {main_heads} と違う"]}
        res.update(a=ja, grid_2x2=ga, counts=summary_counts(data, seeds), invalid_by_exp=invalid_by_exp(data, split),
                   disturb=ds, agree=not diff, diffs=diff[:50])
        code = 0 if not diff else 1
        if diff:
            res["status"] = "disagree"
    OUTD.mkdir(parents=True, exist_ok=True)
    (OUTD / f"{stem}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    txt = sheet(res)
    (OUTD / f"{stem}.md").write_text(txt, encoding="utf-8")
    print(txt)
    if res.get("diffs"):
        print("[r1-judge] A と B が一致しない: " + "; ".join(res["diffs"][:10]), file=sys.stderr)
    return code


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
    p.add_argument("--dry-run-clc", action="store_true", help="CLC の U0 対 N0 で H2 の道具を確かめる（判定に使わない）")
    p.add_argument("--practice", default="", help="練習の実験の並び（例 R1SMOKE,R1SMOKE_B）。判定に使わない")
    p.add_argument("--practice-disturb", default="", help="練習の外乱の実験の並び（例 R1DSMOKE,R1DSMOKE_B）")
    q = sub.add_parser("entry", help="入口の点検だけ（smoke の確かめ。数え・成功は出さない）")
    q.add_argument("--root", default=str(ROOT / "outputs" / "v2eval"))
    q.add_argument("--exp", required=True)
    q.add_argument("--disturb", action="store_true", help="外乱の実験（Lf・Ld、--disturb push）")
    a = ap.parse_args(argv)
    return cmd_entry(a) if a.cmd == "entry" else cmd_check(a)


def cmd_entry(a) -> int:
    """試しの実行の確かめ: 実験の中の条件ごとに、本数・無効の数と、入口の点検の問題だけを出す（3 個とも・立て直しは出さない）。"""
    root = pathlib.Path(a.root)
    conds = tuple(c for c in (DCONDS if a.disturb else CONDS) if (root / a.exp / c).is_dir())
    data = {c: read_condition([(a.exp, root / a.exp / c)]) for c in conds}
    probs = entry_check(data, [], {a.exp: None}, True, disturb="push" if a.disturb else "none")
    missing = [c for c in (DCONDS if a.disturb else CONDS) if c not in conds]
    for c, d in data.items():
        print(f"[entry] {a.exp}/{c}: 試行 {len(d['success'])}、無効 {sum(1 for v in d['invalid'].values() if v)}、"
              f"保存点 {d['ckpts']}、HEAD {d['heads']}")
    for m in missing:
        print(f"[entry] {a.exp}/{m}: 記録が無い")
    for p in probs:
        print(f"[entry] 問題: {p}")
    ok = not probs and not missing and all(d["success"] for d in data.values())
    print(f"[entry] {a.exp}: {'OK' if ok else 'NG'}")
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
