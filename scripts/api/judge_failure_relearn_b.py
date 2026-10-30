"""自分の失敗から学び直す評価の判定を、scripts/api/judge_failure_relearn.py（A）とは別の書き方で数え直す（B）。A と同じ数を返す。

数え: 記録の npz（グリッパの閉・立方体の位置・箱の中・手順・指先）と json（試みの始めの時刻）から、掴み損ねと自力で立て直した
  かを数え直す（natural_recovery.py を読まない）。3 個とも は json の final_in_box の 3 色。無効は json の audit。
統計: 乱数の引き方は手順書（docs/api/r1_protocol.md）のとおり（A と同じ列）。並べ替えは入れ替えの印を 0・1 の行列にして内積で和をとる。ブートストラップは
  引いた添字の回数（bincount）で重みをつける。McNemar は二項の確率を math.comb で足す。Newcombe の方法 10 は φ の補正つきで
  自分で書く。
"""
import json
import math
import pathlib

import numpy as np

LIFT, NEAR, SETTLE = 0.015, 0.06, 2.0
PERM_N, PERM_SEED, PERM_CHUNK = 100_000, 20261020, 10_000
BOOT_N, BOOT_SEED = 10_000, 20261021
Z = 1.959963984540054
ALPHA = 0.05
COL = ("red", "green", "blue")


def episodes_b(js: pathlib.Path):
    """[(掴み損ねか, 自力で立て直したか)]（試みごとに最初の失敗だけ）。"""
    m = json.loads(js.read_text(encoding="utf-8"))
    z = np.load(js.with_suffix(".npz"))
    t, cl, P, IB, ST, TIP = z["sim_time"], z["gripper_closed"].astype(bool), z["cube_pos"], z["cube_in_box"], z["plan_step"], z["fingertip"]
    steps = {int(s["step"]): s for s in m["steps"]}
    out, used = [], set()
    i, n = 0, len(t)
    while i < n:
        if not cl[i]:
            i += 1
            continue
        a = i
        while i < n and cl[i]:
            i += 1
        b = i                                         # [a, b) は閉
        if b >= n:
            break
        k = int(ST[a])
        if k not in steps:
            continue
        st = steps[k]
        c = COL.index(st["color"])
        if IB[a, c]:
            continue
        begins = sorted(float(x["t_begin"]) for x in st["attempts"])
        before = [x for x in begins if x <= float(t[a]) + 1e-9]
        if not before:
            continue
        tb = before[-1]
        if (k, tb) in used:
            continue
        up = P[a:b, :, 2].max(axis=0) - P[a, :, 2]
        if up[c] >= LIFT:
            sel = (t >= t[b - 1]) & (t <= t[b - 1] + SETTLE)
            if IB[b - 1, c] or IB[sel, c].any():
                continue
            used.add((k, tb))
            continue                                  # 落下・置き損ね（掴み損ねではない）
        if (up >= LIFT).any():
            used.add((k, tb))
            continue                                  # 別の立方体
        if math.dist(TIP[a].tolist(), P[a, c].tolist()) >= NEAR:
            continue
        used.add((k, tb))
        after = [x for x in begins if x > tb + 1e-9]
        t_att_end = after[0] if after else float(st.get("t_end", m["t_end"]))
        hit = np.flatnonzero(IB[:, c] & (t > t[a]))
        out.append((True, bool(hit.size) and float(t[hit[0]]) <= t_att_end + 0.5))
    return out


def invalid_b(m: dict) -> bool:
    au = m.get("audit") or {}
    if not au:
        return True
    g2 = au.get("g2") or {}
    return bool((au.get("g1") or {}).get("violations") or g2.get("world_stops") or g2.get("early_use")
                or (au.get("g3") or {}).get("total_violations"))


def read_b(root: pathlib.Path, exps, cond: str) -> dict:
    out = {}
    for e in exps:
        for js in sorted((root / e / cond).glob("run_[0-9][0-9][0-9][0-9].json")):
            if not js.with_suffix(".npz").is_file():
                continue
            m = json.loads(js.read_text(encoding="utf-8"))
            ep = episodes_b(js)
            fb = m.get("final_in_box") or {}
            out[int(m["seed"])] = {"k": sum(1 for _, s in ep if s), "n": len(ep), "ok": all(bool(fb.get(c)) for c in COL),
                                   "bad": invalid_b(m)}
    return out


def disturb_counts_b(root, exps, conds, seeds) -> dict:
    """外乱の評価（記述）の数え: 条件ごとの本数・3 個とも・無効。"""
    root = pathlib.Path(root)
    out = {}
    for c in conds:
        D = read_b(root, exps, c)
        out[c] = {"n": len(seeds), "all_three": sum(D[s]["ok"] for s in seeds), "invalid": sum(D[s]["bad"] for s in seeds)}
    return out


def perm_b(kb, nb, ka, na) -> tuple:
    kb, nb, ka, na = (np.asarray(x, float) for x in (kb, nb, ka, na))
    with np.errstate(divide="ignore", invalid="ignore"):
        obs = ka.sum() / na.sum() - kb.sum() / nb.sum() if na.sum() and nb.sum() else math.nan
    if obs != obs:
        return float(obs), 1.0                                 # 掴み損ねが片方に無い: 検定しない
    rng = np.random.default_rng(PERM_SEED)
    cnt, left = 0, PERM_N
    dk, dn = kb - ka, nb - na
    while left:
        m = min(PERM_CHUNK, left)
        s = (rng.random((m, len(kb))) < 0.5).astype(float)
        ka2, na2 = ka.sum() + s @ dk, na.sum() + s @ dn         # 入れ替えたシードだけ比べる側に基準の値が入る
        kb2, nb2 = kb.sum() - s @ dk, nb.sum() - s @ dn
        with np.errstate(divide="ignore", invalid="ignore"):
            st = ka2 / na2 - kb2 / nb2
        cnt += int((np.abs(st) >= abs(obs) - 1e-12).sum())
        left -= m
    return float(obs), (cnt + 1) / (PERM_N + 1)


def boot_b(kb, nb, ka, na) -> list:
    kb, nb, ka, na = (np.asarray(x, float) for x in (kb, nb, ka, na))
    S = len(kb)
    idx = np.random.default_rng(BOOT_SEED).integers(0, S, (BOOT_N, S))
    vals = []
    for row in idx:
        w = np.bincount(row, minlength=S)
        a_n, b_n = w @ na, w @ nb
        if a_n == 0 or b_n == 0:
            continue
        vals.append((w @ ka) / a_n - (w @ kb) / b_n)
    if not vals:
        return [math.nan, math.nan]
    v = np.array(vals)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def binom_two_sided(x: int, n: int) -> float:
    if n == 0:
        return 1.0
    pr = [math.comb(n, i) / 2 ** n for i in range(n + 1)]
    return float(min(1.0, sum(q for q in pr if q <= pr[x] * (1 + 1e-7))))


def wilson_b(k, n):
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def newcombe_b(n11, a_only, b_only, n00) -> tuple:
    """比べる側 − 基準の差と 95% 信頼区間（Newcombe の方法 10、φ の補正つき）。"""
    n = n11 + a_only + b_only + n00
    pa, pb = (n11 + a_only) / n, (n11 + b_only) / n
    la, ua = wilson_b(n11 + a_only, n)
    lb, ub = wilson_b(n11 + b_only, n)
    den = (n11 + a_only) * (b_only + n00) * (n11 + b_only) * (a_only + n00)
    A = n11 * n00 - a_only * b_only
    A = A - n / 2 if A > n / 2 else (0.0 if A >= 0 else A)
    phi = 0.0 if den == 0 else A / math.sqrt(den)
    d = pa - pb
    lo = d - math.sqrt(max((pa - la) ** 2 - 2 * phi * (pa - la) * (ub - pb) + (ub - pb) ** 2, 0.0))
    hi = d + math.sqrt(max((ua - pa) ** 2 - 2 * phi * (ua - pa) * (pb - lb) + (pb - lb) ** 2, 0.0))
    return d, [lo, hi]


def judge_b(root, exps, conds, family, seeds) -> dict:
    root = pathlib.Path(root)
    D = {c: read_b(root, exps, c) for c in conds}
    counts = {c: {"self_k": sum(D[c][s]["k"] for s in seeds), "miss_n": sum(D[c][s]["n"] for s in seeds),
                  "all_three": sum(D[c][s]["ok"] for s in seeds), "invalid": sum(D[c][s]["bad"] for s in seeds)} for c in conds}
    rows = {}
    for h, (b, a, metric) in family.items():
        if b not in D or a not in D:
            continue
        B, A = D[b], D[a]
        if metric == "self":
            kb, nb = [B[s]["k"] for s in seeds], [B[s]["n"] for s in seeds]
            ka, na = [A[s]["k"] for s in seeds], [A[s]["n"] for s in seeds]
            d, p = perm_b(kb, nb, ka, na)
            rows[h] = {"n_seeds": len(seeds), "base_k": sum(kb), "base_n": sum(nb), "arm_k": sum(ka), "arm_n": sum(na),
                       "diff": d, "p": p, "diff_ci_boot": boot_b(kb, nb, ka, na)}
        else:
            n11 = sum(1 for s in seeds if B[s]["ok"] and A[s]["ok"])
            bo = sum(1 for s in seeds if B[s]["ok"] and not A[s]["ok"])
            co = sum(1 for s in seeds if A[s]["ok"] and not B[s]["ok"])
            n00 = len(seeds) - n11 - bo - co
            d, ci = newcombe_b(n11, co, bo, n00)
            rows[h] = {"n_seeds": len(seeds), "n11": n11, "b": bo, "c": co, "n00": n00, "base_success": n11 + bo,
                       "arm_success": n11 + co, "diff": d, "p": binom_two_sided(min(bo, co), bo + co), "diff_ci_newcombe": ci}
    # Holm
    order = sorted(rows, key=lambda k: (rows[k]["p"], k))
    run = 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * rows[k]["p"]))
        rows[k]["p_holm"] = run
        rows[k]["result"] = "上がった" if (run < ALPHA and rows[k]["diff"] > 0) else "確かめられなかった"
    out = {"counts": counts, "rows": rows}
    names = {"N0": (0, 0), "U0": (1, 0), "Ln": (0, 1), "Lf": (1, 1)}
    if all(c in D for c in names):                     # 参考の 2×2: 組の数とオッズから係数（A は logit の差で書く）
        S = len(seeds)
        k = {c: sum(1 for s in seeds if D[c][s]["ok"]) for c in names}
        if all(0 < v < S for v in k.values()):
            o = {c: k[c] / (S - k[c]) for c in names}
            est = {"b0": math.log(o["N0"]), "bR": math.log(o["U0"] / o["N0"]), "bP": math.log(o["Ln"] / o["N0"]),
                   "bRP": math.log(o["Lf"] * o["N0"] / (o["U0"] * o["Ln"]))}
        else:
            est = {"b0": math.nan, "bR": math.nan, "bP": math.nan, "bRP": math.nan}
        pr = {c: k[c] / S for c in names}
        est.update(diff_R_noP=pr["U0"] - pr["N0"], diff_R_withP=pr["Lf"] - pr["Ln"],
                   interaction_pp=pr["Lf"] - pr["Ln"] - pr["U0"] + pr["N0"])
        out["grid"] = {"cells": {c: k[c] for c in ("N0", "U0", "Ln", "Lf")}, "estimate": est}
    return out
