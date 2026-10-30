"""results/trials.csv（1 行 = 1 試行）から、説明資料の表 1・表 2 と README の数を、results/rtc_trials.csv から付録 A（RTC を足した評価）の
数を数え直す。GPU は要らず、数秒で終わる。

    python scripts\\recount_results.py

使うのは標準ライブラリと numpy だけ。検定は評価の判定と同じ作り:
- 3 個とも箱へ: 同じ初期状態の組の McNemar 正確検定（両側）、差の信頼区間は Newcombe の方法 10、割合は Wilson 法
- 自力の立て直し率: 初期状態ごとの並べ替え検定（10 万回、乱数 20261020）とブートストラップ（1 万回、乱数 20261021）
- つなぎ目の段差（付録 A）: 同じ初期状態の組の Wilcoxon の符号順位検定（両側、差 0 の組は除く、正規近似）
- 事前に決めた 3 つの比較を Holm 法で補正（α = 0.05）。付録 A は 3 つの設定ごとに、成功と段差をそれぞれ Holm 法で補正
列の意味: miss＝掴み損ねの数、self_recovered＝そのうち方策が同じ試みの中で箱に入れた数、
miss_regrip・self_recovered_regrip＝そのうち、持ち上げた後に指を開いてすぐ閉じ直した回（感度分析で除く）。
rtc_trials.csv の seam_median_mm_s＝行動列のつなぎ目の段差の、試行の中央値（mm/s。つなぎ目が無い試行は空）。
"""
import csv
import math
import pathlib

import numpy as np

CSV = pathlib.Path(__file__).resolve().parents[1] / "results" / "trials.csv"
Z = 1.959963984540054
NAMES = {("success_only", 0): "成功データだけ", ("plus_recovery", 0): "＋立て直しデータ",
         ("plus_failure_relearn", 0): "＋自分の失敗から学び直す", ("plus_failure_relearn", 1): "＋LLM の組み直し",
         ("plus_place_goal", 1): "＋配置位置による目標条件付け"}
RTC_CSV = CSV.with_name("rtc_trials.csv")
RTC_NAMES = {"rtc_horizon44_exp": "44 ステップ先まで寄せる（原典の設定）", "rtc_horizon10_linear": "10 ステップ先まで寄せる（LeRobot の既定）",
             "rtc_prefix_only": "推論中の 4 ステップだけそろえる"}


def load():
    data = {}
    with open(CSV, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            key = (r["comparison_set"], r["condition"], int(r["llm"]))
            data.setdefault(key, {})[int(r["seed"])] = {k: int(r[k]) for k in
                                                        ("all_three_in_box", "miss", "self_recovered", "miss_regrip",
                                                         "self_recovered_regrip")}
    return data


def arm(data, sets, cond, llm):
    """いくつかの評価の組をつないだ {初期状態の番号: 行}。"""
    out = {}
    for s in sets:
        out.update(data[(s, cond, llm)])
    return out


# ---------------------------------------------------------------- 統計
def wilson(k, n):
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def mcnemar_exact(b, c):
    """両側の正確な p（食い違った組の中で B(n, 0.5)）。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def newcombe(n11, n10, n01, n00):
    """組にした 2 つの割合の差（腕 − 基準）と 95% 信頼区間（Newcombe の方法 10、φ の補正つき）。"""
    n = n11 + n10 + n01 + n00
    p1, p2 = (n11 + n10) / n, (n11 + n01) / n
    l1, u1 = wilson(n11 + n10, n)
    l2, u2 = wilson(n11 + n01, n)
    den = (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
    a = n11 * n00 - n10 * n01
    a = a - n / 2 if a > n / 2 else (0.0 if a >= 0 else a)
    phi = a / math.sqrt(den) if den else 0.0
    d = p1 - p2
    dl = (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2
    du = (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2
    return d, d - math.sqrt(max(dl, 0.0)), d + math.sqrt(max(du, 0.0))


def success_cmp(base, arm_, seeds):
    n11 = sum(base[s]["all_three_in_box"] and arm_[s]["all_three_in_box"] for s in seeds)
    b = sum(base[s]["all_three_in_box"] and not arm_[s]["all_three_in_box"] for s in seeds)
    c = sum(arm_[s]["all_three_in_box"] and not base[s]["all_three_in_box"] for s in seeds)
    d, lo, hi = newcombe(n11, c, b, len(seeds) - n11 - b - c)
    return {"base": f"{n11 + b}", "arm": f"{n11 + c}", "diff": d, "lo": lo, "hi": hi, "p": mcnemar_exact(b, c),
            "note": f"片方だけ成功 {b} と {c}"}


def kn(rows, seeds, drop_regrip):
    k = np.array([rows[s]["self_recovered"] - (rows[s]["self_recovered_regrip"] if drop_regrip else 0) for s in seeds], float)
    n = np.array([rows[s]["miss"] - (rows[s]["miss_regrip"] if drop_regrip else 0) for s in seeds], float)
    return k, n


def self_cmp(base, arm_, seeds, drop_regrip=False, n_perm=100_000, n_boot=10_000):
    kb, nb = kn(base, seeds, drop_regrip)
    ka, na = kn(arm_, seeds, drop_regrip)
    t_obs = ka.sum() / na.sum() - kb.sum() / nb.sum()
    rng = np.random.default_rng(20261020)                 # 並べ替え: 初期状態ごとに基準と腕を入れ替える
    hits, done, nsum = 0, 0, nb.sum() + na.sum()
    while done < n_perm:
        m = min(10_000, n_perm - done)
        sw = rng.random((m, len(kb))) < 0.5
        k_arm, n_arm = np.where(sw, kb, ka).sum(1), np.where(sw, nb, na).sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = k_arm / n_arm - (kb.sum() + ka.sum() - k_arm) / (nsum - n_arm)
        hits += int(np.sum(np.abs(t) >= abs(t_obs) - 1e-12))
        done += m
    rng = np.random.default_rng(20261021)                 # ブートストラップ: 初期状態を単位に引き直す
    idx = rng.integers(0, len(kb), (n_boot, len(kb)))
    with np.errstate(divide="ignore", invalid="ignore"):
        d = ka[idx].sum(1) / na[idx].sum(1) - kb[idx].sum(1) / nb[idx].sum(1)
    d = d[np.isfinite(d)]
    return {"base": f"{kb.sum():.0f}/{nb.sum():.0f}（{100 * kb.sum() / nb.sum():.1f}%）",
            "arm": f"{ka.sum():.0f}/{na.sum():.0f}（{100 * ka.sum() / na.sum():.1f}%）", "diff": t_obs,
            "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)), "p": (1 + hits) / (1 + n_perm),
            "note": "並べ替え検定・ブートストラップ"}


def holm(ps):
    order = sorted(ps, key=lambda k: (ps[k], k))
    out, run = {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * ps[k]))
        out[k] = run
    return out


def wilcoxon_signed(x, y):
    """組にした差 y − x の Wilcoxon の符号順位検定（両側、差 0 の組は除く、正規近似・同順位の補正つき）の p。"""
    d = np.asarray(y, float) - np.asarray(x, float)
    d = d[d != 0]
    n = len(d)
    a = np.abs(d)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n)
    i = 0
    tie = 0.0
    while i < n:                                          # 同じ値には順位の平均を付ける
        j = i
        while j + 1 < n and a[order[j + 1]] == a[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        tie += (j - i + 1) ** 3 - (j - i + 1)
        i = j + 1
    t = ranks[d > 0].sum()
    mean = n * (n + 1) / 4
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24 - tie / 48)
    return math.erfc(abs(t - mean) / sd / math.sqrt(2))


def load_rtc():
    """results/rtc_trials.csv（付録 A、RTC を足した評価）を {条件: {初期状態の番号: 行}} で。"""
    out = {}
    with open(RTC_CSV, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            row = {k: int(r[k]) for k in ("all_three_in_box", "miss", "self_recovered", "miss_regrip", "self_recovered_regrip")}
            row["seam"] = float(r["seam_median_mm_s"]) if r["seam_median_mm_s"] else None
            out.setdefault(r["condition"], {})[int(r["seed"])] = row
    return out


def fmt_p(p):
    return f"{p:.2g}" if p >= 0.001 else f"{p:.1e}"


def show(title, rows):
    print(f"\n{title}")
    for name, r, extra in rows:
        print(f"  {name}: {r['base']} → {r['arm']}  差 {100 * r['diff']:+.1f} pt（{100 * r['lo']:+.1f}〜{100 * r['hi']:+.1f}）"
              f"  p {fmt_p(r['p'])}{extra}  [{r['note']}]")


def main():
    data = load()
    main_seeds = sorted(data[("main_200", "success_only", 0)])
    print(f"表 1　学び方ごとの結果（同じ {len(main_seeds)} 通りの初期状態）")
    print("  学び方 | 3 個とも箱へ（95% 信頼区間） | 自力の立て直し率 | 握り直しを除く")
    for (cond, llm), name in NAMES.items():
        rows = data[("main_200", cond, llm)]
        k = sum(r["all_three_in_box"] for r in rows.values())
        lo, hi = wilson(k, len(rows))
        sk, sn = sum(r["self_recovered"] for r in rows.values()), sum(r["miss"] for r in rows.values())
        rk = sk - sum(r["self_recovered_regrip"] for r in rows.values())
        rn = sn - sum(r["miss_regrip"] for r in rows.values())
        print(f"  {name} | {k}（{100 * k / len(rows):.1f}%、{100 * lo:.1f}〜{100 * hi:.1f}%） | {sk}/{sn}（{100 * sk / sn:.1f}%）"
              f" | {rk}/{rn}（{100 * rk / rn:.1f}%）")

    m = lambda c, l: data[("main_200", c, l)]               # noqa: E731
    pre = {"立て直しデータを足す（立て直し率）": self_cmp(m("success_only", 0), m("plus_recovery", 0), main_seeds),
           "自分の失敗から学び直す、LLM なし（立て直し率）": self_cmp(m("plus_recovery", 0), m("plus_failure_relearn", 0), main_seeds),
           "自分の失敗から学び直す、LLM あり（3 個とも箱へ）": success_cmp(m("plus_recovery", 1), m("plus_failure_relearn", 1),
                                                                main_seeds)}
    ph = holm({k: v["p"] for k, v in pre.items()})
    show("表 2　事前に決めた 3 つの比較（Holm 法で補正）", [(k, v, f"  Holm 後 {fmt_p(ph[k])}") for k, v in pre.items()])

    s600 = sorted(arm(data, ("main_200", "extra_400"), "plus_failure_relearn", 1))
    show("失敗を分類した後に足した比較（600 通りの初期状態、補正なし）", [
        ("配置位置による目標条件付け（3 個とも箱へ）",
         success_cmp(arm(data, ("main_200", "extra_400"), "plus_failure_relearn", 1),
                     arm(data, ("main_200", "extra_400"), "plus_place_goal", 1), s600), "")])

    show("感度分析: 持ち上げた後の握り直しを掴み損ねから除く（記述）", [
        ("立て直しデータを足す", self_cmp(m("success_only", 0), m("plus_recovery", 0), main_seeds, True), ""),
        ("自分の失敗から学び直す、LLM なし", self_cmp(m("plus_recovery", 0), m("plus_failure_relearn", 0), main_seeds, True), "")])

    sec = lambda c: data[("second_200", c, 0)]              # noqa: E731
    s2 = sorted(sec("plus_failure_relearn"))
    show("4.3 節　追加の試み（3 個とも箱へ）", [
        ("自分の成功した試行も足して学び直す（別の 200 通り、LLM なし）", success_cmp(sec("plus_failure_relearn"), sec("plus_own_success"), s2), ""),
        ("候補から成功の見込みで選ぶ（別の 200 通り、LLM なし）", success_cmp(sec("plus_failure_relearn"), sec("plus_chunk_select"), s2),
         ""),
        ("候補から成功の見込みで選ぶ（600 通り、LLM あり）",
         success_cmp(arm(data, ("main_200", "extra_400"), "plus_failure_relearn", 1),
                     arm(data, ("main_200", "extra_400"), "plus_chunk_select", 1), s600), "")])

    print("\n参考の数（3 個とも箱へ）")
    cells = {f"{NAMES.get((c, 0), c)}{'＋LLM' if l else ''}": sum(r["all_three_in_box"] for r in m(c, l).values())
             for c in ("success_only", "plus_recovery") for l in (0, 1)}
    print("  立て直しデータ × LLM の組み直し:", "、".join(f"{k} {v}" for k, v in cells.items()))
    d = {c: sum(r["all_three_in_box"] for r in data[("disturb_100", c, 1)].values()) for c in ("plus_recovery", "plus_failure_relearn")}
    print(f"  外乱（別の 100 通り、LLM あり）: 学び直す前 {d['plus_recovery']}、学び直した後 {d['plus_failure_relearn']}")
    print(f"  別の 200 通り（LLM なし）: 立て直しデータまで {sum(r['all_three_in_box'] for r in sec('plus_recovery').values())}、"
          f"＋自分の失敗から学び直す {sum(r['all_three_in_box'] for r in sec('plus_failure_relearn').values())}、"
          f"＋自分の成功した試行も足す {sum(r['all_three_in_box'] for r in sec('plus_own_success').values())}")

    rt = load_rtc()
    base = rt["plus_place_goal"]
    sr = sorted(base)
    task = {name: success_cmp(base, rt[c], sr) for c, name in RTC_NAMES.items()}
    th = holm({k: v["p"] for k, v in task.items()})
    show("付録 A　RTC を足す（3 個とも箱へ、3 本を Holm 法で補正）", [(k, v, f"  Holm 後 {fmt_p(th[k])}") for k, v in task.items()])
    print("\n付録 A　つなぎ目の段差（試行の中央値、mm/s。Wilcoxon の符号順位検定、3 本を Holm 法で補正）")
    seam = {}
    for c, name in RTC_NAMES.items():
        ss = [s for s in sr if base[s]["seam"] is not None and rt[c][s]["seam"] is not None]
        x, y = [base[s]["seam"] for s in ss], [rt[c][s]["seam"] for s in ss]
        seam[name] = (len(ss), float(np.median(x)), float(np.median(y)), wilcoxon_signed(x, y))
    sh = holm({k: v[3] for k, v in seam.items()})
    for name, (n, mx, my, p) in seam.items():
        print(f"  {name}: {mx:.1f} → {my:.1f}（{n} 組）  p {fmt_p(p)}  Holm 後 {fmt_p(sh[name])}")
    print("\n付録 A　記述（握り直しを除く掴み損ねと、自力の立て直し）")
    for c, name in {"plus_place_goal": "RTC なし", **RTC_NAMES}.items():
        k, n = kn(rt[c], sr, True)
        print(f"  {name}: 掴み損ね {n.sum():.0f}、自力の立て直し {k.sum():.0f}（{100 * k.sum() / n.sum():.1f}%）")


if __name__ == "__main__":
    main()
