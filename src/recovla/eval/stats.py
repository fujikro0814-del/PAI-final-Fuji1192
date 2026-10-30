"""割合の信頼区間（Wilson 法）と、対にした比較の検定（McNemar 正確検定・Newcombe 法の差の部分・符号順位検定）。"""
import math

import numpy as np
from scipy import stats as _st


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054):
    """割合の 95 % 部分（Wilson 法）。(下限, 上限)、試行が 0 なら (None, None)。"""
    if trials <= 0:
        return None, None
    p = successes / trials
    denom = 1.0 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


# ------------------------------------------------------------------------------ 対にした比較

def mcnemar_exact(n10: int, n01: int) -> float:
    """正確な McNemar 検定の両側 p 値。食い違った対（n10 + n01）の中で n10 が二項分布 B(n, 0.5) に従うかを見る。"""
    n = int(n10) + int(n01)
    if n == 0:
        return 1.0
    return float(_st.binomtest(int(n10), n, 0.5).pvalue)


def _phi_newcombe(n11: int, n10: int, n01: int, n00: int, n: int, correct: bool) -> float:
    """2×2 の φ 係数。周辺の和のどれかが 0 なら 0。

    correct=True のとき、分子 A = n11·n00 − n10·n01 に Newcombe の補正をかける
    （A > n/2 なら A − n/2、0 ≤ A ≤ n/2 なら 0、A < 0 はそのまま）。
    """
    denom = (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
    if denom == 0:
        return 0.0
    a = n11 * n00 - n10 * n01
    if correct:
        a = a - n / 2 if a > n / 2 else (0.0 if a >= 0 else a)
    return a / math.sqrt(denom)


def paired_diff_ci(n11: int, n10: int, n01: int, n00: int, alpha: float = 0.05,
                   *, phi_correction: bool = True) -> tuple[float, float, float]:
    """対応のある 2 つの割合の差 rate_a − rate_b と、その 1 − alpha 部分（Newcombe 1998 の方法 10）。

    rate_a = (n11 + n10) / n、rate_b = (n11 + n01) / n。各割合の Wilson 95% 信頼区間 (l1, u1)・(l2, u2) を
    φ 係数で結ぶ:
        下限 = 差 − √((p1 − l1)² − 2φ(p1 − l1)(u2 − p2) + (u2 − p2)²)
        上限 = 差 + √((u1 − p1)² − 2φ(u1 − p1)(p2 − l2) + (p2 − l2)²)
    φ の補正（phi_correction=True が既定）は Newcombe の方法 10 の定義。
    Newcombe 1998 Table III の 4 例と照合済み（tests/eval/test_stats.py）。
    組が 0 なら (nan, nan, nan)。
    """
    n11, n10, n01, n00 = (int(v) for v in (n11, n10, n01, n00))
    n = n11 + n10 + n01 + n00
    if n <= 0:
        return math.nan, math.nan, math.nan
    z = float(_st.norm.ppf(1.0 - alpha / 2.0))
    p1 = (n11 + n10) / n
    p2 = (n11 + n01) / n
    l1, u1 = wilson_interval(n11 + n10, n, z)
    l2, u2 = wilson_interval(n11 + n01, n, z)
    phi = _phi_newcombe(n11, n10, n01, n00, n, phi_correction)
    diff = p1 - p2
    dl = (p1 - l1) ** 2 - 2.0 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2
    du = (u1 - p1) ** 2 - 2.0 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2
    return diff, diff - math.sqrt(max(dl, 0.0)), diff + math.sqrt(max(du, 0.0))


def wilcoxon_paired(a, b) -> tuple[float, float, int]:
    """符号順位検定（両側、差が 0 の組は除く＝zero_method="wilcox"）。(統計量, p, 対の数)。

    どちらかが NaN の組は除く。組が 0 なら (nan, nan, 0)。差がすべて 0 のときも検定できないので (nan, nan, 対の数)。
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"a と b の形が違う: {a.shape} と {b.shape}")
    keep = np.isfinite(a) & np.isfinite(b)
    n = int(keep.sum())
    if n == 0:
        return math.nan, math.nan, 0
    d = a[keep] - b[keep]
    if not np.any(d != 0):
        return math.nan, math.nan, n
    res = _st.wilcoxon(a[keep], b[keep], zero_method="wilcox")
    return float(res.statistic), float(res.pvalue), n


def pair_by_seed(rows_a: list[dict], rows_b: list[dict], key: str = "seed") -> list[tuple[dict, dict]]:
    """同じシード（配置）の行を対にする（rows_a の並びの順）。片方にしかないシードは除く。

    同じ側に同じシードが 2 回あると組が決まらないので ValueError。
    """
    def index(rows, side):
        out = {}
        for r in rows:
            s = r[key]
            if s in out:
                raise ValueError(f"{side} に {key}={s!r} が 2 回ある")
            out[s] = r
        return out

    ia = index(rows_a, "rows_a")
    ib = index(rows_b, "rows_b")
    return [(ra, ib[s]) for s, ra in ia.items() if s in ib]
