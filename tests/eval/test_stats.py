"""統計の関数（stats.py）の検査。

Newcombe (1998) の方法 10 は、文献（Statistics in Medicine 17, 2635–2650）の Table III の 4 例と照合する。
あわせて、式を別の順で組んだ独立な実装と一致することも確かめる。
"""
import itertools
import math

import numpy as np
import pytest
from scipy import stats as st

from recovla.eval import stats

Z95 = 1.959963984540054


# ------------------------------------------------------------------------------ Wilson

def test_wilson_unchanged():
    # 固定した参照の値（Wilson 法の 95% 信頼区間）
    assert stats.wilson_interval(18, 20) == (0.6989663547715128, 0.9721335187862319)
    assert stats.wilson_interval(61, 100) == (0.5120301704584073, 0.6998312599360558)
    assert stats.wilson_interval(0, 0) == (None, None)


# ------------------------------------------------------------------------------ McNemar

@pytest.mark.parametrize("n10,n01,p", [
    (0, 5, 0.0625),          # 2 · (1/2)^5
    (5, 0, 0.0625),
    (0, 0, 1.0),             # 食い違いなし
    (1, 1, 1.0),
    (3, 7, 352 / 1024),      # 2 · (1 + 10 + 45 + 120) / 2^10
    (0, 10, 2 / 1024),
])
def test_mcnemar_hand_values(n10, n01, p):
    assert stats.mcnemar_exact(n10, n01) == pytest.approx(p, rel=1e-12)


def test_mcnemar_uses_binomtest(monkeypatch):
    calls = []
    real = st.binomtest

    def spy(k, n, p):
        calls.append((k, n, p))
        return real(k, n, p)

    monkeypatch.setattr(stats._st, "binomtest", spy)
    stats.mcnemar_exact(2, 9)
    assert calls == [(2, 11, 0.5)]


# ------------------------------------------------------------------------------ Newcombe の方法 10（独立な実装と照合）

def _wilson_by_roots(x: int, n: int, z: float) -> tuple[float, float]:
    """Wilson の部分を、|p − π| = z √(π(1 − π)/n) の π についての 2 次方程式の 2 根として求める。"""
    p = x / n
    k = z * z / n
    roots = np.sort(np.real(np.roots([1 + k, -(2 * p + k), p * p])))
    return float(roots[0]), float(roots[1])


def _phi_from_data(n11, n10, n01, n00, correct: bool) -> float:
    """0/1 の対の列を作って相関係数として φ を求め、補正は 2×2 の分子に戻してかける。"""
    xa = np.array([1] * n11 + [1] * n10 + [0] * n01 + [0] * n00, dtype=float)
    xb = np.array([1] * n11 + [0] * n10 + [1] * n01 + [0] * n00, dtype=float)
    if xa.std() == 0 or xb.std() == 0:
        return 0.0
    phi = float(np.corrcoef(xa, xb)[0, 1])
    if not correct:
        return phi
    n = len(xa)
    scale = n * n * xa.std() * xb.std()                 # = √((n11+n10)(n01+n00)(n11+n01)(n10+n00))
    a = phi * scale
    a = max(a - n / 2, 0.0) if a >= 0 else a
    return a / scale


def _newcombe10_reference(n11, n10, n01, n00, alpha=0.05, correct=True):
    n = n11 + n10 + n01 + n00
    z = st.norm.isf(alpha / 2)
    pa, pb = (n11 + n10) / n, (n11 + n01) / n
    la, ua = _wilson_by_roots(n11 + n10, n, z)
    lb, ub = _wilson_by_roots(n11 + n01, n, z)
    phi = _phi_from_data(n11, n10, n01, n00, correct)
    # 差の下限は「a の下側の幅」と「b の上側の幅」、上限はその逆（MOVER の組み方で書き直したもの）
    wa_lo, wb_hi = pa - la, ub - pb
    wa_hi, wb_lo = ua - pa, pb - lb
    lower = (pa - pb) - math.sqrt(max(wa_lo ** 2 + wb_hi ** 2 - 2 * phi * wa_lo * wb_hi, 0.0))
    upper = (pa - pb) + math.sqrt(max(wa_hi ** 2 + wb_lo ** 2 - 2 * phi * wa_hi * wb_lo, 0.0))
    return pa - pb, lower, upper


def _tables():
    small = [t for t in itertools.product(range(5), repeat=4) if sum(t) > 0]
    rng = np.random.default_rng(20260925)
    big = [tuple(int(v) for v in rng.integers(0, 60, 4)) for _ in range(300)]
    edge = [(0, 0, 0, 20), (20, 0, 0, 0), (0, 20, 0, 0), (0, 0, 20, 0), (10, 0, 0, 10), (0, 10, 10, 0),
            (18, 0, 2, 0), (61, 10, 3, 26), (1, 0, 0, 0)]
    return small + big + edge


@pytest.mark.parametrize("correct", [True, False])
def test_newcombe10_matches_an_independent_implementation(correct):
    for t in _tables():
        d, lo, hi = stats.paired_diff_ci(*t, phi_correction=correct)
        rd, rlo, rhi = _newcombe10_reference(*t, correct=correct)
        # 幅は平方根の中で打ち消し合う（φ = ±1 で 0 になる）ので、丸めの差が √ で拡大しないよう二乗で比べる
        assert d == pytest.approx(rd, abs=1e-12), t
        assert (d - lo) ** 2 == pytest.approx((rd - rlo) ** 2, abs=1e-12), t
        assert (hi - d) ** 2 == pytest.approx((rhi - rd) ** 2, abs=1e-12), t


def test_newcombe10_alpha():
    for t in [(12, 5, 2, 21), (3, 0, 8, 9)]:
        for alpha in (0.10, 0.01):
            assert stats.paired_diff_ci(*t, alpha=alpha) == pytest.approx(_newcombe10_reference(*t, alpha=alpha),
                                                                          abs=1e-9)


def test_newcombe10_properties():
    for t in _tables():
        d, lo, hi = stats.paired_diff_ci(*t)
        n = sum(t)
        assert d == pytest.approx((t[1] - t[2]) / n)
        assert -1 - 1e-12 <= lo <= d + 1e-12 and d - 1e-12 <= hi <= 1 + 1e-12, t
        # a と b を入れ替えると部分は符号を変えて入れ替わる
        d2, lo2, hi2 = stats.paired_diff_ci(t[0], t[2], t[1], t[3])
        assert (d2, lo2, hi2) == pytest.approx((-d, -hi, -lo), abs=1e-12), t
        # 高い信頼度ほど広い
        _, lo99, hi99 = stats.paired_diff_ci(*t, alpha=0.01)
        assert lo99 <= lo + 1e-12 and hi99 >= hi - 1e-12, t


# Newcombe 1998 Table III（方法 10）の数値例。(n11, n10, n01, n00) → 差の 95% 信頼区間（小数 4 桁）
NEWCOMBE_1998_TABLE_III = [
    ((36, 12, 2, 0), (0.0569, 0.3404)),
    ((36, 14, 0, 0), (0.1528, 0.4167)),
    ((2, 98, 0, 0), (0.9178, 0.9945)),
    ((54, 0, 0, 0), (-0.0664, 0.0664)),
]


@pytest.mark.parametrize("table,ci", NEWCOMBE_1998_TABLE_III)
def test_newcombe10_matches_newcombe_1998_table_iii(table, ci):
    _, lo, hi = stats.paired_diff_ci(*table)
    assert (round(lo, 4), round(hi, 4)) == ci


def test_newcombe10_phi_correction_case():
    """Table III の 4 例は n00 = 0 で φ の補正が効かないので、補正の効く例を別に置く。
    (10, 0, 0, 10) → [−0.0898, 0.0898] は別に書いた独立な実装の値（文献の値ではない）。"""
    d, lo, hi = stats.paired_diff_ci(10, 0, 0, 10)
    assert (d, round(lo, 4), round(hi, 4)) == (0.0, -0.0898, 0.0898)
    # 補正しなければ φ = 1 で幅が 0 に潰れる（補正が効いていることの確かめ）
    assert stats.paired_diff_ci(10, 0, 0, 10, phi_correction=False) == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)


def test_newcombe10_no_pairs():
    assert all(math.isnan(v) for v in stats.paired_diff_ci(0, 0, 0, 0))


def test_newcombe10_zero_phi_equals_combining_wilson_widths():
    """周辺の和が 0（φ = 0）のときは、2 つの Wilson の幅の二乗和の平方根になる（独立な場合の方法 10 と同じ形）。"""
    d, lo, hi = stats.paired_diff_ci(0, 0, 7, 13)          # a は 0/20（周辺の和が 0 → φ = 0）
    la, ua = stats.wilson_interval(0, 20, Z95)
    lb, ub = stats.wilson_interval(7, 20, Z95)
    assert lo == pytest.approx(d - math.hypot(0 - la, ub - 7 / 20), abs=1e-12)
    assert hi == pytest.approx(d + math.hypot(ua - 0, 7 / 20 - lb), abs=1e-12)


# ------------------------------------------------------------------------------ Wilcoxon

def test_wilcoxon_matches_scipy():
    rng = np.random.default_rng(1)
    a = rng.normal(0.3, 0.1, 30)
    b = a - rng.normal(0.05, 0.05, 30)
    b[[3, 7]] = a[[3, 7]]                                   # 差が 0 の対（除かれる）
    res = st.wilcoxon(a, b, zero_method="wilcox")
    assert stats.wilcoxon_paired(a, b) == (float(res.statistic), float(res.pvalue), 30)


def test_wilcoxon_drops_nan_pairs():
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, 25)
    b = rng.normal(0.4, 1, 25)
    a[[0, 5]] = np.nan
    b[[5, 9, 11]] = np.nan
    keep = np.isfinite(a) & np.isfinite(b)
    res = st.wilcoxon(a[keep], b[keep], zero_method="wilcox")
    stat, p, n = stats.wilcoxon_paired(a, b)
    assert n == 21 and stat == pytest.approx(res.statistic) and p == pytest.approx(res.pvalue)


def test_wilcoxon_no_pairs():
    for a, b in [([], []), ([np.nan, 1.0], [2.0, np.nan])]:
        stat, p, n = stats.wilcoxon_paired(a, b)
        assert math.isnan(stat) and math.isnan(p) and n == 0


def test_wilcoxon_all_differences_zero():
    stat, p, n = stats.wilcoxon_paired([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert math.isnan(stat) and math.isnan(p) and n == 3


def test_wilcoxon_shape_mismatch():
    with pytest.raises(ValueError):
        stats.wilcoxon_paired([1.0, 2.0], [1.0])


# ------------------------------------------------------------------------------ シードで対にする

def test_pair_by_seed():
    a = [{"seed": 3, "v": "a3"}, {"seed": 1, "v": "a1"}, {"seed": 2, "v": "a2"}]
    b = [{"seed": 1, "v": "b1"}, {"seed": 4, "v": "b4"}, {"seed": 3, "v": "b3"}]
    pairs = stats.pair_by_seed(a, b)
    assert [(x["v"], y["v"]) for x, y in pairs] == [("a3", "b3"), ("a1", "b1")]
    assert stats.pair_by_seed(a, b, key="v") == []


def test_pair_by_seed_duplicate_raises():
    with pytest.raises(ValueError):
        stats.pair_by_seed([{"seed": 1}, {"seed": 1}], [{"seed": 1}])
