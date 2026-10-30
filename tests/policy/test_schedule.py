"""実行器の時間の流れ（推論と行動列の切り替え、src/recovla/policy/schedule.py）の検査。numpy だけ。

torch が要る照合（推論ごとの雑音・rtc.py との一致）は test_rtc_compare.py。
"""
import numpy as np
import pytest

from recovla.policy.schedule import RuntimeConfig, Schedule, normalize_left_over
from tests.policy.mock_policy import A, chunk, decode, run

H = 50
N = 200


def expected_timeline(s, d, n):
    """naive・rtc の式から直接作った (k_obs, v, o) の列と、k ごとの (i, 添字)。"""
    infos = [(0, 0, 0)] + [(i * s, i * s + d, d) for i in range(1, (n - 1) // s + 1)]
    execs = []
    for k in range(n):
        i = max(n_ for n_, (_, v, _) in enumerate(infos) if v <= k)
        _, v, o = infos[i]
        execs.append((i, o + k - v))
    return infos, execs


# 1. naive・d=0 が sync と一致 -------------------------------------------------------------

@pytest.mark.parametrize("s", [5, 10, 25, 50])
def test_naive_d0_equals_sync(s):
    sync = run(RuntimeConfig("sync", s, 0), N)
    naive = run(RuntimeConfig("naive", s, 0), N)
    rtc = run(RuntimeConfig("rtc", s, 0), N)
    assert naive.executes == sync.executes
    assert rtc.executes == sync.executes
    np.testing.assert_array_equal(np.array(naive.actions), np.array(sync.actions))
    # sync は k = i·s で推論し、添字 0〜s−1 を実行する
    assert sync.executes == [(k // s, k % s) for k in range(N)]
    assert [e["k_obs"] for e in sync.log] == list(range(0, N, s))


# 3. left_over の行（中身。rtc.py との照合は test_rtc_compare.py） ------------------------------

@pytest.mark.parametrize("E", [10, 40])
@pytest.mark.parametrize("s,d", [(10, 0), (10, 2), (10, 4), (25, 3), (40, 10), (50, 0)])
def test_left_over_rows_follow_j(s, d, E):
    tr = run(RuntimeConfig("rtc", s, d, execution_horizon=E), N)
    i = -1
    for dec in tr.decisions:
        if not dec.infer:
            assert dec.left_over is None
            continue
        i += 1
        entry = tr.log[i]
        if i == 0:
            assert dec.left_over is None and entry["left_over_len"] == 0
            continue
        prev = tr.log[i - 1]
        j = prev["offset"] + (entry["k_obs"] - prev["k_valid"])       # j = o_{i−1} + (k_i − v_{i−1})
        L = chunk(i - 1)[j:H]
        assert entry["left_over_len"] == H - j == len(L)
        assert dec.left_over.shape == (E, A) and dec.left_over.dtype == np.float32
        n = min(E, len(L))
        np.testing.assert_array_equal(dec.left_over[:n], L[:n])                  # 先頭から E 行
        assert not dec.left_over[n:].any()                                        # 足りない分は 0
        assert [decode(r) for r in dec.left_over[:n]] == [(i - 1, j + t) for t in range(n)]
        # 残りの先頭は、この手でまだ前の行動列が実行している行（推論の開始時点で未実行の部分）
        assert dec.execute == ((i - 1, j) if d > 0 else (i, 0))
        assert dec.inference_delay == d


def test_left_over_len_and_padding_cases():
    """s=10 では j = s なので残りは 40 行: E=10 は切り詰め、E=40 はそのまま。s=25 で E=40 なら 15 行の 0 詰め。"""
    tr = run(RuntimeConfig("rtc", 10, 2, execution_horizon=10), 30)
    assert [e["left_over_len"] for e in tr.log] == [0, 40, 40]
    tr = run(RuntimeConfig("rtc", 25, 3, execution_horizon=40), 30)
    lo = tr.decisions[25].left_over
    assert tr.log[1]["left_over_len"] == 25 and lo.shape == (40, A)
    assert decode(lo[0]) == (0, 25) and decode(lo[24]) == (0, 49) and not lo[25:].any()


def test_naive_and_sync_pass_no_left_over():
    for mode in ("sync", "naive"):
        tr = run(RuntimeConfig(mode, 10, 0 if mode == "sync" else 3), N)
        assert all(d.left_over is None and d.inference_delay == 0 for d in tr.decisions)
        assert all(e["left_over_len"] == 0 for e in tr.log)


def test_normalize_left_over():
    L = chunk(3)[10:]                                         # 40 行
    np.testing.assert_array_equal(normalize_left_over(L, 10), L[:10])
    np.testing.assert_array_equal(normalize_left_over(L, 40), L)
    out = normalize_left_over(L[:5], 10)
    assert out.shape == (10, A) and out.dtype == L.dtype
    np.testing.assert_array_equal(out[:5], L[:5])
    assert not out[5:].any()
    assert normalize_left_over(L[:0], 10).shape == (10, A)
    with pytest.raises(ValueError):
        normalize_left_over(L[None], 10)


# 4. naive と rtc の時間の流れ ------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["naive", "rtc"])
@pytest.mark.parametrize("d", [0, 1, 2, 3, 4])
def test_naive_rtc_timeline(mode, d):
    s = 10
    tr = run(RuntimeConfig(mode, s, d), N)
    infos, execs = expected_timeline(s, d, N)
    assert [(e["k_obs"], e["k_valid"], e["offset"]) for e in tr.log] == infos
    assert [e["i"] for e in tr.log] == list(range(len(infos)))
    assert tr.executes == execs
    # 実行した行（後処理の後）の中身から逆算しても同じ
    assert [decode(a) for a in tr.actions] == execs

    for k, dec in enumerate(tr.decisions):
        assert dec.k == k
        assert dec.infer == (k % s == 0)
        assert dec.block == (k == 0)
    # 1 つの行動列から実行する範囲: 行動列 0 は 0〜s+d−1、i ≥ 1 は d〜s+d−1（最後の行動列は N で打ち切り）
    rows = {}
    for i, idx in tr.executes:
        rows.setdefault(i, []).append(idx)
    for i, r in rows.items():
        start = 0 if i == 0 else d
        full = list(range(start, s + d))
        assert r == (full if i < len(rows) - 1 else full[:len(r)])
    # 切り替えの直前まで前の行動列の続き: k_i〜k_i+d−1 は行動列 i−1 の添字 s〜s+d−1
    for i in range(1, len(infos)):
        k_i = i * s
        for t in range(d):
            if k_i + t < N:
                assert tr.executes[k_i + t] == (i - 1, s + t)
        if k_i + d < N:
            assert tr.executes[k_i + d] == (i, d)


def test_rtc_and_naive_share_the_timeline():
    for d in range(5):
        a = run(RuntimeConfig("naive", 10, d), N)
        b = run(RuntimeConfig("rtc", 10, d, execution_horizon=40), N)
        assert a.executes == b.executes
        assert [(e["k_obs"], e["k_valid"], e["offset"]) for e in a.log] == \
               [(e["k_obs"], e["k_valid"], e["offset"]) for e in b.log]


# 5. 持ち越さない時点 ----------------------------------------------------------------------------

def assert_no_carry(tr, k, i):
    dec = tr.decisions[k]
    assert dec.infer and dec.block and dec.left_over is None and dec.inference_delay == 0
    assert dec.execute == (i, 0)
    e = tr.log[i]
    assert e == {"i": i, "k_obs": k, "k_valid": k, "offset": 0, "left_over_len": 0, "reset": True,
                 "wall_s": None, "wall_breakdown_s": None}


@pytest.mark.parametrize("mode", ["sync", "naive", "rtc"])
@pytest.mark.parametrize("d", [0, 3])
def test_first_and_after_reset_do_not_carry(mode, d):
    s = 10
    # 31: k=30 の推論の直後で、その行動列が有効になる前（d=3 のとき）、45: 推論と推論の間、
    # 55: 推論の予定の手、56: 続けてもう 1 度
    resets = [31, 45, 55, 56]
    tr = run(RuntimeConfig(mode, s, d, execution_horizon=10), 120, resets=resets)
    assert_no_carry(tr, 0, 0)
    starts = [0] + resets
    by_k = {e["k_obs"]: e for e in tr.log}
    for r in resets:
        assert_no_carry(tr, r, by_k[r]["i"])
    # 番号は試行の中で3 個の連続タスク
    assert [e["i"] for e in tr.log] == list(range(len(tr.log)))
    # 推論は、持ち越さない推論から s 手ごと（次の reset まで）
    expect = []
    for a, b in zip(starts, starts[1:] + [120]):
        expect += list(range(a, b, s))
    assert [e["k_obs"] for e in tr.log] == expect
    assert [e["reset"] for e in tr.log] == [e["k_obs"] in starts for e in tr.log]
    # reset で捨てた、有効になる前の行動列は実行しない（d=3: k=30 の推論の行動列は 33 から有効の予定だった）
    if mode != "sync" and d > 0:
        i30 = by_k[30]["i"]
        assert tr.log[i30]["k_valid"] == 33
        assert all(i != i30 for i, _ in tr.executes)
        assert tr.executes[30] == (i30 - 1, s)                 # k=30 は前の行動列の続き
    # reset の直後の推論の次（持ち越す推論）は、reset 後の行動列の続きを使う
    after = by_k[66]
    if mode == "sync":
        assert after["reset"] is False and after["k_valid"] == 66 and after["offset"] == 0
    else:
        assert after["reset"] is False and after["k_valid"] == 66 + d and after["offset"] == d
        prev = tr.log[after["i"] - 1]
        assert prev["k_obs"] == 56
        if mode == "rtc":
            assert after["left_over_len"] == H - s
            assert decode(tr.decisions[66].left_over[0]) == (prev["i"], s)


def test_reset_before_first_decide_is_harmless():
    sched = Schedule(RuntimeConfig("rtc", 10, 2))
    sched.reset()
    dec = sched.decide(0)
    assert dec.infer and dec.block and dec.left_over is None


# 6. 不正な設定と呼び方 --------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["naive", "rtc"])
@pytest.mark.parametrize("s,d", [(0, 0), (10, -1), (10, 10), (10, 11), (45, 6), (50, 1), (51, 0)])
def test_invalid_config(mode, s, d):
    with pytest.raises(ValueError):
        RuntimeConfig(mode, s, d)


@pytest.mark.parametrize("kwargs", [
    dict(mode="sync", exec_interval=0, delay_steps=0),
    dict(mode="sync", exec_interval=51, delay_steps=0),
    dict(mode="async", exec_interval=10, delay_steps=0),
    dict(mode="naive", exec_interval=10, delay_steps=None),
    dict(mode="naive", exec_interval=10, delay_steps=1.0),
    dict(mode="naive", exec_interval=10, delay_steps=True),
    dict(mode="rtc", exec_interval=10, delay_steps=2, execution_horizon=0),
    dict(mode="rtc", exec_interval=10, delay_steps=2, execution_horizon=51),
    dict(mode="rtc", exec_interval=10, delay_steps=2, chunk_size=0),
])
def test_invalid_config_other(kwargs):
    with pytest.raises(ValueError):
        RuntimeConfig(**kwargs)


def test_valid_edges():
    RuntimeConfig("naive", 1, 0)
    RuntimeConfig("naive", 50, 0)
    RuntimeConfig("rtc", 45, 5, execution_horizon=50)
    # sync は d を使わない（0 とみなす）。設定の d が決まる前（null）でも、s=50 と d>0 の組でもよい
    for d in (None, 0, 2, 5):
        assert RuntimeConfig("sync", 50, d).delay == 0


def test_decide_must_not_skip_k():
    sched = Schedule(RuntimeConfig("naive", 10, 2))
    with pytest.raises(ValueError):
        sched.decide(1)                                 # 0 から始める
    sched.decide(0)
    sched.deliver(chunk(0))
    sched.decide(1)
    for bad in (3, 1, 0, -1, 2.0):
        with pytest.raises(ValueError):
            sched.decide(bad)
    sched.decide(2)                                     # 失敗した呼び出しは状態を変えない


def test_deliver_is_required_and_checked():
    sched = Schedule(RuntimeConfig("rtc", 10, 2))
    with pytest.raises(RuntimeError):
        sched.deliver(chunk(0))                         # 推論を始めていない
    with pytest.raises(RuntimeError):
        sched.action({})                                # decide の前
    sched.decide(0)
    with pytest.raises(ValueError):
        sched.deliver(chunk(0)[:49])                    # (H, A) でない
    with pytest.raises(ValueError):
        sched.deliver(chunk(0)[None])
    with pytest.raises(RuntimeError):
        sched.reset()                                   # deliver の前の reset
    c = chunk(0)
    assert sched.deliver(c) == 0
    with pytest.raises(RuntimeError):
        sched.deliver(chunk(0))                         # 2 度目
    with pytest.raises(KeyError):
        sched.action({})                                # 後処理の後の行動列がない
    c[:] = 0                                            # deliver の後に呼び出し側が行動列を書き換えても影響しない
    for k in range(1, 10):
        sched.decide(k)
    dec = sched.decide(10)
    assert decode(dec.left_over[0]) == (0, 10)
    with pytest.raises(RuntimeError):
        sched.decide(11)                                # 推論 1 の deliver がない


def test_log_frame_is_writable_by_the_caller():
    """wall_s・wall_breakdown_s は None の枠。呼び出し側が log[-1] に書き込めば、そのまま log に残る。"""
    sched = Schedule(RuntimeConfig("rtc", 10, 2))
    sched.decide(0)
    assert list(sched.log[-1]) == ["i", "k_obs", "k_valid", "offset", "left_over_len", "reset",
                                   "wall_s", "wall_breakdown_s"]
    sched.log[-1]["wall_s"] = 0.25
    assert sched.log[0]["wall_s"] == 0.25
