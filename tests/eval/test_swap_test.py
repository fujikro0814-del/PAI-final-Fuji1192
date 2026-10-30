"""指示の差し替え試験（指示の色を替えると VLA の狙う物も替わるか）の判定の純関数の検査。"""
import numpy as np

from recovla.eval.swap_test import MID_DROP_M, chunk_target, majority

CUBES = {"red": [0.40, -0.10], "green": [0.50, 0.00], "blue": [0.45, -0.15]}


def make_chunk(steps, close_at=None, h=50):
    c = np.zeros((h, 7))
    c[:, 6] = -1.0
    for i, d in enumerate(steps):
        c[i, :3] = d
    if close_at is not None:
        c[close_at:, 6] = 1.0
    return c


def test_first_close_point_and_nearest_cube():
    x0 = np.array([0.30, 0.0, 0.35])
    steps = [[0.02, 0.0, -0.01]] * 10                      # 10 ステップで (0.50, 0.0) へ
    r = chunk_target(make_chunk(steps, close_at=9), x0, CUBES)
    assert r["closed"] and r["index"] == 9
    assert np.allclose(r["xy"], [0.50, 0.0]) and r["nearest"] == "green" and not r["mid_drop"]
    r = chunk_target(make_chunk(steps, close_at=4), x0, CUBES)   # 途中で閉じる: (0.40, 0.0) は赤から 10 cm
    assert r["index"] == 4 and np.allclose(r["xy"], [0.40, 0.0]) and r["mid_drop"]


def test_never_closes_uses_chunk_end():
    x0 = np.array([0.45, -0.15, 0.30])
    r = chunk_target(make_chunk([[0.0, 0.0, -0.01]] * 5), x0, CUBES)
    assert not r["closed"] and r["index"] == 49 and r["nearest"] == "blue" and r["dist_m"] < 1e-12


def test_mid_drop_threshold_and_majority():
    assert MID_DROP_M == 0.03
    x0 = np.array([0.50, 0.0299, 0.3])
    assert not chunk_target(make_chunk([], close_at=0), x0, CUBES)["mid_drop"]
    assert majority(["red", "red", "green", "blue", "red"]) == "red"
    assert majority(["red", "red", "green", "green", "blue"]) is None
    assert majority([]) is None
