"""記録からの失敗の検出（recovla.eval.failure_detect）を合成の軌跡で確かめる。"""
import numpy as np

from recovla.eval import failure_detect as FD
from recovla.sim import frames

Z0 = frames.CUBE_REST_Z
DT = 0.05


def _run(n, tip, fingers, closed, cubes, inbox, target=0):
    det = FD.FailureDetector(target)
    for f in range(n):
        det.update(f * DT, tip(f), fingers(f), closed(f), cubes(f), inbox(f))
    return det


def _cubes(target_z=lambda f: Z0, target_xy=(0.4, 0.0)):
    return lambda f: np.array([[target_xy[0], target_xy[1], target_z(f)], [0.4, 0.2, Z0], [0.4, -0.2, Z0]])


NOBOX = lambda f: np.zeros(3, bool)                              # noqa: E731
MOVING_TIP = lambda f: np.array([0.3 + 0.002 * f, 0.1, 0.2])       # noqa: E731  4 cm/s で動き続ける
OPEN = lambda f: np.array([0.04, 0.04])                            # noqa: E731


WINDOW = float(FD._CFG["eval"]["failure_detect"]["grasp_window_s"])


def test_grasp_miss_after_window():
    det = _run(int((WINDOW + 1.0) / DT) + 10, MOVING_TIP, OPEN, lambda f: f >= 10, _cubes(), NOBOX)
    assert [e.kind for e in det.events] == ["grasp_miss"]
    assert abs(det.first.t - (10 * DT + WINDOW)) < 1e-9


def test_no_grasp_miss_when_lifted():
    z = lambda f: Z0 + (0.05 if f >= 20 else 0.0)                  # noqa: E731  閉じて 0.5 s で持ち上がる
    det = _run(80, lambda f: np.array([0.4, 0.0, z(f)]) + [0.002 * (f % 2), 0, 0], OPEN, lambda f: f >= 10,
               _cubes(z), NOBOX)
    assert "grasp_miss" not in [e.kind for e in det.events]


def _carried(f, rest_z):
    """5〜19 フレームは持ち上げて動いている（宙で静止しない）、20 フレーム目から rest_z で静止。"""
    return Z0 + 0.05 + 0.003 * (f - 5) if 5 <= f < 20 else (rest_z if f >= 20 else Z0)


def test_drop_on_table_and_other():
    for rest_z, kind in ((Z0, "drop_table"), (Z0 + 0.05, "drop_other")):
        z = lambda f, rz=rest_z: _carried(f, rz)                   # noqa: E731
        det = _run(60, MOVING_TIP, OPEN, lambda f: False, _cubes(z), NOBOX)     # 静止 2 s（その他の落下は 1.0 s 要る）
        kinds = [e.kind for e in det.events]
        assert kinds == [kind], kinds


def test_other_drop_needs_longer_rest():
    """箱の縁などで 0.5 s 止まってから動く（箱に入る）ものは、その他の落下にしない。"""
    z = lambda f: _carried(f, Z0 + 0.05) if f < 30 else Z0 + 0.05 - 0.004 * (f - 29)   # noqa: E731
    det = _run(45, MOVING_TIP, OPEN, lambda f: False, _cubes(z), NOBOX)
    assert "drop_other" not in [e.kind for e in det.events]


def test_no_drop_while_held_or_in_box():
    z = lambda f: Z0 + 0.1 if f >= 5 else Z0                       # noqa: E731  持ち上げたまま止まっている
    det = _run(60, lambda f: np.array([0.4, 0.0, z(f)]) + [0.002 * (f % 2), 0, 0], OPEN, lambda f: f >= 3,
               _cubes(z), NOBOX)
    assert not [e for e in det.events if e.kind.startswith("drop")]
    det = _run(60, MOVING_TIP, OPEN, lambda f: False, _cubes(lambda f: _carried(f, Z0 + 0.01)),
               lambda f: np.array([f >= 20, False, False]))                           # 箱の中で静止
    assert not [e for e in det.events if e.kind.startswith("drop")]


def test_wrong_color_only_when_open_and_near():
    tip = lambda f: np.array([0.4, 0.2 - 0.004 * max(0, 20 - f), Z0 + 0.03])   # noqa: E731  緑へ近づく
    det = _run(40, tip, OPEN, lambda f: False, _cubes(), NOBOX)
    assert [e.kind for e in det.events] == ["wrong_color"] and det.first.info["cube"] == 1
    det = _run(40, tip, OPEN, lambda f: True, _cubes(), NOBOX)
    assert "wrong_color" not in [e.kind for e in det.events]


def test_stall_needs_three_seconds():
    still = lambda f: np.array([0.3, 0.1, 0.2])                    # noqa: E731
    det = _run(int(2.9 / DT), still, OPEN, lambda f: False, _cubes(), NOBOX)
    assert not det.events
    det = _run(int(3.2 / DT), still, OPEN, lambda f: False, _cubes(), NOBOX)
    assert [e.kind for e in det.events] == ["stall"]


def test_detect_trial_respects_t_end():
    n = 80
    arrays = {"target": np.zeros(n, np.int8), "sim_time": np.arange(n) * DT,
              "fingertip": np.array([MOVING_TIP(f) for f in range(n)]), "fingers": np.array([OPEN(f) for f in range(n)]),
              "gripper_closed": np.array([f >= 10 for f in range(n)]), "cube_pos": np.array([_cubes()(f) for f in range(n)]),
              "cube_in_box": np.zeros((n, 3), bool)}
    assert [e.kind for e in FD.detect_trial(arrays)] == ["grasp_miss"]
    assert FD.detect_trial(arrays, t_end=10 * DT + WINDOW - 0.1) == []
