"""学習と評価で共有する VLA の状態 observation.state の定義（vla_state.py）の確かめ。numpy だけで通る。"""
import pathlib
import re

import numpy as np
import pytest

from recovla.data import vla_state


def frames(n=50, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    if n >= 10:
        q[:5] = [0.0, 1.0, 0.0, 0.0]                                 # ちょうど真下
        q[5:10] = vla_state.quat_mul(np.tile([np.cos(1e-7), np.sin(1e-7), 0, 0], (5, 1)), vla_state.Q_DOWN)
    return {"ee_pos": rng.normal(size=(n, 3)), "ee_quat": q, "fingers": rng.uniform(0, 0.04, (n, 2)),
            "joints": rng.normal(size=(n, 7))}


def test_names_and_layout():
    assert vla_state.STATE_DIM == 15 and len(vla_state.STATE_NAMES) == 15
    assert vla_state.STATE_NAMES[:8] == ["eef_x", "eef_y", "eef_z", "eef_rot_dev_x", "eef_rot_dev_y",
                                         "eef_rot_dev_z", "finger_left", "finger_right_neg"]


def test_values_follow_the_definition():
    f = frames()
    s = vla_state.policy_state(f["ee_pos"], f["ee_quat"], f["fingers"], f["joints"])
    assert s.dtype == np.float32 and s.shape == (50, 15)
    assert np.array_equal(s[:, :3], f["ee_pos"].astype(np.float32))
    assert np.array_equal(s[:, 6], f["fingers"][:, 0].astype(np.float32))
    assert np.array_equal(s[:, 7], (-f["fingers"][:, 1]).astype(np.float32))       # LIBERO と同じ符号
    assert np.array_equal(s[:, 8:], f["joints"].astype(np.float32))
    assert np.all(s[:5, 3:6] == 0)                                                 # 真下なら 0
    assert np.all(np.abs(s[5:10, 3:6]) < 1e-6)                                     # 小さな傾きも有限の値
    tilt = vla_state.quat_mul(np.array([[np.cos(0.05), 0, np.sin(0.05), 0]]), vla_state.Q_DOWN)
    assert np.allclose(vla_state.orientation_deviation(tilt), [[0, 0.1, 0]])
    assert np.allclose(vla_state.orientation_deviation(-tilt), [[0, 0.1, 0]])    # 四元数の符号は効かない


def test_per_frame_is_bit_identical_to_batched():
    f = frames(80, seed=3)
    batched = vla_state.policy_state(f["ee_pos"], f["ee_quat"], f["fingers"], f["joints"])
    for i in range(len(batched)):
        one = vla_state.policy_state_frame({k: v[i] for k, v in f.items()})
        assert np.array_equal(one, batched[i]), i


def test_inputs_are_not_modified():
    f = frames()
    before = {k: v.copy() for k, v in f.items()}
    vla_state.policy_state(f["ee_pos"], f["ee_quat"], f["fingers"], f["joints"])
    for k in f:
        assert np.array_equal(f[k], before[k]), k


def test_shape_errors():
    f = frames(4)
    with pytest.raises(ValueError):
        vla_state.policy_state(f["ee_pos"], f["ee_quat"][:, :3], f["fingers"], f["joints"])
    with pytest.raises(ValueError):
        vla_state.policy_state(f["ee_pos"][:3], f["ee_quat"], f["fingers"], f["joints"])
    with pytest.raises(ValueError):
        vla_state.policy_state_frame({"ee_pos": np.zeros(3), "ee_quat": np.zeros(4), "fingers": np.zeros(2)})


def test_state_is_not_computed_outside_vla_state():
    root = pathlib.Path(vla_state.__file__).parent
    pattern = re.compile(r"def (quat_mul|orientation_deviation)\b|fingers\[:, ?1\] \*=|conj_down|Q_DOWN = ")
    for name in ("convert.py", "vla_observation.py"):
        hits = [ln for ln in (root / name).read_text(encoding="utf-8").splitlines() if pattern.search(ln)]
        assert not hits, f"{name} defines state computation itself: {hits}"
