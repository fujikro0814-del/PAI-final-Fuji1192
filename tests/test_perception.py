"""目標位置の入力（俯瞰画像から求めた狙いの物の位置）の部品: 投影の逆、色の判定、見えないときに保つこと、状態の 18 次元。"""
import numpy as np
import pytest

from recovla.data import vla_state
from recovla.perception import color as PC
from recovla.sim import frames, scene

MODEL = scene.build_model("3cube")
CALIB = PC.overhead_calibration(MODEL)
THR = PC.Thresholds(20, 40, 1)


def test_pixel_to_plane_inverts_project():
    import mujoco
    d = mujoco.MjData(MODEL)
    mujoco.mj_forward(MODEL, d)
    for x, y in ((0.40, -0.10), (0.55, 0.02), (0.45, 0.25), (0.33, -0.20)):
        u, v = frames.project(MODEL, d, "overhead", (x, y, frames.CUBE_REST_Z), CALIB.width, CALIB.height)
        xy = PC.pixel_to_plane(CALIB, u, v, frames.CUBE_REST_Z)
        assert np.allclose(xy, (x, y), atol=1e-9)


def synthetic(blobs):
    img = np.full((256, 256, 3), 120, np.uint8)          # 灰色の机
    img[10:40, 200:250] = (200, 180, 20)                 # 黄色の箱（赤と緑が両方高い）
    for (r0, c0), rgb in blobs:
        img[r0:r0 + 10, c0:c0 + 10] = rgb
    return img


def test_detect_colors_and_box_is_not_a_color():
    img = synthetic([((100, 100), (220, 30, 30)), ((150, 60), (30, 200, 40)), ((50, 50), (30, 30, 210))])
    for color, (r, c) in (("red", (100, 100)), ("green", (150, 60)), ("blue", (50, 50))):
        d = PC.detect(img, color, THR)
        assert d["pixels"] == 100 and d["visible"]
        assert d["u"] == pytest.approx(c + 5) and d["v"] == pytest.approx(r + 5)
    assert not PC.color_mask(synthetic([]), "red", THR).any() and not PC.color_mask(synthetic([]), "green", THR).any()


def test_cue_holds_last_seen_and_resets_on_color_change():
    cue = PC.TargetCue(CALIB, THR)
    seen = cue.update(synthetic([((100, 100), (220, 30, 30))]), "red")
    assert seen[2] == 1.0
    hidden = cue.update(synthetic([]), "red")
    assert hidden[2] == 0.0 and np.allclose(hidden[:2], seen[:2])      # 最後に見えた値を保つ
    other = cue.update(synthetic([]), "green")                         # 色が変わった: 保った値は使わない
    assert other[2] == 0.0 and np.allclose(other[:2], cue.fallback)
    cue.reset()
    assert np.allclose(cue.update(synthetic([]), "red")[:2], cue.fallback)


def test_state_with_cue_and_instruction_color():
    s = np.arange(15, dtype=np.float32)
    out = vla_state.with_cue(s, np.array([0.4, -0.1, 1.0]))
    assert out.shape == (18,) and out.dtype == np.float32 and out[15] == pytest.approx(0.4)
    with pytest.raises(ValueError):
        vla_state.with_cue(s[:14], np.zeros(3))
    assert PC.color_of_instruction("put the green cube in the box") == "green"
    with pytest.raises(ValueError):
        PC.color_of_instruction("put the cube in the box")


def test_thresholds_from_config_fixed():
    thr = PC.Thresholds.from_config()
    assert thr.to_json() == {"min_value": 20, "min_margin": 40, "min_pixels": 1}   # 学習データの配置で決めて固定した値
