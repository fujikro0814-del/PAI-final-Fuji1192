"""VLA に渡す画像の向きの決まり（vla_image_spec.py）の確かめ。numpy だけで通る。期待の画像は添字の計算で直に作り、
モジュールの切り出しを使わない（向きを取り違えた変換が通らないように）。"""
import pathlib
import re

import numpy as np
import pytest

from recovla.data import vla_image_spec as spec

H, W = 5, 7


def asymmetric_image(seed=0):
    """画素がすべて違う画像（どの反転・回転でも配列が変わる）。"""
    return np.arange(H * W * 3, dtype=np.uint8).reshape(H, W, 3) + np.uint8(seed)


def expected(raw, transform):
    out = np.zeros_like(raw)
    for r in range(H):
        for c in range(W):
            src_r = {"identity": r, "fliplr": r, "flipud": H - 1 - r, "rot180": H - 1 - r}[transform]
            src_c = {"identity": c, "fliplr": W - 1 - c, "flipud": c, "rot180": W - 1 - c}[transform]
            out[r, c] = raw[src_r, src_c]
    return out


def test_spec_table_is_the_decided_one():
    assert spec.VIEW_TRANSFORMS == {"overhead": "fliplr", "wrist": "flipud"}
    assert spec.IMAGE_KEYS == {"overhead": "observation.images.image", "wrist": "observation.images.image2"}
    assert spec.SPEC_VERSION == 2


def test_overhead_left_right_and_wrist_up_down_at_value_level():
    raw = asymmetric_image()
    over = spec.to_policy_image("overhead", raw)
    wrist = spec.to_policy_image("wrist", raw)
    assert np.array_equal(over, expected(raw, "fliplr"))
    assert np.array_equal(wrist, expected(raw, "flipud"))
    # ほかの変換とは一致しない
    for t in ("identity", "flipud", "rot180"):
        assert not np.array_equal(over, expected(raw, t))
    for t in ("identity", "fliplr", "rot180"):
        assert not np.array_equal(wrist, expected(raw, t))
    assert over.flags["C_CONTIGUOUS"] and wrist.flags["C_CONTIGUOUS"]
    assert not np.shares_memory(over, raw) and not np.shares_memory(wrist, raw)


def test_policy_images_maps_views_to_keys():
    raw = {"overhead": asymmetric_image(1), "wrist": asymmetric_image(2)}
    out = spec.policy_images(raw)
    assert set(out) == {"observation.images.image", "observation.images.image2"}
    assert np.array_equal(out["observation.images.image"], expected(raw["overhead"], "fliplr"))
    assert np.array_equal(out["observation.images.image2"], expected(raw["wrist"], "flipud"))


@pytest.mark.parametrize("views", [{"wrist"}, {"overhead"}, {"overhead", "wrist", "side"}, set()])
def test_policy_images_rejects_missing_or_extra_views(views):
    with pytest.raises(spec.ImageSpecError):
        spec.policy_images({v: asymmetric_image() for v in views})


def test_unknown_view_transform_and_shape_raise():
    with pytest.raises(spec.ImageSpecError):
        spec.to_policy_image("side", asymmetric_image())
    with pytest.raises(spec.ImageSpecError):
        spec.apply_net_transform("fliplr+rot180", asymmetric_image())
    with pytest.raises(spec.ImageSpecError):
        spec.to_policy_image("wrist", np.zeros((H, W), np.uint8))


@pytest.mark.parametrize("tables", [
    ({"wrist": "flipud"}, spec.IMAGE_KEYS, spec.CAMERAS),                       # 1 つの視点だけの変換
    ({"overhead": "fliplr"}, spec.IMAGE_KEYS, spec.CAMERAS),
    (spec.VIEW_TRANSFORMS, {"overhead": "observation.images.image"}, spec.CAMERAS),   # 1 つの視点だけの鍵
    (spec.VIEW_TRANSFORMS, spec.IMAGE_KEYS, ("overhead",)),
    ({"overhead": "fliplr", "wrist": "upside-down"}, spec.IMAGE_KEYS, spec.CAMERAS),  # 知らない名前
])
def test_validate_rejects_partial_or_unknown_tables(tables):
    with pytest.raises(spec.ImageSpecError):
        spec.validate(*tables)


def test_module_validates_its_tables_at_import():
    src = pathlib.Path(spec.__file__).read_text(encoding="utf-8")
    assert re.search(r"^validate\(\)$", src, re.M), "vla_image_spec must call validate() at import time"
    spec.validate()


def test_record_roundtrip_and_rejections():
    rec = spec.spec_record()
    spec.check_record(dict(rec))
    with pytest.raises(spec.ImageSpecError):          # 版 1 のデータ: 指定の欄がない
        spec.check_record({"converter_version": 1, "images": "flipped left-right"})
    with pytest.raises(spec.ImageSpecError):
        spec.check_record({**rec, "image_spec_version": 1})
    with pytest.raises(spec.ImageSpecError):          # 両方の視点を左右反転
        spec.check_record({**rec, "image_transforms": {"overhead": "fliplr", "wrist": "fliplr"}})
    with pytest.raises(spec.ImageSpecError):          # 視点が 1 つ欠ける
        spec.check_record({**rec, "image_transforms": {"wrist": "flipud"}})


def test_no_image_transform_written_outside_the_spec():
    root = pathlib.Path(spec.__file__).parent
    pattern = re.compile(r"fliplr\(|flipud\(|rot90\(|\[::-1\]|\[:, ?::-1\]|\[::-1, ?::-1\]")
    for name in ("convert.py", "vla_observation.py"):
        hits = [ln for ln in (root / name).read_text(encoding="utf-8").splitlines() if pattern.search(ln)]
        assert not hits, f"{name} transforms images itself: {hits}"


def test_cameras_match_the_recorder():
    pytest.importorskip("mujoco")
    pytest.importorskip("cv2")
    from recovla.sim import control
    assert tuple(control.CAMERAS) == spec.CAMERAS
