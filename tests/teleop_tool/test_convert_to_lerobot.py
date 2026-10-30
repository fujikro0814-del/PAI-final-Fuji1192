"""記録から LeRobot の学習データへの変換（recovla.data.convert）の確かめ。lerobot が要る。

合成の生の記録を tmp_path に作って変換する。
"""
import json
import pathlib

import numpy as np
import pytest

pytest.importorskip("lerobot")
from PIL import Image  # noqa: E402

from recovla.data import convert as conv  # noqa: E402


def _axis_quat(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    return np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis])


def make_raw_episode(root: pathlib.Path, episode_id=0, n=21, size=16):
    ep = root / "2026-09-16" / f"ep_{episode_id:06d}"
    rng = np.random.default_rng(episode_id)
    for cam in ("overhead", "wrist"):
        (ep / cam).mkdir(parents=True)
        for i in range(n):
            img = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
            Image.fromarray(img).save(ep / cam / f"{i:06d}.png")
    x_des = np.cumsum(rng.normal(0, 0.003, (n, 3)), axis=0) + [0.3, 0.0, 0.3]
    quats = np.array([conv.quat_mul(_axis_quat([1, 0, 0], a)[None], conv.Q_DOWN)[0]
                      for a in rng.normal(0, 0.002, n)])
    quats[::3] *= -1.0          # 四元数の符号は効かない
    closed = np.zeros(n, bool)
    closed[8:15] = True
    data = {
        "ee_pos": x_des + 0.004, "ee_quat": quats,
        "fingers": np.stack([np.linspace(0.04, 0.02, n)] * 2, axis=1),
        "joints": rng.normal(0, 1, (n, 7)), "x_des": x_des, "gripper_closed": closed,
    }
    np.savez(ep / "data.npz", **data)
    meta = {"episode_id": episode_id, "n_frames": n, "record_hz": 20, "instruction": "put the cube in the box",
            "cameras": {"width": size}, "placement_id": 1, "operator": "01", "session": 1,
            "success": True, "retakes": 0}
    (ep / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return ep, meta, data


def test_episode_arrays_definitions(tmp_path):
    ep, meta, data = make_raw_episode(tmp_path, n=21)
    arr = conv.episode_arrays(meta, data)
    assert arr["raw_index"].tolist() == list(range(0, 20, 2))            # (21-1)//2 = 10
    a, s = arr["action"], arr["state"]
    assert a.dtype == np.float32 and s.dtype == np.float32
    assert np.allclose(a[:, :3], data["x_des"][2:21:2] - data["x_des"][0:20:2], atol=1e-7)
    assert np.all(a[:, 3:6] == 0)
    assert a[:, 6].tolist() == [1.0 if c else -1.0 for c in data["gripper_closed"][2:21:2]]
    assert np.allclose(s[:, :3], data["ee_pos"][0:20:2], atol=1e-6)
    assert np.allclose(s[:, 6], data["fingers"][0:20:2, 0]) and np.all(s[:, 7] <= 0)
    assert np.abs(s[:, 3:6]).max() < 0.01                                  # ほぼ真下（π に跳ばない）


def test_orientation_deviation():
    assert np.allclose(conv.orientation_deviation(conv.Q_DOWN[None]), 0.0)
    tilt = conv.quat_mul(_axis_quat([0, 1, 0], 0.1)[None], conv.Q_DOWN)
    assert np.allclose(conv.orientation_deviation(tilt), [[0, 0.1, 0]])
    assert np.allclose(conv.orientation_deviation(-tilt), [[0, 0.1, 0]])


@pytest.mark.torch
def test_convert_and_verify(tmp_path):
    raw = tmp_path / "raw"
    for eid in (0, 1):
        make_raw_episode(raw, episode_id=eid, n=21 + 4 * eid)
    out = tmp_path / "lerobot" / "unit"
    episodes = conv.raw_episodes(raw)
    conv.convert(episodes, out, "unit")
    assert conv.verify(out, tmp_path / "exports") is True
    info = json.loads((out / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["fps"] == 10 and info["codebase_version"] == "v3.0"
    assert info["features"]["observation.state"]["shape"] == [15]
    assert info["features"]["observation.images.image"]["dtype"] == "image"
    assert sorted(p.name for p in (tmp_path / "exports").iterdir()) == [
        "ep_000000_lerobot.npz", "ep_000001_lerobot.npz"]
    with pytest.raises(FileExistsError):
        conv.convert(episodes, out, "unit")


@pytest.mark.torch
def test_the_ways_of_calling_the_converter(tmp_path):
    """フォルダを並べる呼び方と --raw-dir の呼び方。"""
    raw = tmp_path / "raw"
    for i in (0, 1):
        make_raw_episode(raw, episode_id=i, n=21 + 4 * i)
    listed = conv.main(["--out", str(tmp_path / "ds_list"), "--name", "list",
                        str(raw / "2026-09-16" / "ep_000000")])
    assert listed == 0
    record = json.loads((tmp_path / "ds_list" / "meta" / "conversion.json").read_text(encoding="utf-8"))
    assert "manifest" not in record and len(record["sources"]) == 1
    assert conv.main(["--out", str(tmp_path / "ds_dir"), "--name", "dir", "--raw-dir", str(raw)]) == 0
    record = json.loads((tmp_path / "ds_dir" / "meta" / "conversion.json").read_text(encoding="utf-8"))
    assert "manifest" not in record and len(record["sources"]) == 2


@pytest.mark.torch
def test_dataset_images_are_transformed_per_view_at_value_level(tmp_path):
    """俯瞰は左右反転、手首は上下反転。期待の画像は vla_image_spec を使わず numpy の切り出しで作り、
    両方を左右反転する形とは違うことも確かめる。"""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    raw = tmp_path / "raw"
    ep, meta, data = make_raw_episode(raw, n=21)
    out = tmp_path / "ds"
    conv.convert([ep], out, "unit_views")
    ds = LeRobotDataset("local/unit_views", root=out)
    arr = conv.episode_arrays(meta, data)
    for k, i in enumerate(arr["raw_index"]):
        item = ds[k]
        for cam, key, want_fn in (("overhead", "observation.images.image", lambda a: a[:, ::-1]),
                                  ("wrist", "observation.images.image2", lambda a: a[::-1, :])):
            raw_img = np.asarray(Image.open(ep / cam / f"{i:06d}.png").convert("RGB"))
            got = np.round(item[key].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            assert np.array_equal(got, want_fn(raw_img)), (cam, k)
            if cam == "wrist":
                assert not np.array_equal(got, raw_img[:, ::-1]), "wrist still flipped left-right"
    conversion = json.loads((out / "meta" / "conversion.json").read_text(encoding="utf-8"))
    assert conversion["converter_version"] == 2
    assert conversion["image_spec_version"] == 2
    assert conversion["image_transforms"] == {"overhead": "fliplr", "wrist": "flipud"}
    assert conversion["image_keys"] == {"overhead": "observation.images.image",
                                        "wrist": "observation.images.image2"}


@pytest.mark.torch
def test_verify_writes_raw_vs_policy_image_and_rejects_version1(tmp_path, capsys):
    raw = tmp_path / "raw"
    ep, _, _ = make_raw_episode(raw, n=21, size=16)
    out = tmp_path / "ds"
    conv.convert([ep], out, "unit_check")
    check_png = tmp_path / "check.png"
    assert conv.verify(out, None, check_png) is True
    img = np.asarray(Image.open(check_png))
    assert img.shape[1] == 2 * 16 and img.shape[0] >= 2 * 16          # 2 つの視点 × [生 | 方策の入力]
    default_png = out.parent / f"{out.name}_verify_raw_vs_policy.png"
    assert conv.verify(out, None) is True and default_png.is_file()
    # 版 1 の記録（視点ごとの指定なし）は検証に落ちる
    cpath = out / "meta" / "conversion.json"
    c = json.loads(cpath.read_text(encoding="utf-8"))
    for field in ("image_spec_version", "image_transforms", "image_keys"):
        c.pop(field)
    c["converter_version"] = 1
    cpath.write_text(json.dumps(c), encoding="utf-8")
    assert conv.verify(out, None, tmp_path / "check_v1.png") is False
    assert "FAIL image spec" in capsys.readouterr().out

