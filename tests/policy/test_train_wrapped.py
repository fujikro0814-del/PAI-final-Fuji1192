"""学習の包み（recovla.policy.train_wrapped）の部品: エピソードの始めのフレームを重くする並べ方、起動器の LoRA・重みの設定、
目標位置の入力のずらし、表からの速い読み出し。"""
import json
import re

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("lerobot")

from recovla.policy import train_launcher as tl                     # noqa: E402
from recovla.policy.train_wrapped import WeightedEpisodeAwareSampler   # noqa: E402

pytestmark = pytest.mark.torch

# 学習データに近い形: 300 本、1 本 120〜170 フレーム
RNG = np.random.default_rng(0)
LENGTHS = RNG.integers(120, 171, size=300)
TO = np.cumsum(LENGTHS)
FROM = TO - LENGTHS


def sampler(weight=5.0, frames=20, seed=1000):
    return WeightedEpisodeAwareSampler(FROM.tolist(), TO.tolist(), shuffle=True, seed=seed,
                                       first_frames=frames, weight=weight)


def test_expected_share_formula():
    s = sampler()
    n_first = 20 * len(LENGTHS)
    n_rest = int(LENGTHS.sum()) - n_first
    assert s.expected_first_share == pytest.approx(5 * n_first / (5 * n_first + n_rest))
    assert s.stats()["uniform_first_share"] == pytest.approx(n_first / LENGTHS.sum())


def test_drawn_share_matches_expected():
    """実際に引かれた「最初の 2 秒のフレーム」の割合が期待値に合う（3 周、二項の 4 標準偏差以内）。"""
    s = sampler()
    idx = [i for _ in range(3) for i in s]
    n = len(idx)
    assert n == 3 * int(LENGTHS.sum())                                  # 1 周の長さは全フレーム数のまま
    p = s.expected_first_share
    got = s.stats()["drawn_first_share"]
    assert abs(got - p) < 4 * np.sqrt(p * (1 - p) / n), (got, p)
    # 引いたフレームを、エピソードの中の位置に戻して数え直しても同じ割合になる（数え方の検算）
    ep = np.searchsorted(TO, np.asarray(idx), side="right")
    pos = np.asarray(idx) - FROM[ep]
    assert (pos < 20).mean() == pytest.approx(got)
    assert (pos >= 0).all() and (pos < LENGTHS[ep]).all()


def test_weight_one_is_uniform():
    s = sampler(weight=1.0)
    idx = [i for i in s]
    p = s.stats()["uniform_first_share"]
    assert abs(s.stats()["drawn_first_share"] - p) < 4 * np.sqrt(p * (1 - p) / len(idx))


def test_deterministic_per_seed_and_epoch():
    a, b = sampler(), sampler()
    ea1, eb1 = list(a), list(b)
    assert ea1 == eb1                                                   # 同じシード・同じ周回なら同じ列
    assert list(a) != ea1                                               # 次の周回は別の列
    assert list(sampler(seed=1001)) != ea1


# --- 起動器 -----------------------------------------------------------------------------------------------

LORA = {"layers": list(range(8, 16)), "modules": ["q_proj", "v_proj"], "r": 16, "alpha": 32}
MODULE_NAMES = [f"model.vlm_with_expert.vlm.model.text_model.layers.{i}.self_attn.{m}"
                for i in range(16) for m in ("q_proj", "k_proj", "v_proj", "o_proj")]


def config(tmp_path, **over):
    cfg = {"dataset": str(tmp_path / "ds"), "train_scope": "expert", "batch_size": 32, "steps": 50, **over}
    p = tmp_path / "c.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return tl.load_config(p)


def test_lora_flags_target_exactly_layers_8_to_15_q_v(tmp_path):
    cfg = config(tmp_path, lora=LORA, first_frames_weight={"frames": 20, "weight": 5})
    cmd = tl.build_command(tl.wrapper_prefix(cfg, python="py"), "SNAP", cfg, tmp_path / "out", "job")
    cut = cmd.index("--")
    assert cmd[:cut] == ["py", "-m", "recovla.policy.train_wrapped", "--reference", "--first-frames=20",
                         "--first-weight=5"]
    args = dict(a.split("=", 1) for a in cmd[cut + 1:])
    assert len(args) == len(cmd) - cut - 1                              # 同じ鍵を 2 度渡さない
    target = args["--peft.target_modules"]
    hits = [n for n in MODULE_NAMES if re.fullmatch(target, n)]
    assert hits == [f"model.vlm_with_expert.vlm.model.text_model.layers.{i}.self_attn.{m}"
                    for i in range(8, 16) for m in ("q_proj", "v_proj")]
    assert json.loads(args["--peft.full_training_modules"]) == tl.FULL_TRAINING_MODULES
    assert args["--peft.r"] == "16" and args["--peft.lora_alpha"] == "32" and args["--peft.method_type"] == "LORA"
    assert args["--policy.train_expert_only"] == "true"                # VLM 本体は固定のまま
    assert not any(a.startswith("--policy.optimizer_lr") for a in cmd)  # 学習率は変えない


@pytest.mark.parametrize("over, words", [
    ({"lora": {"layers": [16], "modules": ["q_proj"], "r": 16, "alpha": 32}}, "lora.layers"),
    ({"lora": {"layers": [8], "modules": ["gate"], "r": 16, "alpha": 32}}, "lora.modules"),
    ({"lora": {"layers": [8], "modules": ["q_proj"], "r": 0, "alpha": 32}}, "lora.r"),
    ({"lora": {"layers": [8], "modules": ["q_proj"], "r": 16}}, "lora must be"),
    ({"first_frames_weight": {"frames": 0, "weight": 5}}, "first_frames_weight"),
    ({"extra_args": ["--peft.r=8"]}, "--peft.r"),
])
def test_lora_config_rejects(tmp_path, over, words):
    with pytest.raises(tl.LaunchError, match=re.escape(words)):
        config(tmp_path, **over)


def test_plain_config_is_not_wrapped(tmp_path):
    assert not tl.wrapped(config(tmp_path))


# --- 目標位置の入力のずらし（学習データの増やし方） --------------------------------------------------------------

from recovla.policy.train_wrapped import CueAugment   # noqa: E402


def test_cue_augment_distribution_matches_spec():
    """足したずれの分布が指定どおり: 確率 0.5、大きさ U(0, 2 cm)、向き一様（二項・一様の 4 標準偏差以内）。"""
    aug = CueAugment(1000, 0.5, 0.02, (15, 16), (0.07, 0.1))
    offs, ons = [], []
    for _ in range(400):
        o, on = aug.draw(32)
        offs.append(o)
        ons.append(on)
    off, on = np.concatenate(offs), np.concatenate(ons)
    n = len(on)
    assert abs(on.mean() - 0.5) < 4 * np.sqrt(0.25 / n)
    assert np.all(off[~on] == 0.0)
    mag = np.linalg.norm(off[on], axis=1)
    m = on.sum()
    assert mag.min() >= 0 and mag.max() <= 0.02
    assert abs(mag.mean() - 0.01) < 4 * 0.02 / np.sqrt(12 * m)                 # 一様の平均と標準誤差
    hist = np.histogram(mag, bins=10, range=(0, 0.02))[0]
    assert np.all(np.abs(hist - m / 10) < 4 * np.sqrt(m / 10))
    ang = np.arctan2(off[on, 1], off[on, 0]) % (2 * np.pi)
    ah = np.histogram(ang, bins=12, range=(0, 2 * np.pi))[0]
    assert np.all(np.abs(ah - m / 12) < 4 * np.sqrt(m / 12))
    s = aug.stats()
    assert s["samples"] == n and s["applied"] == m and sum(s["magnitude_hist_20"]) == m
    # 同じシードなら同じ列
    a2 = CueAugment(1000, 0.5, 0.02, (15, 16), (0.07, 0.1))
    assert np.array_equal(a2.draw(32)[0], offs[0])


def test_cue_augment_touches_only_cue_dims_in_normalized_space():
    import torch
    aug = CueAugment(7, 1.0, 0.02, (15, 16), (0.05, 0.1))
    st = torch.zeros((4, 17))
    out = aug.apply({"observation.state": st.clone(), "action": torch.ones((4, 50, 7))})
    d = out["observation.state"]
    assert torch.all(d[:, :15] == 0)                                             # 目標位置の入力以外は変えない
    assert torch.all(out["action"] == 1)                                         # 学習データの行動は変えない
    rng = np.random.default_rng(np.random.SeedSequence([7, 54]))
    on = rng.random(4) < 1.0
    mag, ang = rng.uniform(0, 0.02, 4), rng.uniform(0, 2 * np.pi, 4)
    assert np.allclose(d[:, 15].numpy(), np.cos(ang) * mag / 0.05, atol=1e-6)   # 標準偏差で割って足す
    assert np.allclose(d[:, 16].numpy(), np.sin(ang) * mag / 0.1, atol=1e-6)


def test_launcher_passes_fast_query(tmp_path):
    cfg = config(tmp_path, fast_query=True)
    assert tl.wrapped(cfg)
    pre = tl.wrapper_prefix(cfg, python="py")
    assert "--fast-query" in pre and pre[-1] == "--"
    assert not tl.wrapped(config(tmp_path, fast_query=False))
    with pytest.raises(tl.LaunchError, match="fast_query"):
        config(tmp_path, fast_query=1)


def test_fast_query_matches_lerobot_path():
    """行動・状態の表からの読み出しが、LeRobot の元の経路と型・形・値まで同じ。画像の列は元の経路へ回す。"""
    import types
    import datasets
    import torch
    from lerobot.datasets.dataset_reader import DatasetReader
    from lerobot.datasets.io_utils import hf_transform_to_torch
    from recovla.policy import train_wrapped as tw
    rng = np.random.default_rng(3)
    n = 40
    hf = datasets.Dataset.from_dict({"action": rng.normal(size=(n, 7)).tolist(),
                                     "observation.state": rng.normal(size=(n, 17)).tolist(),
                                     "index": list(range(n)), "flag": rng.integers(0, 5, n).tolist()})
    hf.set_transform(hf_transform_to_torch)
    meta = types.SimpleNamespace(video_keys=[], features={"action": {"dtype": "float32"},
                                                           "observation.state": {"dtype": "float32"},
                                                           "flag": {"dtype": "int64"}})
    reader = types.SimpleNamespace(hf_dataset=hf, _meta=meta, _absolute_to_relative_idx=None)
    q = {"action": [5, 6, 7, 7, 7], "observation.state": [9], "flag": [0, 39]}
    want = DatasetReader._query_hf_dataset(reader, q)
    got = tw.fast_query_hf_dataset(reader, q)
    assert set(got) == set(want)
    for k in want:
        assert got[k].dtype == want[k].dtype and got[k].shape == want[k].shape and torch.equal(got[k], want[k])
    reader._absolute_to_relative_idx = {i + 100: i for i in range(n)}            # エピソードを選んだときの番号の付け替え
    q2 = {"action": [110, 111]}
    assert torch.equal(tw.fast_query_hf_dataset(reader, q2)["action"], DatasetReader._query_hf_dataset(reader, q2)["action"])


def test_launcher_passes_cue_augment(tmp_path):
    cfg = config(tmp_path, cue_augment={"prob": 0.5, "max_m": 0.02})
    assert tl.wrapped(cfg)
    pre = tl.wrapper_prefix(cfg, python="py")
    assert "--cue-aug-prob=0.5" in pre and "--cue-aug-max-m=0.02" in pre and pre[-1] == "--"
    with pytest.raises(tl.LaunchError, match="cue_augment"):
        config(tmp_path, cue_augment={"prob": 0.5, "max_m": 0.5})
