"""torch が要る検査: 推論ごとの雑音と、LeRobot 同梱の rtc.py との照合。

torch が無い環境では飛ばす。
"""
import numpy as np
import pytest

pytestmark = pytest.mark.torch   # torch が要る検査の目印
torch = pytest.importorskip("torch")
rtc = pytest.importorskip("lerobot.rollout.inference.rtc")

from recovla.policy.schedule import RuntimeConfig, normalize_left_over, noise_generator  # noqa: E402
from tests.policy.mock_policy import A, chunk, run  # noqa: E402

H = 50
N = 200
SEED = 198000               # 検査用のシード
NOISE_SHAPE = (1, H, 32)


def draw(seed, i):
    return torch.randn(NOISE_SHAPE, generator=noise_generator(seed, i))


# 2. 乱数 ----------------------------------------------------------------------------------------

def test_same_seed_and_i_give_the_same_noise():
    for i in (0, 1, 7, 19):
        assert torch.equal(draw(SEED, i), draw(SEED, i))
    torch.manual_seed(0)
    a = draw(SEED, 3)
    torch.manual_seed(1)                                # 大域のシードに左右されない
    assert torch.equal(a, draw(SEED, 3))


def test_different_i_or_seed_give_different_noise():
    noises = [draw(SEED, i) for i in range(20)]
    for a in range(20):
        for b in range(a + 1, 20):
            assert not torch.equal(noises[a], noises[b])
    assert not torch.equal(draw(SEED, 0), draw(SEED + 1, 0))


def test_generator_is_cpu_and_matches_the_seed_sequence():
    g = noise_generator(SEED, 5)
    assert g.device.type == "cpu"
    ss = np.random.SeedSequence(SEED, spawn_key=(2, 5))
    assert g.initial_seed() == int(ss.generate_state(1, dtype=np.uint64)[0])


def test_i_th_noise_is_the_same_across_modes():
    """方式ごとに推論の回数が違っても、i 回目の推論どうしは同じ雑音になる。"""
    cfgs = [RuntimeConfig("sync", 50, None), RuntimeConfig("sync", 10, None),
            RuntimeConfig("naive", 10, 2), RuntimeConfig("rtc", 10, 2), RuntimeConfig("rtc", 10, 0)]
    per_mode = []
    for cfg in cfgs:
        tr = run(cfg, N, resets=[123])
        assert [e["i"] for e in tr.log] == list(range(len(tr.log)))    # 推論の番号は 0 から3 個の連続タスク
        per_mode.append({e["i"]: draw(SEED, e["i"]) for e in tr.log})
    for a in per_mode:
        for b in per_mode:
            for i in set(a) & set(b):
                assert torch.equal(a[i], b[i])


# 3. rtc.py の _normalize_prev_actions_length との照合 ---------------------------------------------

@pytest.mark.parametrize("E", [10, 40])
@pytest.mark.parametrize("steps", [0, 1, 5, 9, 10, 11, 25, 39, 40, 41, 50])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_normalize_equals_rtc_py(E, steps, dtype):
    L = chunk(2)[H - steps:].astype(dtype)
    ours = normalize_left_over(L, E)
    theirs = rtc._normalize_prev_actions_length(torch.from_numpy(L.copy()), target_steps=E)
    assert ours.shape == tuple(theirs.shape) and ours.dtype == theirs.numpy().dtype
    np.testing.assert_array_equal(ours, theirs.numpy())


@pytest.mark.parametrize("E", [10, 40])
@pytest.mark.parametrize("s,d", [(10, 0), (10, 2), (10, 4), (25, 3), (40, 10), (50, 0)])
def test_schedule_left_over_equals_rtc_py(s, d, E):
    """Schedule が渡す残りが、前の行動列[j:H] を rtc.py の関数で揃えたものと同じ。"""
    tr = run(RuntimeConfig("rtc", s, d, execution_horizon=E), N)
    infers = [dec for dec in tr.decisions if dec.infer]
    assert len(infers) == len(tr.log) >= 2
    for i in range(1, len(tr.log)):
        prev, entry = tr.log[i - 1], tr.log[i]
        j = prev["offset"] + (entry["k_obs"] - prev["k_valid"])
        theirs = rtc._normalize_prev_actions_length(torch.from_numpy(chunk(i - 1)[j:H]), target_steps=E)
        assert infers[i].left_over.shape == (E, A)
        np.testing.assert_array_equal(infers[i].left_over, theirs.numpy())
