"""乱数のシードから乱数列を作る決まり（common/seeds.py）の検査。numpy だけで通る。"""
import numpy as np

from recovla.common import seeds


def test_stream_is_the_named_spawn_key():
    for name, sid in seeds.STREAM_ID.items():
        want = np.random.Generator(np.random.PCG64(np.random.SeedSequence(123, spawn_key=(sid,)))).random(5)
        assert np.array_equal(seeds.stream(123, name).random(5), want)
    assert seeds.STREAM_ID == {"layout": 0, "induce": 1, "noise": 2, "script": 3, "inject": 4, "order": 5,
                               "sensor_setup": 6, "sensor_frame": 7, "latency": 8}


def test_streams_are_independent_and_reproducible():
    a, b = seeds.streams(7), seeds.streams(7)
    draws = {n: a[n].random(4) for n in a}
    assert all(np.array_equal(draws[n], b[n].random(4)) for n in b)
    vals = [tuple(v) for v in draws.values()]
    assert len(set(vals)) == len(vals)


def test_torch_seed_is_generate_state_uint64():
    ss = seeds.seed_sequence(110003, "noise", 4)
    assert ss.spawn_key == (2, 4)
    s = seeds.torch_seed(ss)
    assert isinstance(s, int) and 0 <= s < 2 ** 64
    assert s == int(np.random.SeedSequence(110003, spawn_key=(2, 4)).generate_state(1, dtype=np.uint64)[0])
    assert s != seeds.torch_seed(seeds.seed_sequence(110003, "noise", 5))


def test_script_and_inject_keys():
    want = np.random.Generator(np.random.PCG64(np.random.SeedSequence(20000, spawn_key=(3, 2, 1)))).random(3)
    assert np.array_equal(seeds.script_rng(20000, "blue", 1).random(3), want)
    want = np.random.Generator(np.random.PCG64(np.random.SeedSequence(20000, spawn_key=(4, 0, 0)))).random(3)
    assert np.array_equal(seeds.inject_rng(20000, "red", 0).random(3), want)
