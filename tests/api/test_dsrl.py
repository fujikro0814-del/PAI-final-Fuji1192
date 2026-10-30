"""VLA の初期雑音の空間での強化学習（recovla.rl.dsrl）: 雑音の形・範囲、SensorPolicy への差し込み、小さな問題で SAC が学ぶこと。"""
import types

import numpy as np
import torch

from recovla.rl import dsrl as D


def test_features_and_actor_bounds():
    f = D.features(np.arange(17, dtype=np.float32), np.array([0.1, 0.2, 1.0]), 25)
    assert f.shape == (D.FEAT_DIM,) and f[-2] == 1.0 and f[-1] == 0.5
    assert D.features(np.zeros(17), None, 0)[-2] == 0.0
    with torch.no_grad():
        w, logp = D.Actor()(torch.randn(64, D.FEAT_DIM))
    assert w.shape == (64, D.LAT_DIM) and float(w.abs().max()) <= D.B_W and torch.isfinite(logp).all()


def test_attach_passes_one_noise_repeated_over_the_chunk(tmp_path):
    got = {}

    def infer(obs, gen, rtc=None, noise=None):
        got["noise"] = noise
        return np.zeros((50, 7))

    pol = types.SimpleNamespace(policy=types.SimpleNamespace(config=types.SimpleNamespace(chunk_size=50, max_action_dim=32)),
                                last_cue=np.array([0.3, 0.0, 1.0]), infer=infer)
    steer = D.Steer(None, seed=1)
    D.attach(pol, steer)
    D.attach(pol, steer)                       # 2 回目は包み直さない
    steer.begin_episode()
    pol.infer({"observation.state": np.zeros(17, np.float32)}, None)
    pol.infer({"observation.state": np.ones(17, np.float32)}, None)
    n = got["noise"]
    assert n.shape == (1, 50, 32) and np.all(n[0] == n[0, 0]) and len(steer.rows) == 2
    assert steer.rows[1][0][-1] == 1 / 50 and np.array_equal(steer.rows[1][1], n[0, 0])


def test_save_and_reload_actor(tmp_path):
    a = D.Actor()
    p = tmp_path / "actor.pt"
    D.save_actor(a, p, {"version": 7})
    s = D.Steer(p, deterministic=True)
    assert s.meta["version"] == 7
    f = torch.randn(1, D.FEAT_DIM)
    assert torch.allclose(s.actor(f, True)[0], a(f, True)[0])


def test_sac_learns_a_one_step_bandit():
    rng = np.random.default_rng(0)
    sac = D.SAC(seed=0, batch=128)
    for _ in range(1500):
        f = rng.normal(size=(1, D.FEAT_DIM)).astype(np.float32)
        with torch.no_grad():
            w = sac.actor(torch.from_numpy(f))[0].numpy()
        sac.add_episode(f, w, float(w[0, 0] > 0.5))
        sac.update(2)
    with torch.no_grad():
        w = sac.actor(torch.randn(256, D.FEAT_DIM), True)[0]
    assert float((w[:, 0] > 0.5).float().mean()) > 0.9


def test_logp_penalises_saturation_without_a_cap():
    """上限に張り付くほど logp が上がる（1e-6 の下駄を足す形では、罰が 1 次元あたり約 14 で頭打ちになり発散した）。"""
    a = D.Actor()
    with torch.no_grad():
        for p in a.net.parameters():
            p.zero_()
        lp = []
        for m in (5.0, 10.0, 20.0):
            a.net[-1].bias[:D.LAT_DIM] = m          # 平均を大きく、std は exp(0)=1
            torch.manual_seed(0)
            lp.append(float(a(torch.zeros(1, D.FEAT_DIM), True)[1]))
    assert all(np.isfinite(lp)) and lp[0] < lp[1] < lp[2] and lp[2] > D.LAT_DIM * 30


def test_sac_stays_bounded_on_sparse_episodes():
    """27 推論の試行・最後だけ 0/1 の報酬（実際の学習と同じ形）で、Q が 0〜1 の近く、alpha が上限の内に留まる。"""
    rng = np.random.default_rng(1)
    sac = D.SAC(seed=1)
    T, info = 27, {}
    for _ in range(80):
        f = rng.normal(size=(T, D.FEAT_DIM)).astype(np.float32)
        with torch.no_grad():
            w = sac.actor(torch.from_numpy(f))[0].numpy()
        sac.add_episode(f, w, float(rng.random() < 0.6 + 0.3 * float(w[:, 0].mean())))
        if len(sac) >= 1000:
            info = sac.update(2 * T)
    assert info and all(np.isfinite(v) for v in info.values())
    assert -0.05 <= info["q_mean"] <= 1.05 and info["alpha"] <= D.ALPHA_MAX * 1.0001 and info["w_sat"] < 0.5
    assert np.allclose(sac.G[-T:], D.GAMMA ** np.arange(T - 1, -1, -1) * sac.R[-1])
