"""行動列の候補を K 本作り、学習した成功の見込みで選ぶ仕組み（recovla.rl.select）の小さな確かめ。VLA は使わない。"""
import types

import numpy as np
import torch

from recovla.rl import select as S


def _state(rng=None):
    s = np.zeros(18, np.float32) if rng is None else rng.normal(size=18).astype(np.float32)
    s[17] = 1.0
    return s


def test_control_fraction_and_shapes():
    frac = np.mean([S.is_control(s) for s in range(80000, 90000)])
    assert abs(frac - S.CONTROL_FRAC) < 0.02
    st = _state()
    st[15:17] = [0.3, 0.1]                                  # 目標位置の入力
    post = np.zeros((50, 7))
    post[:, 0] = 0.01                                       # x に 1 行 1 cm
    a = S.chunk_features(post, st)
    assert a.shape == (S.ACT_FEAT,) and S.state_features(st, 3).shape == (S.STATE_FEAT,)
    assert np.isclose(a[0], 0.05) and np.isclose(a[6], 0.20)          # 5 行後・20 行後の x の動き
    assert np.isclose(a[9], 0.3 - 0.10) and np.isclose(a[11], 0.3 - 0.20)   # 10・20 行後の、目標位置の入力との x の差
    assert S.auc([0.9, 0.8, 0.1], [1, 1, 0]) == 1.0
    assert np.isnan(S.auc([0.5], [1]))


def _data(rng, n_ep=1500):
    X, y, W = [], [], []
    for e in range(n_ep):
        x = rng.normal(size=(1, S.SEL_FEAT)).astype(np.float32)
        r = float(x[0, S.STATE_FEAT] > 0)                   # 成功は行動列の特徴の 0 番の向きで決まる（1 推論の試行）
        X.append(x)
        y.append(np.full(1, r, np.float32))
        W.append(S.boot_weights(e)[:, None])
    return np.concatenate(X), np.concatenate(y), np.concatenate(W, 1)


def test_scorer_learns_a_rule():
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    X, y, W = _data(rng)
    sc = S.Scorer()
    sc.mu.copy_(torch.from_numpy(X.mean(0)))
    sc.sd.copy_(torch.from_numpy(X.std(0) + 1e-3))
    S.fit(sc, X, y, W, steps=300, rng=rng, opts=S.make_opts(sc))
    Xv, yv, _ = _data(np.random.default_rng(1), n_ep=500)
    _, p = sc.score(torch.from_numpy(Xv))
    assert S.auc(p.numpy(), yv) > 0.8


class _FakePolicy:
    """SensorPolicy の形だけ。行動列 = 雑音（前処理・後処理はそのまま）。"""

    def __init__(self):
        self.torch, self.device = torch, "cpu"
        self.policy = types.SimpleNamespace(config=types.SimpleNamespace(chunk_size=50, max_action_dim=32),
                                            predict_action_chunk=lambda b, noise, **kw: noise + 0 * b["observation.state"][:, :1, None])
        self.pre = self.post = lambda x: x
        self.input_check, self.last_cue, self.last_raw = {}, None, None

    def _check_inputs(self, obs, batch):
        return {}


class _Fixed(S.Scorer):
    """5 行後の x の動きが大きいほど見込みが高い。"""

    def score(self, x, kappa=S.KAPPA):
        v = x[:, S.STATE_FEAT]
        return v, torch.sigmoid(v)


def test_attach_control_is_the_base_and_select_takes_the_best():
    pol = _FakePolicy()
    sel = S.Select(None, k=8)
    S.attach(pol, sel)
    obs = {"observation.state": _state()[None], "task": "t"}
    base = torch.randn((1, 50, 32), generator=torch.Generator().manual_seed(3))[0].numpy()
    ctrl = next(s for s in range(80000, 80100) if S.is_control(s))
    sel.begin_episode(ctrl)
    out = pol.infer(obs, torch.Generator().manual_seed(3))
    assert np.allclose(pol.last_raw, base) and np.allclose(out, base[:, :7])
    assert sel.rows[-1]["k"] == 1

    sel.sc = _Fixed()
    other = next(s for s in range(80000, 80100) if not S.is_control(s))
    sel.begin_episode(other)
    out = pol.infer(obs, torch.Generator().manual_seed(3))
    assert sel.rows[-1]["k"] == 8
    assert np.allclose(out, pol.last_raw[:, :7])
    assert sel.rows[-1]["a"][0] >= base[:5, 0].sum() - 1e-6          # 候補 0（土台）より悪くない


def test_untrained_random_scorer_runs_the_k_path():
    sel = S.Select("random", k=4, control="never")
    sel.begin_episode(80000)
    assert sel.choosing
    j = sel.pick(_state(np.random.default_rng(0)), np.random.default_rng(1).normal(size=(4, 50, 7)))
    assert 0 <= j < 4 and sel.rows[-1]["k"] == 4
