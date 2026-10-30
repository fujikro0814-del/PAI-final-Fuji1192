"""行動列の候補から成功の見込みで選ぶ（条件 Udc）。VLA を固定し、推論ごとにふつうの雑音から K 本の行動列を作り、
学習した成功の見込みが最も高い 1 本を選ぶ。V-GPS（Nakamoto ほか、CoRL 2024）の「方策の出した行動を価値で選び直す」を、
自分の試行から学ぶ形にしたもの。見込みは雑音でなく行動列の上で学ぶ（DSRL-NA、Wagenmaker ほか 2025 と同じ）。

    sel = Select(scorer_path, k=8)               # 見込みの重み（無ければ選ばない＝候補 0 だけ）
    attach(sensor_policy, sel)                    # SensorPolicy.infer を、K 本作って選ぶものに替える
    sel.begin_episode(seed); ...; sel.rows        # 推論ごとの (観測の特徴, 選んだ行動列の特徴, 選んだ番号)
    sc = Scorer; fit(sc, X, y, groups)          # 試行の成功（0/1）を当てる分類（5 つの網の集まり）

- 候補はすべて、VLA の学習と同じふつうの雑音から作る（学習の分布から外れない）。候補 0 は元の VLA と同じ乱数
- 見込みは「その行動列で試行が成功したか」の分類で、TD を使わない（見積もりの誤差が膨らまない）
- 5 つの網の平均から、ばらつきを KAPPA 倍引いて選ぶ（自信のない候補を避ける）
- 見込みの入力は少しの量（手先・目標位置の入力との差・行動列の 5/10/20 行後までの動き・その後の目標位置の入力との差・グリッパ）。
  状態を全部入れると「どの試行か」を覚えてしまい、行動列の違いを学ばない
- 試行の CONTROL_FRAC は、シードで決まる対照（選ばない＝元の VLA そのもの）。同じ条件で選ぶ効果を比べられる
"""
import io
import os
import pathlib
import time

import numpy as np
import torch
import torch.nn as nn


K_CAND = 8
GRIP_ROWS = (0, 4, 8, 12, 16)         # 行動列の先の 20 行のうち、グリッパの指令を見る行
STATE_FEAT = 11                       # 手先 xyz・向きのずれ 3・指・目標位置の入力の旗・推論の番号・目標位置の入力 − 手先 (x, y)
ACT_FEAT = 9 + 4 + len(GRIP_ROWS)     # 5・10・20 行後までの動き (xyz × 3)・その後の目標位置の入力との差 (xy × 2)・グリッパ
SEL_FEAT = STATE_FEAT + ACT_FEAT
SC_HIDDEN = 128
ENSEMBLE = 5
KAPPA = 0.5
CONTROL_FRAC = 0.25
MIN_EPISODES = 40                     # これだけ溜まるまで見込みの重みを書かない（その間は選ばない）


def state_features(state, i: int) -> np.ndarray:
    """方策の状態（手先 3・向きのずれ 3・指 2・関節 7・目標位置の入力 x y [旗]）→ STATE_FEAT。
    関節や絶対の位置は入れない（見込みが「どの試行か」を覚えてしまう）。"""
    s = np.asarray(state, np.float32).reshape(-1)
    flag = float(s[17]) if len(s) > 17 else 1.0
    return np.concatenate([s[0:3], s[3:6], s[6:7], [flag, min(int(i), 200) / 50.0], s[15:17] - s[0:2]]).astype(np.float32)


def chunk_features(post, state) -> np.ndarray:
    """後処理の後の行動列 (H, 7)（dx dy dz [m]、回転 0、グリッパ ±1）→ ACT_FEAT。"""
    p = np.asarray(post, np.float32)
    s = np.asarray(state, np.float32).reshape(-1)
    cum = np.cumsum(p[:20, :3], 0)
    d = cum[[4, 9, 19]]
    rel = s[15:17][None] - (s[0:2][None] + d[1:, :2])
    return np.concatenate([d.reshape(-1), rel.reshape(-1), p[list(GRIP_ROWS), 6]]).astype(np.float32)


def is_control(seed: int) -> bool:
    """シードで決まる対照（選ばない試行）。実行の順・機械によらない。"""
    return ((int(seed) * 2654435761) % 2 ** 32) / 2 ** 32 < CONTROL_FRAC


class Scorer(nn.Module):
    """成功の見込みの分類の集まり。入力は平均・標準偏差で揃えてから使う（重みと一緒に保存）。"""

    def __init__(self, n: int = ENSEMBLE):
        super().__init__()
        self.nets = nn.ModuleList(
            nn.Sequential(nn.Linear(SEL_FEAT, SC_HIDDEN), nn.LayerNorm(SC_HIDDEN), nn.ReLU(),
                          nn.Linear(SC_HIDDEN, SC_HIDDEN), nn.ReLU(), nn.Linear(SC_HIDDEN, 1)) for _ in range(n))
        self.register_buffer("mu", torch.zeros(SEL_FEAT))
        self.register_buffer("sd", torch.ones(SEL_FEAT))

    def logits(self, x) -> torch.Tensor:
        z = (x - self.mu) / self.sd
        return torch.stack([m(z).squeeze(-1) for m in self.nets])          # (E, N)

    @torch.no_grad()
    def score(self, x, kappa: float = KAPPA):
        p = torch.sigmoid(self.logits(x))
        return p.mean(0) - kappa * p.std(0, unbiased=False), p.mean(0)


def boot_weights(seed: int, n: int = ENSEMBLE) -> np.ndarray:
    """試行ごとの、網ごとの重み（Poisson(1)、シードで決まる）。網ごとに違う記録を見る（ブートストラップ）。"""
    return np.random.default_rng(int(seed) + 7919).poisson(1.0, n).astype(np.float64)


def fit(sc: Scorer, X: np.ndarray, y: np.ndarray, W: np.ndarray, steps: int, rng, opts, batch: int = 512) -> float:
    """各網を、遷移の重み W (E, N)（試行の Poisson の重み）で引いた記録で学ぶ。opts は網ごとの AdamW（続けて学ぶため持ち回す）。"""
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    loss = 0.0
    for e, net in enumerate(sc.nets):
        cdf = np.cumsum(W[e])
        if cdf[-1] <= 0:
            continue
        for _ in range(int(steps)):
            idx = torch.from_numpy(np.minimum(np.searchsorted(cdf, rng.random(batch) * cdf[-1], side="right"), len(X) - 1))
            z = (Xt[idx] - sc.mu) / sc.sd
            lo = nn.functional.binary_cross_entropy_with_logits(net(z).squeeze(-1), yt[idx])
            opts[e].zero_grad()
            lo.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opts[e].step()
            loss = float(lo.detach())
    return loss


def make_opts(sc: Scorer, lr: float = 1e-3):
    return [torch.optim.AdamW(n.parameters(), lr=lr, weight_decay=1e-2) for n in sc.nets]


def save_scorer(sc: Scorer, path, meta: dict) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    torch.save({"state_dict": sc.state_dict(), "meta": meta}, buf)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(buf.getvalue())
    os.replace(tmp, path)


def auc(p, y) -> float:
    """順位で数える AUC（同順位は半分）。片方の組が無ければ nan。"""
    p, y = np.asarray(p, float), np.asarray(y, float)
    pos, neg = p[y > 0.5], p[y <= 0.5]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


class Select:
    """推論ごとに K 本の候補から 1 本を選び、(特徴, 選んだ行動列の特徴, 番号, 見込み) を溜める。"""

    def __init__(self, scorer_path=None, k: int = K_CAND, kappa: float = KAPPA, control: str = "seed"):
        self.path = None if scorer_path is None else pathlib.Path(scorer_path)
        self.k, self.kappa = int(k), float(kappa)
        self.control_mode = control             # "seed"（シードで決める）・"never"（評価: いつも選ぶ）・"always"（いつも選ばない）
        self.sc, self.mtime = None, None
        self.meta = {"version": "none"}
        self.rows, self.control, self.seed = [], False, 0
        self.reload()

    def reload(self) -> bool:
        if self.path is not None and str(self.path) == "random":     # 動作確認用: 学ぶ前の重みで選ぶ
            if self.sc is None:
                self.sc, self.meta = Scorer().eval(), {"version": "random"}
            return False
        if self.path is None or not self.path.is_file():
            return False
        m = self.path.stat().st_mtime
        if m == self.mtime:
            return False
        for _ in range(5):
            try:
                d = torch.load(self.path, map_location="cpu", weights_only=False)
                break
            except Exception:
                time.sleep(0.5)
        else:
            return False
        sc = Scorer(len([k for k in d["state_dict"] if k.endswith(".0.weight")]))
        sc.load_state_dict(d["state_dict"])
        self.sc, self.meta, self.mtime = sc.eval(), d.get("meta", {}), m
        return True

    def begin_episode(self, seed: int) -> None:
        self.rows, self.seed = [], int(seed)
        self.control = {"seed": is_control(seed), "never": False, "always": True}[self.control_mode]

    @property
    def choosing(self) -> bool:
        return self.sc is not None and not self.control

    def pick(self, state, posts: np.ndarray) -> int:
        """state: 方策の状態（目標位置の入力込み）、posts: 候補の後処理の後の行動列 (K, H, 7)。選んだ番号を返す。"""
        f = state_features(state, len(self.rows))
        x = np.stack([np.concatenate([f, chunk_features(p, state)]) for p in posts]).astype(np.float32)
        s = pm = None
        j = 0
        if self.choosing and len(posts) > 1:
            s, pm = self.sc.score(torch.from_numpy(x), self.kappa)
            j = int(torch.argmax(s))
        self.rows.append({"f": f, "a": x[j, STATE_FEAT:], "j": j, "k": len(posts),
                          "p": None if pm is None else float(pm[j]), "p_mean": None if pm is None else float(pm.mean())})
        return j


def attach(pol, sel: Select) -> None:
    """SensorPolicy の infer を、K 本作って選ぶものに替える。候補 0 は元の VLA と同じ乱数。選ばない試行は 1 本だけ作る。"""
    pol._select = sel
    if getattr(pol, "_select_attached", False):
        return
    pol._select_attached = True

    def infer(obs, noise_generator, rtc_kwargs=None):
        t, cfgp, s = pol.torch, pol.policy.config, pol._select
        batch = {k: t.from_numpy(v) for k, v in obs.items() if isinstance(v, np.ndarray)}
        batch["task"] = obs["task"]
        batch = pol.pre(batch)
        if pol.input_check is None:
            pol.input_check = pol._check_inputs(obs, batch)
        H, A = cfgp.chunk_size, cfgp.max_action_dim
        k = s.k if s.choosing else 1
        noise = t.randn((1, H, A), generator=noise_generator)
        if k > 1:
            g = t.Generator().manual_seed(int(s.seed) * 1000 + len(s.rows))
            noise = t.cat([noise, t.randn((k - 1, H, A), generator=g)])
        noise = noise.to(pol.device)
        bk = {}
        for key, v in batch.items():
            if k > 1 and t.is_tensor(v) and v.dim() > 0 and v.shape[0] == 1:
                v = v.repeat(k, *([1] * (v.dim() - 1)))
            elif k > 1 and isinstance(v, list) and len(v) == 1:
                v = v * k
            bk[key] = v
        kw = {}
        if rtc_kwargs:
            prev = t.as_tensor(rtc_kwargs["prev_chunk_left_over"], dtype=t.float32)[None].to(pol.device)
            kw = {"inference_delay": int(rtc_kwargs["inference_delay"]), "prev_chunk_left_over": prev.repeat(k, 1, 1)}
        with t.no_grad():
            chunk = pol.policy.predict_action_chunk(bk, noise=noise, **kw)
        raws = chunk.detach().cpu().numpy()
        posts = pol.post(chunk).detach().cpu().numpy().reshape(k, -1, chunk.shape[-1])[:, :, :7].astype(np.float64)
        st = np.asarray(obs["observation.state"], np.float32).reshape(-1)
        if len(st) == 17:                  # 状態に見えているかの旗が無いモデル: 旗は last_cue から
            st = np.concatenate([st, [1.0 if pol.last_cue is None else float(np.asarray(pol.last_cue).reshape(-1)[2])]])
        j = s.pick(st, posts)
        pol.last_raw = raws[j]
        return posts[j]

    pol.infer = infer


def episode_arrays(rows, reward: float, control: bool) -> dict:
    return {"feat": np.stack([r["f"] for r in rows]).astype(np.float32),
            "act": np.stack([r["a"] for r in rows]).astype(np.float32),
            "j": np.array([r["j"] for r in rows], np.int16), "k": np.array([r["k"] for r in rows], np.int16),
            "p": np.array([np.nan if r["p"] is None else r["p"] for r in rows], np.float32),
            "reward": np.float32(reward), "control": np.bool_(control)}
