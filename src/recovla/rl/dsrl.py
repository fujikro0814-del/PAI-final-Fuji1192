"""VLA（SmolVLA）を固定し、推論ごとの初期雑音を小さな方策で選ぶ強化学習（DSRL、Wagenmaker ほか arXiv:2506.15799）。
試したが採用しなかった方式（成功率が下がった）。

    steer = Steer(actor_path)                     # 小さな方策（無ければ初期の重み）。reload で新しい重みを読み直す
    attach(sensor_policy, steer)                  # SensorPolicy.infer の雑音を steer の選んだものに替える
    steer.begin_episode; ...; rows = steer.rows # 推論ごとの (観測の特徴, 選んだ雑音 w)
    learner = SAC; learner.add_episode(feats, ws, reward); learner.update(n)

- 行動: 雑音 w（LAT_DIM = 32、tanh × B_W）。行動列の 50 行すべてに同じ w を使う（原論文と同じ）
- 観測の特徴: 方策の状態（17 次元: 手先・向き・指・関節・目標位置の入力の x y）＋目標位置の入力が見えているか＋推論の番号 / 50
- 学び方: SAC（2 つの Q、温度は自動）。報酬は試行の最後に成功なら 1、それ以外 0
- 安定のため: tanh の補正は数値の安定した形、Q の目標にはエントロピーを入れず [0, 1] に切る（報酬が 0/1 のため）、
  Huber の損失と勾配の上限、alpha は小さく始めて上限 ALPHA_MAX
"""
import io
import json
import os
import pathlib
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LAT_DIM = 32
B_W = 1.0
STATE_DIM = 17
FEAT_DIM = STATE_DIM + 2
GAMMA = 0.99
HIDDEN = 256
ALPHA_INIT = 0.01
ALPHA_MAX = 0.1
GRAD_CLIP = 1.0
MC_MIX = 0.5             # Q の目標に混ぜる、試行の結果そのもの（γ^(残りの推論) × 報酬）の割合。Q の高すぎる見積もりを抑える


def features(state, cue, i: int) -> np.ndarray:
    """方策の状態（17）＋目標位置の入力の旗（見えていれば 1）＋推論の番号 / 50 → FEAT_DIM の float32。"""
    s = np.asarray(state, np.float32).reshape(-1)[:STATE_DIM]
    flag = 0.0 if cue is None else float(np.asarray(cue).reshape(-1)[2])
    return np.concatenate([s, [flag, min(int(i), 200) / 50.0]]).astype(np.float32)


def mlp(n_in: int, n_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(n_in, HIDDEN), nn.LayerNorm(HIDDEN), nn.ReLU(),
                         nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
                         nn.Linear(HIDDEN, n_out))


class Actor(nn.Module):
    """tanh で押さえた正規分布。出力は [-B_W, B_W]^LAT_DIM。"""

    def __init__(self):
        super().__init__()
        self.net = mlp(FEAT_DIM, 2 * LAT_DIM)

    def forward(self, f, deterministic: bool = False):
        mu, log_std = self.net(f).chunk(2, dim=-1)
        log_std = log_std.clamp(-5.0, 1.0)
        std = log_std.exp()
        u = mu if deterministic else mu + std * torch.randn_like(mu)
        a = torch.tanh(u)
        logp = (-0.5 * ((u - mu) / std) ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(-1)
        # log(1 - tanh(u)^2) = 2 (log 2 - u - softplus(-2u))。端に張り付くほど罰が大きくなる
        logp = logp - (2.0 * (np.log(2.0) - u - F.softplus(-2.0 * u))).sum(-1)
        return B_W * a, logp


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.q1, self.q2 = mlp(FEAT_DIM + LAT_DIM, 1), mlp(FEAT_DIM + LAT_DIM, 1)

    def forward(self, f, w):
        x = torch.cat([f, w / B_W], -1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


def save_actor(actor: Actor, path, meta: dict) -> None:
    """一時ファイルに書いてから置き換える（読む側が書きかけを読まない）。"""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    torch.save({"state_dict": actor.state_dict(), "meta": meta}, buf)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(buf.getvalue())
    os.replace(tmp, path)


class Steer:
    """試行を集める側の小さな方策。推論ごとに雑音を選び、(特徴, w) を溜める。"""

    def __init__(self, actor_path=None, deterministic: bool = False, seed: int = 0):
        self.path = None if actor_path is None else pathlib.Path(actor_path)
        self.deterministic = bool(deterministic)
        self.gen = torch.Generator().manual_seed(int(seed))
        torch.manual_seed(int(seed))
        self.actor = Actor().eval()
        self.mtime = None
        self.meta = {"version": "init"}
        self.rows = []
        self.reload()

    def reload(self) -> bool:
        if self.path is None or not self.path.is_file():
            return False
        m = self.path.stat().st_mtime
        if m == self.mtime:
            return False
        for _ in range(5):
            try:
                d = torch.load(self.path, map_location="cpu", weights_only=False)
                break
            except Exception:              # 置き換えの瞬間に当たったら少し待つ
                time.sleep(0.5)
        else:
            return False
        self.actor.load_state_dict(d["state_dict"])
        self.meta, self.mtime = d.get("meta", {}), m
        return True

    def begin_episode(self) -> None:
        self.rows = []

    @torch.no_grad()
    def choose(self, state, cue) -> np.ndarray:
        f = features(state, cue, len(self.rows))
        w, _ = self.actor(torch.from_numpy(f)[None], self.deterministic)
        w = w[0].numpy().astype(np.float32)
        self.rows.append((f, w))
        return w


def attach(pol, steer: Steer) -> None:
    """SensorPolicy の infer を、steer の選んだ雑音（行動列の全行に同じ w、使わない次元にも同じ w）で回すものに替える。"""
    if getattr(pol, "_steer", None) is not None:
        pol._steer = steer
        return
    pol._steer = steer
    orig = pol.infer

    def infer(obs, noise_generator, rtc_kwargs=None):
        cfgp = pol.policy.config
        w = pol._steer.choose(obs["observation.state"], pol.last_cue)
        if cfgp.max_action_dim != LAT_DIM:
            raise ValueError(f"max_action_dim {cfgp.max_action_dim} != LAT_DIM {LAT_DIM}")
        noise = np.tile(w, (cfgp.chunk_size, 1))[None]
        return orig(obs, noise_generator, rtc_kwargs, noise=noise)

    pol.infer = infer


def episode_arrays(rows, reward: float) -> dict:
    f = np.stack([r[0] for r in rows]).astype(np.float32)
    w = np.stack([r[1] for r in rows]).astype(np.float32)
    return {"feat": f, "w": w, "reward": np.float32(reward)}


class SAC:
    """方策オフの SAC。試行ごとに (feat (T, FEAT_DIM), w (T, LAT_DIM), 最後の報酬) を受け取る。"""

    def __init__(self, lr: float = 3e-4, batch: int = 256, tau: float = 0.005, seed: int = 0, capacity: int = 1_000_000):
        torch.manual_seed(int(seed))
        self.rng = np.random.default_rng(int(seed))
        self.actor, self.critic, self.target = Actor(), Critic(), Critic()
        self.target.load_state_dict(self.critic.state_dict())
        self.log_alpha = torch.tensor(float(np.log(ALPHA_INIT)), requires_grad=True)
        self.target_entropy = -float(LAT_DIM)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.opt_t = torch.optim.Adam([self.log_alpha], lr=lr)
        self.batch, self.tau, self.cap = int(batch), float(tau), int(capacity)
        self.F = np.zeros((0, FEAT_DIM), np.float32)
        self.W = np.zeros((0, LAT_DIM), np.float32)
        self.R = np.zeros(0, np.float32)
        self.F2 = np.zeros((0, FEAT_DIM), np.float32)
        self.D = np.zeros(0, np.float32)
        self.G = np.zeros(0, np.float32)
        self.mc_mix = float(MC_MIX)
        self.n_updates = 0
        self.n_episodes = 0

    def add_episode(self, feat, w, reward: float) -> None:
        T = len(feat)
        if T == 0:
            return
        r = np.zeros(T, np.float32)
        r[-1] = float(reward)
        d = np.zeros(T, np.float32)
        d[-1] = 1.0
        f2 = np.concatenate([feat[1:], feat[-1:]])
        g = (GAMMA ** np.arange(T - 1, -1, -1) * float(reward)).astype(np.float32)
        for name, x in (("F", feat), ("W", w), ("R", r), ("F2", f2), ("D", d), ("G", g)):
            setattr(self, name, np.concatenate([getattr(self, name), np.asarray(x, np.float32)])[-self.cap:])
        self.n_episodes += 1

    def __len__(self) -> int:
        return len(self.R)

    def update(self, n: int) -> dict:
        if len(self) < self.batch:
            return {}
        info = {}
        for _ in range(int(n)):
            idx = self.rng.integers(0, len(self), self.batch)
            f, w, r, f2, d, g = (torch.from_numpy(getattr(self, k)[idx]) for k in ("F", "W", "R", "F2", "D", "G"))
            alpha = self.log_alpha.exp().detach()
            with torch.no_grad():
                w2, _ = self.actor(f2)
                q1t, q2t = self.target(f2, w2)
                # エントロピーは Q の目標に入れない（成功の確率としての Q、0〜1 に切る）
                y = (r + GAMMA * (1 - d) * torch.min(q1t, q2t)).clamp(0.0, 1.0)
                y = (1.0 - self.mc_mix) * y + self.mc_mix * g
            q1, q2 = self.critic(f, w)
            lc = F.smooth_l1_loss(q1, y) + F.smooth_l1_loss(q2, y)
            self.opt_c.zero_grad()
            lc.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), GRAD_CLIP)
            self.opt_c.step()
            wn, logp = self.actor(f)
            q1n, q2n = self.critic(f, wn)
            la = (alpha * logp - torch.min(q1n, q2n)).mean()
            self.opt_a.zero_grad()
            la.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), GRAD_CLIP)
            self.opt_a.step()
            lt = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
            self.opt_t.zero_grad()
            lt.backward()
            self.opt_t.step()
            with torch.no_grad():
                self.log_alpha.clamp_(max=float(np.log(ALPHA_MAX)))
                for p, pt in zip(self.critic.parameters(), self.target.parameters()):
                    pt.mul_(1 - self.tau).add_(self.tau * p)
            self.n_updates += 1
            info = {"loss_critic": float(lc.detach()), "loss_actor": float(la.detach()), "alpha": float(alpha),
                    "q_mean": float(q1.detach().mean()), "entropy": float(-logp.detach().mean()),
                    "w_sat": float((wn.detach().abs() > 0.95 * B_W).float().mean())}
        return info

    def state(self) -> dict:
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict(), "target": self.target.state_dict(),
                "log_alpha": self.log_alpha.item(), "n_updates": self.n_updates, "n_episodes": self.n_episodes}

    def load_state(self, s: dict) -> None:
        self.actor.load_state_dict(s["actor"])
        self.critic.load_state_dict(s["critic"])
        self.target.load_state_dict(s["target"])
        with torch.no_grad():
            self.log_alpha.fill_(float(s["log_alpha"]))
        self.n_updates, self.n_episodes = int(s["n_updates"]), int(s["n_episodes"])


def write_episode(path, arrays: dict, summary: dict) -> None:
    """試行 1 本: npz（特徴・雑音・報酬）と、横に json（試行の要約）。npz は一時ファイルから置き換える。"""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    path.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, default=float), encoding="utf-8")
    os.replace(tmp, path)
