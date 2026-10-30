"""方策の実行器の時間の流れ（docs/interfaces/runner.md）。方策も物理も呼ばない純粋な部品。

いつ推論を始め、いつ結果が有効になり、行動列のどの行を実行し、RTC に前の行動列のどこを渡すかだけを決める。
runner.py は、decide(k) → （infer なら）方策を呼んで deliver(行動列) → action(後処理後の行動列) の順に使う。

記号: k は 10 Hz の行動の番号、H は行動列の長さ、s は実行間隔、d は遅延、E は RTC で前の行動列に合わせる範囲。
推論 i は k_i で観測を取り、行動列 i は手 v_i から添字 o_i で実行を始める。
"""
from __future__ import annotations

import numbers
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

MODES = ("sync", "naive", "rtc")
NOISE_STREAM_ID = 2          # 雑音の乱数列の番号。再現のため固定し、変えない


def _is_int(x) -> bool:
    return isinstance(x, numbers.Integral) and not isinstance(x, bool)


@dataclass(frozen=True)
class RuntimeConfig:
    mode: Literal["sync", "naive", "rtc"]
    exec_interval: int          # s
    delay_steps: int            # d（sync では無視して 0 とみなす。None でもよい）
    chunk_size: int = 50        # H
    execution_horizon: int = 10 # E（rtc だけ）

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode は {MODES} のどれか: {self.mode!r}")
        H, s = self.chunk_size, self.exec_interval
        if not _is_int(H) or H < 1:
            raise ValueError(f"chunk_size（H）は 1 以上の整数: {H!r}")
        if not _is_int(s) or not 1 <= s <= H:
            raise ValueError(f"exec_interval（s）は 1 <= s <= H={H} の整数: {s!r}")
        if self.mode != "sync":
            d = self.delay_steps
            if not _is_int(d) or not 0 <= d < s:
                raise ValueError(f"delay_steps（d）は 0 <= d < s={s} の整数: {d!r}")
            if s + d > H:
                raise ValueError(f"s + d <= H でなければならない: s={s}, d={d}, H={H}")
        if self.mode == "rtc":
            E = self.execution_horizon
            # E > H は LeRobot の RTC の処理（modeling_rtc.py）で行動列と形が合わず落ちる
            if not _is_int(E) or not 1 <= E <= H:
                raise ValueError(f"execution_horizon（E）は 1 <= E <= H={H} の整数: {E!r}")

    @property
    def delay(self) -> int:
        """実際に使う d（sync は 0）。"""
        return 0 if self.mode == "sync" else int(self.delay_steps)


@dataclass(frozen=True)
class Decision:
    k: int
    infer: bool                         # この k の観測で推論を始める
    block: bool                         # 推論の間、物理を止めて待つ（最初・reset 後と、sync の全推論）
    left_over: Optional[np.ndarray]     # rtc の i >= 1 のときだけ、(E, A) に揃えた前の行動列の残り。それ以外は None
    inference_delay: int                # 方策に渡す d（rtc のとき。それ以外は 0）
    execute: tuple[int, int]            # この k で実行する (行動列の番号 i, 行動列の中の添字)


@dataclass
class _Chunk:
    i: int
    v: int                  # 有効になる手
    o: int                  # 有効になったときに実行する添字
    data: Optional[np.ndarray] = None   # deliver された行動列（後処理の前、(H, A)）

    def index_at(self, k: int) -> int:
        return self.o + (k - self.v)


def normalize_left_over(prev: np.ndarray, target_steps: int) -> np.ndarray:
    """前の行動列の残り (T, A) を target_steps 行に揃える。長ければ先頭から切り詰め、短ければ後ろを 0 で詰める。

    lerobot 0.6.1 の lerobot/rollout/inference/rtc.py の _normalize_prev_actions_length を numpy で書いたもの。
    """
    prev = np.asarray(prev)
    if prev.ndim != 2:
        raise ValueError(f"(T, A) の 2 次元の配列でなければならない: 形 {prev.shape}")
    steps, action_dim = prev.shape
    if steps >= target_steps:
        return prev[:target_steps].copy()
    padded = np.zeros((target_steps, action_dim), dtype=prev.dtype)
    padded[:steps] = prev
    return padded


class Schedule:
    """sync・naive・rtc の時間の流れ。1 試行につき 1 つ作り、k = 0, 1, 2, … の順に decide を呼ぶ。"""

    def __init__(self, cfg: RuntimeConfig):
        self.cfg = cfg
        self._log: list[dict] = []
        self._next_k = 0                        # 次に decide に渡すべき k
        self._carry = False                     # 次の推論で前の行動列を持ち越すか（最初と reset 後は偽）
        self._next_infer_k: Optional[int] = None  # 次に推論を始める手（持ち越さないときは次の decide で始める）
        self._active: Optional[_Chunk] = None   # いま実行している行動列
        self._incoming: Optional[_Chunk] = None # 推論済みで、まだ有効になっていない行動列
        self._pending: Optional[_Chunk] = None  # decide が infer=True を返し、deliver を待っている行動列
        self._last: Optional[Decision] = None

    def reset(self) -> None:
        """次の推論を「持ち越さない」扱いにする（指示の切り替え・やり直しの直後）。

        次の decide(k) はその k で推論を始め（block=True、v = k、o = 0、left_over なし）、そこから s 手ごとに戻る。
        持っていた行動列（有効になる前のものを含む）は捨てる。行動列の番号 i は試行の中で3 個の連続タスクのまま続ける。
        """
        if self._pending is not None:
            raise RuntimeError(f"塊 {self._pending.i} の deliver の前に reset した")
        self._carry = False
        self._next_infer_k = None
        self._active = None
        self._incoming = None

    def decide(self, k: int) -> Decision:
        if not _is_int(k) or k != self._next_k:
            raise ValueError(f"decide は k を 0 から 1 ずつ増やして呼ぶ（次は {self._next_k}）: {k!r}")
        if self._pending is not None:
            raise RuntimeError(f"塊 {self._pending.i} の deliver がないまま次の手に進んだ")
        k = int(k)
        self._next_k = k + 1
        cfg, H, s, d = self.cfg, self.cfg.chunk_size, self.cfg.exec_interval, self.cfg.delay

        # 前の推論の行動列が、この手から有効になる
        if self._incoming is not None and k >= self._incoming.v:
            self._active, self._incoming = self._incoming, None

        infer = not self._carry or k == self._next_infer_k
        block, left_over, inference_delay = False, None, 0
        if infer:
            i = len(self._log)
            if not self._carry or cfg.mode == "sync":
                new = _Chunk(i, v=k, o=0)
                block = True
                left_over_len = 0
            else:
                new = _Chunk(i, v=k + d, o=d)
                prev = self._active
                j = prev.index_at(k)            # j = o_{i−1} + (k_i − v_{i−1})
                left_over_len = H - j
                if cfg.mode == "rtc":
                    left_over = normalize_left_over(prev.data[j:H], cfg.execution_horizon)
                    inference_delay = d
                else:
                    left_over_len = 0           # naive は何も渡さない
            self._log.append({"i": i, "k_obs": k, "k_valid": new.v, "offset": new.o,
                              "left_over_len": left_over_len, "reset": not self._carry,
                              "wall_s": None, "wall_breakdown_s": None})
            self._pending = new
            self._carry = True
            self._next_infer_k = k + s
            if new.v == k:
                self._active, self._incoming = new, None
            else:
                self._incoming = new

        idx = self._active.index_at(k)
        assert 0 <= idx < H, (k, self._active)
        self._last = Decision(k=k, infer=infer, block=block, left_over=left_over,
                              inference_delay=inference_delay, execute=(self._active.i, idx))
        return self._last

    def deliver(self, chunk: np.ndarray) -> int:
        """直前の infer=True に対する行動列（(H, A)、後処理の前）を渡す。返り値は行動列の番号 i。"""
        if self._pending is None:
            raise RuntimeError("deliver を待っている推論がない（decide が infer=True を返した後に 1 度だけ呼ぶ）")
        chunk = np.array(chunk, copy=True)
        if chunk.ndim != 2 or chunk.shape[0] != self.cfg.chunk_size:
            raise ValueError(f"塊は (H={self.cfg.chunk_size}, A) の 2 次元の配列: 形 {chunk.shape}")
        self._pending.data = chunk
        i, self._pending = self._pending.i, None
        return i

    def action(self, chunk_post: dict[int, np.ndarray]) -> np.ndarray:
        """直前の decide の execute に当たる行を、後処理の後の行動列（行動列の番号 → (H, A)）から返す。"""
        if self._last is None:
            raise RuntimeError("decide の前に action を呼んだ")
        i, idx = self._last.execute
        if i not in chunk_post:
            raise KeyError(f"後処理の後の塊 {i} がない（k={self._last.k}）")
        return np.asarray(chunk_post[i])[idx]

    @property
    def log(self) -> list[dict]:
        """推論ごとの記録。infer=True の decide の時点で末尾に足す。

        wall_s・wall_breakdown_s は None の枠だけ。runner.py が測って log[-1] に書き込む（同じ dict を返す）。
        """
        return self._log


def noise_generator(seed: int, i: int):
    """推論 i の雑音を引く torch.Generator（CPU）。SeedSequence(seed, spawn_key=(2, i)) から作る。

    torch のシードは、その SeedSequence の generate_state の最初の 64 ビットの語。
    雑音は torch.randn((1, H, 32), generator=g) のように CPU で引いてから方策の device に移す。
    """
    import torch                # torch はここだけで使う（時間の流れの部分は torch なしで動く）

    ss = np.random.SeedSequence(seed, spawn_key=(NOISE_STREAM_ID, i))
    g = torch.Generator(device="cpu")
    g.manual_seed(int(ss.generate_state(1, dtype=np.uint64)[0]))
    return g
