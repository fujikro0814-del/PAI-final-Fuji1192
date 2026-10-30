"""検査用の模擬の方策と、Schedule を回す枠（runner.py の呼び方をなぞる）。

行動列の中身は (推論の番号 i, 行動列の中の添字) から一意に決まる: 値[添字, a] = 1000·(i + 1) + 添字 + a/64。
+1 は、0 詰めの行（値 0）と行動列 0 の添字 0 を見分けるため。a/64 は 2 進で割り切れるので、float32 でも誤差なく逆算できる。
"""
from dataclasses import dataclass, field

import numpy as np

from recovla.policy.schedule import RuntimeConfig, Schedule

A = 32          # 方策の出力の次元（詰め物込み）
A_EXEC = 7      # 実行する先頭の次元


def chunk(i: int, H: int = 50) -> np.ndarray:
    """推論 i が返す行動列（後処理の前、(H, A)、float32）。"""
    idx = np.arange(H, dtype=np.float32)[:, None]
    a = np.arange(A, dtype=np.float32)[None, :]
    return (1000.0 * (i + 1) + idx + a / 64.0).astype(np.float32)


def postprocess(c: np.ndarray) -> np.ndarray:
    """後処理の模擬（先頭 7 次元を取り、形の違う値にする）。前後を取り違えると値でわかる。"""
    return -c[:, :A_EXEC].astype(np.float64)


def decode(row: np.ndarray) -> tuple[int, int]:
    """後処理の前の行（またはその -1 倍）から (i, 添字) を逆算する。"""
    v = abs(float(row[0]))
    return int(v // 1000) - 1, int(round(v % 1000))


@dataclass
class Trace:
    decisions: list = field(default_factory=list)
    executes: list = field(default_factory=list)       # k ごとの (i, 添字)
    actions: list = field(default_factory=list)        # k ごとの action の行（後処理の後）
    log: list = field(default_factory=list)


def run(cfg: RuntimeConfig, n: int, resets=()) -> Trace:
    """k = 0..n−1 を回す。resets に入っている k では、decide の前に reset を呼ぶ。"""
    sched = Schedule(cfg)
    post: dict[int, np.ndarray] = {}
    tr = Trace()
    for k in range(n):
        if k in resets:
            sched.reset()
        dec = sched.decide(k)
        if dec.infer:
            i = sched.log[-1]["i"]
            c = chunk(i, cfg.chunk_size)
            assert sched.deliver(c) == i
            post[i] = postprocess(c)
        tr.decisions.append(dec)
        tr.executes.append(dec.execute)
        tr.actions.append(sched.action(post))
    tr.log = sched.log
    return tr
