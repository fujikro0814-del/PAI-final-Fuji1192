"""指示の差し替え試験（指示文の色を変えたとき、方策がその色の立方体へ向かうか）の判定の純関数。

1 つの行動列（後処理の後の (H, 7)）について、今の x_des から手先参照位置の差分を積算した経路で、グリッパの行動が
初めて正（閉）になる地点（閉じなければ行動列の終点）の水平位置を求め、最も近い立方体を判定する。どの立方体からも
MID_DROP_M 以上離れていれば「中間落ち」。
"""
import collections

import numpy as np

from recovla.common import config

MID_DROP_M = float(config.load()["eval"]["swap_mid_drop_m"])


def chunk_target(chunk, x0, cubes_xy: dict, mid_drop_m: float = MID_DROP_M) -> dict:
    chunk = np.asarray(chunk, float)
    path = np.asarray(x0, float) + np.cumsum(chunk[:, :3], axis=0)
    closed = np.flatnonzero(chunk[:, 6] > 0.0)
    i = int(closed[0]) if closed.size else len(chunk) - 1
    p = path[i, :2]
    d = {c: float(np.hypot(*(p - np.asarray(xy, float)))) for c, xy in cubes_xy.items()}
    near = min(d, key=d.get)
    return {"index": i, "closed": bool(closed.size), "xy": [float(v) for v in p], "nearest": near,
            "dist_m": d[near], "mid_drop": d[near] >= mid_drop_m}


def majority(labels) -> object:
    """最多の票。同数の 1 位が 2 つ以上なら None（多数決が決まらない＝正答に数えない）。"""
    top = collections.Counter(labels).most_common()
    if not top:
        return None
    return top[0][0] if len(top) == 1 or top[0][1] > top[1][1] else None
