"""物・手・置き場所の関係の判定。閾値は推定した量（大きさ・不確かさ）からの相対で書き、
定数の cm を使わない。失敗の検知（api/monitors）とスクリプト方策（api/skills_script）が使う。

    near_hand(o, tip)          # 手に近い: |手先 − 物の中心| ≤ 物の半径 × k ＋ unc_k × σ
    on_table(o, table_z)       # 机の上: 中心の高さ ≤ 机 ＋ 見かけの高さ × 0.5 ＋ 余白（見かけの高さ × table_frac ＋ unc_k × σ）
    outside(o, place)          # 箱の外: 推定した内側の領域から unc_k × (σ_物 ＋ σ_箱) 以上外
    inside(o, place)           # 箱の中: 内側の領域の中（深さ ≥ 0）
    on_surface(o, place)       # 皿の上: 物の底が上面に載って円の内側（下の注）
    in_place(o, place)         # 置き場所の種類で分ける: 箱らしい物（container）は inside、皿らしい物（surface）は on_surface
物の半径 = 辺の対角の半分。見かけの高さ = 向き R の各軸の、机の法線への射影の和。σ = 中心の共分散のいちばん大きい標準偏差。
大きさ・領域の無い推定（None）では、判定できないので None を返す（呼び手が扱う）。
皿の上（on_surface）: 物の底の中心（中心から向き R の上向きの軸に沿って辺の半分下）が、上面の円の半径 − unc_k × (σ_物 ＋ σ_皿)
の内側で、上面からの高さが 見かけの高さ × on_frac ＋ unc_k × (σ_物 ＋ σ_皿) 以内、かつ立っている（upright。知覚の口が
物の種類の知識で決める。カップなら上が開いて見えた）。
"""
import math
from typing import Optional

import numpy as np

UNC_K = 2.0                         # 余白に掛ける不確かさの倍数
NEAR_K = 1.0                        # 手に近いの半径の倍数
TABLE_FRAC = 0.25                   # 机の上の余白の、見かけの高さに対する割合（斜めの物の中心の高さのずれを見込む）
ON_FRAC = 0.1                       # 皿の上: 底と上面の高さの差の許しの、見かけの高さに対する割合


def sigma(o) -> float:
    if o is None or getattr(o, "cov", None) is None:
        return 0.0
    return float(math.sqrt(max(float(np.linalg.eigvalsh(np.asarray(o.cov, float)).max()), 0.0)))


def radius(o) -> Optional[float]:
    e = getattr(o, "extent", None)
    return None if e is None else 0.5 * float(np.linalg.norm(e))


def height(o) -> Optional[float]:
    """机の法線（+z）の向きの見かけの高さ。"""
    e, R = getattr(o, "extent", None), getattr(o, "R", None)
    if e is None:
        return None
    if R is None:
        return float(e[2])
    return float(np.sum(np.abs(np.asarray(R, float)[2, :]) * np.asarray(e, float)))


def width_min(o) -> Optional[float]:
    """机に沿う向きの小さい方の幅（向きつきの外接箱の、机に沿う 2 軸の短い方）。"""
    e, R = getattr(o, "extent", None), getattr(o, "R", None)
    if e is None:
        return None
    if R is None:
        return float(min(e[0], e[1]))
    R = np.asarray(R, float)
    k = int(np.argmax(np.abs(R[2, :])))
    return float(min(e[i] for i in range(3) if i != k))


def near_hand(o, tip, k: float = NEAR_K, unc_k: float = UNC_K) -> Optional[bool]:
    r = radius(o)
    if o is None or tip is None or r is None:
        return None
    return bool(np.linalg.norm(np.asarray(o.pos, float) - np.asarray(tip, float)) <= k * r + unc_k * sigma(o))


def on_table(o, table_z, frac: float = TABLE_FRAC, unc_k: float = UNC_K) -> Optional[bool]:
    h = height(o)
    if o is None or table_z is None or h is None:
        return None
    return bool(float(o.pos[2]) - float(table_z) <= 0.5 * h + frac * h + unc_k * sigma(o))


def outside(o, place, unc_k: float = UNC_K) -> Optional[bool]:
    d = None if (o is None or place is None) else place.depth(o.pos)
    if d is None:
        return None
    return bool(d <= -unc_k * (sigma(o) + place.sigma))


def inside(o, place) -> Optional[bool]:
    d = None if (o is None or place is None) else place.depth(o.pos)
    return None if d is None else bool(d >= 0.0)


def place_kind(place) -> str:
    """置き場所の推定の種類: 領域が皿らしい物（PlateGeom、kind "surface"）なら "surface"、ほかは "container"。"""
    return getattr(getattr(place, "region", None), "kind", "container")


def on_surface(o, place, upright: bool = True, frac: float = ON_FRAC, unc_k: float = UNC_K) -> Optional[bool]:
    e, h = getattr(o, "extent", None), height(o)
    reg = None if place is None or not getattr(place, "ok", False) else getattr(place, "region", None)
    if o is None or reg is None or e is None or h is None:
        return None
    R = np.eye(3) if getattr(o, "R", None) is None else np.asarray(o.R, float)
    bottom = np.asarray(o.pos, float) - R[:, 2] * (0.5 * float(np.asarray(e, float)[2]))
    s = sigma(o) + place.sigma
    r = reg.on_top(bottom, unc_k * s, frac * h + unc_k * s)
    return bool(r["on"] and upright)


def in_place(o, place, upright: bool = True) -> Optional[bool]:
    """置き場所の種類で分けた「目標の場所にある」: container は inside、surface は on_surface。"""
    if place_kind(place) == "surface":
        return on_surface(o, place, upright)
    return inside(o, place)
