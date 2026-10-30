"""置く位置の入力（docs/api/r2_protocol.md 6.4 節）の実行時の値。

学習データでは、生成プログラムが運んだ立方体がエピソードの終わりに止まった位置（data/convert.place_label）。生成プログラムは
箱を 2×2 に分けたマスの空いた所に置くので、実行時も同じ形の候補から選ぶ:
  候補  推定した箱の内寸を 2×2 に分けたマスの中心（api/geometry.BoxGeom.slot。既知の寸法は読まない）
  選び  箱の中の物（目標の物を除く）の推定の中心から、水平にいちばん遠いマス。箱が空なら決まった順の最初（BoxGeom.slot の順）
シミュレーションの真の値は使わない。手順・やり直しの始めに 1 回決め、その手順の間は変えない（腕が箱の上で俯瞰の画像を隠す前に決める）。

    xy, info = free_cell(est, place, target)     # est は WorldEstimate（prt.wm.geometry）。箱が推定できていなければ xy は None
"""
import numpy as np

N_CELLS = 2                      # 1 辺のマスの数（スクリプトの置き場所と同じ 2×2）


def free_cell(est, place: str, target: str):
    """→ ((2,) の世界の x, y か None, 記録用の dict)。"""
    ps = None if est is None else est.places.get(place)
    if ps is None or not ps.ok or ps.region is None:
        return None, {"source": "no_box"}
    box = ps.region
    others = [(name, np.asarray(o.pos, float)[:2]) for name, o in est.objects.items()
              if name != target and o.status != "lost" and o.in_place.get(place)]
    cells = [np.asarray(box.slot(i, N_CELLS), float)[:2] for i in range(N_CELLS * N_CELLS)]
    clear = [min((float(np.linalg.norm(c - p)) for _, p in others), default=float("inf")) for c in cells]
    k = int(np.argmax(clear))                     # 同じ値なら最初（argmax は最初を返す）
    return cells[k].copy(), {"source": "estimated", "cell": k, "in_place": [n for n, _ in others],
                             "clearance_m": None if not others else round(clear[k], 4),
                             "cells": [[round(float(v), 4) for v in c] for c in cells]}
