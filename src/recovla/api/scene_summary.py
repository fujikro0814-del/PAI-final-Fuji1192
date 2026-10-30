"""手順の組み直しの LLM に渡す場面の要約を、推定の形（api/estimated の WorldEstimate）で直す（手順書 docs/api/r1_protocol.md 5 節の 2）。

    out = refine(base, est, place="box", held=set, tip=None)

base は実行器の要約（perception_summary の table・in_box・in_hand・lost）。est が無ければ base に空の欄を足すだけ。
直すこと（どれも推定の大きさ・箱の底の推定からの相対。既知の寸法は使わない）:
- stacked_in_box: 箱の中と出た色のうち、箱の中の別の物の上に重なっているもの（in_box から外す）。次のどちらか:
  (1) 物の下の端（中心 − 見かけの高さの半分）が、推定した箱の底から見かけの高さの半分より上
  (2) 水平に重なる箱の中の別の物があり、その物の上面より、下の物の見かけの高さの半分以上高い
  （(2) は、置かれた物の推定が下の面まで延ばされ、(1) で見分けられないときのため）
- near_box_wall: 箱の内側の柱の外で、外の縁から物の幅の半分の 1.5 倍以内（壁ぎわ・縁の上）。印だけ（欄は動かさない）
置き場所の形はタスクの推定の形のまま（箱 BoxGeom・上に置く面 PlateGeom。面では底 = 上面、縁 = 推定の円の縁）。欄の名前は
in_box・near_box_wall のまま（場所の中・上と、その縁の意味。組み直しの指示文がタスクの言葉で説明する）
- not_in_hand_after_hold: 持っていたはず（held。実行器がこの手順で手の中と見た色）なのに、今は手の中でも手のそばでもなく、
  箱の中（重なりでない）でもない色
"""
import numpy as np

from recovla.api import relations as REL

KEYS = ("table", "in_box", "in_hand", "lost")
EXTRA = ("stacked_in_box", "near_box_wall", "not_in_hand_after_hold")
WALL_K = 1.5                        # 壁ぎわ: 外の縁から物の幅の半分のこの倍以内


def floor_z(region) -> float:
    """推定した置き場所の底の、世界の高さ。箱（BoxGeom）は縁の中心から縁の法線の向きに floor、上に置く面（PlateGeom など
    floor の無い形）は上面の中心の高さ。"""
    if not hasattr(region, "floor"):
        return float(np.asarray(region.center, float)[2])
    return float((np.asarray(region.center, float) + np.asarray(region.R, float) @ np.array([0.0, 0.0, region.floor]))[2])


def top_z(o) -> float:
    return float(o.pos[2]) + 0.5 * float(REL.height(o))


def wall_gap(o, region) -> float:
    """物の中心から外の縁までの水平の距離（中なら負）。箱は外の縁の長方形、面は円の縁（推定の半径）。"""
    q = region.local(o.pos)
    if not hasattr(region, "outer"):
        return float(np.hypot(q[0], q[1]) - float(region.radius))
    return float(max(abs(q[0]) - region.outer[0], abs(q[1]) - region.outer[1]))


def stacked(name, objs: dict, pl) -> bool:
    o, reg = objs[name], pl.region
    h = REL.height(o)
    if h is None or reg is None or not reg.over_xy(o.pos):
        return False
    margin = REL.UNC_K * (REL.sigma(o) + pl.sigma)
    if float(o.pos[2]) - 0.5 * h - floor_z(reg) > 0.5 * h + margin:
        return True
    for k, u in objs.items():
        if k == name or u.status == "lost" or REL.height(u) is None or not reg.over_xy(u.pos):
            continue
        wo, wu = REL.width_min(o), REL.width_min(u)
        if np.linalg.norm(np.asarray(o.pos[:2], float) - np.asarray(u.pos[:2], float)) >= 0.5 * (wo + wu):
            continue                                                  # 水平に重ならない
        if top_z(o) - top_z(u) > 0.5 * REL.height(u) + margin:
            return True
    return False


def near_wall(o, pl) -> bool:
    reg = pl.region
    if reg is None or REL.width_min(o) is None or reg.over_xy(o.pos):
        return False
    return wall_gap(o, reg) <= WALL_K * 0.5 * REL.width_min(o) + REL.UNC_K * (REL.sigma(o) + pl.sigma)


def refine(base: dict, est, place: str = "box", held=(), tip=None) -> dict:
    out = {k: list(base.get(k, ())) for k in KEYS}
    for k in EXTRA:
        out[k] = []
    pl = None if est is None else est.places.get(place)
    if pl is None or not pl.ok or pl.region is None:
        return out
    objs = {n: o for n, o in est.objects.items() if o.status != "lost"}
    for c in sorted(objs):
        o = objs[c]
        if c in out["in_box"] and stacked(c, objs, pl):
            out["in_box"].remove(c)
            out["stacked_in_box"].append(c)
        if c not in out["in_hand"] and near_wall(o, pl):
            out["near_box_wall"].append(c)
    for c in sorted(set(held)):
        o = objs.get(c)
        if c in out["in_hand"] or c in out["in_box"]:
            continue
        near = None if o is None else (True if o.near_hand is True else REL.near_hand(o, tip))
        if near is not True:
            out["not_in_hand_after_hold"].append(c)
    return out
