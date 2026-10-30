"""評価のための目標の判定（真値を使う）。実行系の版は api/task_spec.goal_met（知覚の推定を使う）。

    in_place_truth(spec, cube_pos, box)        # {物: 箱の成功の体積の中か}（sim/frames.in_box と同じ体積）
    on_place_truth(spec, objs, place, rule)    # {物: 皿などの上に載って立っているか}（目標 all_on。カップと皿の場面）
    in_place_of_meta(spec, meta)               # 試行の記録（harness/task_loop の meta["final_in_place"]、無ければ
                                               # meta["final_in_box"] を色で引く）から
    goal_met_truth(spec, in_place)             # 目標が満たされたか

all_on の「載って立っている」（configs/scenes/cup_plate.yaml の success）: 物の底の中心が置き場所の上面の円の内側
（半径 − radius_margin_m）、底の高さと上面の差が contact_tol_m 以内、物の軸と置き場所の法線のなす角が max_tilt_deg 以下。
"""
import math

import numpy as np

from recovla.api import task_spec as TS
from recovla.sim import frames


def _color(spec, name: str) -> str:
    return TS.cue_of(spec, TS.StepSpec("pick_place", name, spec.goal.place))


def in_place_truth(spec, cube_pos: dict, box) -> dict:
    """cube_pos: {物の名前: 真の中心 (3,)}、box: 箱の中心（真値）。"""
    return {n: bool(frames.in_box(cube_pos[n], box)) for n in spec.goal.objects}


def _rot(quat) -> np.ndarray:
    w, x, y, z = (float(v) for v in quat)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def on_surface_truth(bottom, quat, place: dict, rule: dict) -> dict:
    """1 つの物: bottom = 底の中心（真値、世界）、quat = 物の向き（体の z が物の軸）。place = {"center": 上面の中心 (3,),
    "normal": 上面の法線 (3,), "radius": 上面の円の半径}。rule = configs/scenes/cup_plate.yaml の success。"""
    c, n = np.asarray(place["center"], float), np.asarray(place["normal"], float)
    n = n / np.linalg.norm(n)
    d = np.asarray(bottom, float) - c
    h = float(d @ n)                                       # 上面からの高さ
    rho = float(np.linalg.norm(d - h * n))                 # 上面の中心からの水平の距離
    tilt = math.degrees(math.acos(float(np.clip(_rot(quat)[:, 2] @ n, -1.0, 1.0))))
    inside = rho <= float(place["radius"]) - float(rule["radius_margin_m"])
    touch = abs(h) <= float(rule["contact_tol_m"])
    upright = tilt <= float(rule["max_tilt_deg"])
    return {"on": bool(inside and touch and upright), "rho_m": rho, "height_m": h, "tilt_deg": tilt,
            "inside": bool(inside), "touch": bool(touch), "upright": bool(upright)}


def on_place_truth(spec, objs: dict, place: dict, rule: dict) -> dict:
    """objs: {物の名前: (底の中心 (3,), 向きの四元数 (4,))}（真値）。{物: 載って立っているか}。"""
    return {n: on_surface_truth(objs[n][0], objs[n][1], place, rule)["on"] for n in spec.goal.objects}


def in_place_of_meta(spec, meta: dict) -> dict:
    """task_loop の記録から {物: 目標の場所にあるか}。final_in_place（物の名前で引ける欄）があればそれ、無ければ final_in_box
    （{色: 真偽}）を物の名前で引き直す。"""
    fin = meta.get("final_in_place")
    if fin is not None:
        return {n: bool(fin[n]) for n in spec.goal.objects}
    fin = meta["final_in_box"]
    return {n: bool(fin[_color(spec, n)]) for n in spec.goal.objects}


def goal_met_truth(spec, in_place: dict) -> bool:
    if spec.goal.kind in ("all_in", "all_on"):
        return all(bool(in_place[n]) for n in spec.goal.objects)
    raise ValueError(spec.goal.kind)
