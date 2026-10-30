"""座標と幾何の集約。3 色の場面の座標・寸法はここからだけ読む。

- 世界座標（MuJoCo）。x は台座から前、y は左、z は上。机の上面は z = sim.table_top_z
- 指先の中心: hand 体の原点から hand の z 軸の向きに sim.fingertip_offset
- 箱: 体 goal_box の原点（静的な体）。成功の体積は中心から ±scene.box.success_inner_half、0 < z < wall_top_z
- 箱の中の置き場所: 箱の中心からの相対 scene.box_slots
- 画像の反転は recovla.data.vla_image_spec が唯一の定義（ここでは扱わない）
"""
import numpy as np
import mujoco

from recovla.common import config
from recovla.common.seeds import COLORS

_CFG = config.load()
_SCENE = _CFG["scene"]

CUBE_SIZE = float(_SCENE["cube_size"])
CUBE_HALF = CUBE_SIZE / 2.0
TABLE_TOP_Z = float(_CFG["sim"]["table_top_z"])
CUBE_REST_Z = TABLE_TOP_Z + CUBE_HALF                       # 机の上で静止した立方体の中心の高さ
FINGERTIP_OFFSET = float(_CFG["sim"]["fingertip_offset"])
BOX_BODY = "goal_box"
BOX_INNER_HALF = float(_SCENE["box"]["inner_half"])          # 壁の内面まで
BOX_SUCCESS_HALF = float(_SCENE["box"]["success_inner_half"])
BOX_WALL_TOP_Z = float(_SCENE["box"]["wall_top_z"])
BOX_FLOOR_Z = float(_SCENE["box"]["floor_z"])                # 箱の底の上面
BOX_SLOTS = np.array(_SCENE["box_slots"], dtype=float)       # (4, 2) 箱の中心からの相対
BOX_WALLS = ("box_wall_xp", "box_wall_xn", "box_wall_yp", "box_wall_yn")
ROBOT_BODIES = tuple(_CFG["contact"]["robot_bodies"])


def cube_body(color: str) -> str:
    return f"cube_{color}"


def cube_geom(color: str) -> str:
    return f"cube_{color}_geom"


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])


def quat_yaw(quat) -> float:
    """z 軸まわりの回転角（立方体がほぼ水平のとき）。"""
    w, x, y, z = quat
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def tilt_deg(quat) -> float:
    """立方体の傾き: 物体の z 軸と世界の z 軸のなす角のうち、面の向きの対称性（90° ごと）を除いた最小のもの。"""
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(quat, float))
    up = np.abs(m.reshape(3, 3)[2, :])                   # 各軸の世界 z 成分
    return float(np.degrees(np.arccos(np.clip(up.max(), -1.0, 1.0))))


def fingertip_center(data, hand_body_id: int) -> np.ndarray:
    return data.xpos[hand_body_id] + data.xmat[hand_body_id].reshape(3, 3)[:, 2] * FINGERTIP_OFFSET


def box_pos(model) -> np.ndarray:
    return np.array(model.body(BOX_BODY).pos, dtype=float)


def rect_corners(center_xy, half_xy, yaw: float = 0.0) -> np.ndarray:
    """水平面の長方形の 4 隅（反時計回り）。"""
    c, s = np.cos(yaw), np.sin(yaw)
    hx, hy = float(half_xy[0]), float(half_xy[1])
    local = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
    return np.asarray(center_xy, float)[:2] + local @ np.array([[c, s], [-s, c]])


def cube_footprint(pos, quat) -> np.ndarray:
    """立方体の水平面の足跡（ヨーだけを使う。傾きが小さいときの近似。着地の検査は傾き 10° 以下のときだけ意味を持つ）。"""
    return rect_corners(pos, (CUBE_HALF, CUBE_HALF), quat_yaw(quat))


def _seg_point_dist(p, a, b) -> float:
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-18), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _separated(P, Q) -> bool:
    """分離軸の判定（凸なので、両方の辺の法線だけを調べればよい）。"""
    for poly in (P, Q):
        for i in range(len(poly)):
            e = poly[(i + 1) % len(poly)] - poly[i]
            n = np.array([-e[1], e[0]])
            if (P @ n).max() < (Q @ n).min() or (Q @ n).max() < (P @ n).min():
                return True
    return False


def polygon_distance(P, Q) -> float:
    """凸多角形どうしの水平の最短距離（重なれば 0）。mj_geomDistance の箱どうしは専用の関数を通り最大 19 mm 誤る
    （recovla.sim.contact の説明）ので、着地の隙間はこちらで測る。"""
    P, Q = np.asarray(P, float), np.asarray(Q, float)
    if not _separated(P, Q):
        return 0.0
    d = min(_seg_point_dist(p, Q[j], Q[(j + 1) % len(Q)]) for p in P for j in range(len(Q)))
    return float(min(d, min(_seg_point_dist(q, P[j], P[(j + 1) % len(P)]) for q in Q for j in range(len(P)))))


def wall_footprints(model) -> dict:
    """箱の壁 4 枚の水平面の足跡（世界座標）。"""
    box = box_pos(model)
    return {w: rect_corners(box[:2] + model.geom(w).pos[:2], model.geom(w).size[:2]) for w in BOX_WALLS}


def box_outer_half(model) -> float:
    """箱の外寸の半分（中心から壁の外面まで）。壁は箱の体の座標で、xp・xn は x、yp・yn は y の向き。"""
    out = []
    for w in BOX_WALLS:
        g = model.geom(w)
        axis = 0 if w.endswith(("xp", "xn")) else 1
        out.append(abs(float(g.pos[axis])) + float(g.size[axis]))
    return float(max(out))


def in_box(pos, box) -> bool:
    """成功の体積の中か（静止は問わない）。"""
    rel = np.asarray(pos, float) - np.asarray(box, float)
    return bool(abs(rel[0]) < BOX_SUCCESS_HALF and abs(rel[1]) < BOX_SUCCESS_HALF and 0.0 < rel[2] < BOX_WALL_TOP_Z)


def over_box_interior(pos, box) -> bool:
    """水平位置が箱の内寸の内側か（高さは問わない）。"""
    rel = np.asarray(pos, float)[:2] - np.asarray(box, float)[:2]
    return bool(abs(rel[0]) < BOX_INNER_HALF and abs(rel[1]) < BOX_INNER_HALF)


def slot_xy(box, slot: int) -> np.ndarray:
    return np.asarray(box, float)[:2] + BOX_SLOTS[int(slot)]


def slot_occupied(box, slot: int, others_xy, radius: float) -> bool:
    """他の立方体（水平位置）が置き場所の中心から radius 以内にあるか。"""
    c = slot_xy(box, slot)
    return any(np.hypot(*(np.asarray(p, float)[:2] - c)) < radius for p in others_xy)


def project(model, data, camera: str, point, width: int, height: int):
    """世界の点を、そのカメラの（反転前の）画像の画素 (u, v) に透視投影する（fovy は縦の画角）。カメラの後ろなら None。"""
    cam = model.camera(camera).id
    pos = data.cam_xpos[cam]
    rot = data.cam_xmat[cam].reshape(3, 3)              # 列: カメラの x（右）・y（上）・z（後ろ向き）
    rel = rot.T @ (np.asarray(point, float) - pos)
    if rel[2] >= 0.0:
        return None
    f = 0.5 * height / np.tan(np.radians(model.cam_fovy[cam]) / 2.0)
    u = width / 2.0 + f * rel[0] / -rel[2]
    v = height / 2.0 - f * rel[1] / -rel[2]
    return float(u), float(v)
