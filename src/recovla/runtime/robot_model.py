"""実行系が読むロボットのモデル。

読んでよいのは assets/mjcf/robot_only.xml（panda.xml と制御器の目標の印だけ）。場面の XML（机・立方体・箱・俯瞰カメラ）を
読まないことを、静的な検査（scripts/check_real_boundary.py）が確かめる。実行系の中で mujoco のモデルを読むのはこのファイルだけ。
"""
import pathlib

import mujoco

ALLOWED = "robot_only.xml"


def load(robot_xml: str) -> mujoco.MjModel:
    p = pathlib.Path(robot_xml)
    if p.name != ALLOWED:
        raise ValueError(f"実行系が読めるロボットの模型は {ALLOWED} だけ: {p}")
    return mujoco.MjModel.from_xml_path(str(p))


HAND_BODIES = ("hand", "left_finger", "right_finger")
N_CUBES = 3


def belief_model(robot_xml: str, cube_size: float, box_outer: float, box_wall: float, box_height: float,
                 box_floor: float) -> mujoco.MjModel:
    """安全フィルタの距離のモデル（実行系が知覚から作る世界）: ロボットのモデルに、知覚した立方体（mocap の箱、既知の寸法）と箱の壁
    （mocap の体に既知の寸法の壁 4 枚）を足したもの。シミュレーション側の距離のモデル（sim/contact.distance_model）と同じく、指先の当て板を
    同じ寸法の箱形のメッシュにし、native CCD を使う（箱どうしの距離の誤りを避ける）。物理には使わない。"""
    p = pathlib.Path(robot_xml)
    if p.name != ALLOWED:
        raise ValueError(f"実行系が読めるロボットの模型は {ALLOWED} だけ: {p}")
    spec = mujoco.MjSpec()
    spec.from_file(str(p))
    n = 0
    for name in HAND_BODIES:
        body = spec.find_body(name)
        g = body.first_geom()
        while g is not None:
            if g.type == mujoco.mjtGeom.mjGEOM_BOX and g.contype:
                s = list(g.size)
                mesh = spec.add_mesh()
                mesh.name = f"belief_pad_{n}"
                mesh.uservert = [float(v) for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)
                                 for v in (i * s[0], j * s[1], k * s[2])]
                g.type = mujoco.mjtGeom.mjGEOM_MESH
                g.meshname = mesh.name
                n += 1
            g = body.next_geom(g)
    wb = spec.worldbody
    h = 0.5 * cube_size
    for i in range(N_CUBES):
        b = wb.add_body()
        b.name = f"belief_cube_{i}"
        b.mocap = True
        b.pos = [0.0, 0.0, -1.0 - i]
        g = b.add_geom()
        g.name = f"belief_cube_{i}_geom"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [h, h, h]
        g.contype = g.conaffinity = 0
    bx = wb.add_body()
    bx.name = "belief_box"
    bx.mocap = True
    bx.pos = [0.0, 0.0, -5.0]
    half, t = 0.5 * box_outer, 0.5 * box_wall
    zc, zh = 0.5 * (box_floor + box_height), 0.5 * (box_height - box_floor)
    for nm, pos, size in (("xp", [half - t, 0, zc], [t, half, zh]), ("xn", [-(half - t), 0, zc], [t, half, zh]),
                          ("yp", [0, half - t, zc], [half, t, zh]), ("yn", [0, -(half - t), zc], [half, t, zh])):
        g = bx.add_geom()
        g.name = f"belief_wall_{nm}"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.pos, g.size = pos, size
        g.contype = g.conaffinity = 0
    model = spec.compile()
    model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_NATIVECCD)
    return model
