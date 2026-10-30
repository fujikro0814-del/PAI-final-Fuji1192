"""手・指と障害物の接触と最短距離（記録の contact_robot・min_dist の列）。

列（K = 7、並びは固定）: 立方体 red・green・blue、箱の壁 xp・xn・yp・yn。どの列が「障害物」か（机上の目標外の
立方体と壁。目標の立方体と箱の中の立方体は対象外）は、手順ごとに obstacle_mask で決め、記録には全列を残す。

- 接触: 物理ステップごとに MuJoCo の接触の一覧（d.contact）を見て、ロボット側の全ての衝突形状（hand・指のメッシュと
  指先の当て板）と各列の形状の組が 1 度でもあれば真。窓（20 Hz のフレーム）の中で OR する
- 立方体どうしの接触: 同じ窓の中で、色 a と色 b の立方体が触れたか（対称）
- 最短距離: フレームの時点で mj_geomDistance を、**距離の計算専用のモデルの写し**（distance_model）で求める。
  contact.distmax_m より遠ければその値。写しでは次の 2 点だけを変える（物理に使うモデルは変えない）:
    1. native CCD を有効にする（mujoco 3.2.3 の既定の libccd は、近い組で最大 30 mm 誤り、離れているのに負の値を
       返すことがある。native CCD は凸包どうしの厳密な距離との差が 0.054 mm 以内）
    2. 指先の当て板（箱）を、同じ寸法の箱形のメッシュにする（箱どうしの距離は CCD を通らない専用の関数で計算され、
       最大 19 mm 誤る。当て板は指のメッシュから最大 9.6 mm はみ出しているので、外せない）
  写しは物理の状態を持たないので、フレームごとに qpos を写して mj_kinematics で位置だけを求める
"""
import mujoco
import numpy as np

from recovla.common import config
from recovla.common.seeds import COLORS
from recovla.sim import frames

_CFG = config.load()
_C = _CFG["contact"]
COLUMNS = tuple(f"cube_{c}" for c in COLORS) + tuple(w.replace("box_", "") for w in frames.BOX_WALLS)


def _box_corners(size) -> list:
    s = np.asarray(size, float)
    return [float(v) for i in (-1, 1) for j in (-1, 1) for k in (-1, 1) for v in (i * s[0], j * s[1], k * s[2])]


def distance_model():
    """距離の計算専用のモデル（native CCD・当て板をメッシュに）。形状の番号と体の木は元のモデルと同じ。"""
    spec = mujoco.MjSpec()
    spec.from_file(str(config.path(_CFG["paths"]["scene"])))
    n = 0
    for name in _C["robot_bodies"]:
        g = spec.find_body(name).first_geom()
        while g is not None:
            if g.type == mujoco.mjtGeom.mjGEOM_BOX and g.contype:
                mesh = spec.add_mesh()
                mesh.name = f"distance_pad_{n}"
                mesh.uservert = _box_corners(g.size)
                g.type = mujoco.mjtGeom.mjGEOM_MESH
                g.meshname = mesh.name
                n += 1
            g = spec.find_body(name).next_geom(g)
    model = spec.compile()
    model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_NATIVECCD)
    return model


class ContactMeter:
    def __init__(self, model):
        self.model = model
        bodies = {model.body(b).id for b in _C["robot_bodies"]}
        self.robot_geoms = np.array([g for g in range(model.ngeom)
                                     if model.geom_bodyid[g] in bodies and model.geom_contype[g]], dtype=int)
        self.dmodel = distance_model()
        self.ddata = mujoco.MjData(self.dmodel)
        if self.dmodel.ngeom != model.ngeom or self.dmodel.nq != model.nq or any(
                self.dmodel.geom_bodyid[g] != model.geom_bodyid[g] for g in range(model.ngeom)):
            raise RuntimeError("distance model does not match the scene (geom order)")
        self.distance_geoms = self.robot_geoms                     # 手・指のメッシュと当て板（写しでは全部メッシュ）
        self.column_geoms = np.array([model.geom(frames.cube_geom(c)).id for c in COLORS]
                                     + [model.geom(w).id for w in frames.BOX_WALLS], dtype=int)
        self.cube_geoms = self.column_geoms[:len(COLORS)]
        self.distmax = float(_C["distmax_m"])
        self._robot_set = np.zeros(model.ngeom, dtype=bool)
        self._robot_set[self.robot_geoms] = True
        self._column_of = np.full(model.ngeom, -1, dtype=int)
        self._column_of[self.column_geoms] = np.arange(len(self.column_geoms))
        self._fromto = np.zeros(6)
        self.reset_window()

    @staticmethod
    def obstacle_mask(target: str, in_box_colors) -> np.ndarray:
        """障害物の列: 机上の目標外の立方体と、壁 4 枚。"""
        mask = np.ones(len(COLUMNS), dtype=bool)
        for i, c in enumerate(COLORS):
            if c == target or c in in_box_colors:
                mask[i] = False
        return mask

    def reset_window(self) -> None:
        self.window_robot = np.zeros(len(COLUMNS), dtype=bool)
        self.window_cube_cube = np.zeros((len(COLORS), len(COLORS)), dtype=bool)

    def on_step(self, data) -> None:
        """物理ステップの後に呼ぶ。今の接触を窓に OR する。"""
        n = data.ncon
        if n == 0:
            return
        g = data.contact.geom[:n]
        g1, g2 = g[:, 0], g[:, 1]
        c1, c2 = self._column_of[g1], self._column_of[g2]
        r1, r2 = self._robot_set[g1], self._robot_set[g2]
        hit = np.concatenate([c2[r1 & (c2 >= 0)], c1[r2 & (c1 >= 0)]])
        if hit.size:
            self.window_robot[hit] = True
        cc = (c1 >= 0) & (c1 < len(COLORS)) & (c2 >= 0) & (c2 < len(COLORS))
        if cc.any():
            self.window_cube_cube[c1[cc], c2[cc]] = True
            self.window_cube_cube[c2[cc], c1[cc]] = True

    def distances(self, data) -> np.ndarray:
        """(K,) 各列とロボット（手・指の全ての衝突形状）の最短距離 [m]。data の qpos を距離のモデルに写して測る。"""
        dd = self.ddata
        np.copyto(dd.qpos, data.qpos)
        mujoco.mj_kinematics(self.dmodel, dd)
        out = np.full(len(COLUMNS), self.distmax)
        for k, og in enumerate(self.column_geoms):
            for rg in self.distance_geoms:
                d = mujoco.mj_geomDistance(self.dmodel, dd, int(rg), int(og), self.distmax, self._fromto)
                if d < out[k]:
                    out[k] = d
        return out

    def take_window(self):
        """窓の接触（robot (K,), cube_cube (3, 3)）を返して、窓を空にする。"""
        out = (self.window_robot.copy(), self.window_cube_cube.copy())
        self.reset_window()
        return out
