"""実行系の安全フィルタ。シミュレーション側（recovla.sim.safety）と同じ二次計画で、障害物を知覚から作る。

    sf = PerceptionSafetyFilter(setup, cfg["safety_filter"], extra_margin)
    sf.start_trial(target)
    sf.set_world(world_model)                    # 知覚の結果（10 Hz）: 目標以外の机上の立方体と、推定した箱の壁を置く
    sf.begin_read(q, width, x_cmd)               # 入力の読み取りごと（50 Hz）: 測った関節角で距離と向きを測る
    dx = sf.filter(x_cmd, dx)                    # 物理ステップごと（Motion の差し込み口）

- 障害物: 知覚した目標以外の立方体（「見えている」か「隠れて保持中」で、箱の中でも手の中でもないもの。既知の寸法の箱を
  推定した位置と向きに置く）と、推定した箱の壁 4 枚（既知の寸法）。距離は知覚から作る世界のモデル（runtime.robot_model.belief_model）
- 余裕: d_min と検出の距離に、知覚した障害物の表面位置の誤差の p95（学習用の配置で測った値）を足す
- 手先の遅延: 参照位置での見積もりと実際の手の距離（測った関節角の順運動学）の小さい方
"""
import math

import mujoco
import numpy as np

from recovla.runtime import robot_model
from recovla.runtime.qp import project

_FAR = np.array([0.0, 0.0, -10.0])


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


class PerceptionSafetyFilter:
    def __init__(self, setup, sf_cfg: dict, extra_margin: float):
        s = setup
        self.m = robot_model.belief_model(s.robot_xml, s.cube_size, s.box_outer, s.box_wall, s.box_height, s.box_floor)
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.extra = float(extra_margin)
        self.d_min = float(sf_cfg["d_min_m"]) + self.extra
        self.d_detect = float(sf_cfg["d_detect_m"]) + self.extra
        self.gamma = float(sf_cfg["gamma"])
        self.enabled = bool(sf_cfg["enabled"])
        bodies = {m.body(b).id for b in robot_model.HAND_BODIES}
        self.robot_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in bodies and m.geom_contype[g]]
        self.cube_geoms = [m.geom(f"belief_cube_{i}_geom").id for i in range(robot_model.N_CUBES)]
        self.cube_mocap = [m.body_mocapid[m.body(f"belief_cube_{i}").id] for i in range(robot_model.N_CUBES)]
        self.wall_geoms = [m.geom(f"belief_wall_{w}").id for w in ("xp", "xn", "yp", "yn")]
        self.box_mocap = m.body_mocapid[m.body("belief_box").id]
        self.arm_qadr = np.array([m.joint(f"joint{i}").qposadr[0] for i in range(1, 8)])
        self.finger_qadr = np.array([m.joint(n).qposadr[0] for n in ("finger_joint1", "finger_joint2")])
        self.hand_id = m.body("hand").id
        self._fromto = np.zeros(6)
        self.target = None
        self.gate = False
        self.reset_trial()

    def reset_trial(self) -> None:
        self._rows, self._x0, self._last_n = [], None, {}
        self.n_active_steps, self.total_change_m, self.infeasible_steps = 0, 0.0, 0
        self.obstacles = []
        for mc in self.cube_mocap:
            self.d.mocap_pos[mc] = _FAR
        self.d.mocap_pos[self.box_mocap] = _FAR
        self.window_active = False

    def start_trial(self, target: str) -> None:
        self.reset_trial()
        self.target = target

    def on(self) -> bool:
        return self.enabled and self.gate and self.target is not None

    def set_world(self, wm) -> None:
        self.obstacles = []
        for i, (c, e) in enumerate(sorted(wm.cubes.items())):
            mc = self.cube_mocap[i]
            use = c != self.target and not e.in_box and e.status in ("seen", "held")
            self.d.mocap_pos[mc] = e.pos if use else _FAR
            self.d.mocap_quat[mc] = _yaw_quat(e.yaw)
            if use:
                self.obstacles.append((self.cube_geoms[i], c))
        if wm.box.get("ok"):
            self.d.mocap_pos[self.box_mocap] = [wm.box["xy"][0], wm.box["xy"][1], wm.table_z]
            self.d.mocap_quat[self.box_mocap] = _yaw_quat(wm.box["yaw"])
            self.obstacles += [(g, "wall") for g in self.wall_geoms]
        else:
            self.d.mocap_pos[self.box_mocap] = _FAR

    def summary(self) -> dict:
        return {"enabled": self.enabled, "d_min_m": self.d_min, "d_detect_m": self.d_detect, "extra_margin_m": self.extra,
                "gamma": self.gamma, "active_physics_steps": self.n_active_steps,
                "total_change_m": self.total_change_m, "infeasible_steps": self.infeasible_steps}

    def begin_read(self, q, width: float, x_cmd) -> None:
        self._rows = []
        self._x0 = np.array(x_cmd, float)
        if not self.on() or not self.obstacles:
            return
        m, d = self.m, self.d
        d.qpos[self.arm_qadr] = q
        d.qpos[self.finger_qadr] = 0.5 * float(width)
        mujoco.mj_kinematics(m, d)
        lag = self._x0 - d.xpos[self.hand_id]
        dmax = self.d_detect + 0.05
        for og, key0 in self.obstacles:
            rows_k = []
            for rg in self.robot_geoms:
                best = mujoco.mj_geomDistance(m, d, int(rg), int(og), dmax, self._fromto)
                if best >= dmax:
                    continue
                v = self._fromto[:3] - self._fromto[3:]
                nv = np.linalg.norm(v)
                key = (int(og), int(rg))
                if abs(best) > 1e-4 and nv > 1e-9:
                    n = v / nv if best > 0 else -v / nv
                    self._last_n[key] = n
                elif key in self._last_n:
                    n = self._last_n[key]
                else:
                    continue
                d_ref = min(best, best + float(n @ lag))
                if d_ref < self.d_detect:
                    rows_k.append((n, d_ref, int(og)))
            rows_k.sort(key=lambda r: r[1])
            for r in rows_k:
                if all(float(r[0] @ q_[0]) <= 0.95 for q_ in self._rows if q_[2] == r[2]):
                    self._rows.append(r)

    def filter(self, x_cmd, dx):
        if not self._rows or not self.on():
            return dx
        moved = np.asarray(x_cmd, float) - self._x0
        A = np.array([n for n, _, _ in self._rows])
        d_now = np.array([dd + float(n @ moved) for n, dd, _ in self._rows])
        b = -self.gamma * (d_now - self.d_min)
        x, ok = project(dx, A, b)
        ch = float(np.linalg.norm(x - dx))
        if ch > 0.0:
            self.window_active = True
            self.n_active_steps += 1
            self.total_change_m += ch
        if not ok:
            self.infeasible_steps += 1
        return x
