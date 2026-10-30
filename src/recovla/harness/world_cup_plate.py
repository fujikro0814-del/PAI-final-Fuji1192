"""別のタスク「カップを皿に置く」の世界（評価とデータ生成の側。真の状態を持つ）。

    world, suite = make_cup_plate_eval_world(cfg, spec=spec)   # 評価: v3_real の重ね（腕のサーボ・ハンドの遅延・幅のセンサ）を入れる
    rig = CupPlateGenRig(cfg)                                  # データ生成: harness/real_matched.V3RealGenRig と同じ（生成プログラムは真の状態で動く）
    lay = world.sample_layout(seed)                            # sim/scene.sample_cup_plate_layout
    world.reset(lay)
    world.in_place_truth                                     # {物: 皿の上に載って立っているか}（goal_truth.on_place_truth）
    world.truth("red")                                         # 生成プログラムが読む真の状態（expert/script.Truth。cube_pos はタスクの物の中心）
    setup = world.setup_info(cfg)                              # VLA の側の知覚・完了判定へ渡す既知の情報（下の注）

場面の口は CupPlateSim（SimRig を継ぎ、場面を scene_cup_plate.xml にして 3 つの口と真の状態を替える）にまとめ、評価の世界
CupPlateWorldRig(WorldRig, CupPlateSim) とデータ生成の世界 CupPlateGenRig(V3RealGenRig, CupPlateSim) が共に使う（Python の
継承の順で、WorldRig・DrivenRig などの下に CupPlateSim が入り、その下が SimRig になる）:
  _init_objects  タスクの物（形 → 場面の部品: cup → カップ）の体と自由関節の速さの番地（cube_ids・cube_vadr の名前のまま。
                 箱は無い: box = None）
  _make_meter    手・指と物の接触を窓に OR するだけの計器（立方体と箱の列・最短距離の写しのモデルは無い）
  _apply_layout  カップの自由関節と皿の固定の体の位置（sim/scene.apply_cup_plate_layout）
  truth          生成プログラムが読む真の状態（cube_pos・cube_quat・cube_linvel = タスクの物の中心・向き・速さ、box = 皿の上面の中心）
試行の繰り返しは harness/task_loop の run_place_task_trial（run_task_trial が世界の place_task を見て渡す）。

VLA の側へ渡す既知の情報（setup_info、「VLA 側の既知の値」。configs/scenes/cup_plate.yaml の vla_setup）: 知覚
（runtime/perception）・完了判定（runtime/judge.JudgeV2）・ハンドの grasp は立方体と箱の既知の寸法を読むので、
  cube_size  = カップの外径（ハンドの grasp の幅。runtime/runner・harness/driven が読む）
  箱の欄     = 皿: 内寸 = 上面の円に内接する正方形、壁 = box_wall、壁の上端 = 皿の上面の高さ、底 = 皿の上面 ＋ 高さ − 外径
             （JudgeV2 の「底 ＋ 一辺 = 置いた物の上面」が合う）、公称の位置 = 皿の置き方の範囲の中心（試行の皿の位置は渡さない）
  目標位置の入力の平面 cue_plane_z = 机 ＋ cue_height（カップの中心。立方体は cube_size の半分）。方策の目標位置の入力に使う
手で書いたスクリプト方策（CL2）はこれらの欄を読まない（tests/api/test_script_policy.py・test_cup_plate.py で確かめる）。
"""
import dataclasses
import math

import mujoco
import numpy as np

from recovla.api.harness import goal_truth as GT
from recovla.harness.setup import nominal_setup
from recovla.harness.real_matched import V3RealGenRig
from recovla.harness.world import WorldRig
from recovla.runtime.types import frozen_array
from recovla.sim import frames
from recovla.sim import scene as S
from recovla.sim.rig import SimRig

SHAPE_PART = {"cup": "cup"}                   # 物の形 → configs/scenes/cup_plate.yaml の部品
PLACE_PART = {"plate": "plate"}               # 場所の finder の種類 → 部品
DEFAULT_TASK = "configs/tasks/cup_plate.yaml"


class PlaceContactMeter:
    """ContactMeter（sim/contact.py。立方体 3 個と箱の壁の列）の代わり。列は場面の物（names）。手・指と物の接触、物どうしの
    接触を窓に OR する。最短距離の写しのモデルは作らない（distances は distmax で埋める）。g1_audit が読む dmodel・ddata を持つ。"""

    def __init__(self, model, bodies: dict, distmax: float = 0.1):
        self.model = model
        self.dmodel, self.ddata = model, mujoco.MjData(model)
        self.columns = tuple(bodies)
        robot = {model.body(b).id for b in frames.ROBOT_BODIES}
        self._robot = np.zeros(model.nbody, bool)
        self._robot[list(robot)] = True
        self._col = np.full(model.nbody, -1, int)
        for i, b in enumerate(bodies.values()):
            self._col[model.body(b).id] = i
        self.distmax = float(distmax)
        self.reset_window()

    def reset_window(self) -> None:
        n = len(self.columns)
        self.window_robot = np.zeros(n, bool)
        self.window_obj_obj = np.zeros((n, n), bool)

    def on_step(self, data) -> None:
        for k in range(data.ncon):
            c = data.contact[k]
            b1, b2 = self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]
            i1, i2 = self._col[b1], self._col[b2]
            if self._robot[b1] and i2 >= 0:
                self.window_robot[i2] = True
            elif self._robot[b2] and i1 >= 0:
                self.window_robot[i1] = True
            elif i1 >= 0 and i2 >= 0 and i1 != i2:
                self.window_obj_obj[i1, i2] = self.window_obj_obj[i2, i1] = True

    def take_window(self) -> tuple:
        out = (self.window_robot.copy(), self.window_obj_obj.copy())
        self.reset_window()
        return out

    def distances(self, data) -> np.ndarray:
        return np.full(len(self.columns), self.distmax)


def _spec_of(spec):
    if spec is not None:
        return spec
    from recovla.api import task_spec as TS
    return TS.load_task(DEFAULT_TASK)


class CupPlateSim(SimRig):
    """カップと皿の場面の口（単独では使わず、WorldRig・V3RealGenRig と組み合わせる）。_init_place を親の __init__ の前に呼ぶ。"""
    scene_variant = "cup_plate"

    def _init_place(self, spec=None, scene_cfg: dict = None) -> None:
        spec = _spec_of(spec)
        self.place_task = spec                                   # harness/task_loop.run_task_trial が見る印
        self.sc = scene_cfg or S.cup_plate_config()
        self.obj_body = {o.name: self.sc[SHAPE_PART[o.shape]]["body"] for o in spec.objects}
        self.place_body = {p.name: self.sc[PLACE_PART[p.finder.partition(":")[0]]]["body"] for p in spec.places}

    # ---------------------------------------------------------------- SimRig の口
    def _make_meter(self):
        return PlaceContactMeter(self.model, self.obj_body)

    def _init_objects(self) -> None:
        m = self.model
        self.box = None
        self.obj_names = tuple(self.obj_body)
        self.cube_ids = np.array([m.body(b).id for b in self.obj_body.values()])
        self.cube_vadr = np.array([int(m.jnt_dofadr[m.body(b).jntadr[0]]) for b in self.obj_body.values()])

    def _apply_layout(self, layout) -> None:
        S.apply_cup_plate_layout(self.model, self.data, layout, self.sc)

    def sample_layout(self, seed: int, start: str = "home") -> S.CupPlateLayout:
        return S.sample_cup_plate_layout(seed, self.sc, start=start)

    def cubes_in_box(self, data=None) -> tuple:
        return tuple(n for n, v in self.in_place_truth(data).items() if v)

    def target_index(self, target: str) -> int:
        """生成プログラムの目標（色 "red" か物の名前 "cup"）→ タスクの物の並びの添字。"""
        from recovla.api import task_spec as TS
        names = self.obj_names
        if target in names:
            return names.index(target)
        return names.index(TS.step_of_cue(self.place_task, target).obj)

    def truth(self, target: str, data=None):
        """生成プログラムが読む真の状態（expert/script.Truth）。cube_pos はタスクの物の真の中心、box は皿の上面の中心（世界）。"""
        from recovla.expert.script import Truth
        d = self.data if data is None else data
        m = self.model
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, self.hand_id, self._vel6, 0)
        names = self.obj_names
        poses = [self.object_pose(n, d) for n in names]
        return Truth(t=float(d.time), target=self.target_index(target),
                     cube_pos=np.array([self.object_center(n, d) for n in names]),
                     cube_quat=np.array([p[1] for p in poses]), cube_linvel=np.array([p[2] for p in poses]),
                     fingers=d.qpos[self.finger_qadr].copy(), finger_vel=d.qvel[self.finger_vadr].copy(),
                     gripper_closed=bool(self.controller.gripper_closed), hand_pos=d.xpos[self.hand_id].copy(),
                     hand_vel=self._vel6[3:].copy(), fingertip=frames.fingertip_center(d, self.hand_id),
                     x_cmd=self.integrator.x_cmd.copy(), box=self.place_geometry(data=d)["center"].copy())

    # ---------------------------------------------------------------- 真の状態（評価）
    def place_geometry(self, name: str = None, data=None) -> dict:
        """置き場所（皿）の上面: {"center", "normal", "radius", "top_z"}（真の値、世界）。"""
        d = self.data if data is None else data
        name = name or self.place_task.goal.place
        b = self.model.body(self.place_body[name]).id
        R = d.xmat[b].reshape(3, 3)
        p = self.sc["plate"]
        c = d.xpos[b] + R[:, 2] * float(p["thickness"])
        return {"center": c.copy(), "normal": R[:, 2].copy(), "radius": float(p["radius"]), "top_z": float(c[2])}

    def object_pose(self, name: str, data=None) -> tuple:
        """物の (底の中心, 向きの四元数, 体の原点の速さ (3,))（真の値、世界）。体の原点が底の中心（scene_cup_plate.xml）。"""
        d = self.data if data is None else data
        b = self.model.body(self.obj_body[name])
        v = int(self.model.jnt_dofadr[b.jntadr[0]])
        return d.xpos[b.id].copy(), d.xquat[b.id].copy(), d.qvel[v:v + 3].copy()

    def object_height(self, name: str) -> float:
        return float(self.sc[SHAPE_PART[self.place_task.object(name).shape]]["height"])

    def object_center(self, name: str, data=None) -> np.ndarray:
        """物の真の中心（底の中心から軸の向きに高さの半分）。"""
        d = self.data if data is None else data
        b = self.model.body(self.obj_body[name]).id
        return d.xpos[b] + d.xmat[b].reshape(3, 3)[:, 2] * (0.5 * self.object_height(name))

    def on_detail(self, name: str, data=None) -> dict:
        bottom, quat, _ = self.object_pose(name, data)
        return GT.on_surface_truth(bottom, quat, self.place_geometry(data=data), self.sc["success"])

    def in_place_truth(self, data=None) -> dict:
        place = self.place_geometry(data=data)
        objs = {n: self.object_pose(n, data)[:2] for n in self.place_task.goal.objects}
        return GT.on_place_truth(self.place_task, objs, place, self.sc["success"])

    # ---------------------------------------------------------------- 既知の情報（VLA の側の既知の値）
    def setup_info(self, cfg: dict):
        """nominal_setup の cube_size と箱の欄をカップと皿にしたもの（このファイルの始めの注）。試行の皿の位置は渡さない。"""
        p, lay, v = self.sc["plate"], self.sc["layout"], self.sc["vla_setup"]
        top = float(cfg["sim"]["table_top_z"]) + float(p["thickness"])
        wall = float(v["box_wall"])
        inner_half = float(p["radius"]) / math.sqrt(2.0)
        nominal = [float(np.mean(lay["plate_region"]["x"])), float(np.mean(lay["plate_region"]["y"]))]
        base = nominal_setup(cfg)
        return dataclasses.replace(base, cube_size=float(v["object_width"]), box_outer=2.0 * (inner_half + wall),
                                   box_wall=wall, box_height=top,
                                   box_floor=top + float(v["object_height"]) - float(v["object_width"]),
                                   box_nominal_xy=frozen_array(nominal))

    def cue_height(self) -> float:
        """方策の目標位置の入力の投影の平面の、机からの高さ（カップの中心。VLA 側の既知の値）。"""
        return float(self.sc["vla_setup"]["cue_height"])

    def vla_known(self, cfg: dict) -> dict:
        """記録に残す「VLA 側の既知の値」（setup_info で入れた値と、目標位置の入力の平面の高さ）。"""
        su = self.setup_info(cfg)
        return {"cube_size": su.cube_size, "box_outer": su.box_outer, "box_wall": su.box_wall, "box_height": su.box_height,
                "box_floor": su.box_floor, "box_nominal_xy": [float(v) for v in su.box_nominal_xy],
                "cue_height": self.cue_height(), "cue_plane_z": float(su.table_z + self.cue_height()),
                "note": "旧い知覚・JudgeV2・ハンドの grasp・方策の手がかりが読む値。cube_size = カップの外径（grasp の幅）、箱の欄は皿"
                        "（内接する正方形・上面の高さ・底 = 上面 ＋ 高さ − 外径・置き方の範囲の中心）、手がかりの平面 = カップの中心"}


class CupPlateWorldRig(WorldRig, CupPlateSim):
    """評価の世界。WorldRig（腕とハンドの作動のモデル・監査）に、カップと皿の場面の口。"""

    def __init__(self, render: bool = False, cfg: dict = None, audit_fk: bool = True, gravcomp: bool = None, spec=None,
                 scene_cfg: dict = None):
        self._init_place(spec, scene_cfg)
        super().__init__(render=render, cfg=cfg, audit_fk=audit_fk, gravcomp=gravcomp)


class CupPlateGenRig(V3RealGenRig, CupPlateSim):
    """データ生成の世界（harness/real_matched.V3RealGenRig と同じ: 生成プログラムの指令は実行系と同じ口、記録はセンサのモデル）。
    ハンドの grasp の幅と目標位置の入力の平面はカップのもの（setup_info・cue_height）。"""

    def __init__(self, cfg: dict = None, spec=None, scene_cfg: dict = None):
        self._init_place(spec, scene_cfg)
        super().__init__(cfg)
        self.setup_info = CupPlateSim.setup_info(self, self.cfg)       # harness/driven の grasp の幅（cube_size）をカップの外径に

    def episode_meta(self) -> dict:
        out = super().episode_meta()
        su = self.setup_believed
        out["cue_calibration"]["plane_z"] = float(su.table_z + self.cue_height())
        out["task"] = self.place_task.name
        out["vla_known"] = {"cube_size": su.cube_size, "cue_height": self.cue_height()}
        return out


def make_cup_plate_eval_world(cfg: dict, gravcomp=None, spec=None):
    """評価の世界（CupPlateWorldRig）とセンサのモデル。harness/real_matched.make_eval_world と同じ重ねを入れる。"""
    from recovla.harness import real_matched as V
    world = CupPlateWorldRig(render=False, cfg=cfg, gravcomp=gravcomp, spec=spec)
    V.install_actuation(world, cfg)
    V.install_contact(world, cfg)
    return world, V.make_suite(world, cfg)
