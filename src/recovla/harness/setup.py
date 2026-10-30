"""既知の情報（SetupInfo）を作る。実行系に渡すのは、較正誤差を含む「信じている値」。

    nominal_setup(cfg)              # 誤差なし（生成プログラムの駆動・検査用）
    trial_setup(cfg, seed)          # 試行のシードから較正誤差を引いた値（センサのモデルで作る。recovla.harness.sensors）
"""
import numpy as np

from recovla.common import config
from recovla.runtime.types import SetupInfo, frozen_array

ROBOT_XML = "assets/mjcf/robot_only.xml"
BOX_WALL = 0.01                     # 箱の壁の厚さ（scene_3cube.xml の壁の半分の厚さ 0.005 の 2 倍。tests でモデルと照合する）


def nominal_setup(cfg: dict, cameras: dict = None, table_z: float = None, table_normal=None) -> SetupInfo:
    sc, sim, act = cfg["scene"], cfg["sim"], cfg["actuation"]
    box = sc["box"]
    return SetupInfo(
        robot_xml=str(config.path(ROBOT_XML)),
        table_z=float(sim["table_top_z"] if table_z is None else table_z),
        table_normal=frozen_array([0.0, 0.0, 1.0] if table_normal is None else table_normal),
        cube_size=float(sc["cube_size"]),
        box_outer=2.0 * float(box["inner_half"]) + 2.0 * BOX_WALL,
        box_wall=BOX_WALL,
        box_height=float(box["wall_top_z"]),
        box_floor=float(box["floor_z"]),
        box_nominal_xy=frozen_array(box["pos"][:2]),
        cameras=dict(cameras or {}),
        retreat_pose=frozen_array(cfg["expert"]["retreat_pose"]),
        workspace={k: tuple(float(v) for v in sim["workspace"][k]) for k in ("x", "y", "z")},
        cue_fallback_xy=frozen_array([np.mean(sc["region"]["x"]), np.mean(sc["region"]["y"])]),
        grasp_force=float(act["grasp_force"]),
        gripper_speed=float(act["gripper_speed"]),
    )
