"""実行系とシミュレーションの間の入れ物（実行系はシミュレーションの真値に触れず、ここの入れ物だけを受け取る）。

実行系（方策の入口・上位層・完了判定・安全フィルタ・実行器・再試行・知覚）が受け取るのは、ここの入れ物だけ:
  SensorFrame   宣言したセンサの出力（関節の角度・速度、グリッパの開き幅と is_grasped、俯瞰と手首の RGB-D）
  SetupInfo     宣言した既知の情報（ロボットのモデル、机の面、寸法、カメラの内部パラメータ、較正誤差を含む外部パラメータ）
実行系が出すのは RobotIO の口（libfranka と同じ形）だけ:
  command_joints(q)                                  関節の位置の指令（1 kHz 相当。1 回の呼び出しで 1 刻み）
  gripper_move(width, speed) / gripper_grasp(...)    ハンドの口
配列は作ったところ（評価の枠の sense）で書き込み不可にする。numpy だけに依る。
"""
import dataclasses
from typing import Optional, Protocol

import numpy as np


def frozen_array(a, dtype=np.float64) -> np.ndarray:
    out = np.array(a, dtype=dtype, copy=True)
    out.flags.writeable = False
    return out


@dataclasses.dataclass(frozen=True)
class JointState:
    t: float                     # 測った時刻 [s]（シミュレーションの時刻）
    q: np.ndarray                # (7,) [rad]
    dq: np.ndarray               # (7,) [rad/s]
    q_d: np.ndarray = None       # (7,) ロボットが報告する最後の関節の位置の指令（libfranka の RobotState.q_d）。
                                 # 試行の始めに、実行系の指令をここから始める（指令の跳びを作らない）


@dataclasses.dataclass(frozen=True)
class GripperState:
    t: float                     # ハンドが状態を更新した時刻
    width: float                 # 開き幅 [m]
    is_grasped: bool             # ハンドが報告する把持の判定


@dataclasses.dataclass(frozen=True)
class CameraFrame:
    name: str                    # "overhead" | "wrist"
    t_capture: float             # 撮った時刻（ハードウェアの時刻印に相当）
    t_arrival: float             # 実行系に届いた時刻（撮った時刻＋カメラの遅延）
    rgb: np.ndarray              # (256, 256, 3) uint8。方策に渡す切り出し（生の向き。反転は vla_image_spec）
    depth: Optional[np.ndarray]  # (H, W) float32 [m]、欠けは 0。深度の画像（カメラの内部パラメータは SetupInfo）
    seq: int                     # フレームの3 個の連続タスク番号


@dataclasses.dataclass(frozen=True)
class SensorFrame:
    t: float                     # いまの時刻（実行系の時計）
    joints: JointState
    gripper: GripperState
    cameras: dict                # {name: CameraFrame}。t までに届いた最新のフレーム（まだ 1 フレームも届いていなければ無い）


@dataclasses.dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclasses.dataclass(frozen=True)
class Pose:
    """剛体の変換（位置と回転行列）。p_parent = R @ p_child + t。"""
    t: np.ndarray                # (3,)
    R: np.ndarray                # (3, 3)

    def apply(self, p: np.ndarray) -> np.ndarray:
        return np.asarray(p) @ self.R.T + self.t

    def inverse(self) -> "Pose":
        return Pose(frozen_array(-self.R.T @ self.t), frozen_array(self.R.T))

    def compose(self, other: "Pose") -> "Pose":
        return Pose(frozen_array(self.R @ other.t + self.t), frozen_array(self.R @ other.R))


@dataclasses.dataclass(frozen=True)
class CameraSetup:
    """カメラの既知の情報。カメラの座標は MuJoCo と同じ（x 右、y 上、z 後ろ向き。光軸は −z）。"""
    name: str
    rgb_crop: Intrinsics         # 方策に渡す RGB（切り出して 256×256 にしたもの）の内部パラメータ
    depth: Intrinsics            # 深度の画像の内部パラメータ
    extrinsic: Pose              # 俯瞰: 世界 ← カメラ。手首: hand ← カメラ（hand-eye）。どちらも較正誤差を含む実行系が持つ値
    mount: str                   # "world" | "hand"
    depth_unit: float            # 量子化の単位 [m]
    min_z: float                 # 最小距離 [m]


@dataclasses.dataclass(frozen=True)
class SetupInfo:
    robot_xml: str               # ロボットのモデル（場面を含まない）
    table_z: float               # 机の面の高さ（較正誤差を含む実行系が持つ値）
    table_normal: np.ndarray     # (3,) 机の面の法線（同上）
    cube_size: float             # 立方体の一辺 [m]
    box_outer: float             # 箱の外寸（正方形）[m]
    box_wall: float              # 壁の厚さ [m]
    box_height: float            # 壁の上端の机の面からの高さ [m]
    box_floor: float             # 底の上面の机の面からの高さ [m]
    box_nominal_xy: np.ndarray   # 固定具の位置（既知。ずれの監視にだけ使う）
    cameras: dict                # {name: CameraSetup}
    retreat_pose: np.ndarray     # 待機位置（手先）
    workspace: dict              # {"x": (lo, hi), "y": ..., "z": ...}
    cue_fallback_xy: np.ndarray  # 目標位置の入力が一度も見えていないときの (x, y)（課題の配置の範囲の中心。課題の定義）
    grasp_force: float           # 把持力 [N/本]（作動のモデル）
    gripper_speed: float         # 開き幅の速さ [m/s]


class RobotIO(Protocol):
    def sense(self, cameras: bool = True) -> SensorFrame: ...
    def command_joints(self, q: np.ndarray) -> None: ...
    def gripper_move(self, width: float, speed: float) -> None: ...
    def gripper_grasp(self, width: float, speed: float, force: float, eps_inner: float, eps_outer: float) -> None: ...
