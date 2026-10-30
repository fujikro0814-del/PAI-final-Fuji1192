"""制御ループの部品: 手先の速度の指令から、追従器・微分逆運動学（DLS）・関節の PD までをつなぐ。

数値は configs/default.yaml（sim・controller）から読む（tests/test_c_port.py が記録済みの値と照合する）。

信号の流れ:

    pad (ScriptPad) -> vel
      -> VelocityCommandIntegrator（物理の 1 ステップごとに x_cmd += vel * timestep）
      -> 制御器のクラッチ（常に接続、倍率 1.0） -> desired_pos = x_des
      -> 追従器 -> IK -> PD
"""
import contextlib
import dataclasses
import io

import mujoco
import numpy as np

from recovla.common import config
from recovla.sim.controller_ik import TeleopControllerIK
from recovla.sim.device import DeviceInput, DeviceState

CFG = config.load()
_SIM = CFG["sim"]

SCENE_PATH = str(config.path(CFG["paths"]["scene_g0"]))     # 立方体 1 個の場面
HOME_QPOS = [float(q) for q in _SIM["home_qpos"]]
COLLECT_WORKSPACE = dict(
    workspace_x=tuple(_SIM["workspace"]["x"]),
    workspace_y=tuple(_SIM["workspace"]["y"]),
    workspace_z=tuple(_SIM["workspace"]["z"]),
)
START_POS = np.array(_SIM["start_pos"], dtype=float)
SETTLE_SECONDS = float(_SIM["settle_s"])
FINGER_OPEN = float(_SIM["finger_open"])     # finger_joint1/2 の可動域の上限 [m]
RECORD_EVERY = int(_SIM["record_every"])     # 20 Hz のフレーム 1 つあたりの物理ステップ数（500 / 20）
IMAGE_SIZE = int(_SIM["image_size"])
CAMERAS = tuple(_SIM["cameras"])
FINGERTIP_OFFSET = float(_SIM["fingertip_offset"])   # hand の原点から指先の中心まで（hand の z 方向）[m]
TABLE_TOP_Z = float(_SIM["table_top_z"])
STEPS_PER_PAD_READ = int(_SIM["steps_per_pad_read"])

_IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])


def check_timestep(model) -> None:
    """モデルの timestep が設定（sim.timestep）と違えば止める。"""
    if float(model.opt.timestep) != float(_SIM["timestep"]):
        raise RuntimeError(f"model timestep {model.opt.timestep} != configs sim.timestep {_SIM['timestep']}")


# ------------------------------------------------------------------ 制御器

def make_controller(model, data, **overrides):
    """DLS の制御器（調整値は configs の controller）。overrides は個々の引数の上書き（テスト用）。"""
    c = CFG["controller"]
    kwargs = dict(
        ik_gain=float(c["ik_gain"]),
        max_joint_step=float(c["max_joint_step"]),
        rot_weight=float(c["rot_weight"]),
        lock_orientation=bool(c["lock_orientation"]),
        null_gain=float(c["null_gain"]),
        ik_damping=float(c["ik_damping"]),
        tracker_omega=float(c["tracker_omega"]),
        max_target_vel=float(c["max_target_vel"]),
        q_des_leash=float(c["q_des_leash"]),
        max_dx_norm=float(c["max_dx_norm"]),
        scale=float(c["scale"]),
        **COLLECT_WORKSPACE,
    )
    kwargs.update(overrides)
    return TeleopControllerIK(model, data, **kwargs)


def apply_home_pose(model, data) -> None:
    """腕の 7 関節に HOME_QPOS を書き、派生量を更新する（関節は名前で引くので、立方体の qpos の並びによらない）。"""
    for i, q in enumerate(HOME_QPOS, start=1):
        data.qpos[model.joint(f"joint{i}").qposadr[0]] = q
    mujoco.mj_forward(model, data)


# ---------------------------------------------------------------- x_des

class VelocityCommandIntegrator(DeviceInput):
    """入力の速度の指令を積分して、制御器のクラッチが読む絶対位置 x_cmd にする。

    refresh は入力を読む（入力の読み取りごと）。read は controller.update が物理の 1 ステップごとに呼び、
    最新の速度を 1 ステップぶん積分する。
    """

    def __init__(self, inner: DeviceInput, timestep: float,
                 workspace_x, workspace_y, workspace_z):
        self.inner = inner
        self.timestep = float(timestep)
        self.workspace_x = workspace_x
        self.workspace_y = workspace_y
        self.workspace_z = workspace_z
        self.x_cmd = np.zeros(3)
        self.latest = DeviceState(pos=np.zeros(3), quat=_IDENTITY_QUAT.copy(),
                                  button_grip=False, button_clutch=True,
                                  vel=np.zeros(3))
        # False: 腕は入力を無視する（速度 0、グリッパのボタンも無視）。refresh での読み取りは続ける
        self.enabled = True
        # command_filter(x_cmd, step) -> step: 安全フィルタ（recovla.sim.safety）。None ならフィルタなし
        self.command_filter = None

    def start(self) -> None:
        self.inner.start()

    def stop(self) -> None:
        self.inner.stop()

    def reset(self, pos) -> None:
        self.x_cmd = self._clamp(np.array(pos, dtype=float))

    def refresh(self) -> DeviceState:
        self.latest = self.inner.read()
        return self.latest

    def read(self) -> DeviceState:
        s = self.latest
        if self.enabled and s.vel is not None:
            step = np.asarray(s.vel, dtype=float) * self.timestep
            if self.command_filter is not None:          # 安全フィルタの差し込み口
                step = self.command_filter(self.x_cmd, step)
            self.x_cmd = self._clamp(self.x_cmd + step)
        return DeviceState(pos=self.x_cmd.copy(), quat=_IDENTITY_QUAT.copy(),
                           button_grip=bool(self.enabled and s.button_grip),
                           button_clutch=True)

    def _clamp(self, pos: np.ndarray) -> np.ndarray:
        pos[0] = np.clip(pos[0], *self.workspace_x)
        pos[1] = np.clip(pos[1], *self.workspace_y)
        pos[2] = np.clip(pos[2], *self.workspace_z)
        return pos


def make_collect_controller(model, data):
    """データ生成用の制御器（make_controller に制御の周期を物理の 1 ステップに合わせたもの）。

    先に HOME_QPOS を data に書く: IK の零空間の姿勢の目標 q_neutral は作った時点の関節角で、
    新しい MjData はすべて 0 の直立の姿勢のため（呼ぶ側はその後でエピソードの状態を置き直す）。"""
    apply_home_pose(model, data)
    controller = make_controller(model, data)
    controller.ctrl_dt = float(model.opt.timestep)
    return controller


def make_integrator(device: DeviceInput, model, controller):
    return VelocityCommandIntegrator(
        device, model.opt.timestep, controller.workspace_x,
        controller.workspace_y, controller.workspace_z)


# ---------------------------------------------------------- 開始の状態

@dataclasses.dataclass
class StartState:
    qpos: np.ndarray
    ctrl: np.ndarray
    q_des: np.ndarray


class _Hold(DeviceInput):
    """クラッチを離した入力: 制御器は設定済みの desired_pos を保つ。"""

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def read(self) -> DeviceState:
        return DeviceState(pos=np.zeros(3), quat=_IDENTITY_QUAT.copy(),
                           button_grip=False, button_clutch=False)


def _set_fingers(model, data, opening: float) -> None:
    for name in ("finger_joint1", "finger_joint2"):
        data.qpos[model.joint(name).qposadr[0]] = opening


def _open_gripper(controller) -> None:
    for idx in controller.gripper_indices:
        controller.data.ctrl[idx] = controller.model.actuator_ctrlrange[idx][1]
    controller.gripper_closed = False


def settle_start_state(model, seconds: float = SETTLE_SECONDS,
                       tol: float = 0.010, qvel_tol: float = 1e-3, pos=None) -> StartState:
    """HOME_QPOS から、データ生成用の制御器（物理の周期の追従器＋IK、グリッパは開）で手を START_POS へ動かし、
    静止させる。得られる qpos・ctrl・q_des はサーボの重力による定常のずれを含むので、これに置き直したエピソードは
    静止した状態から始まる。

    tol は意図して緩い: DLS の IK は desired_pos と手の間に定常のずれを残す（減衰があると零空間の姿勢の項が
    厳密に零空間に入らない。START_POS で 5 mm、作業空間の中で 5〜17 mm、主に +z。null_gain=0 なら 0）。
    ここでは START_POS の近くで止まったことだけを確かめる。

    pos: 別の手の位置で静止させる（3 色の場面の待機姿勢）。None なら START_POS。"""
    goal = START_POS if pos is None else np.asarray(pos, dtype=float)
    data = mujoco.MjData(model)
    apply_home_pose(model, data)
    _set_fingers(model, data, FINGER_OPEN)
    mujoco.mj_forward(model, data)
    with contextlib.redirect_stdout(io.StringIO()):  # IK の診断の出力を捨てる
        controller = make_collect_controller(model, data)
        _open_gripper(controller)
        controller.sync_target_to_hand()
        controller.desired_pos = controller._clamp_workspace(goal.copy())
        hold = _Hold()
        for _ in range(int(round(seconds / model.opt.timestep))):
            controller.update(hold)
            mujoco.mj_step(model, data)
    err = float(np.linalg.norm(data.xpos[controller.hand_body_id] - goal))
    speed = float(np.linalg.norm(data.qvel[controller.dof_indices]))
    if err > tol or speed > qvel_tol:
        raise RuntimeError(
            f"start state did not settle after {seconds} s: hand "
            f"{err * 1000:.2f} mm from {goal}, |qvel| {speed:.2e} rad/s")
    return StartState(qpos=data.qpos.copy(), ctrl=data.ctrl.copy(),
                      q_des=controller.q_des.copy())


def reset_episode(model, data, controller, integrator, start: StartState,
                  placement=None) -> None:
    """静止させた開始の状態をそのまま戻し（placement＝recorder.Placement があれば立方体をそこに置く）、x_des をそこに合わせる。"""
    from recovla.record import recorder          # recorder がこのモジュールの定数を読むので、ここで読み込む
    data.qpos[:] = start.qpos
    data.qvel[:] = 0.0
    data.qacc_warmstart[:] = 0.0
    data.ctrl[:] = start.ctrl
    data.time = 0.0
    if placement is not None:
        recorder.apply_placement(model, data, placement)
    mujoco.mj_forward(model, data)
    controller.q_des = start.q_des.copy()
    controller.gripper_closed = False
    controller.sync_target_to_hand()
    integrator.reset(controller.target_pos)
    # ここで同じ基準でクラッチをつなぐ。最初の update に任せると、積分器が 1 ステップ進んだ後に基準を取り、
    # エピソードの間ずっと desired_pos が x_cmd から 1 ステップぶん（vel * dt）ずれる
    controller.device_ref_pos = integrator.x_cmd.copy()
    controller.target_ref_pos = controller.target_pos.copy()
    controller.device_ref_quat = _IDENTITY_QUAT.copy()
    controller.target_ref_quat = controller.target_quat.copy()
    controller.prev_clutch = True
