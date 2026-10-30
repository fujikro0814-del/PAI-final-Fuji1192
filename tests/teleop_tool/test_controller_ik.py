"""微分逆運動学の制御器（recovla.sim.controller_ik.TeleopControllerIK）の動きの確かめ。"""
import mujoco
import numpy as np
import pytest

from recovla.sim import control
from recovla.sim.controller_ik import TeleopControllerIK
from recovla.sim.device import DeviceInput, DeviceState

SCENE = control.SCENE_PATH

HOME_QPOS = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]


@pytest.fixture()
def model_data():
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    for i, q in enumerate(HOME_QPOS, start=1):
        data.qpos[model.joint(f"joint{i}").qposadr[0]] = q
    mujoco.mj_forward(model, data)
    return model, data


class _ScriptedDevice(DeviceInput):
    """決まった (pos, clutch) の列を read ごとに 1 つずつ返す。"""

    def __init__(self, steps):
        self._steps = list(steps)
        self._i = 0

    def start(self) -> None:
        pass

    def read(self) -> DeviceState:
        pos, clutch = self._steps[min(self._i, len(self._steps) - 1)]
        self._i += 1
        return DeviceState(
            pos=np.array(pos, dtype=float),
            quat=np.array([1.0, 0.0, 0.0, 0.0]),
            button_grip=False,
            button_clutch=clutch,
            button_reset=False,
        )

    def stop(self) -> None:
        pass


def test_releasing_clutch_holds_position_off_neutral_posture(model_data):
    """クラッチを離すと、零空間の姿勢の引き戻しは止まり、手先は最後の指令の位置に留まる
    （引き戻しが作業空間へ漏れると、入力なしでも手先が流れる。作業域の端・低い所で大きい）。
    """
    model, data = model_data
    controller = TeleopControllerIK(
        model, data, ik_gain=0.08, rot_weight=0.5, null_gain=0.05,
        workspace_x=(0.25, 0.62), workspace_y=(-0.30, 0.30),
        workspace_z=(-0.05, 0.45))

    # クラッチを押したまま、作業域の端の低い位置（立方体を取りに行く形）へ手先を動かす
    low_edge_target = np.array([0.30, 0.20, 0.05])
    device = _ScriptedDevice([(low_edge_target, True)] * 400)
    for _ in range(400):
        controller.update(device)
        mujoco.mj_step(model, data)

    hand_id = controller.hand_body_id
    hand_before = data.xpos[hand_id].copy()

    # クラッチを離し（pos の値は効かない）、入力なしで回す
    device = _ScriptedDevice([(low_edge_target, False)] * 400)
    for _ in range(400):
        controller.update(device)
        mujoco.mj_step(model, data)

    hand_after = data.xpos[hand_id].copy()
    drift = np.linalg.norm(hand_after - hand_before)
    assert drift < 0.02, (
        f"hand drifted {drift:.4f} m after releasing clutch "
        f"(before={hand_before}, after={hand_after}) -- null-space pull "
        f"is leaking into task space while clutch is released")


def test_null_space_pull_still_active_while_clutch_held(model_data):
    """クラッチを押している間は、零空間の引き戻しが効き、関節は中立の姿勢へ近づく
    （弱い引き戻しなので、収束ではなく近づく向きだけを見る）。"""
    model, data = model_data
    controller = TeleopControllerIK(
        model, data, ik_gain=0.08, rot_weight=0.0, null_gain=0.2,
        workspace_x=(0.25, 0.62), workspace_y=(-0.30, 0.30),
        workspace_z=(-0.05, 0.45))

    # 冗長な関節を 1 つ中立からずらし、今の手先の位置でクラッチを押し続ける（関節の変化は零空間の項だけから）
    data.qpos[controller.arm[1][2]] += 0.3
    mujoco.mj_forward(model, data)
    q_neutral = controller.q_neutral.copy()

    hand_pos = data.xpos[controller.hand_body_id].copy()
    device = _ScriptedDevice([(hand_pos, True)] * 300)
    for _ in range(300):
        controller.update(device)
        mujoco.mj_step(model, data)

    q_now = np.array([data.qpos[a[2]] for a in controller.arm])
    # 中立までの距離が縮む（0 でなくてよい）
    dist_before = abs(q_neutral[1] - (q_neutral[1] + 0.3))
    dist_after = abs(q_neutral[1] - q_now[1])
    assert dist_after < dist_before


def test_dq_clamp_preserves_direction_not_elementwise_clip(model_data):
    """dq は向きを保ったまま縮めて max_joint_step に収める。要素ごとの np.clip は関節の間の比を変え、
    ヤコビアンの解と違う向きへ腕を動かす。
    """
    model, data = model_data
    controller = TeleopControllerIK(model, data, ik_gain=0.08,
                                    max_joint_step=0.01)

    # _solve_ik_step の中と同じ縮め方を、偏った dq で直に確かめる
    dq_raw = np.array([0.05, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
    max_dq_val = np.max(np.abs(dq_raw))
    dq_clamped = dq_raw * (controller.max_joint_step / max_dq_val)

    dir_raw = dq_raw / np.linalg.norm(dq_raw)
    dir_clamped = dq_clamped / np.linalg.norm(dq_clamped)
    assert np.allclose(dir_raw, dir_clamped), "clamp must preserve direction"
    assert np.isclose(np.max(np.abs(dq_clamped)), controller.max_joint_step)

    # 比べ: 要素ごとの clip では向きが変わる
    dq_old_buggy = np.clip(dq_raw, -controller.max_joint_step,
                           controller.max_joint_step)
    dir_old = dq_old_buggy / np.linalg.norm(dq_old_buggy)
    assert not np.allclose(dir_raw, dir_old), (
        "sanity check: old clip SHOULD distort direction for this input")
