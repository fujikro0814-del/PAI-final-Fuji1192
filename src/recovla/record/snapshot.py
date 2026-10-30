"""シミュレーションの途中の状態の保存と復元（記録の開始状態と、途中からの保存・再生に使う）。

保存するもの:
  - 物理: mj_getState(mjSTATE_INTEGRATION)（qpos・qvel・act・ctrl・mocap・warmstart など）
  - 制御器: q_des、target_pos・target_vel・desired_pos・target_quat、gripper_closed・prev_grip・
    prev_clutch、クラッチの基準 4 つ
  - 積分器: x_cmd、enabled
  - 物理ステップの番号（20 Hz・10 Hz の位相）
パッドの状態は保存しない（復元の後、最初の入力の読み取りで上書きされる）。
"""
import dataclasses

import mujoco
import numpy as np

SPEC = mujoco.mjtState.mjSTATE_INTEGRATION
CONTROLLER_FIELDS = ("q_des", "target_pos", "target_vel", "desired_pos", "target_quat",
                     "device_ref_pos", "target_ref_pos", "device_ref_quat", "target_ref_quat")
FLAG_FIELDS = ("gripper_closed", "prev_grip", "prev_clutch")


@dataclasses.dataclass
class SimSnapshot:
    state: np.ndarray
    controller: dict
    flags: dict
    x_cmd: np.ndarray
    integrator_enabled: bool
    step: int

    def to_arrays(self, prefix: str = "start_") -> dict:
        """npz に入れる形（None の基準は NaN で埋める）。"""
        out = {f"{prefix}state": self.state, f"{prefix}x_cmd": self.x_cmd,
               f"{prefix}step": np.int64(self.step), f"{prefix}integrator_enabled": np.bool_(self.integrator_enabled)}
        for k, v in self.controller.items():
            out[f"{prefix}ctl_{k}"] = np.full(4 if "quat" in k else 3, np.nan) if v is None else np.asarray(v)
        for k, v in self.flags.items():
            out[f"{prefix}flag_{k}"] = np.bool_(v)
        return out

    @classmethod
    def from_arrays(cls, arrays: dict, prefix: str = "start_") -> "SimSnapshot":
        ctl = {}
        for k in CONTROLLER_FIELDS:
            v = np.asarray(arrays[f"{prefix}ctl_{k}"], dtype=float)
            ctl[k] = None if np.isnan(v).all() else v.copy()
        return cls(state=np.asarray(arrays[f"{prefix}state"]).copy(), controller=ctl,
                   flags={k: bool(arrays[f"{prefix}flag_{k}"]) for k in FLAG_FIELDS},
                   x_cmd=np.asarray(arrays[f"{prefix}x_cmd"], dtype=float).copy(),
                   integrator_enabled=bool(arrays[f"{prefix}integrator_enabled"]), step=int(arrays[f"{prefix}step"]))


def capture(model, data, controller, integrator, step: int) -> SimSnapshot:
    state = np.zeros(mujoco.mj_stateSize(model, SPEC))
    mujoco.mj_getState(model, data, state, SPEC)
    ctl = {k: (None if getattr(controller, k) is None else np.array(getattr(controller, k), dtype=float))
           for k in CONTROLLER_FIELDS}
    return SimSnapshot(state, ctl, {k: bool(getattr(controller, k)) for k in FLAG_FIELDS},
                       np.array(integrator.x_cmd, dtype=float), bool(integrator.enabled), int(step))


def restore(model, data, controller, integrator, snap: SimSnapshot) -> None:
    """保存した時点の状態に戻す。integrator は x_cmd を持つもの（積分器、または再生用の ScriptedPad）。"""
    mujoco.mj_setState(model, data, snap.state, SPEC)
    mujoco.mj_forward(model, data)
    # mujoco 3.2.3 の mj_forward は qacc_warmstart を書き換える。派生量（xpos・ヤコビアン）は
    # そのまま使い、状態（warmstart を含む）だけをもう一度書き戻す
    mujoco.mj_setState(model, data, snap.state, SPEC)
    for k, v in snap.controller.items():
        setattr(controller, k, None if v is None else np.array(v, dtype=float))
    for k, v in snap.flags.items():
        setattr(controller, k, bool(v))
    controller.data.mocap_pos[controller.mocap_id] = controller.target_pos
    controller.data.mocap_quat[controller.mocap_id] = controller.target_quat
    integrator.x_cmd = np.array(snap.x_cmd, dtype=float)
    if hasattr(integrator, "enabled"):
        integrator.enabled = bool(snap.integrator_enabled)
