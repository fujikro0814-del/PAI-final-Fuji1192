"""記録したエピソードの再生確認（立方体 1 個の場面）。

記録した行動を順に同じ追従器と IK に通すと、記録した手先の軌跡が再現され、再生した画像も記録と一致するはず。
エピソードのフォルダには何も書かない。方式:

    step     物理ステップごとの x_des とグリッパの指令（data.npz の step_*）。開始状態と記録が揃っているかの確認で、
             完全で一致するはず（画像は GPU のゆらぎの範囲で）。
    window   方策が出す 20 Hz の行動だけ: action_xyz[i] = x_des[i+1] - x_des[i] をフレーム i の後の 25 ステップに均等に配り、
             フレーム i+1 のグリッパの指令をその部分の最初のステップから当てる。
    last_step, shifted
             わざと誤った定義（陰性対照）: 部分の最後のステップの x_des の変化だけ、と 1 部分遅延した行動。
             これが失敗しなければ、確認は何も示していない。
"""
import dataclasses
import json
import pathlib

import mujoco
import numpy as np

from recovla.common import config
from recovla.record import recorder
from recovla.sim import control
from recovla.sim.device import DeviceInput, DeviceState

MODES = ("step", "window", "last_step", "shifted")
WINDOW_TOL_M = float(config.load("g0")["g0"]["replay_tol_m"])   # 手先の軌跡の許容（5 mm）
IMAGE_TOL = int(config.load("g0")["g0"]["replay_image_tol"])    # GPU のゆらぎ: 一部の画素で ±1〜2 階調


@dataclasses.dataclass
class Episode:
    path: pathlib.Path
    meta: dict
    data: dict

    @property
    def n_frames(self) -> int:
        return int(self.meta["n_frames"])

    def image(self, camera: str, i: int) -> np.ndarray:
        return recorder.read_png(self.path / camera / f"{i:06d}.png")


def load_episode(path) -> Episode:
    path = pathlib.Path(path)
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    with np.load(path / "data.npz") as npz:
        data = {k: npz[k] for k in npz.files}
    return Episode(path, meta, data)


class ScriptedPad(DeviceInput):
    """制御器に x_des の絶対値とグリッパの閉の目標を渡す。目標が制御器の状態と違えば 1 ステップだけボタンを押す
    （制御器は立ち上がりで切り替わる）。reset_episode が積分器に使う x_cmd・reset を同じ形で持つ。"""

    def __init__(self, controller):
        self.controller = controller
        self.x_cmd = np.zeros(3)
        self.closed_target = False
        self._pressed = False

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def reset(self, pos) -> None:
        self.x_cmd = np.array(pos, dtype=float)

    def read(self) -> DeviceState:
        press = (self.closed_target != self.controller.gripper_closed) and not self._pressed
        self._pressed = press
        return DeviceState(pos=self.x_cmd.copy(), quat=np.array([1.0, 0.0, 0.0, 0.0]),
                           button_grip=press, button_clutch=True)


def _placement(meta) -> recorder.Placement:
    c = meta["cube_init"]
    return recorder.Placement(meta.get("placement_id"), int(meta["placement_seed"]),
                              float(c["x"]), float(c["y"]), float(c["yaw"]))


def _action_schedule(start, actions: np.ndarray, steps_per_action: int):
    """各行動 (dx, dy, dz, rx, ry, rz, gripper ±1) を steps_per_action 個の物理ステップに均等に配る。グリッパは最初のステップから。"""
    pos = np.asarray(start, dtype=float).copy()
    for a in actions:
        begin = pos.copy()
        for j in range(1, steps_per_action + 1):
            pos = begin + a[:3] * (j / steps_per_action)
            yield pos, bool(a[6] > 0.0)


def _schedule(ep: Episode, mode: str, actions=None, frame_stride: int = 1):
    """ステップ 1..n_steps の (x_des, gripper_closed) の目標。"""
    d, every = ep.data, int(ep.meta["record_every"])
    if mode == "actions":  # 外から渡す行動（LeRobot から読み戻したものなど）
        yield from _action_schedule(d["x_des"][0], actions, frame_stride * every)
        return
    n_steps = int(ep.meta["n_steps"])
    if mode == "step":
        closed = d["step_gripper_cmd"] <= 0.5 * 255.0
        for k in range(1, n_steps + 1):
            yield d["step_x_des"][k], bool(closed[k])
        return
    x, g = d["x_des"], d["gripper_closed"]
    n = ep.n_frames
    acts = np.zeros((n - 1, 7))
    for i in range(n - 1):
        if mode == "window":
            acts[i, :3], grip = x[i + 1] - x[i], g[i + 1]
        elif mode == "last_step":
            s = d["step_x_des"]
            acts[i, :3], grip = s[(i + 1) * every] - s[(i + 1) * every - 1], g[i + 1]
        elif mode == "shifted":
            acts[i, :3] = x[i] - x[i - 1] if i > 0 else np.zeros(3)
            grip = g[i]
        else:
            raise ValueError(mode)
        acts[i, 6] = 1.0 if grip else -1.0
    yield from _action_schedule(x[0], acts, every)


def replay(ep: Episode, mode: str, renderer: mujoco.Renderer, model=None,
           actions=None, frame_stride: int = 1) -> dict:
    """1 回再生し、再生したフレーム（data.npz と同じ項目）とカメラごとの画像を返す。mode "actions" は `actions` (K, 7) を
    実行する（1 つが記録の frame_stride フレーム分。10 fps なら 2）。フレームは記録のフレームごとに取る。"""
    model = model or mujoco.MjModel.from_xml_path(control.SCENE_PATH)
    data = mujoco.MjData(model)
    controller = control.make_collect_controller(model, data)
    pad = ScriptedPad(controller)
    start = control.StartState(qpos=ep.data["start_qpos"], ctrl=ep.data["start_ctrl"],
                               q_des=ep.data["start_q_des"])
    control.reset_episode(model, data, controller, pad, start, _placement(ep.meta))
    cameras = tuple(ep.meta["cameras"]["names"])
    sampler = recorder.FrameSampler(model, controller, renderer, cameras)
    every = int(ep.meta["record_every"])

    frames, images = [], {c: [] for c in cameras}

    def capture(step):
        f, imgs = sampler.capture(data, step)
        frames.append(f)
        for c, im in zip(cameras, imgs):
            images[c].append(im)

    capture(0)
    for k, (x_des, closed) in enumerate(_schedule(ep, mode, actions, frame_stride), start=1):
        pad.x_cmd = np.asarray(x_des, dtype=float)
        pad.closed_target = closed
        controller.update(pad)
        mujoco.mj_step(model, data)
        if k % every == 0:
            capture(k)
    out = {name: np.array([f[name] for f in frames]) for name in recorder.FRAME_FIELDS}
    out["images"] = images
    box = model.body(recorder.BOX_BODY).pos  # 動かない物体
    out["success"] = recorder.cube_in_box(out["cube_pos"][-1], box)
    return out


def compare(ep: Episode, rep: dict, images: bool = True) -> dict:
    d = ep.data
    n = min(ep.n_frames, len(rep["ee_pos"]))  # 10 fps の再生は最後の 0.1 s の区切りで終わる
    ee_err = np.linalg.norm(rep["ee_pos"][:n] - d["ee_pos"][:n], axis=1)
    result = {
        "n_frames": n,
        "ee_err_max_m": float(ee_err.max()),
        "ee_err_mean_m": float(ee_err.mean()),
        "ee_err_final_m": float(ee_err[-1]),
        "joint_err_max_rad": float(np.abs(rep["joints"][:n] - d["joints"][:n]).max()),
        "x_des_err_max_m": float(np.abs(rep["x_des"][:n] - d["x_des"][:n]).max()),
        "gripper_mismatch_frames": int((rep["gripper_closed"][:n] != d["gripper_closed"][:n]).sum()),
        "cube_final_err_m": float(np.linalg.norm(rep["cube_pos"][n - 1] - d["cube_pos"][n - 1])),
        "success_recorded": bool(ep.meta["success"]),
        "success_replayed": bool(rep["success"]),
        "bit_exact_state": bool(np.array_equal(rep["joints"][:n], d["joints"][:n])
                                and np.array_equal(rep["ee_pos"][:n], d["ee_pos"][:n])),
    }
    if images:
        for cam, frames in rep["images"].items():
            worst, frac = 0, 0.0
            for i in range(n):
                diff = np.abs(frames[i].astype(int) - ep.image(cam, i).astype(int))
                worst = max(worst, int(diff.max()))
                frac = max(frac, float((diff.max(axis=2) > IMAGE_TOL).mean()))
            result[f"img_{cam}_max_diff"] = worst
            result[f"img_{cam}_worst_frame_fraction_over_tol"] = frac
    return result
