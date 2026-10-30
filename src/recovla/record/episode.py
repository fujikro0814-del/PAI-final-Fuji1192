"""3 色の場面のエピソードの記録。

1 本のエピソードは 1 つのフォルダ `<回>/<名前>/`（名前は `<種類>_<配置のシード>_<色>_r<作り直しの回数>`）:

    overhead/000000.png ...   256x256 RGB（描画ありのときだけ）
    wrist/000000.png ...
    data.npz                  フレームごとの配列（FRAME_FIELDS）、物理ステップごとの x_des・グリッパの指令、開始状態
    meta.json

保存中は `<名前>.partial`、保存が済んだら名前を変える。フレームの取り方・行動の定義は立方体 1 個の記録器
（recovla.record.recorder）と同じ: フレーム i は物理ステップ 25i、状態と画像は同じ時点（scratch に複写して mj_forward）、
行動は保存しない（変換で x_des の差から作る）。
"""
import json
import os
import pathlib
import queue
import shutil
import threading

import mujoco
import numpy as np

from recovla.common.seeds import COLORS
from recovla.expert.script import phase_of
from recovla.record.recorder import write_png
from recovla.sim import contact

SCHEMA_VERSION = 1
K = len(contact.COLUMNS)
FRAME_FIELDS = {
    # 名前: (1 フレームの形, 型)
    "step": ((), np.int64),
    "sim_time": ((), np.float64),
    "ee_pos": ((3,), np.float64),
    "ee_quat": ((4,), np.float64),
    "ee_axisangle": ((3,), np.float64),
    "fingers": ((2,), np.float64),
    "joints": ((7,), np.float64),
    "joint_vel": ((7,), np.float64),
    "x_des": ((3,), np.float64),
    "tracker_target": ((3,), np.float64),
    "gripper_cmd": ((), np.float64),
    "gripper_closed": ((), np.bool_),
    "fingertip": ((3,), np.float64),
    "cube_pos": ((3, 3), np.float64),          # [色, xyz]
    "cube_quat": ((3, 4), np.float64),
    "cube_linvel": ((3, 3), np.float64),
    "cube_in_box": ((3,), np.bool_),
    "target": ((), np.int8),
    "phase": ((), np.int8),
    "contact_robot": ((K,), np.bool_),        # 直前の窓（25 物理ステップ）に手・指が列 k に触れた
    "contact_cube_cube": ((3, 3), np.bool_),
    "min_dist": ((K,), np.float64),           # フレームの時点の最短距離（手・指のメッシュ）
    "safety_active": ((), np.bool_),
}


def capture_frame(rig, target: str, target_rest_s: float, pp, render: bool = True):
    """今の状態の 1 フレーム（状態・真値・接触の窓・最短距離・段階）と画像。接触の窓はここで空になる。"""
    s = rig.forward_scratch()
    tr = rig.truth(target, s)
    h = rig.hand_id
    quat = s.xquat[h].copy()
    if quat[0] < 0.0:
        quat = -quat
    axisangle = np.zeros(3)
    mujoco.mju_quat2Vel(axisangle, quat, 1.0)
    c_robot, c_cc = rig.meter.take_window()
    frame = {
        "step": rig.step, "sim_time": s.time, "ee_pos": s.xpos[h].copy(), "ee_quat": quat,
        "ee_axisangle": axisangle, "fingers": s.qpos[rig.finger_qadr].copy(),
        "joints": s.qpos[rig.arm_qadr].copy(), "joint_vel": s.qvel[rig.arm_vadr].copy(),
        "x_des": rig.controller.desired_pos.copy(), "tracker_target": rig.controller.target_pos.copy(),
        "gripper_cmd": float(rig.data.ctrl[rig.grip_act]), "gripper_closed": bool(rig.controller.gripper_closed),
        "fingertip": tr.fingertip, "cube_pos": tr.cube_pos, "cube_quat": tr.cube_quat,
        "cube_linvel": tr.cube_linvel,
        "cube_in_box": np.array([c in rig.cubes_in_box(s) for c in COLORS]),
        "target": COLORS.index(target), "phase": int(phase_of(tr, target_rest_s, pp)),
        "contact_robot": c_robot, "contact_cube_cube": c_cc, "min_dist": rig.meter.distances(s),
        "safety_active": bool(rig.safety.take_window()) if hasattr(rig, "safety") else False,
    }
    return frame, (rig.render(s) if render else [])


class SceneEpisodeWriter:
    """PNG は別のスレッドで書き（recorder.EpisodeWriter と同じ）、配列はメモリに貯める。"""

    def __init__(self, run_dir, name: str, cameras):
        self.final_dir = pathlib.Path(run_dir) / name
        self.dir = self.final_dir.with_name(name + ".partial")
        if self.final_dir.exists() or self.dir.exists():
            raise FileExistsError(self.final_dir)
        self.cameras = tuple(cameras)
        self.dir.mkdir(parents=True)
        for cam in self.cameras:
            (self.dir / cam).mkdir()
        self.frames = []
        self.step_x_des = []
        self.step_gripper_cmd = []
        self._queue = queue.Queue()
        self._error = None
        self._thread = threading.Thread(target=self._work, daemon=True)
        self._thread.start()

    def _work(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                write_png(*item)
            except Exception as exc:  # noqa: BLE001
                if self._error is None:
                    self._error = repr(exc)
            finally:
                self._queue.task_done()

    def add_frame(self, frame: dict, images) -> None:
        i = len(self.frames)
        self.frames.append(frame)
        for cam, rgb in zip(self.cameras, images):
            self._queue.put((self.dir / cam / f"{i:06d}.png", rgb))

    def add_step(self, x_des, gripper_cmd: float) -> None:
        self.step_x_des.append(np.array(x_des, dtype=np.float64))
        self.step_gripper_cmd.append(float(gripper_cmd))

    def _stop(self) -> None:
        if self._thread.is_alive():
            self._queue.join()
            self._queue.put(None)
            self._thread.join()

    def discard(self) -> None:
        self._stop()
        shutil.rmtree(self.dir, ignore_errors=True)

    def finalize(self, meta: dict, extra_arrays: dict, record_every: int) -> pathlib.Path:
        self._stop()
        if self._error:
            raise RuntimeError(f"{self.final_dir.name}: {self._error}")
        n = len(self.frames)
        arrays = {name: np.array([f[name] for f in self.frames], dtype=dtype).reshape((n,) + shape)
                  for name, (shape, dtype) in FRAME_FIELDS.items()}
        n_steps = (n - 1) * record_every
        arrays["timestamp"] = np.arange(n, dtype=np.float64) * float(meta["record_dt"])
        arrays["step_x_des"] = np.array(self.step_x_des[:n_steps + 1], dtype=np.float64)
        arrays["step_gripper_cmd"] = np.array(self.step_gripper_cmd[:n_steps + 1], dtype=np.float64)
        arrays.update(extra_arrays)
        np.savez(self.dir / "data.npz", **arrays)
        meta = dict(meta, schema_version=SCHEMA_VERSION, n_frames=n, n_steps=n_steps,
                    contact_columns=list(contact.COLUMNS))
        (self.dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        rename_with_retry(self.dir, self.final_dir)
        return self.final_dir


def rename_with_retry(src, dst, attempts: int = 50, wait_s: float = 0.1) -> None:
    """フォルダの改名。Windows では、書いた直後のファイルを他のプロセス（ウイルス対策・索引）が一瞬つかんで
    アクセス拒否になることがあるので、少し待ってやり直す。"""
    import time
    for i in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(wait_s)
