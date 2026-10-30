"""立方体 1 個の場面のエピソードの記録（数値は configs/g0.yaml）。

保存した 1 本のエピソードの形:

    <日または回のフォルダ>/ep_000123/
        overhead/000000.png ...    256x256 RGB、フレームごとに 1 枚
        wrist/000000.png ...
        data.npz                   フレームごと・物理ステップごとの配列（下）
        meta.json

記録中は ep_XXXXXX.partial に書き、保存で名前を変える。捨てるときは .partial を消す。

フレーム: フレーム i は物理ステップ i * RECORD_EVERY で取る（フレーム 0 = リセットの状態）。画像と状態は同じ時点のもの:
動いている MjData を scratch の MjData に複写して mj_forward する。mj_step の直後の位置（xpos・カメラ）は qpos より
1 ステップ古く、動いている側で mj_forward を呼ぶと、次の IK に入る値が変わって軌道が変わるため。

行動は切り出して保存しない。x_des（controller.desired_pos）を、フレームごと・物理ステップごとに絶対値で保存する。
フレーム i と組になる 20 Hz の行動は、その後の部分の和:
    action_xyz[i] = x_des[i + 1] - x_des[i]          (i = 0 .. n_frames - 2)
    action_gripper[i] = フレーム i + 1 のグリッパの指令
LeRobot・LIBERO と同じく、時刻 t の行動は t を観測した後に実行するもの。最後のフレームには行動がない。
"""
import dataclasses
import datetime
import hashlib
import json
import os
import pathlib
import queue
import shutil
import threading
from typing import Optional

import cv2
import mujoco
import numpy as np

from recovla.common import config

_CFG = config.load("g0")
_G0 = _CFG["g0"]
_PLACE = _G0["placement"]

SCHEMA_VERSION = 1
INSTRUCTION = _G0["instruction"]

# 立方体の初期位置の範囲と、学習用の固定の配置（エピソードごとに新しい配置を引かず、数個の配置を繰り返す）
N_PLACEMENTS = int(_PLACE["n"])
CUBE_X_RANGE = tuple(float(v) for v in _PLACE["x_range"])
CUBE_Y_RANGE = tuple(float(v) for v in _PLACE["y_range"])
_YAW_HALF = np.pi / (180.0 / float(_PLACE["yaw_half_range_deg"]))    # 45° → np.pi / 4.0 と完全で同じ値
CUBE_YAW_RANGE = (-_YAW_HALF, _YAW_HALF)
CUBE_HALF = float(_CFG["scene"]["cube_size"]) / 2.0
PLACEMENT_MIN_SEPARATION = float(_PLACE["min_separation"])   # 学習用の配置どうしの間隔 [m]
TRAIN_SEED_LIMIT = int(_PLACE["train_seed_limit"])           # 学習のシードは 0〜2999、評価は 100000 以上
CUBE_BODY = _PLACE["cube_body"]

# 成功 = 立方体が箱の中で止まっている（箱の内側 ±0.05 m、壁の上端 0.06 m）
BOX_BODY = "goal_box"
BOX_INNER_HALF_XY = float(_CFG["scene"]["box"]["success_inner_half"])
BOX_WALL_TOP_Z = float(_CFG["scene"]["box"]["wall_top_z"])

PNG_COMPRESSION = int(_CFG["sim"]["png_compression"])        # 速さ優先（PNG はどの段階でも可逆）


# ------------------------------------------------------------- 配置

@dataclasses.dataclass(frozen=True)
class Placement:
    placement_id: Optional[int]   # 評価のシードでは None
    seed: int
    x: float
    y: float
    yaw: float

    def quat(self) -> np.ndarray:
        return np.array([np.cos(self.yaw / 2), 0.0, 0.0, np.sin(self.yaw / 2)])


def placement_from_seed(seed: int, placement_id: Optional[int] = None) -> Placement:
    """初期位置の範囲から一様に 1 回引く。評価は学習で使わないシード（100000 以上）で呼ぶ。"""
    rng = np.random.default_rng(seed)
    return Placement(placement_id, int(seed), float(rng.uniform(*CUBE_X_RANGE)),
                     float(rng.uniform(*CUBE_Y_RANGE)),
                     float(rng.uniform(*CUBE_YAW_RANGE)))


def training_placements(n: int = N_PLACEMENTS,
                        min_separation: float = PLACEMENT_MIN_SEPARATION) -> list:
    """学習用の固定の配置: シード 0, 1, 2, ... の順に引き、残したどの配置とも min_separation 以上離れたものだけを
    n 個になるまで残す（決定的）。"""
    kept = []
    for seed in range(TRAIN_SEED_LIMIT):
        p = placement_from_seed(seed)
        if all(np.hypot(p.x - q.x, p.y - q.y) >= min_separation for q in kept):
            kept.append(dataclasses.replace(p, placement_id=len(kept)))
            if len(kept) == n:
                return kept
    raise RuntimeError("could not find enough separated training placements")


def apply_placement(model, data, placement: Placement) -> None:
    body = model.body(CUBE_BODY).id
    jnt = model.body_jntadr[body]
    qadr, vadr = model.jnt_qposadr[jnt], model.jnt_dofadr[jnt]
    data.qpos[qadr:qadr + 3] = [placement.x, placement.y, CUBE_HALF]
    data.qpos[qadr + 3:qadr + 7] = placement.quat()
    data.qvel[vadr:vadr + 6] = 0.0


def cube_in_box(cube_pos, box_pos) -> bool:
    rel = np.asarray(cube_pos, float) - np.asarray(box_pos, float)
    return bool(abs(rel[0]) < BOX_INNER_HALF_XY and abs(rel[1]) < BOX_INNER_HALF_XY
                and 0.0 < rel[2] < BOX_WALL_TOP_Z)


# ------------------------------------------------------------- 配置の順

def next_placement(placements: list, saved: dict) -> Placement:
    """保存したエピソードの少ない配置から、同数なら番号の小さい順に選ぶ（繰り返しの数をそろえる）。"""
    return min(placements, key=lambda p: (saved.get(p.placement_id, 0), p.placement_id))


def _now_iso() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def file_sha256(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


# -------------------------------------------------------------- フレームの取得

FRAME_FIELDS = {
    # 名前: (1 フレームの形, 型)
    "step": ((), np.int64),               # リセットからの物理ステップ数
    "sim_time": ((), np.float64),         # シミュレーションの時刻 [s]
    "ee_pos": ((3,), np.float64),         # 手先の原点、world [m]
    "ee_quat": ((4,), np.float64),        # 手先、world、w>=0
    "ee_axisangle": ((3,), np.float64),   # 同じ姿勢の回転ベクトル
    "fingers": ((2,), np.float64),        # finger_joint1, finger_joint2 [m]
    "joints": ((7,), np.float64),         # joint1..7 [rad]
    "joint_vel": ((7,), np.float64),      # [rad/s]
    "x_des": ((3,), np.float64),          # controller.desired_pos（行動の元）
    "tracker_target": ((3,), np.float64), # controller.target_pos（追従器の後）
    "gripper_cmd": ((), np.float64),      # アクチュエータの ctrl 0..255（0 = 閉）
    "gripper_closed": ((), np.bool_),
    "cube_pos": ((3,), np.float64),       # 真値（判定と記録だけに使う）
    "cube_quat": ((4,), np.float64),
}


class FrameSampler:
    """同じ時点の画像と状態を取る（モジュールの説明を参照）。"""

    def __init__(self, model, controller, renderer: mujoco.Renderer, cameras):
        self.model = model
        self.controller = controller
        self.renderer = renderer
        self.cameras = tuple(cameras)
        self.scratch = mujoco.MjData(model)
        self.arm_qadr = np.array([model.joint(f"joint{i}").qposadr[0] for i in range(1, 8)])
        self.arm_vadr = np.array([model.joint(f"joint{i}").dofadr[0] for i in range(1, 8)])
        self.finger_qadr = np.array([model.joint(n).qposadr[0]
                                     for n in ("finger_joint1", "finger_joint2")])
        self.cube = model.body(CUBE_BODY).id
        self.grip = controller.gripper_indices[0]

    def capture(self, data, step: int):
        s = self.scratch
        np.copyto(s.qpos, data.qpos)
        np.copyto(s.qvel, data.qvel)
        # mocap はわざと複写しない（scratch はモデルの既定の mocap の姿勢のまま）。mocap の目標の球はカメラから
        # 隠れ（geom group 5）物理もないが、影と反射があると、その位置で影の縁の俯瞰画像の画素が約 19 個変わり、
        # 追従器の目標が観測に漏れるため
        np.copyto(s.ctrl, data.ctrl)
        s.time = data.time
        mujoco.mj_forward(self.model, s)

        h = self.controller.hand_body_id
        quat = s.xquat[h].copy()
        if quat[0] < 0.0:
            quat = -quat
        axisangle = np.zeros(3)
        mujoco.mju_quat2Vel(axisangle, quat, 1.0)
        frame = {
            "step": step,
            "sim_time": s.time,
            "ee_pos": s.xpos[h].copy(),
            "ee_quat": quat,
            "ee_axisangle": axisangle,
            "fingers": s.qpos[self.finger_qadr].copy(),
            "joints": s.qpos[self.arm_qadr].copy(),
            "joint_vel": s.qvel[self.arm_vadr].copy(),
            "x_des": self.controller.desired_pos.copy(),
            "tracker_target": self.controller.target_pos.copy(),
            "gripper_cmd": float(data.ctrl[self.grip]),
            "gripper_closed": bool(self.controller.gripper_closed),
            "cube_pos": s.xpos[self.cube].copy(),
            "cube_quat": s.xquat[self.cube].copy(),
        }
        images = []
        for cam in self.cameras:
            self.renderer.update_scene(s, camera=cam)
            images.append(self.renderer.render().copy())
        return frame, images


# ---------------------------------------------------------------- 書き出し

def write_png(path, rgb) -> None:
    """Windows の cv2.imwrite は ASCII 以外のパスを開けないので、メモリで符号化して Python で書く。"""
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])
    if not ok:
        raise RuntimeError(f"PNG encoding failed for {path}")
    pathlib.Path(path).write_bytes(buf.tobytes())


def read_png(path) -> np.ndarray:
    """RGB の画像を読む（write_png と同じく ASCII 以外のパスでも読める）。"""
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def episode_dir_name(episode_id: int) -> str:
    return f"ep_{int(episode_id):06d}"


class EpisodeWriter:
    """PNG を別のスレッドで <day>/ep_XXXXXX.partial に書き（PNG の符号化は GIL を離す）、制御のループを止めない。
    フレームと物理ステップの配列はメモリに貯める。

    episode_id=None: 番号は保存のときに決める。記録は <day>/ep_pending_<時刻>_<pid>.partial に書き（ep_N の名前でないので
    数え上げ・変換・重複の検査に入らない）、finalize(..., episode_id=N) で ep_N に改名する。保存に失敗したら
    mark_failed(N) で ep_N.partial に改名し、使った番号を残す。"""

    def __init__(self, root, episode_id, day: str, cameras, record_every: int,
                 nominal_dt: float):
        self.episode_id = episode_id
        self.cameras = tuple(cameras)
        self.record_every = record_every
        self.nominal_dt = nominal_dt
        self.day_dir = pathlib.Path(root) / day
        self.pending = episode_id is None
        if self.pending:
            stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            self.final_dir = None
            self.dir = self.day_dir / f"ep_pending_{stamp}_{os.getpid()}.partial"
            if self.dir.exists():
                raise FileExistsError(self.dir)
        else:
            self.final_dir = self.day_dir / episode_dir_name(episode_id)
            self.dir = self.final_dir.with_name(self.final_dir.name + ".partial")
            if self.final_dir.exists() or self.dir.exists():
                raise FileExistsError(self.final_dir)
        for cam in self.cameras:
            (self.dir / cam).mkdir(parents=True)
        self.frames = []
        self.step_x_des = []
        self.step_gripper_cmd = []
        self.frozen_frames = {cam: 0 for cam in self.cameras}   # _count_frozen を参照
        self._previous_images = None
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
                path, rgb = item
                write_png(path, rgb)
            except Exception as exc:  # noqa: BLE001
                if self._error is None:
                    self._error = repr(exc)
            finally:
                self._queue.task_done()

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    def add_frame(self, frame: dict, images) -> None:
        i = len(self.frames)
        self.frames.append(frame)
        for cam, rgb in zip(self.cameras, images):
            self._queue.put((self.dir / cam / f"{i:06d}.png", rgb))
        self._count_frozen(images)

    def _count_frozen(self, images) -> None:
        """カメラの停止を見つける材料: 視点ごとに、前のフレームと画素がすべて同じだった回数。止まったカメラは同じ画像を
        送り続けるが、このタスクでは手が常に動く。ここでは数えるだけで、判定はしない。"""
        if self._previous_images is not None:
            for cam, rgb, last in zip(self.cameras, images, self._previous_images):
                if np.array_equal(rgb, last):
                    self.frozen_frames[cam] += 1
        self._previous_images = list(images)

    def add_step(self, x_des, gripper_cmd: float) -> None:
        """リセット時（添字 0）と、物理ステップのたびに呼ぶ。"""
        self.step_x_des.append(np.array(x_des, dtype=np.float64))
        self.step_gripper_cmd.append(float(gripper_cmd))

    def _stop_thread(self) -> None:
        self._queue.join()
        self._queue.put(None)
        self._thread.join()

    def abandon(self) -> None:
        """PNG のスレッドを止め、ディスクにあるものはそのまま残す（保存が終わらなかったとき）。"""
        if self._thread.is_alive():
            self._stop_thread()

    def discard(self) -> None:
        self._stop_thread()
        shutil.rmtree(self.dir)

    def mark_failed(self, episode_id: int) -> pathlib.Path:
        """番号が未定の記録の保存に失敗した後、ep_N.partial として残す（できる範囲で）。"""
        if not self.pending or not self.dir.name.startswith("ep_pending_") or not self.dir.exists():
            return self.dir
        target = self.day_dir / (episode_dir_name(episode_id) + ".partial")
        try:
            os.replace(self.dir, target)
            self.dir = target
        except OSError:
            pass
        return self.dir

    def finalize(self, n_keep: int, meta: dict,
                 extra_arrays: Optional[dict] = None, episode_id=None) -> pathlib.Path:
        """フレーム [0, n_keep) と、残した最後のフレームまでの物理ステップを残し、data.npz（開始状態などの extra_arrays）と
        meta.json を書いて .partial から改名する。番号が未定の記録はここで episode_id が要る。"""
        if self.pending:
            if episode_id is None:
                raise ValueError("a pending recording needs the episode_id assigned at save time")
            self.episode_id = int(episode_id)
            final = self.day_dir / episode_dir_name(episode_id)
            if final.exists() or final.with_name(final.name + ".partial").exists():
                raise FileExistsError(final)
            self.final_dir = final
        self._stop_thread()
        if self._error:
            raise RuntimeError(f"episode {self.episode_id}: {self._error}")
        n_keep = max(1, min(n_keep, len(self.frames)))
        for cam in self.cameras:
            for i in range(n_keep, len(self.frames)):
                (self.dir / cam / f"{i:06d}.png").unlink(missing_ok=True)
        frames = self.frames[:n_keep]
        arrays = {}
        for name, (shape, dtype) in FRAME_FIELDS.items():
            arrays[name] = np.array([f[name] for f in frames], dtype=dtype).reshape(
                (n_keep,) + shape)
        # 名目の 0.05 s の格子（実際の時刻は sim_time）
        arrays["timestamp"] = np.arange(n_keep, dtype=np.float64) * self.nominal_dt
        n_steps = (n_keep - 1) * self.record_every
        arrays["step_x_des"] = np.array(self.step_x_des[:n_steps + 1], dtype=np.float64)
        arrays["step_gripper_cmd"] = np.array(self.step_gripper_cmd[:n_steps + 1],
                                              dtype=np.float64)
        arrays.update(extra_arrays or {})
        np.savez(self.dir / "data.npz", **arrays)
        meta = dict(meta, n_frames=n_keep, n_steps=n_steps)
        (self.dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(self.dir, self.final_dir)
        return self.final_dir
