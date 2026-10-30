"""合成の試行の記録（評価器が書く trial の json・npz と同じ形）を作る補助。

何も起きない試行（立方体は静止、手は止まっている、行動は 0、接触なし）を土台に、検査ごとに
跳び・時刻・接触・移動を前もって決めて書き込む。write で trial_NNNN.json と trial_NNNN.npz を書く。
"""
import copy
import json
import pathlib

import numpy as np

COLORS = ("red", "green", "blue")
OBSTACLES = ["cube_red", "cube_green", "cube_blue", "wall_xp", "wall_xn", "wall_yp", "wall_yn"]
STEP_DT = 0.002          # 物理 1 ステップ
FRAME_STEPS = 25         # フレーム 20 Hz
CHUNK = 10               # 行動列の実行間隔（行動の数）
CUBE_Z = 0.02            # 机の上の立方体の中心の高さ
CUBE0 = np.array([[0.40, -0.10, CUBE_Z], [0.45, 0.00, CUBE_Z], [0.50, -0.10, CUBE_Z]])
EE0 = np.array([0.307, 0.0, 0.35])
FINGERTIP_OFFSET = 0.1034


def frame_time(i):
    """フレーム i の時刻 [s]（物理ステップ 25i の時点）。"""
    return np.asarray(i) * FRAME_STEPS * STEP_DT


class SynthTrial:
    def __init__(self, n_frames: int = 400, trial: int = 0, seed: int = 110000):
        n = n_frames
        self.n = n
        i = np.arange(n)
        k = i // 2
        self.arrays = {
            "step": (i * FRAME_STEPS).astype(np.int64),
            "sim_time": frame_time(i).astype(np.float64),
            "ee_pos": np.tile(EE0, (n, 1)),
            "ee_quat": np.tile([0.0, 1.0, 0.0, 0.0], (n, 1)),
            "fingertip": np.tile(EE0 - [0, 0, FINGERTIP_OFFSET], (n, 1)),
            "fingers": np.full((n, 2), 0.04),
            "x_des": np.tile(EE0, (n, 1)),
            "gripper_closed": np.zeros(n, dtype=bool),
            "cube_pos": np.tile(CUBE0, (n, 1, 1)),
            "cube_quat": np.tile([1.0, 0.0, 0.0, 0.0], (n, 3, 1)),
            "cube_linvel": np.zeros((n, 3, 3)),
            "cube_in_box": np.zeros((n, 3), dtype=bool),
            "target": np.zeros(n, dtype=np.int8),
            "phase": np.zeros(n, dtype=np.int8),
            "action": np.concatenate([np.zeros((n, 6)), -np.ones((n, 1))], axis=1),
            "chunk_id": (k // CHUNK).astype(np.int32),
            "chunk_index": (k % CHUNK).astype(np.int32),
            "chunk_switch": (i % 2 == 0) & (k % CHUNK == 0),
            "contact_robot": np.zeros((n, len(OBSTACLES)), dtype=bool),
            "contact_cube_cube": np.zeros((n, 3, 3), dtype=bool),
            "min_dist": np.full((n, len(OBSTACLES)), 0.10),
            "safety_active": np.zeros(n, dtype=bool),
            "induce_active": np.zeros(n, dtype=bool),
        }
        m = int(k[-1] // CHUNK + 1)
        self.arrays.update({
            "chunk_k_valid": (np.arange(m) * CHUNK).astype(np.int32),
            "chunk_xdes_pred": np.tile(EE0, (m, 50, 1)),
            "chunk_grip_pred": -np.ones((m, 50)),
        })
        t_end = float(frame_time(n - 1))
        self.meta = {
            "record_version": 1,
            "trial": trial, "seed": seed,
            "experiment": "E2",
            "condition": "R1_rtc_s10_d2",
            "model": {"name": "R1", "checkpoint": "synthetic", "step": 30000},
            "runtime": {"mode": "rtc", "exec_interval": 10, "delay_steps": 2, "rtc_guidance_horizon": 10,
                        "rtc_schedule": "EXP", "safety_filter": False},
            "layout": {"kind": "empty", "start": "home",
                       "cubes": {c: [float(CUBE0[j, 0]), float(CUBE0[j, 1]), 0.0] for j, c in enumerate(COLORS)},
                       "prefilled": []},
            "steps": [{"target": "red", "instruction": "put the red cube in the box",
                       "t_start": 0.0, "t_end": t_end, "success": False, "t_success": None}],
            "success": False,
            "time_limit_s": 30.0,
            "induce": {"kind": None, "params": {}, "fired": False, "t_fire": None,
                       "established": False, "t_established": None, "t_failure": None, "reason": None},
            "obstacles": list(OBSTACLES),
            "inference": [{"i": j, "k_obs": j * CHUNK, "k_valid": j * CHUNK, "offset": 0, "left_over_len": 0,
                           "reset": j == 0, "wall_s": 0.2, "wall_breakdown_s": {}} for j in range(m)],
            "input_check": {},
            "code_version": {},
        }

    # ------------------------------------------------------------------ 書き込みの補助

    def set_success(self, t_success: float):
        self.meta["success"] = True
        self.meta["steps"][-1].update(success=True, t_success=t_success, t_end=t_success)

    def set_induce(self, kind="P2", t_fire=None, t_failure=None, established=True):
        self.meta["induce"].update(kind=kind, fired=t_fire is not None, t_fire=t_fire,
                                   established=established, t_established=t_failure if established else None,
                                   t_failure=t_failure if established else None,
                                   reason=None if established else "lifted")

    def set_velocity(self, v_per_action: np.ndarray, ee_gain: float = 1.0):
        """10 Hz の行動の手先参照速度 v_k [m/s]（(K, 3)、K = フレームの数 / 2）を決め、action・x_des・ee_pos を書く。

        x_des はフレーム 2k で行動 k の直前の値、フレーム 2k+1 はその中点。ee_pos は x_des の速度の ee_gain 倍で
        フレームごとに進む（フレーム i の後の部分の手先速度 = ee_gain · v_{i // 2}）。
        """
        v = np.asarray(v_per_action, dtype=float)
        n = self.n
        k = np.arange(n) // 2
        dx = v * 0.1                                                     # 行動 = 0.1 s の間の x_des の変化
        a = self.arrays["action"]
        a[:, :3] = dx[k]
        before = np.vstack([np.zeros((1, 3)), np.cumsum(dx, axis=0)])[:len(v)]   # 行動 k の直前までの和
        half = (np.arange(n) % 2)[:, None] * 0.5
        self.arrays["x_des"] = EE0 + before[k] + half * dx[k]
        u = ee_gain * v[k] * 0.05                                        # フレーム i の後の部分の移動
        self.arrays["ee_pos"] = EE0 + np.vstack([np.zeros((1, 3)), np.cumsum(u, axis=0)[:-1]])
        self.arrays["fingertip"] = self.arrays["ee_pos"] - [0, 0, FINGERTIP_OFFSET]

    def move_cube(self, color: str, frames: np.ndarray, pos: np.ndarray):
        """立方体をフレーム frames で pos に置き、以後はそこに留める。linvel は位置の差分から書く。"""
        c = COLORS.index(color)
        frames = np.asarray(frames)
        cp = self.arrays["cube_pos"]
        cp[frames, c] = pos
        cp[frames[-1] + 1:, c] = cp[frames[-1], c]
        self._velocity_from_position(c)

    def _velocity_from_position(self, c: int):
        cp = self.arrays["cube_pos"][:, c]
        vel = np.zeros_like(cp)
        vel[1:] = np.diff(cp, axis=0) / (FRAME_STEPS * STEP_DT)
        self.arrays["cube_linvel"][:, c] = vel

    def touch(self, obstacle: str, frames):
        self.arrays["contact_robot"][np.asarray(frames), OBSTACLES.index(obstacle)] = True

    def cube_cube(self, a: str, b: str, frames):
        ia, ib = COLORS.index(a), COLORS.index(b)
        self.arrays["contact_cube_cube"][np.asarray(frames), ia, ib] = True
        self.arrays["contact_cube_cube"][np.asarray(frames), ib, ia] = True

    def write(self, folder) -> pathlib.Path:
        folder = pathlib.Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        stem = folder / f"trial_{self.meta['trial']:04d}"
        stem.with_suffix(".json").write_text(json.dumps(self.meta, ensure_ascii=False), encoding="utf-8")
        np.savez(stem.with_suffix(".npz"), **self.arrays)
        return stem.with_suffix(".json")

    def copy(self) -> "SynthTrial":
        return copy.deepcopy(self)
