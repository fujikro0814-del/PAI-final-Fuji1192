"""センサのモデル（シミュレーションの側。configs/sensor_v1.yaml の sensor）。実行系はこの出力だけを見る。

    suite = SensorSuite(world_model, cfg)
    setup = suite.start_trial(seed, world_data, base_setup)   # 試行ごとの値（較正誤差・系統誤差・照明・位相）を引き、信じている既知の情報を返す
    suite.on_physics_step(world_data)                         # 物理の 1 ステップごと（mj_step の後）: 撮る時刻なら状態を控える
    frame = suite.sense(world_data, hand, q_d, cameras)       # いまの SensorFrame（届いた最新のフレーム、関節、グリッパ）

- カメラ: 30 fps で撮る（カメラごとに位相を試行ごとに引く）。撮った時刻の状態（qpos）を控え、フレームごとに遅延を引いて届く時刻を決める。
  描画と雑音は、実行系が使うフレーム（届いた最新のフレーム）だけを、控えた状態から遅延して作る。雑音の乱数はフレームの番号から作るので、
  描く順や描くかどうかによらず同じフレームは同じ画像になる
- 較正誤差: 描くカメラは本当の位置のまま。実行系に渡す外部パラメータ（SetupInfo）の側に誤差を入れる
- 深度: 視差の空間に雑音（公称の RMS の式）と試行ごとの系統のずれを足して深度に戻し、隠れのシード範囲・縁・穴・最小距離で欠けにし、単位で丸める
- 色: 生の画素の大きさで露出・白の釣り合い・読み出しとショットの雑音を足し、切り出して 256×256 に縮める（面積平均）
"""
import math

import cv2
import mujoco
import numpy as np

from recovla.common import seeds
from recovla.runtime.types import (CameraFrame, CameraSetup, GripperState, Intrinsics, JointState, Pose, SensorFrame,
                                   frozen_array)

CAM_INDEX = {"overhead": 0, "wrist": 1}


def intrinsics(width: int, height: int, fovy_deg: float) -> Intrinsics:
    f = 0.5 * height / math.tan(math.radians(fovy_deg) / 2.0)
    return Intrinsics(int(width), int(height), f, f, width / 2.0, height / 2.0)


def small_rotation(rng, sigma_deg: float) -> np.ndarray:
    """軸ごとに σ の回転ベクトルからの回転行列（Rodrigues）。"""
    w = np.radians(rng.normal(0.0, sigma_deg, 3))
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


# ------------------------------------------------------------------------------ 画像のモデル
def depth_model(z: np.ndarray, f: float, baseline: float, rng, sys_px: float, cam: dict, dn: dict) -> np.ndarray:
    """描いた深度 z (H, W) [m] → 雑音・欠け・量子化を入れた深度（欠けは 0）。"""
    z = np.asarray(z, np.float64)
    H, W = z.shape
    valid = z > 1e-6
    disp = np.where(valid, f * baseline / np.maximum(z, 1e-6), 0.0)
    # 隠れのシード範囲（左の撮像素子が基準。右の撮像素子から見えない画素）: u' > u に d(u') − d(u) ≥ u' − u の点があれば隠れる
    g = disp - np.arange(W)[None, :]
    cm = np.maximum.accumulate(g[:, ::-1], axis=1)[:, ::-1]
    nxt = np.full_like(cm, -np.inf)
    nxt[:, :-1] = cm[:, 1:]
    occluded = nxt >= g + 0.5
    # 視差の雑音（空間に相関、σ を保つ）と系統のずれ
    n = rng.standard_normal((H, W))
    s = float(dn["corr_px"])
    if s > 0:
        n = cv2.GaussianBlur(n, (0, 0), s)
        n /= max(float(n.std()), 1e-12)
    d2 = disp + sys_px + float(cam["subpixel"]) * n
    out = np.where(valid & (d2 > 1e-6), f * baseline / np.maximum(d2, 1e-6), 0.0)
    # 縁の空飛ぶ画素: 段差の大きい縁の画素を、近い側と遠い側の間の値にする
    zmax = cv2.dilate(z.astype(np.float32), np.ones((3, 3), np.uint8)).astype(np.float64)
    zmin = cv2.erode(z.astype(np.float32), np.ones((3, 3), np.uint8)).astype(np.float64)
    edge = valid & ((zmax - zmin) >= float(dn["edge_step"])) & (rng.random((H, W)) < float(dn["edge_prob"]))
    out = np.where(edge, zmin + rng.random((H, W)) * (zmax - zmin), out)
    # 穴（3×3）
    centers = rng.random((H, W)) < float(dn["hole_area"]) / 9.0
    holes = cv2.dilate(centers.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    bad = ~valid | occluded | holes | (out < float(cam["min_z"])) | (out > float(cam["max_z"]))
    out = np.where(bad, 0.0, out)
    unit = float(cam["depth_unit"])
    out = np.round(out / unit) * unit
    return out.astype(np.float32)


def color_model(rgb: np.ndarray, rng, gain: float, wb: np.ndarray, cn: dict, crop: int = None, out: int = 256) -> np.ndarray:
    img = np.asarray(rgb, np.float64)
    if crop:
        H, W = img.shape[:2]
        y0, x0 = (H - crop) // 2, (W - crop) // 2
        img = img[y0:y0 + crop, x0:x0 + crop]
    img = img * gain * wb[None, None, :]
    sigma = np.sqrt(float(cn["read_sigma"]) ** 2 + float(cn["shot_gain"]) * np.clip(img, 0, None))
    img = np.clip(np.round(img + rng.standard_normal(img.shape) * sigma), 0, 255).astype(np.uint8)
    if img.shape[0] != out:
        img = cv2.resize(img, (out, out), interpolation=cv2.INTER_AREA)
    return img


# ---------------------------------------------------------------------------------- センサ一式
class _Stream:
    def __init__(self, name: str, period: float, phase: float):
        self.name, self.period = name, period
        self.next_t = phase
        self.seq = -1
        self.captures = []          # (seq, t_capture, t_arrival, qpos)
        self.cache = {}


class SensorSuite:
    def __init__(self, model, cfg: dict):
        self.model = model
        self.cfg = cfg
        self.sc = cfg["sensor"]
        self.scratch = mujoco.MjData(model)
        self.cam_ids = {n: model.camera(c["model_camera"]).id for n, c in self.sc["cameras"].items()}
        self.renderers = {}
        self.seed = None

    def _renderer(self, w: int, h: int):
        key = (w, h)
        if key not in self.renderers:
            self.renderers[key] = mujoco.Renderer(self.model, h, w)
        return self.renderers[key]

    def close(self) -> None:
        for r in self.renderers.values():
            r.close()
        self.renderers = {}

    # ----------------------------------------------------------------------- 試行の始め
    def start_trial(self, seed: int, data, base_setup):
        """試行ごとの値を引き、較正誤差を含む SetupInfo（実行系に渡す）を返す。"""
        import dataclasses
        self.seed = int(seed)
        rng = seeds.stream(seed, "sensor_setup")
        self.lat_rng = np.random.default_rng(seeds.seed_sequence(seed, "latency", 0))   # カメラの遅延（計算の時間は 1）
        self.joint_rng = np.random.default_rng(seeds.seed_sequence(seed, "sensor_frame", 99))
        sc, cal = self.sc, self.sc["calibration"]
        period = 1.0 / float(sc["frame_hz"])
        self.streams = {n: _Stream(n, period, float(rng.uniform(0, period))) for n in sc["cameras"]}
        cn = sc["color_noise"]
        self.gain = {n: float(rng.uniform(*cn["exposure_gain"])) for n in sc["cameras"]}
        self.wb = {n: rng.uniform(*cn["white_balance"], 3) for n in sc["cameras"]}
        # 深度の系統のずれ（視差 [px]）: 公称の Z 精度を 4σ とみなす
        self.sys_px = {}
        cams = {}
        mujoco.mj_forward(self.model, data)
        for n, c in sc["cameras"].items():
            dI = intrinsics(c["depth"]["width"], c["depth"]["height"], c["depth"]["fovy"])
            self.sys_px[n] = float(rng.normal(0.0, float(c["sys_sigma_px"])))
            true = self._true_extrinsic(n, data)
            e = cal[n]
            dR = small_rotation(rng, float(e["rot_sigma_deg"]))
            dt = rng.normal(0.0, float(e["trans_sigma"]), 3)
            believed = Pose(frozen_array(true.R @ dt + true.t), frozen_array(true.R @ dR))
            col = c["color"]
            crop_fovy = col["fovy"] if not col.get("crop") else math.degrees(
                2 * math.atan(math.tan(math.radians(col["fovy"]) / 2) * col["crop"] / col["height"]))
            cams[n] = CameraSetup(n, intrinsics(col["out"], col["out"], crop_fovy), dI, believed, c["mount"],
                                  float(c["depth_unit"]), float(c["min_z"]))
        self.true_extrinsics = {n: self._true_extrinsic(n, data) for n in sc["cameras"]}
        self.believed = cams
        t = cal["table"]
        n_tab = small_rotation(rng, float(t["tilt_sigma_deg"])) @ np.array([0.0, 0.0, 1.0])
        g = sc["gripper"]
        self.grip_period = 1.0 / float(g["rate_hz"])
        self.grip_next = float(rng.uniform(0, self.grip_period))
        self.grip_state = None
        self.trial_values = {"gain": self.gain, "white_balance": {k: v.tolist() for k, v in self.wb.items()},
                             "sys_px": self.sys_px, "table_dz": None, "phase": {n: s.next_t for n, s in self.streams.items()}}
        dz = float(rng.normal(0.0, float(t["z_sigma"])))
        self.trial_values["table_dz"] = dz
        self.trial_values["extrinsic_error"] = {
            n: {"dt_m": (cams[n].extrinsic.t - self.true_extrinsics[n].t).tolist(),
                "drot_deg": float(np.degrees(np.arccos(np.clip((np.trace(self.true_extrinsics[n].R.T @ cams[n].extrinsic.R) - 1) / 2, -1, 1))))}
            for n in cams}
        return dataclasses.replace(base_setup, cameras=cams, table_z=base_setup.table_z + dz,
                                   table_normal=frozen_array(n_tab))

    def prime(self, data) -> None:
        """試行の始め（時刻 0）の前からカメラは流れていて、場面は止まっている。時刻 0 の状態のフレームを 1 つ、時刻 0 で届いた
        ものとして置く（撮った時刻は遅延の分だけ前）。これがないと、最初のフレームが届くまで実行系が 30〜100 ms 待つ。"""
        lat = self.sc["latency_ms"]
        for s in self.streams.values():
            L = float(self.lat_rng.uniform(*lat)) / 1000.0
            s.seq += 1
            s.captures.append((s.seq, float(data.time) - L, float(data.time), data.qpos.copy()))

    def _true_extrinsic(self, name: str, data) -> Pose:
        cid = self.cam_ids[name]
        if self.sc["cameras"][name]["mount"] == "world":
            return Pose(frozen_array(data.cam_xpos[cid]), frozen_array(data.cam_xmat[cid].reshape(3, 3)))
        m = self.model
        body = m.cam_bodyid[cid]
        R_body = data.xmat[body].reshape(3, 3)
        R_cam = data.cam_xmat[cid].reshape(3, 3)
        t = R_body.T @ (data.cam_xpos[cid] - data.xpos[body])
        return Pose(frozen_array(t), frozen_array(R_body.T @ R_cam))

    # ------------------------------------------------------------------------ 物理の 1 ステップごと
    def on_physics_step(self, data) -> None:
        t = float(data.time)
        lat = self.sc["latency_ms"]
        for s in self.streams.values():
            if t >= s.next_t - 1e-9:
                s.seq += 1
                t_arr = t + float(self.lat_rng.uniform(*lat)) / 1000.0
                s.captures.append((s.seq, t, t_arr, data.qpos.copy()))
                if len(s.captures) > 16:
                    s.captures.pop(0)
                s.next_t += s.period
        if t >= self.grip_next - 1e-9:
            self._grip_due = True
            self.grip_next += self.grip_period

    def _render(self, name: str, qpos) -> tuple:
        c = self.sc["cameras"][name]
        m, d = self.model, self.scratch
        d.qpos[:] = qpos
        mujoco.mj_forward(m, d)
        cid = self.cam_ids[name]
        fovy0 = float(m.cam_fovy[cid])
        try:
            col, dep = c["color"], c["depth"]
            r = self._renderer(col["width"], col["height"])
            m.cam_fovy[cid] = col["fovy"]
            r.disable_depth_rendering()
            r.update_scene(d, camera=cid)
            rgb = r.render().copy()
            r = self._renderer(dep["width"], dep["height"])
            m.cam_fovy[cid] = dep["fovy"]
            r.enable_depth_rendering()
            r.update_scene(d, camera=cid)
            z = r.render().copy()
            r.disable_depth_rendering()
        finally:
            m.cam_fovy[cid] = fovy0
        return rgb, z

    def _frame(self, name: str, cap) -> CameraFrame:
        seq, t_cap, t_arr, qpos = cap
        s = self.streams[name]
        if seq in s.cache:
            return s.cache[seq]
        c = self.sc["cameras"][name]
        rng = np.random.default_rng(seeds.seed_sequence(self.seed, "sensor_frame", CAM_INDEX[name], seq))
        rgb_raw, z = self._render(name, qpos)
        rgb = color_model(rgb_raw, rng, self.gain[name], self.wb[name], self.sc["color_noise"],
                          c["color"].get("crop"), int(c["color"]["out"]))
        dI = self.believed[name].depth
        depth = depth_model(z, dI.fx, float(c["baseline"]), rng, self.sys_px[name], c, self.sc["depth_noise"])
        rgb.flags.writeable = False
        depth.flags.writeable = False
        fr = CameraFrame(name, float(t_cap), float(t_arr), rgb, depth, int(seq))
        s.cache = {seq: fr}
        return fr

    def sense(self, data, hand, q_d, cameras: bool = True) -> SensorFrame:
        """cameras が偽なら画像を作らない（腕の 500 Hz の制御は関節とグリッパだけを読む。画像は方策・知覚の 10 Hz で読む）。"""
        t = float(data.time)
        cams = {}
        for n, s in (self.streams.items() if cameras else ()):
            arrived = [c for c in s.captures if c[2] <= t + 1e-9]
            if arrived:
                cams[n] = self._frame(n, arrived[-1])
        js = self.sc["joints"]
        q = data.qpos[self._arm_qadr()] + self.joint_rng.normal(0, float(js["q_sigma"]), 7)
        dq = data.qvel[self._arm_vadr()] + self.joint_rng.normal(0, float(js["dq_sigma"]), 7)
        joints = JointState(t, frozen_array(q), frozen_array(dq), frozen_array(q_d))
        if getattr(self, "_grip_due", False) or self.grip_state is None:
            g = self.sc["gripper"]
            w = hand.width() + float(self.joint_rng.normal(0, float(g["width_sigma"])))
            w = round(w / float(g["width_unit"])) * float(g["width_unit"])
            self.grip_state = GripperState(t, float(w), bool(hand.is_grasped()))
            self._grip_due = False
        return SensorFrame(t, joints, self.grip_state, cams)

    def _arm_qadr(self):
        if not hasattr(self, "_qadr"):
            m = self.model
            self._qadr = np.array([m.joint(f"joint{i}").qposadr[0] for i in range(1, 8)])
            self._vadr = np.array([m.joint(f"joint{i}").dofadr[0] for i in range(1, 8)])
        return self._qadr

    def _arm_vadr(self):
        self._arm_qadr()
        return self._vadr
