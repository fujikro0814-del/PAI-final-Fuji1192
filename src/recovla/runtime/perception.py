"""知覚。センサの値と既知の情報だけから、机の面・箱・立方体を推定する（シミュレーションの真値は使わない）。

    per = Perception(setup, params, color_thr)
    per.record_joints(sensor.joints)            # 関節の記録（手首のフレームを撮った時刻の関節角を引くため。500 Hz で呼ぶ）
    chk = per.check_table(frame, width)         # 起動時（俯瞰）: 自分の深度から平面を当てはめて保ち、実行系が持つ値とのずれを返す
    per.init_box([(frame, width), ...])         # 試行の始め（待機の姿勢）: 箱の位置と向きを数フレームの平均で推定
    wm = per.update(frames, gripper, t, tip)    # 10 Hz: 立方体の推定の更新（融合・隠れの保持）→ WorldModel

流れ:
  深度 → 点群（俯瞰は実行系が持つ外部パラメータ、手首は撮った時刻の関節角の順運動学と実行系が持つ hand-eye）
  → ロボット自身の除去（測った関節角でロボットだけのモデルを実行系が持つカメラの姿勢から描き、margin だけ太らせたマスク）
  → 机の面からの高さ（そのカメラ自身の深度から当てはめた平面で測る。俯瞰は起動時、手首はフレームごと）→ table_h より上の点
  → 色（方策の RGB の切り出しへ投影して、目標位置の入力と同じ閾値で分類）
  → 箱（黄、壁の上端の高さの点を xy に投影し、既知の寸法の枠を当てはめる）・立方体（色ごとの点の上面から中心と向き）
  → 融合（測定の誤差の逆数で重み）と隠れの保持
真値は一切使わない。精度は評価の枠が試行の後に真値と照らす。
"""
import dataclasses
import math

import cv2
import mujoco
import numpy as np

from recovla.runtime import cue as C
from recovla.runtime import robot_model

COLORS = ("red", "green", "blue")


@dataclasses.dataclass
class Params:
    self_margin_m: float = 0.01        # 自己除去の太らせ方 [m]（学習用の配置で決めた）
    table_h: float = 0.015             # 机の面からの高さの閾値 [m]（学習用の配置で決めた）
    box_band: tuple = (-0.025, 0.012)  # 壁の上端の高さからのシード範囲（箱の点）
    # 立方体の位置の測定の誤差の目安（融合の重み）[m]。深度の雑音に加えて、較正誤差による位置のずれを入れる:
    # 俯瞰は補正の後に残る向き（0.5°）と水平の位置（4 mm）で約 1 cm、手首は hand-eye（3 mm・0.5°、20 cm 先で約 2 mm）で約 4 mm
    sigma_overhead: float = 0.010
    sigma_wrist: float = 0.004
    moved_tol: float = 0.025           # 推定から測定がこれ以上離れたら「動いた」とみなして置き換える
    min_points: int = 15               # 色の点がこれ以上で「見えた」
    lost_s: float = 1.0                # 見えない時間がこれを超えたら「失った」
    top_band: float = 0.006            # 立方体の上面とみなす点のシード範囲（最高点からの高さ）。1 cm より側面の点が混ざりにくい
    correct_overhead: bool = True      # 起動時に机の面で俯瞰の外部パラメータ（傾き 2 つと高さ）を直す

    @classmethod
    def from_config(cls, c: dict) -> "Params":
        kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in (c or {}).items() if k in cls.__dataclass_fields__}
        return cls(**kw)


@dataclasses.dataclass
class CubeEstimate:
    color: str
    pos: np.ndarray                    # (3,) 中心
    yaw: float
    var: float                         # 位置の分散の目安
    t_seen: float
    in_box: bool
    status: str                        # "seen" | "held"（隠れて保持）| "in_hand" | "lost"
    source: str                        # "overhead" | "wrist"


@dataclasses.dataclass
class WorldModel:
    t: float
    table_z: float
    box: dict                          # {"xy": (2,), "yaw": float, "ok": bool}
    cubes: dict                        # {color: CubeEstimate}
    flags: dict                        # 知覚の失敗の知らせ（止まり方の判定に使う）


# ------------------------------------------------------------------------------ 幾何
def fit_plane(pts: np.ndarray, rng, iters: int = 120, tol: float = 0.004):
    """RANSAC で平面（法線は上向き、点）→ (n, c, 内点の数)。見つからなければ None。"""
    if len(pts) < 50:
        return None
    if len(pts) > 8000:
        pts = pts[rng.choice(len(pts), 8000, replace=False)]
    best, best_n = None, -1
    for _ in range(iters):
        s = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        if n[2] < 0:
            n = -n
        cnt = int(np.sum(np.abs((pts - s[0]) @ n) < tol))
        if cnt > best_n:
            best, best_n = (n, s[0]), cnt
    if best is None:
        return None
    n, p = best
    inl = pts[np.abs((pts - p) @ n) < tol]
    c = inl.mean(axis=0)
    _, _, vt = np.linalg.svd(inl - c, full_matrices=False)
    n = vt[-1] if vt[-1][2] > 0 else -vt[-1]
    return n, c, int(len(inl))


def _rot_between(a, b) -> np.ndarray:
    """単位ベクトル a を b に重ねる最小の回転。"""
    a, b = np.asarray(a, float) / np.linalg.norm(a), np.asarray(b, float) / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    s = float(np.linalg.norm(v))
    if s < 1e-12:
        return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / s ** 2)


def pixel_rays(intr) -> np.ndarray:
    """(H, W, 3): 深度 1 のときのカメラ座標（x 右、y 上、z 後ろ向き。画素の中心）。"""
    u = (np.arange(intr.width) + 0.5 - intr.cx) / intr.fx
    v = -(np.arange(intr.height) + 0.5 - intr.cy) / intr.fy
    U, V = np.meshgrid(u, v)
    return np.stack([U, V, -np.ones_like(U)], axis=-1)


def project(points_cam: np.ndarray, intr):
    """カメラ座標の点 (N, 3) → 画素 (u, v)（整数の添字）と、前にあって画像の中か。"""
    z = -points_cam[:, 2]
    ok = z > 1e-6
    u = np.full(len(points_cam), -1)
    v = np.full(len(points_cam), -1)
    u[ok] = np.floor(points_cam[ok, 0] / z[ok] * intr.fx + intr.cx).astype(int)
    v[ok] = np.floor(-points_cam[ok, 1] / z[ok] * intr.fy + intr.cy).astype(int)
    ok &= (u >= 0) & (u < intr.width) & (v >= 0) & (v < intr.height)
    return u, v, ok


def rot_to_quat(R) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(-1))
    return q


class SelfRenderer:
    """ロボットだけのモデルを、実行系が持つカメラの姿勢から描く（自己除去のマスク）。"""

    def __init__(self, model_path: str):
        self.m = robot_model.load(model_path)
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.arm_qadr = np.array([m.joint(f"joint{i}").qposadr[0] for i in range(1, 8)])
        self.finger_qadr = np.array([m.joint(n).qposadr[0] for n in ("finger_joint1", "finger_joint2")])
        self.cam = m.camera("probe").id
        self.mocap = m.body_mocapid[m.body("probe_mount").id]
        self.hand = m.body("hand").id
        self.renderers = {}

    def set_q(self, q, width: float) -> None:
        self.d.qpos[self.arm_qadr] = q
        self.d.qpos[self.finger_qadr] = 0.5 * float(width)
        mujoco.mj_kinematics(self.m, self.d)

    def hand_pose(self):
        return self.d.xpos[self.hand].copy(), self.d.xmat[self.hand].reshape(3, 3).copy()

    def depth(self, pose_R, pose_t, intr) -> np.ndarray:
        """ロボットの深度（ロボットのない画素は inf）。set_q の後に呼ぶ。"""
        key = (intr.width, intr.height)
        if key not in self.renderers:
            self.renderers[key] = mujoco.Renderer(self.m, intr.height, intr.width)
            self.renderers[key].enable_depth_rendering()
        r = self.renderers[key]
        self.d.mocap_pos[self.mocap] = pose_t
        self.d.mocap_quat[self.mocap] = rot_to_quat(pose_R)
        self.m.cam_fovy[self.cam] = math.degrees(2 * math.atan(0.5 * intr.height / intr.fy))
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_camlight(self.m, self.d)
        r.update_scene(self.d, camera=self.cam)
        z = r.render().astype(np.float64)
        far = self.m.vis.map.zfar * self.m.stat.extent
        z[z >= 0.99 * far] = np.inf
        return z


_SELF_RENDERERS = {}


def shared_self_renderer(model_path: str) -> SelfRenderer:
    """プロセスで 1 つだけ作って使い回す。mujoco 3.2.3 の Renderer は、破棄のとき別の Renderer の GL 文脈を壊す（sim/render.py
    の close_renderer の注記）。試行ごとに作って捨てると、捨てた時点で評価器の描画（別の Renderer）が壊れ、以後の全フレームが空になる。"""
    if model_path not in _SELF_RENDERERS:
        _SELF_RENDERERS[model_path] = SelfRenderer(model_path)
    return _SELF_RENDERERS[model_path]


# ------------------------------------------------------------------------------ 知覚
class Perception:
    def __init__(self, setup, params: Params = None, color_thr: C.Thresholds = None):
        self.setup = setup
        self.p = params or Params()
        self.thr = color_thr
        self.selfr = shared_self_renderer(setup.robot_xml)
        self.rays ={n: pixel_rays(c.depth) for n, c in setup.cameras.items()}
        self.hist_t, self.hist_q = [], []
        self.planes = {}                       # {カメラ: (法線, 点)}（固定のカメラだけ、起動時に当てはめて保つ）
        self.corrections = {}                  # {カメラ: (回転, 中心, 平行移動)}（机の面での外部パラメータの補正）
        self.rng = np.random.default_rng(0)
        self.n0 = np.asarray(setup.table_normal, float)
        self.p0 = np.array([0.0, 0.0, setup.table_z])
        self.box = {"xy": np.asarray(setup.box_nominal_xy, float).copy(), "yaw": 0.0, "ok": False}
        self.cubes = {}
        self.flags = {}
        self._unseen = {c: 0.0 for c in COLORS}
        self._last_t = None

    # ----------------------------------------------------------------- 関節の記録
    def record_joints(self, joints) -> None:
        self.hist_t.append(float(joints.t))
        self.hist_q.append(np.asarray(joints.q, float))
        if len(self.hist_t) > 400:
            del self.hist_t[:100]
            del self.hist_q[:100]

    def q_at(self, t: float) -> np.ndarray:
        if not self.hist_t:
            raise RuntimeError("no joint history")
        i = int(np.searchsorted(self.hist_t, t))
        i = min(max(i, 0), len(self.hist_t) - 1)
        return self.hist_q[i]

    # --------------------------------------------------------------- 点群
    def camera_pose(self, name: str, t_capture: float, width: float):
        """世界 ← カメラ（実行系が持つ値）。手首は撮った時刻の関節角の順運動学。ロボットの描画の姿勢も合わせて置く。"""
        cam = self.setup.cameras[name]
        q = self.q_at(t_capture)
        self.selfr.set_q(q, width)
        if cam.mount == "world":
            R, t = np.asarray(cam.extrinsic.R), np.asarray(cam.extrinsic.t)
            corr = self.corrections.get(name)
            if corr is not None:                               # 机の面での補正（傾き 2 つと高さ）
                Rc, c, d = corr
                return Rc @ R, Rc @ (t - c) + c + d
            return R, t
        hp, hR = self.selfr.hand_pose()
        return hR @ np.asarray(cam.extrinsic.R), hR @ np.asarray(cam.extrinsic.t) + hp

    def points(self, frame, width: float, margin: float = None):
        """1 フレームの点群: 世界の点 (N, 3)、色の添字（-1 なし・0〜2 は COLORS・3 は箱の黄）、机の面からの高さ、自己除去の後の
        画素の添字。margin は自己除去の太らせ方 [m]（既定は params）。"""
        name = frame.name
        cam = self.setup.cameras[name]
        R, t = self.camera_pose(name, frame.t_capture, width)
        z = np.asarray(frame.depth, np.float64)
        valid = z > 0
        # 自己除去: ロボットの描画のマスクを、距離に応じた画素の数だけ太らせる
        zr = self.selfr.depth(R, t, cam.depth)
        robot = np.isfinite(zr)
        m = self.p.self_margin_m if margin is None else margin
        if robot.any() and m > 0:
            r_px = int(math.ceil(m * cam.depth.fx / max(float(np.median(zr[robot])), 0.05)))
            if r_px > 0:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_px + 1, 2 * r_px + 1))
                robot = cv2.dilate(robot.astype(np.uint8), k).astype(bool)
        keep = valid & ~robot
        ii = np.flatnonzero(keep.reshape(-1))
        pc = self.rays[name].reshape(-1, 3)[ii] * z.reshape(-1)[ii][:, None]
        pw = pc @ R.T + t
        # 机の面からの高さは、そのカメラ自身の深度から当てはめた平面で測る（カメラの較正誤差・深度の系統のずれが、
        # 物の点と机の点に同じように効くので打ち消される）。俯瞰は起動時に当てはめた平面、手首はフレームごとに当てはめる
        plane = self.planes.get(name) if cam.mount == "world" else None     # 俯瞰は起動時の平面（補正の後は既知の平面）
        if plane is None:
            hb = (pw - self.p0) @ self.n0
            f = fit_plane(pw[np.abs(hb) < 0.03], self.rng)
            plane = (f[0], f[1]) if f is not None else (self.n0, self.p0)
        h = (pw - plane[1]) @ plane[0]
        # 色: 同じ姿勢の RGB の切り出しへ投影し、目標位置の入力と同じ閾値で分類
        u, v, ok = project(pc, cam.rgb_crop)
        col = np.full(len(pw), -1, dtype=np.int8)
        rgb = frame.rgb
        if ok.any():
            px = rgb[v[ok], u[ok]].astype(np.int16)
            lab = np.full(ok.sum(), -1, dtype=np.int8)
            for ci, c in enumerate(COLORS):
                ch = C.CHANNEL[c]
                others = np.max(np.delete(px, ch, axis=1), axis=1)
                lab[(px[:, ch] >= self.thr.min_value) & (px[:, ch] - others >= self.thr.min_margin)] = ci
            yellow = (px[:, 0] >= 120) & (px[:, 1] >= 90) & (px[:, 2] <= 0.6 * np.minimum(px[:, 0], px[:, 1]))
            lab[(lab < 0) & yellow] = 3
            col[ok] = lab
        return {"pw": pw, "h": h, "col": col, "idx": ii, "robot_mask": robot, "R": R, "t": t, "plane": plane}

    # ------------------------------------------------------------------ 机の面
    def check_table(self, frame, width: float) -> dict:
        """起動時（俯瞰）: 自分の深度から平面を当てはめ、実行系が持つ机の面とのずれ（高さ・傾き）を返す（止まり方の判定に使う）。
        correct が真なら、当てはめた平面が既知の平面に重なるように、俯瞰カメラの実行系が持つ外部パラメータの傾き 2 つと
        高さを直す（向き（鉛直まわり）と水平の位置は直せない）。以後の高さの基準は既知の平面になる。"""
        name = frame.name
        self.planes.pop(name, None)
        self.corrections.pop(name, None)
        P = self.points(frame, width)
        n, c = P["plane"]
        wc = np.array([np.mean(self.setup.workspace["x"]), np.mean(self.setup.workspace["y"])])
        z_fit = c[2] - (n[0] * (wc[0] - c[0]) + n[1] * (wc[1] - c[1])) / n[2]
        z_bel = self.p0[2] - (self.n0[0] * wc[0] + self.n0[1] * wc[1]) / self.n0[2]
        ang = math.degrees(math.acos(float(np.clip(n @ self.n0, -1, 1))))
        out = {"dz_m": float(z_fit - z_bel), "angle_deg": ang, "corrected": False}
        if self.p.correct_overhead:
            Rc = _rot_between(n, self.n0)                      # n を既知の法線へ回す（c を中心に）
            d = self.n0 * float((self.p0 - c) @ self.n0)       # c を既知の平面の上へ
            self.corrections[name] = (Rc, c, d)
            self.planes[name] = (self.n0.copy(), c + d)
            out["corrected"] = True
        else:
            self.planes[name] = (n, c)
        return out

    # -------------------------------------------------------------------- 箱
    def fit_box(self, P) -> dict:
        s = self.setup
        top = s.box_height
        sel = (P["col"] == 3) & (P["h"] > top + self.p.box_band[0]) & (P["h"] < top + self.p.box_band[1])
        xy = P["pw"][sel, :2]
        if len(xy) < 30:
            return {"ok": False, "n": int(len(xy))}
        half = 0.5 * (s.box_outer - s.box_wall)                  # 壁の中心線
        rect = cv2.minAreaRect(xy.astype(np.float32))
        c0 = np.array(rect[0], float)
        th0 = math.radians(rect[2]) % (math.pi / 2)

        def cost(x):
            cx, cy, th = x
            ct, st = math.cos(th), math.sin(th)
            d = xy - (cx, cy)
            qx, qy = d[:, 0] * ct + d[:, 1] * st, -d[:, 0] * st + d[:, 1] * ct
            e = np.abs(np.maximum(np.abs(qx), np.abs(qy)) - half)
            return float(np.sum(np.minimum(e, 0.01) ** 2))

        from scipy.optimize import minimize
        best = None
        for dth in (0.0, math.pi / 4):
            r = minimize(cost, [c0[0], c0[1], th0 + dth], method="Nelder-Mead",
                         options={"xatol": 1e-5, "fatol": 1e-10, "maxiter": 400})
            if best is None or r.fun < best.fun:
                best = r
        cx, cy, th = best.x
        return {"ok": True, "xy": np.array([cx, cy]), "yaw": float(th % (math.pi / 2)), "n": int(len(xy)),
                "rms": float(math.sqrt(best.fun / len(xy)))}

    def init_box(self, frames_and_widths) -> dict:
        """試行の始めに数フレームで箱を推定し、平均を保持する。既知の位置（設置の前提）からのずれは監視に使う。"""
        fits = [f for f in (self.fit_box(self.points(fr, w)) for fr, w in frames_and_widths) if f["ok"]]
        if not fits:
            self.box = {"xy": self.box["xy"], "yaw": 0.0, "ok": False}
            self.flags["box_not_found"] = True
            return self.box
        xy = np.mean([f["xy"] for f in fits], axis=0)
        yaws = np.array([f["yaw"] for f in fits])
        yaw = float(np.angle(np.mean(np.exp(4j * yaws))) / 4) % (math.pi / 2)
        self.box = {"xy": xy, "yaw": yaw, "ok": True, "n_frames": len(fits)}
        dev = float(np.linalg.norm(xy - np.asarray(self.setup.box_nominal_xy)))
        ydev = math.degrees(min(yaw, math.pi / 2 - yaw))
        self.box.update({"dev_m": dev, "dev_yaw_deg": ydev})
        if dev > 0.02 or ydev > 3.0:
            self.flags["box_moved"] = True
        return self.box

    def in_box_xy(self, xy, shrink: float = 0.0) -> bool:
        s = self.setup
        b = self.box
        d = np.asarray(xy) - b["xy"]
        ct, st = math.cos(b["yaw"]), math.sin(b["yaw"])
        qx, qy = d[0] * ct + d[1] * st, -d[0] * st + d[1] * ct
        inner = 0.5 * s.box_outer - s.box_wall - shrink
        return max(abs(qx), abs(qy)) <= inner

    # ------------------------------------------------------------------- 立方体
    def measure_cubes(self, P) -> dict:
        s = self.setup
        out = {}
        above = P["h"] > self.p.table_h
        for ci, c in enumerate(COLORS):
            sel = above & (P["col"] == ci)
            pts = P["pw"][sel]
            hh = P["h"][sel]
            if len(pts) < self.p.min_points:
                continue
            med = np.median(pts[:, :2], axis=0)
            keep = np.linalg.norm(pts[:, :2] - med, axis=1) < 0.035
            pts, hh = pts[keep], hh[keep]
            if len(pts) < self.p.min_points:
                continue
            top = np.percentile(hh, 95)
            tp = pts[hh > top - self.p.top_band]
            xy = tp[:, :2].mean(axis=0) if len(tp) >= 5 else pts[:, :2].mean(axis=0)
            yaw = 0.0
            if len(tp) >= 8:
                yaw = math.radians(cv2.minAreaRect(tp[:, :2].astype(np.float32))[2]) % (math.pi / 2)
            n, c0 = P["plane"]
            zc = c0[2] - (n[0] * (xy[0] - c0[0]) + n[1] * (xy[1] - c0[1])) / n[2]   # その xy の机の面の高さ
            pos = np.array([xy[0], xy[1], zc + top - 0.5 * s.cube_size])
            out[c] = {"pos": pos, "yaw": yaw, "n": int(len(pts)), "top_h": float(top)}
        return out

    def update(self, frames: dict, gripper, joints_t: float, hand_tip=None) -> WorldModel:
        """10 Hz。frames: {name: CameraFrame}（届いた最新のフレーム）。gripper: GripperState。hand_tip: 指先の位置（順運動学）。"""
        t = float(joints_t)
        dt = 0.0 if self._last_t is None else t - self._last_t
        self._last_t = t
        seen = set()
        for name in ("overhead", "wrist"):
            fr = frames.get(name)
            if fr is None or fr.depth is None:
                continue
            P = self.points(fr, gripper.width)
            sig = self.p.sigma_wrist if name == "wrist" else self.p.sigma_overhead
            for c, m in self.measure_cubes(P).items():
                seen.add(c)
                self._fuse(c, m, sig, t, name)
        for c in COLORS:
            e = self.cubes.get(c)
            if e is None:
                continue
            if c in seen:
                self._unseen[c] = 0.0
                continue
            held = (gripper.is_grasped and hand_tip is not None and np.linalg.norm(e.pos - hand_tip) < 0.04)
            if held:
                e.pos, e.status, e.t_seen = np.asarray(hand_tip, float).copy(), "in_hand", t
                continue
            self._unseen[c] += dt
            e.status = "held" if self._unseen[c] < self.p.lost_s else "lost"
        for e in self.cubes.values():
            e.in_box = bool(self.box["ok"] and self.in_box_xy(e.pos[:2]) and e.status != "in_hand")
        return WorldModel(t, float(self.p0[2]), dict(self.box), {c: dataclasses.replace(e) for c, e in self.cubes.items()},
                          dict(self.flags))

    def _fuse(self, c, m, sig, t, source) -> None:
        e = self.cubes.get(c)
        if e is None or np.linalg.norm(m["pos"] - e.pos) > self.p.moved_tol or e.status in ("lost", "in_hand"):
            self.cubes[c] = CubeEstimate(c, m["pos"], m["yaw"], sig ** 2, t, False, "seen", source)
            return
        var = e.var + (0.002 ** 2)                               # 触れない限り動かないが、少しの揺らぎを許す
        w = var / (var + sig ** 2)
        e.pos = e.pos + w * (m["pos"] - e.pos)
        e.var = (1 - w) * var
        e.yaw = m["yaw"] if source == "wrist" else e.yaw
        e.t_seen, e.status, e.source = t, "seen", source
