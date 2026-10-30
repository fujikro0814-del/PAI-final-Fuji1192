"""記録した腕の状態を方策の observation.state に変える定義（ここだけで定義する）。

学習データの変換（convert.episode_arrays）と評価の入口（vla_observation）の両方が使う。

observation.state（15 次元、float32）:

  eef_x, eef_y, eef_z               ee_pos: 手先の原点 [m, world]
  eef_rot_dev_x, _y, _z             真下向きからの姿勢のずれ [rad]
  finger_left, finger_right_neg     fingers: finger_joint1, -finger_joint2 [m]（LIBERO の符号）
  joint1 .. joint7                  関節 [rad]

姿勢は q * conj(Q_DOWN) の world 座標の回転ベクトル（Q_DOWN = 真下向き）で、3 つとも 0 の近くにくる。
LIBERO の絶対の回転ベクトルは使わない: 手が下向きだと x 成分が pi の近くになり（[-pi, pi] では +pi と -pi の間で跳ぶ）、
LeRobot 0.6.1 は分散を float32 で mean(x^2) - mean(x)^2 と計算するため、pi の近くで 3e-4 だけ動く値は
桁落ちで分散が 0 になり、正規化した値が 1e5 に達する。依存は numpy だけ。
"""
import numpy as np

STATE_VERSION = 1
Q_DOWN = np.array([0.0, 1.0, 0.0, 0.0])   # w, x, y, z: world の x 軸まわりに pi
STATE_NAMES = (["eef_x", "eef_y", "eef_z", "eef_rot_dev_x", "eef_rot_dev_y",
                "eef_rot_dev_z", "finger_left", "finger_right_neg"]
               + [f"joint{i}" for i in range(1, 8)])
STATE_DIM = len(STATE_NAMES)
FRAME_FIELDS = {"ee_pos": 3, "ee_quat": 4, "fingers": 2, "joints": 7}   # 記録の項目 → 幅


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N, 4) の w,x,y,z の四元数の積（b は (4,) でもよい）。"""
    b = np.broadcast_to(b, a.shape)
    aw, ax, ay, az = a.T
    bw, bx, by, bz = b.T
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], axis=1)


def orientation_deviation(quat: np.ndarray) -> np.ndarray:
    """(N, 4) の手先の四元数 → (N, 3) の q * conj(Q_DOWN) の world 座標の回転ベクトル（角度 [0, pi]、0 = 真下向き）。"""
    conj_down = Q_DOWN * np.array([1.0, -1.0, -1.0, -1.0])
    d = quat_mul(np.asarray(quat, dtype=np.float64), conj_down)
    d[d[:, 0] < 0] *= -1.0
    s = np.linalg.norm(d[:, 1:], axis=1)
    angle = 2.0 * np.arctan2(s, d[:, 0])
    scale = np.where(s > 1e-12, angle / np.maximum(s, 1e-12), 2.0)  # 小さい角度の極限
    return d[:, 1:] * scale[:, None]


def policy_state(ee_pos, ee_quat, fingers, joints) -> np.ndarray:
    """(N, ...) の記録の項目 → (N, 15) float32 の observation.state。"""
    arrays = {"ee_pos": np.asarray(ee_pos), "ee_quat": np.asarray(ee_quat),
              "fingers": np.asarray(fingers), "joints": np.asarray(joints)}
    n = None
    for name, width in FRAME_FIELDS.items():
        a = arrays[name]
        if a.ndim != 2 or a.shape[1] != width:
            raise ValueError(f"{name}: expected (N, {width}), got {a.shape}")
        if n is None:
            n = a.shape[0]
        elif a.shape[0] != n:
            raise ValueError(f"{name}: {a.shape[0]} rows, expected {n}")
    fingers = arrays["fingers"].copy()
    fingers[:, 1] *= -1.0
    state = np.concatenate([arrays["ee_pos"], orientation_deviation(arrays["ee_quat"]),
                            fingers, arrays["joints"]], axis=1)
    return state.astype(np.float32)


# 目標位置の入力。15 次元の後ろに足して 18 次元にする。値は recovla.perception.color.TargetCue
# （俯瞰画像の指示色の画素の重心を机の面へ投影した x, y [m, world] と、見えているかの旗 1 / 0）。学習の変換と評価の
# 入口は、どちらも conversion.json の target_cue を見てこの 3 つを足す。旗は学習データで一定になりうるので、
# 正規化の値を固定する（CUE_FIXED_STATS。変換が meta/stats.json に書き、検証が例外として扱う）
CUE_NAMES = ["cue_x", "cue_y", "cue_visible"]
CUE_FIXED_STATS = {"cue_visible": {"mean": 0.5, "std": 0.5}}      # 1 → +1、0 → −1
# 学習データで旗が 0 のフレームが 1% 未満なら、旗は方策の入力から外して記録だけに残す（17 次元）
CUE_NAMES_NO_FLAG = ["cue_x", "cue_y"]


# 置く位置の入力（docs/api/r2_protocol.md 6.4 節）。目標位置の入力の後ろに足す 2 つ（x, y [m, world]）。
# 学習: 目標の立方体がエピソードの終わりに止まった位置（シミュレーションの値、後から付ける正解。1 本の中で一定）。
# 評価: 手順・やり直しの始めに、推定した箱と箱の中の物から選んだ空いたマスの中心（recovla.api.place_cue）
PLACE_NAMES = ["place_x", "place_y"]


def with_place(state: np.ndarray, place) -> np.ndarray:
    """(N, D) と (N, 2)（または (D,) と (2,)）→ D + 2 次元の float32。"""
    s, p = np.asarray(state, np.float32), np.asarray(place, np.float32)
    if p.shape[-1] != len(PLACE_NAMES):
        raise ValueError(f"place {p.shape}")
    if p.ndim < s.ndim:
        p = np.broadcast_to(p, s.shape[:-1] + p.shape[-1:])
    return np.concatenate([s, p], axis=-1)


def cue_names(keep_flag: bool = True) -> list:
    return list(CUE_NAMES if keep_flag else CUE_NAMES_NO_FLAG)


def with_cue(state: np.ndarray, cue: np.ndarray, keep_flag: bool = True) -> np.ndarray:
    """(N, 15) と (N, 3)（または (15,) と (3,)）→ 18 次元（keep_flag が偽なら旗を落として 17 次元）の float32。"""
    s, c = np.asarray(state, np.float32), np.asarray(cue, np.float32)
    if s.shape[-1] != STATE_DIM or c.shape[-1] != len(CUE_NAMES) or s.shape[:-1] != c.shape[:-1]:
        raise ValueError(f"state {s.shape} / cue {c.shape}")
    return np.concatenate([s, c if keep_flag else c[..., :2]], axis=-1)


def policy_state_frame(frame: dict) -> np.ndarray:
    """1 フレーム {ee_pos (3,), ee_quat (4,), fingers (2,), joints (7,)} → (15,) float32。"""
    missing = set(FRAME_FIELDS) - set(frame)
    if missing:
        raise ValueError(f"proprio frame is missing {sorted(missing)}")
    rows = {}
    for name, width in FRAME_FIELDS.items():
        a = np.asarray(frame[name])
        if a.shape != (width,):
            raise ValueError(f"{name}: expected ({width},), got {a.shape}")
        rows[name] = a[None]
    return policy_state(rows["ee_pos"], rows["ee_quat"], rows["fingers"], rows["joints"])[0]
