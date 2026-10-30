"""実機の記録（1 kHz の腕・グリッパー）を、学習データ（R1v3、LeRobot 10 fps）と同じ列の 10 Hz の表にする。

    python3 real_to_ds.py --arm arm.csv --gripper gripper.csv --replay-ds n_20000_red_r0_ds.csv --out real_ds.csv [--frame base|task]

python3 と numpy だけで動く（リポジトリを読まない）。学習データの定義（src/recovla/data/convert.py・vla_state.py）から、
要る関数だけを写した（quat_mul・orientation_deviation。tests/test_real_replay.py で元と一致を確かめる）。

列（docs/real/replay_format.md の 4 節）:
  frame_index, timestamp          フレームの番号と、エピソードの始めからの時刻（frame_index × 0.1 s）
  t_stream                        流し始めからの時刻（実機の記録の t と同じ時計）。再生の _ds.csv の値をそのまま使う
  observation.state の 15 列       eef_x, eef_y, eef_z（hand の原点＝フランジ [m]）, eef_rot_dev_x/_y/_z（真下からの向きのずれ
                                  [rad]）, finger_left, finger_right_neg（開き幅の半分と、その符号を反転したもの [m]）,
                                  joint1..joint7 [rad]
  action の 7 列                   dx, dy, dz（次の 0.1 s の手先の指令の変化 [m]）, drx, dry, drz（= 0）, gripper（+1 閉 / −1 開）
  meas_dx, meas_dy, meas_dz       （実機だけ）測った手先の次の 0.1 s の変化 [m]。参考
画像（observation.images.*）と、画像から作る目標位置の入力（cue_x, cue_y）は、カメラがないので出さない。

フレームを取る時刻: シミュレーションの記録器は、物理の 1 ステップの後、その時刻 T に、その時点で届いている最新の測定（センサのモデル）を
読む（record/episode.py・harness/gen_sensed.py）。ここでも、各フレームの時刻 T（再生の _ds.csv の t_stream）に、t ≤ T の最後の
腕の行と最後のグリッパーの読みを使う。
行動（指令の変化）は、実機が同じ指令の列を流したので、再生の _ds.csv の値をそのまま写す（実機の側に x_des はない）。
座標: --frame base（既定）は実機の台座の座標。task は台座の座標を縦の軸まわりに +130° 回したもの（シミュレーションの座標と同じ）。
"""
import argparse
import csv
import math
import sys

import numpy as np

Q1_SHIFT = -math.radians(130.0)                   # 実機の q1 = シミュレーションの q1 − 130°
F_T_EE_Z = 0.1034                                 # O_T_EE（Franka Hand の既定）はフランジから z に 0.1034 m
Q_DOWN = np.array([0.0, 1.0, 0.0, 0.0])
STATE_NAMES = (["eef_x", "eef_y", "eef_z", "eef_rot_dev_x", "eef_rot_dev_y", "eef_rot_dev_z",
                "finger_left", "finger_right_neg"] + [f"joint{i}" for i in range(1, 8)])
ACTION_NAMES = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]
DS_COLS = ["frame_index", "timestamp", "t_stream"] + STATE_NAMES + ACTION_NAMES
REAL_EXTRA = ["meas_dx", "meas_dy", "meas_dz"]


# ------------------------------------------------------------------ 学習データの定義（vla_state.py から写した）

def quat_mul(a, b) -> np.ndarray:
    """(N, 4) w,x,y,z のハミルトン積（b は (4,) でもよい）。"""
    a = np.asarray(a, float)
    b = np.broadcast_to(np.asarray(b, float), a.shape)
    aw, ax, ay, az = a.T
    bw, bx, by, bz = b.T
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], axis=1)


def orientation_deviation(quat) -> np.ndarray:
    """(N, 4) 手先の四元数 → 世界の座標の q * conj(Q_DOWN) の回転ベクトル（0 は真下向き）。"""
    conj_down = Q_DOWN * np.array([1.0, -1.0, -1.0, -1.0])
    d = quat_mul(np.asarray(quat, dtype=np.float64), conj_down)
    d[d[:, 0] < 0] *= -1.0
    s = np.linalg.norm(d[:, 1:], axis=1)
    angle = 2.0 * np.arctan2(s, d[:, 0])
    scale = np.where(s > 1e-12, angle / np.maximum(s, 1e-12), 2.0)
    return d[:, 1:] * scale[:, None]


def policy_state(ee_pos, ee_quat, fingers, joints) -> np.ndarray:
    fingers = np.array(fingers, float, copy=True)
    fingers[:, 1] *= -1.0
    return np.concatenate([np.asarray(ee_pos, float), orientation_deviation(ee_quat), fingers,
                           np.asarray(joints, float)], axis=1).astype(np.float32)


# ------------------------------------------------------------------ 座標

def rz_quat(theta: float) -> np.ndarray:
    return np.array([math.cos(theta / 2.0), 0.0, 0.0, math.sin(theta / 2.0)])


def rz(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def mat_to_quat(R) -> np.ndarray:
    """回転行列 (3, 3) → 四元数 w,x,y,z（w ≥ 0）。"""
    R = np.asarray(R, float)
    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def flange_from_otee(otee16) -> tuple:
    """libfranka の O_T_EE（列ごとの 16 個）→ フランジ（シミュレーションの hand の原点）の位置と四元数。"""
    T = np.asarray(otee16, float).reshape(4, 4).T
    R = T[:3, :3]
    return T[:3, 3] - R[:, 2] * F_T_EE_Z, mat_to_quat(R)


# ------------------------------------------------------------------ CSV

def read_csv(path) -> dict:
    """列名 → np.ndarray（数の列）。数でない列は文字の配列。"""
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.reader(f)
        head = next(rd)
        data = list(rd)
    out = {}
    for i, h in enumerate(head):
        col = [r[i] for r in data]
        try:
            out[h] = np.array([float(x) if x != "" else np.nan for x in col])
        except ValueError:
            out[h] = np.array(col)
    return out


def write_csv(path, rows, cols) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow(r)


def fmt(x, nd: int = 10) -> str:
    return f"{float(x):.{nd}f}"


# ------------------------------------------------------------------ 変換

def last_at_or_before(t_src, t_query) -> np.ndarray:
    """各 t_query について、t_src ≤ t_query の最後の行の番号（無ければ 0）。"""
    idx = np.searchsorted(np.asarray(t_src, float), np.asarray(t_query, float) + 1e-9, side="right") - 1
    return np.clip(idx, 0, len(t_src) - 1)


def to_frame(pos, quat, joints, frame: str) -> tuple:
    """台座の座標の値を、frame（base はそのまま、task は +130° 回してシミュレーションの座標）へ。"""
    if frame == "base":
        return pos, quat, joints
    th = -Q1_SHIFT
    j = np.array(joints, float, copy=True)
    j[:, 0] += th
    return pos @ rz(th).T, quat_mul(np.broadcast_to(rz_quat(th), quat.shape).copy(), quat), j


def real_to_ds(arm: dict, grip: dict, replay_ds: dict, frame: str = "base") -> tuple:
    """実機の記録 → (行の列, 列名)。フレームの時刻・行動は再生の _ds.csv から。"""
    T = np.asarray(replay_ds["t_stream"], float)
    ia = last_at_or_before(arm["t"], T)
    otee = np.stack([np.asarray(arm[f"O_T_EE{i}"], float) for i in range(16)], axis=1)[ia]
    pq = [flange_from_otee(o) for o in otee]
    pos = np.array([p for p, _ in pq])
    quat = np.array([q for _, q in pq])
    joints = np.stack([np.asarray(arm[f"q{i}"], float) for i in range(7)], axis=1)[ia]
    if grip is not None and len(grip.get("t", [])):
        w = np.asarray(grip["width"], float)[last_at_or_before(grip["t"], T)]
    else:
        w = np.full(len(T), np.nan)
    fingers = np.stack([0.5 * w, 0.5 * w], axis=1)
    pos_f, quat_f, joints_f = to_frame(pos, quat, joints, frame)
    state = policy_state(pos_f, quat_f, fingers, joints_f)
    meas = np.vstack([pos_f[1:] - pos_f[:-1], np.full((1, 3), np.nan)])
    act_cols = ACTION_NAMES
    if frame == "task":                                   # 再生の _ds.csv の行動は台座の座標: 水平を +130° 回す
        a = np.stack([np.asarray(replay_ds[c], float) for c in ("dx", "dy", "dz")], axis=1) @ rz(-Q1_SHIFT).T
    else:
        a = np.stack([np.asarray(replay_ds[c], float) for c in ("dx", "dy", "dz")], axis=1)
    rows = []
    for k in range(len(T)):
        rows.append([int(replay_ds["frame_index"][k]), fmt(replay_ds["timestamp"][k], 3), fmt(T[k], 3)]
                    + [fmt(x) for x in state[k]] + [fmt(x) for x in a[k]]
                    + [fmt(replay_ds[c][k]) for c in act_cols[3:]] + [fmt(x) for x in meas[k]])
    return rows, DS_COLS + REAL_EXTRA


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", required=True)
    ap.add_argument("--gripper", default=None)
    ap.add_argument("--replay-ds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frame", choices=("base", "task"), default="base")
    a = ap.parse_args(argv)
    rds = read_csv(a.replay_ds)
    rows, cols = real_to_ds(read_csv(a.arm), read_csv(a.gripper) if a.gripper else None, rds, a.frame)
    write_csv(a.out, rows, cols)
    print(f"{len(rows)} こま → {a.out}（座標 {a.frame}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
