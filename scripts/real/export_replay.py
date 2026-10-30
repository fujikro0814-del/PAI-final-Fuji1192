"""生成プログラムの指令の再生（実機の試験 A8）の書き出し: 合わせる前のシミュレーションの学習データ（R1v3）と同じ生成を
実行系の腕の層で回し、実機に流す関節の指令の列を作る。

    python scripts/real/export_replay.py                      # 既定の 19 本（通常 10・立て直し A・B・C 各 3）を outputs/real/replay_v1/ へ
    python scripts/real/export_replay.py --q1-max -1.66 --out outputs/real/replay_v1

中身:
- 世界は合わせる前の生成と同じ（expert/generate.py の rig "v3"＝SensedDrivenRig、configs の v2 の層＋expert_v3）。
  各本は R1v3 のデータを作った生成（outputs/gen/F_data_v3_20261005-133737）と同じ配置のシード・色・やり直しの回数で
  run_attempt を 1 回だけ回し、所要時間・フレーム数がその記録と一致することを確かめる（違えば止めて報告）。
- 記録は物理の 1 ステップ（500 Hz）ごと: 制限層（runtime/limiter.py＝libfranka の limitRate）の出力の 1 kHz の関節の指令 2 つ、
  シミュレーションの測った関節角・速さ、指の開き幅。ハンドの指令（move・grasp）はハンドの記録（hand.log）から。
- 制限層の出力はもともと 1 kHz なので補間はしない。シミュレーションの測った値（500 Hz）は 1 kHz の行に線形に補間する。
- 立て直し（A・B・C）は試行の最初（開始姿勢で止まった状態）から全部を書く。学習に入ったのは記録を始めた後だけで、その行は
  recorded = 1。実機は動いている途中から流し始められないため。
- 生成プログラムが終わった後、腕が止まりきるまで（制限層の速さ・加速度が 0）手先を保つ指令を続け、さらに 0.2 s 止めて終える（tail = 1）。
- 実機への変換: q1 を −130°（−2.2689 rad）ずらすだけ。ほかの関節は同じ。手先は libfranka の O_T_EE の決まり
  （フランジ＝シミュレーションの hand の原点から、hand の z 軸に 0.1034 m。向きは hand と同じ）で、ずらした関節角の順運動学。
- 確かめ（ずらした後の指令の列、書き出した桁で計算）: 関節の可動域、q1 の範囲（--q1-min/--q1-max）、libfranka 0.9.2 の
  rate_limiting.h の関節の速度・加速度・躍度の上限（余裕を掛けない値）、指の幅 0〜0.08 m、最後に止まっていること。
  外れた本は書き出さずに理由を manifest に残す（切り詰めて通すことはしない）。
- 学習データと同じ形の 10 Hz の表（<名前>_ds.csv）: 記録器のフレーム（センサのモデルの値）から、data/convert.py の episode_arrays と
  同じ取り方で observation.state（15、目標位置の入力 cue_x・cue_y は画像から作るので除く）と action（7）。座標は実機の台座。
  シミュレーションの座標で計算した値が、R1v3 の学習データのその本の行と一致することを確かめる（違えば除外）。
書くもの: outputs/real/replay_v1/<名前>.csv・<名前>_gripper.csv・<名前>_ds.csv・manifest.json。列は docs/real/replay_format.md。
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from real_to_ds import DS_COLS, quat_mul, read_csv, rz, rz_quat  # noqa: E402,F401  (read_csv はテストが使う)

DT = 1e-3
Q1_SHIFT = -math.radians(130.0)
F_T_EE_Z = 0.1034
# libfranka 0.9.2 include/franka/rate_limiting.h（余裕を掛けない上限）と、Panda の関節の可動域（データシート）
MAX_VEL = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])
MAX_ACC = np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0])
MAX_JERK = np.array([7500.0, 3750.0, 5000.0, 6250.0, 7500.0, 10000.0, 10000.0])
Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_MAX = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
# 手先の上限（runtime/motion.py の CART。関節の位置の指令の口では実機は関節の上限で判定するので、参考として出す）
CART = (1.7, 13.0, 6500.0)
DECIMALS = 10                                  # 書き出す桁（rad・m）。丸めの躍度への影響は 4·0.5e-10/1e-9 = 2 rad/s³ 以下
GEN_RUN = "outputs/gen/F_data_v3_20261005-133737"
DATASET = "outputs/datasets/R1v3_20261005-133737"
# R1v3 の学習データの本（outputs/manifests/R1v3_20261005-133737.json にある名前）。(種類, 配置のシード, 色, やり直しの回数)
TRIALS = [("n", 20000, "red", 0), ("n", 20000, "green", 0), ("n", 20000, "blue", 0),
          ("n", 20001, "red", 0), ("n", 20001, "green", 0), ("n", 20001, "blue", 0),
          ("n", 20002, "red", 0), ("n", 20002, "green", 0), ("n", 20002, "blue", 0),
          ("n", 20003, "red", 0),
          ("A", 30000, "red", 0), ("A", 30001, "green", 0), ("A", 30002, "blue", 0),
          ("B", 31000, "red", 1), ("B", 31001, "green", 0), ("B", 31002, "blue", 0),
          ("C", 32000, "red", 0), ("C", 32001, "green", 0), ("C", 32002, "blue", 0)]

COLS = (["t", "tail", "recorded"] + [f"q_cmd{i}" for i in range(7)] + [f"q_sim{i}" for i in range(7)]
        + [f"dq_sim{i}" for i in range(7)] + ["ee_cmd_x", "ee_cmd_y", "ee_cmd_z", "ee_sim_x", "ee_sim_y", "ee_sim_z",
                                               "ee_sim_qw", "ee_sim_qx", "ee_sim_qy", "ee_sim_qz", "width_sim"])
GRIP_COLS = ["t", "kind", "width", "speed", "force", "eps_inner", "eps_outer"]


# ------------------------------------------------------------------ 変換と確かめ（純粋な関数。テストで照らす）

def to_real(q) -> np.ndarray:
    """シミュレーションの関節角（…, 7）を実機の関節角へ: q1 だけ −130° ずらす。"""
    q = np.array(q, dtype=float, copy=True)
    q[..., 0] += Q1_SHIFT
    return q


def derivatives(q, q_start) -> tuple:
    """1 kHz の指令の列の速度・加速度・躍度（libfranka と同じく前の指令との差。始めは q_start で止まっていたとする）。"""
    full = np.vstack([np.repeat(np.asarray(q_start, float)[None], 3, axis=0), np.asarray(q, float)])
    dq = np.diff(full, axis=0) / DT
    ddq = np.diff(dq, axis=0) / DT
    dddq = np.diff(ddq, axis=0) / DT
    return dq[2:], ddq[1:], dddq


def check_stream(q_real, q_start_real, widths=(), q1_range=(-2.80, -1.66)) -> dict:
    """実機の上限・範囲の確かめ。返り値の ok が偽なら reasons に理由。比は上限に対する最大（1 以下なら中）。"""
    q = np.asarray(q_real, float)
    dq, ddq, dddq = derivatives(q, q_start_real)
    reasons = []
    allq = np.vstack([np.asarray(q_start_real, float)[None], q])
    lo_viol = allq < Q_MIN
    hi_viol = allq > Q_MAX
    if lo_viol.any() or hi_viol.any():
        js = sorted(set(np.where(lo_viol | hi_viol)[1].tolist()))
        reasons.append(f"関節の可動域の外: 関節 {[j + 1 for j in js]}")
    q1lo, q1hi = float(q1_range[0]), float(q1_range[1])
    if allq[:, 0].min() < q1lo - 1e-12 or allq[:, 0].max() > q1hi + 1e-12:
        reasons.append(f"q1 が範囲 [{q1lo}, {q1hi}] の外: {allq[:, 0].min():.4f}〜{allq[:, 0].max():.4f}")
    ratios = {"dq": (np.abs(dq) / MAX_VEL).max(axis=0), "ddq": (np.abs(ddq) / MAX_ACC).max(axis=0),
              "dddq": (np.abs(dddq) / MAX_JERK).max(axis=0)}
    for k, r in ratios.items():
        if (r > 1.0).any():
            reasons.append(f"{k} が上限を超える: 関節 {[int(j) + 1 for j in np.where(r > 1.0)[0]]}（最大 {r.max():.3f} 倍）")
    w = np.asarray(list(widths), float)
    if w.size and ((w < 0).any() or (w > 0.08 + 1e-12).any()):
        reasons.append("指の幅が 0〜0.08 m の外")
    end_v = float(np.abs(dq[-1]).max()) if len(dq) else 0.0
    end_a = float(np.abs(ddq[-1]).max()) if len(ddq) else 0.0
    if end_v > 1e-6 or end_a > 1e-3:
        reasons.append(f"最後に止まっていない（速さ {end_v:.2e} rad/s、加速度 {end_a:.2e} rad/s²）")
    full = {"dq": np.abs(dq) / MAX_VEL, "ddq": np.abs(ddq) / MAX_ACC, "dddq": np.abs(dddq) / MAX_JERK}
    return {"ok": not reasons, "reasons": reasons,
            "ratio_max": {k: [round(float(x), 4) for x in v] for k, v in ratios.items()},
            # 上限の 9 割を超えた刻みの割合（どれかの関節）。制限層は余裕 0.99 で上限いっぱいの加減速を使う
            "frac_over_90pct": {k: round(float((v > 0.9).any(axis=1).mean()), 4) for k, v in full.items()},
            "ratio_max_all": {k: round(float(v.max()), 4) for k, v in ratios.items()},
            "q1_range_used": [round(float(allq[:, 0].min()), 4), round(float(allq[:, 0].max()), 4)],
            "end_speed": end_v, "end_accel": end_a}


def cart_ratios(ee) -> dict:
    """手先の位置の列（1 kHz）の速度・加速度・躍度の最大と、CART の上限に対する比（参考）。"""
    ee = np.asarray(ee, float)
    v = np.diff(ee, axis=0) / DT
    a = np.diff(v, axis=0) / DT
    j = np.diff(a, axis=0) / DT
    mx = [float(np.linalg.norm(x, axis=1).max()) if len(x) else 0.0 for x in (v, a, j)]
    return {"max": [round(m, 4) for m in mx], "ratio": [round(m / c, 4) for m, c in zip(mx, CART)]}


def write_csv(path, rows, cols) -> str:
    path = pathlib.Path(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow(r)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fmt(x) -> str:
    return f"{float(x):.{DECIMALS}f}"


# ------------------------------------------------------------------ シミュレーションで回す

class FK:
    """ロボットだけのモデルで、実機の決まりの手先（O_T_EE: hand の原点から hand の z に 0.1034 m）を出す。"""

    def __init__(self, motion):
        import mujoco
        self.mj = mujoco
        self.m = motion.model
        self.d = mujoco.MjData(self.m)
        self.qadr = motion.arm_qadr
        self.hid = motion.hand_id

    def __call__(self, q) -> tuple:
        self.d.qpos[self.qadr] = q
        self.mj.mj_kinematics(self.m, self.d)
        R = self.d.xmat[self.hid].reshape(3, 3)
        return self.d.xpos[self.hid] + R[:, 2] * F_T_EE_Z, self.d.xquat[self.hid].copy()


def run_trial(rig, kind, seed, color, retry, layout_kind) -> dict:
    from recovla.expert import generate as G
    cap = {"cmds": [], "meas": [], "start_q": None, "t0": None}
    orig_reset, orig_apply, orig_step = rig.reset, rig.apply_joint_commands, rig.physics_step

    def reset(layout):
        orig_reset(layout)
        cap["start_q"] = rig.motion.limiter.q.copy()
        cap["t0"] = float(rig.data.time)

    def apply(cmds):
        cap["cmds"].extend(np.array(c, float) for c in cmds)
        orig_apply(cmds)

    def step(on_step=None):
        orig_step(on_step)
        d = rig.data
        cap["meas"].append((float(d.time), d.qpos[rig.arm_qadr].copy(), d.qvel[rig.arm_vadr].copy(), rig.hand.width()))

    rig.reset, rig.apply_joint_commands, rig.physics_step = reset, apply, step
    try:
        spec = G.EpisodeSpec(int(seed), color, layout_kind, kind)
        t_wall = time.perf_counter()
        summ = G.run_attempt(rig, spec, int(retry), None, False)
        n_main = len(cap["cmds"])
        # 生成プログラムの後: 手先を保つ指令で、制限層の速さ・加速度が 0 になるまで続け、さらに 0.2 s 止める
        lim, hold_after = rig.motion.limiter, None
        for k in range(int(5.0 / (rig.steps_per_read * rig.timestep))):
            if hold_after is None and np.abs(lim.v).max() < 1e-9 and np.abs(lim.a).max() < 1e-9:
                hold_after = k
            if hold_after is not None and k - hold_after >= int(round(0.2 / (rig.steps_per_read * rig.timestep))):
                break
            rig.pad_read(None, False, None)
        wall = time.perf_counter() - t_wall
        hand_log = list(rig.hand.log)
    finally:
        rig.reset, rig.apply_joint_commands, rig.physics_step = orig_reset, orig_apply, orig_step
        for name in ("reset", "apply_joint_commands", "physics_step"):
            rig.__dict__.pop(name, None)
    frames = summ.pop("_frames", None) or []
    return {"summary": summ, "frames": frames, "cmds": np.array(cap["cmds"]), "n_main": n_main, "meas": cap["meas"],
            "start_q": cap["start_q"], "t0": cap["t0"], "hand_log": hand_log, "wall_s": round(wall, 2),
            "converged": hold_after is not None}


def build_rows(res, fk) -> tuple:
    """1 kHz の行。指令 i は流し始めから (i+1)·1 ms。シミュレーションの測った値は、物理の 1 ステップ n の後（指令 2n+1 の時刻）。"""
    cmds = to_real(res["cmds"])
    n = len(cmds)
    t = (np.arange(n) + 1) * DT
    meas_t = np.array([(k + 1) * 2 * DT for k in range(len(res["meas"]))])
    q_sim = to_real(np.array([m[1] for m in res["meas"]]))
    dq_sim = np.array([m[2] for m in res["meas"]])
    w_sim = np.array([m[3] for m in res["meas"]])
    interp = lambda arr: np.stack([np.interp(t, meas_t, arr[:, j]) for j in range(arr.shape[1])], axis=1)
    q_sim_i, dq_sim_i = interp(q_sim), interp(dq_sim)
    w_i = np.interp(t, meas_t, w_sim)
    t_rec = res["summary"].get("t_record_start")
    rows, ee_cmd = [], []
    cmds_r = np.round(cmds, DECIMALS)
    for i in range(n):
        e_c, _ = fk(cmds_r[i])
        e_s, quat = fk(q_sim_i[i])
        ee_cmd.append(e_c)
        tail = int(i >= res["n_main"])
        # 試行の時計（d.time）は reset で 0。物理の 1 ステップ n（d.time = 2n ms から）の指令は 2n+1・2n+2 ms の行
        recorded = int(res["summary"]["kind"] == "n" or (t_rec is not None and res["t0"] + t[i] > float(t_rec) + 1e-9))
        rows.append([f"{t[i]:.3f}", tail, recorded] + [fmt(x) for x in cmds_r[i]] + [fmt(x) for x in q_sim_i[i]]
                    + [fmt(x) for x in dq_sim_i[i]] + [fmt(x) for x in e_c] + [fmt(x) for x in e_s]
                    + [fmt(x) for x in quat] + [fmt(w_i[i])])
    return rows, cmds_r, np.array(ee_cmd)


def gripper_rows(res) -> list:
    out = []
    for (tt, kind, args) in res["hand_log"]:
        tr = float(tt) - res["t0"]
        a = list(args) + [None] * (5 - len(args))
        if kind == "move":
            out.append([f"{tr:.3f}", "move", fmt(a[0]), fmt(a[1]), "", "", ""])
        else:
            out.append([f"{tr:.3f}", "grasp"] + [fmt(x) for x in a[:5]])
    return out


def ds_arrays(frames, frame: str = "base") -> dict:
    """記録器のフレーム（20 Hz、センサのモデルの値）→ 学習データと同じ 10 fps の state（15）と action（7）。取り方は
    data/convert.py の episode_arrays と同じ（フレーム k = 記録 2k、行動 k = x_des[2k+2] − x_des[2k]、開閉は 2k+2 の状態）。
    frame "base" は実機の台座の座標（位置・x_des・手先の向きを縦の軸まわりに −130°、q1 − 130°）、"sim" はそのまま。"""
    from recovla.data import vla_state
    n = len(frames)
    k = (n - 1) // 2
    idx = np.arange(k) * 2
    nxt = idx + 2
    get = lambda key: np.array([np.asarray(frames[i][key], float) for i in range(n)])
    ee_pos, ee_quat, fingers, joints, x_des = get("ee_pos"), get("ee_quat"), get("fingers"), get("joints"), get("x_des")
    closed = np.array([bool(f["gripper_closed"]) for f in frames])
    t = np.array([float(f["sim_time"]) for f in frames])
    if frame == "base":
        R = rz(Q1_SHIFT)
        ee_pos, x_des = ee_pos @ R.T, x_des @ R.T
        ee_quat = quat_mul(np.broadcast_to(rz_quat(Q1_SHIFT), ee_quat.shape).copy(), ee_quat)
        joints = to_real(joints)
    state = vla_state.policy_state(ee_pos[idx], ee_quat[idx], fingers[idx], joints[idx])
    grip = np.where(closed[nxt], 1.0, -1.0)[:, None]
    action = np.concatenate([x_des[nxt] - x_des[idx], np.zeros((k, 3)), grip], axis=1).astype(np.float32)
    return {"t": t[idx], "state": state, "action": action}


def ds_from_dataset(state15, action, t) -> dict:
    """学習データの行（シミュレーションの座標）をそのまま、実機の台座の座標へ回す。向きのずれ（回転ベクトル v）は
    q = exp(v)·Q_DOWN で手先の四元数に戻し、縦の軸まわりに −130° 回してから同じ定義で計算し直す。"""
    from recovla.data import vla_state
    s = np.asarray(state15, float)
    v = s[:, 3:6]
    ang = np.linalg.norm(v, axis=1)
    axis = np.where(ang[:, None] > 1e-12, v / np.maximum(ang, 1e-12)[:, None], 0.0)
    d = np.concatenate([np.cos(ang / 2)[:, None], np.sin(ang / 2)[:, None] * axis], axis=1)
    quat = quat_mul(d, vla_state.Q_DOWN)
    R = rz(Q1_SHIFT)
    quat_b = quat_mul(np.broadcast_to(rz_quat(Q1_SHIFT), quat.shape).copy(), quat)
    fingers = s[:, 6:8].copy()
    fingers[:, 1] *= -1.0
    state = vla_state.policy_state(s[:, 0:3] @ R.T, quat_b, fingers, to_real(s[:, 8:15]))
    a = np.asarray(action, float).copy()
    a[:, 0:3] = a[:, 0:3] @ R.T
    return {"t": np.asarray(t, float), "state": state, "action": a.astype(np.float32)}


def ds_rows(arr) -> list:
    rows = []
    for k in range(len(arr["t"])):
        rows.append([k, f"{k * 0.1:.3f}", f"{arr['t'][k]:.3f}"] + [fmt(x) for x in arr["state"][k]]
                    + [fmt(x) for x in arr["action"][k]])
    return rows


class Dataset:
    """R1v3 の学習データ（LeRobot）の state・action（画像は読まない）。書き出しが学習データと同じかの照合に使う。"""

    def __init__(self, ds_dir: pathlib.Path):
        import pyarrow.parquet as pq
        conv = json.loads((ds_dir / "meta" / "conversion.json").read_text(encoding="utf-8"))
        self.episode = {s["raw_episode"]: int(s["episode_index"]) for s in conv["sources"]}
        cols = ["observation.state", "action", "episode_index", "frame_index", "timestamp"]
        parts = [pq.read_table(f, columns=cols) for f in sorted((ds_dir / "data").rglob("*.parquet"))]
        self.ep = np.concatenate([p.column("episode_index").to_numpy() for p in parts])
        self.fi = np.concatenate([p.column("frame_index").to_numpy() for p in parts])
        self.ts = np.concatenate([p.column("timestamp").to_numpy() for p in parts])
        self.state = np.concatenate([np.stack(p.column("observation.state").to_numpy(zero_copy_only=False)) for p in parts])
        self.action = np.concatenate([np.stack(p.column("action").to_numpy(zero_copy_only=False)) for p in parts])
        self.dir = ds_dir

    def match(self, name: str, arr) -> dict:
        """シミュレーションの座標の state（15）・action が、学習データのその本と一致するか（最大の差）。"""
        if name not in self.episode:
            return {"found": False}
        m = self.ep == self.episode[name]
        order = np.argsort(self.fi[m])
        st, ac, ts = self.state[m][order], self.action[m][order], self.ts[m][order]
        same_n = len(st) == len(arr["state"])
        out = {"found": True, "episode_index": self.episode[name], "frames_dataset": int(len(st)),
               "frames_export": int(len(arr["state"])), "same_frames": bool(same_n)}
        if same_n:
            out["state15_max_abs_diff"] = float(np.abs(st[:, :15] - arr["state"]).max())
            out["action_max_abs_diff"] = float(np.abs(ac - arr["action"]).max())
            out["timestamp_max_abs_diff"] = float(np.abs(ts - np.arange(len(ts)) * 0.1).max())
            out["state0_max_abs_diff"] = float(np.abs(st[0, :15] - arr["state"][0]).max())
        out["rows"] = (st[:, :15], ac)
        return out


def git_head() -> dict:
    """今のコミットと、書きかけがあるか（code_version は PATH の git を探すので、同梱の git も試す）。"""
    import subprocess
    for exe in ("git", str(ROOT / ".tools" / "git" / "cmd" / "git.exe"),
                str(ROOT.parent / "recovery_vla" / ".tools" / "git" / "cmd" / "git.exe")):
        try:
            head = subprocess.run([exe, "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30)
            if head.returncode != 0:
                continue
            st = subprocess.run([exe, "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True, timeout=60)
            return {"commit": head.stdout.strip(), "dirty": bool(st.stdout.strip())}
        except (OSError, subprocess.SubprocessError):
            continue
    return {"commit": None, "dirty": None}


def recorded_summary(gen_dir: pathlib.Path) -> dict:
    """生成の記録（generation.jsonl）の各本の試み: (種類, シード, 色, やり直し) → 所要時間・フレーム数・成否・配置の種類。"""
    out = {}
    for line in (gen_dir / "generation.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        for a in r["attempts"]:
            out[(a["kind"], a["layout_seed"], a["color"], a["retry"])] = {
                "duration_s": a["duration_s"], "n_frames": a["n_frames"], "success": a["success"],
                "layout_kind": r.get("layout_kind") or a["layout"].get("kind")}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="outputs/real/replay_v1")
    ap.add_argument("--q1-min", type=float, default=-2.80)
    ap.add_argument("--q1-max", type=float, default=-1.66)
    ap.add_argument("--gen-run", default=None, help=f"段階 3 の生成の記録（既定 {GEN_RUN}。この worktree になければ本線の場所）")
    ap.add_argument("--only", default=None, help="名前（例 n_20000_red_r0）をカンマで。既定は全部")
    ap.add_argument("--dataset", default=None, help=f"照合する学習データ（既定 {DATASET}。この worktree になければ本線の場所）")
    args = ap.parse_args(argv)

    from recovla.common import code_version
    from recovla.expert import generate as G

    out = pathlib.Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    if out.exists() and any(out.iterdir()):
        print(f"書き出し先 {out} が空でない。別の場所を指定する")
        return 2
    out.mkdir(parents=True, exist_ok=True)
    gen_dir = pathlib.Path(args.gen_run) if args.gen_run else ROOT / GEN_RUN
    if not gen_dir.is_dir():
        gen_dir = ROOT.parent / "recovery_vla" / GEN_RUN
    rec = recorded_summary(gen_dir)
    ds_dir = pathlib.Path(args.dataset) if args.dataset else ROOT / DATASET
    if not ds_dir.is_dir():
        ds_dir = ROOT.parent / "recovery_vla" / DATASET
    dataset = Dataset(ds_dir)
    trials = TRIALS
    if args.only:
        keep = set(args.only.split(","))
        trials = [tr for tr in TRIALS if f"{tr[0]}_{tr[1]}_{tr[2]}_r{tr[3]}" in keep]

    G._init_worker(False, "v3")
    rig = G._RIG
    fk = FK(rig.motion)
    cfg_sha = {n: hashlib.sha256((ROOT / "configs" / f"{n}.yaml").read_bytes()).hexdigest()
               for n in ("default", "sensor_v1", "runtime_v2", "runtime_v2_color", "runtime_v2_derived",
                         "runtime_v2_judge", "expert_v3", "real_v1") if (ROOT / "configs" / f"{n}.yaml").is_file()}
    manifest = {"schema": "recovery_vla.real_replay/1",
                "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                "code_version": code_version.code_version(), "git_head": git_head(), "config_sha256": cfg_sha,
                "generation_reference": str(gen_dir.relative_to(gen_dir.parents[2])).replace("\\", "/"),
                "q1_shift_rad": Q1_SHIFT, "q1_range_rad": [args.q1_min, args.q1_max], "dt_s": DT,
                "ee_convention": "libfranka O_T_EE (Franka Hand: flange + 0.1034 m along hand z, rotation = hand)",
                "limits": {"max_vel": MAX_VEL.tolist(), "max_acc": MAX_ACC.tolist(), "max_jerk": MAX_JERK.tolist(),
                           "q_min": Q_MIN.tolist(), "q_max": Q_MAX.tolist(), "source": "libfranka 0.9.2 rate_limiting.h"},
                "dataset_reference": DATASET,
                "ds_view": {"fps": 10, "columns": DS_COLS, "frame": "base (real robot base = Rz(-130 deg) * sim world)",
                            "omitted": ["observation.images.image", "observation.images.image2", "cue_x", "cue_y"],
                            "note": "state/action follow data/convert.py episode_arrays and data/vla_state.py; "
                                    "the sim-frame values equal the R1v3 dataset rows (dataset_match per trial)"},
                "trials": []}
    try:
        for kind, seed, color, retry in trials:
            name = f"{kind}_{seed}_{color}_r{retry}"
            ref = rec.get((kind, seed, color, retry))
            entry = {"name": name, "kind": kind, "seed": seed, "color": color, "retry": retry,
                     "instruction": G.instruction(color)}
            if ref is None or not ref["success"]:
                entry.update({"exported": False, "reasons": ["生成の記録に成功した試みがない"]})
                manifest["trials"].append(entry)
                continue
            res = run_trial(rig, kind, seed, color, retry, ref["layout_kind"])
            s = res["summary"]
            same = (s["success"] and abs(s["duration_s"] - ref["duration_s"]) < 1e-6 and s["n_frames"] == ref["n_frames"])
            rows, cmds_r, ee_cmd = build_rows(res, fk)
            grows = gripper_rows(res)
            start_real = np.round(to_real(res["start_q"]), DECIMALS)
            widths = [float(r[2]) for r in grows]
            chk = check_stream(cmds_r, start_real, widths, (args.q1_min, args.q1_max))
            if not same:
                chk["ok"] = False
                chk["reasons"].append(f"生成の記録と一致しない（所要時間 {s['duration_s']} 対 {ref['duration_s']}、"
                                      f"こま {s['n_frames']} 対 {ref['n_frames']}、成功 {s['success']}）")
            if not res["converged"]:
                chk["ok"] = False
                chk["reasons"].append("台本の後、5 s 以内に止まりきらなかった")
            # 学習データとの照合: 行動（指令の変化・開閉）は一致、フレーム数・最初のフレームの状態も一致すること。2 フレーム目からの状態は
            # センサのモデルの雑音の引き方が描画の有無で変わる（生成は描画あり、ここは描画なし）ので、雑音の大きさ（2 mm・2 mrad）
            # 以内であることだけを見る。書き出す表は学習データの行そのもの（を台座の座標へ回したもの）
            sim_arr = ds_arrays(res["frames"], "sim")
            ds_match = dataset.match(name, sim_arr)
            rows_ds = ds_match.pop("rows", None)
            if not (ds_match.get("same_frames") and ds_match["action_max_abs_diff"] < 1e-6
                    and ds_match["state0_max_abs_diff"] < 1e-6 and ds_match["state15_max_abs_diff"] < 2e-3):
                chk["ok"] = False
                chk["reasons"].append(f"学習データの本と一致しない: {ds_match}")
                ds = ds_arrays(res["frames"], "base")
            else:
                ds = ds_from_dataset(rows_ds[0], rows_ds[1], sim_arr["t"])
                back = ds_arrays(res["frames"], "base")             # 回し方の確かめ: 同じ回し方をシミュレーションの値に掛けたもの
                ds_match["rotation_check_action_max_abs_diff"] = float(np.abs(back["action"] - ds["action"]).max())
            ee_z = ee_cmd[:, 2]
            entry.update({
                "layout_kind": ref["layout_kind"], "start_pose": s["layout"].get("start"),
                "start_q_real": start_real.tolist(), "start_ee_real": [round(float(v), 6) for v in fk(start_real)[0]],
                "same_as_generation": bool(same), "duration_s": s["duration_s"], "n_frames": s["n_frames"],
                "sim_t0": res["t0"],
                "t_record_start": s.get("t_record_start"), "stream_s": round(len(cmds_r) * DT, 3),
                "tail_s": round((len(cmds_r) - res["n_main"]) * DT, 3), "check": chk,
                "ee_cart_cmd": cart_ratios(ee_cmd), "ee_z_cmd_range_m": [round(float(ee_z.min()), 4), round(float(ee_z.max()), 4)],
                "gripper_events": len(grows), "sim_wall_s": res["wall_s"], "exported": bool(chk["ok"]),
                "dataset_match": ds_match, "ds_frames": int(len(ds["t"])),
                "ds_t_stream_range": [round(float(ds["t"][0]), 3), round(float(ds["t"][-1]), 3)]})
            if chk["ok"]:
                entry["ds_csv"] = f"{name}_ds.csv"
                entry["ds_csv_sha256"] = write_csv(out / f"{name}_ds.csv", ds_rows(ds), DS_COLS)
                entry["csv"] = f"{name}.csv"
                entry["csv_sha256"] = write_csv(out / f"{name}.csv", rows, COLS)
                entry["gripper_csv"] = f"{name}_gripper.csv"
                entry["gripper_csv_sha256"] = write_csv(out / f"{name}_gripper.csv", grows, GRIP_COLS)
            else:
                entry["reasons"] = chk["reasons"]
            manifest["trials"].append(entry)
            print(f"{name}: {'書き出し' if chk['ok'] else '除外'} 流す {entry['stream_s']} s、比 {chk['ratio_max_all']}、"
                  f"q1 {chk['q1_range_used']}" + ("" if chk["ok"] else f"、理由 {chk['reasons']}"), flush=True)
    finally:
        rig.close()
    manifest["n_exported"] = sum(1 for e in manifest["trials"] if e.get("exported"))
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"書き出し {manifest['n_exported']}/{len(manifest['trials'])} 本 → {out}")
    return 0 if manifest["n_exported"] == len(manifest["trials"]) else 1


if __name__ == "__main__":
    sys.exit(main())
