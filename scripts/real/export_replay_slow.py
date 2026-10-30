"""生成プログラムの指令の再生の、実機での初回用の遅い版（replay_v1_slow）: 書き出した 1 kHz の関節の指令を速さ 0.5 倍
（時間 2 倍）に引き延ばし、同じ指令をシミュレーションの腕のサーボにそのまま流した応答を付ける。

    python scripts/real/export_replay_slow.py                 # outputs/real/replay_v1 → outputs/real/replay_v1_slow

なぜ: 通常の速さの列は関節の加速度・躍度が上限の 0.99 倍に届く。実機の通信は約 0.16% 抜けるので、上限の近くで 1 つ抜けると
不連続の反射で止まりうる。0.5 倍なら加速度は 0.25 倍、躍度は 0.125 倍。

作り方:
1. 引き延ばし: 各本の q_cmd（1 kHz）を、開始姿勢（t = 0 で止まっている）を前に足して 3 次のスプライン（両端の速さ 0）で
   つなぎ、時刻 t/2 で読む（行 i は (i+1)·1 ms、読む元の時刻は (i+1)·0.5 ms）。行数は 2 倍。グリッパーの指令の時刻も 2 倍。
   速さ・力などの値は変えない（指の動く速さは実機のまま）。
2. シミュレーションの応答: 生成プログラムは回さず（判断し直さない）、同じ配置・同じ開始姿勢から、指令を物理の 1 ステップ（500 Hz）ごとに
   2 つずつ、生成と同じ口（WorldRig.apply_joint_commands → physics_step の位置のサーボ）に流す。グリッパーの指令は記録の時刻に
   ハンドの口へ出す。**立方体は無し**（実機と同じ。立方体の接触を切り、重力を打ち消してその場に浮かせる）。
3. この口が元の応答を再現することを、1.0 倍・立方体ありで確かめる（元の書き出しの q_sim・手先・指の幅との最大の差。manifest の
   reproduction）。参考に 1.0 倍・立方体なしの差も出す。
4. 列は通常版と同じ（docs/real/replay_format.md）。学習データの列の表（_ds.csv）は作らない（引き延ばした列に対応する学習データの
   行がない）。
"""
import argparse
import datetime
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import export_replay as X  # noqa: E402
from real_to_ds import read_csv  # noqa: E402

SCALE = 2.0                                       # 時間の倍率（速さ 0.5 倍）


def stretch(q_real, start_real, scale: float = SCALE) -> np.ndarray:
    """1 kHz の指令の列を時間 scale 倍に引き延ばす（3 次のスプライン、両端の速さ 0）。返り値の行 i は (i+1)·1 ms。"""
    from scipy.interpolate import CubicSpline
    q = np.asarray(q_real, float)
    n = len(q)
    t = np.concatenate([[0.0], (np.arange(n) + 1) * X.DT])
    y = np.vstack([np.asarray(start_real, float)[None], q])
    cs = CubicSpline(t, y, axis=0, bc_type="clamped")
    m = int(round(n * scale))
    ts = (np.arange(m) + 1) * X.DT / scale
    return np.round(cs(ts), X.DECIMALS)


def to_sim(q_real) -> np.ndarray:
    q = np.array(q_real, float, copy=True)
    q[..., 0] -= X.Q1_SHIFT
    return q


def remove_cubes(rig) -> None:
    """立方体を無しにする: 接触を切り、重力を打ち消してその場に浮かせる（指は空を掴む＝実機と同じ）。"""
    m = rig.model
    for c in ("red", "green", "blue"):
        g = m.geom(f"cube_{c}_geom").id
        m.geom_contype[g] = 0
        m.geom_conaffinity[g] = 0
        m.body_gravcomp[m.body(f"cube_{c}").id] = 1.0


def restore_cubes(rig, saved) -> None:
    m = rig.model
    for c, (ct, ca, gc) in saved.items():
        g = m.geom(f"cube_{c}_geom").id
        m.geom_contype[g], m.geom_conaffinity[g] = ct, ca
        m.body_gravcomp[m.body(f"cube_{c}").id] = gc


def open_loop(rig, seed, layout_kind, cmds_sim, events, cubes: bool) -> dict:
    """同じ配置・開始姿勢から、指令（シミュレーションの関節角、1 kHz）を 500 Hz の 1 ステップに 2 つずつ流す。events は
    (時刻, 口, 引数) で、その時刻の物理の 1 ステップの前にハンドへ出す（生成の pad_read と同じ順）。"""
    from recovla.sim import scene
    m = rig.model
    saved = {c: (int(m.geom_contype[m.geom(f"cube_{c}_geom").id]), int(m.geom_conaffinity[m.geom(f"cube_{c}_geom").id]),
                 float(m.body_gravcomp[m.body(f"cube_{c}").id])) for c in ("red", "green", "blue")}
    try:
        if not cubes:
            remove_cubes(rig)
        rig.reset(scene.sample_layout(int(seed), layout_kind))
        start_q = rig.motion.limiter.q.copy()
        t0 = float(rig.data.time)
        ev = sorted(events, key=lambda e: e[0])
        k = 0
        meas = []
        for n in range(len(cmds_sim) // 2):
            while k < len(ev) and ev[k][0] <= float(rig.data.time) - t0 + 1e-9:
                _, kind, args = ev[k]
                (rig.hand.move if kind == "move" else rig.hand.grasp)(*args)
                k += 1
            rig.apply_joint_commands([cmds_sim[2 * n], cmds_sim[2 * n + 1]])
            rig.physics_step(None)
            d = rig.data
            meas.append((float(d.time), d.qpos[rig.arm_qadr].copy(), d.qvel[rig.arm_vadr].copy(), rig.hand.width()))
        hand_log = list(rig.hand.log)
    finally:
        restore_cubes(rig, saved)
    return {"meas": meas, "start_q": start_q, "t0": t0, "hand_log": hand_log, "events_sent": k}


def events_from_csv(path, scale: float = 1.0) -> list:
    g = read_csv(path)
    out = []
    for i in range(len(g["t"])):
        t = float(g["t"][i]) * scale
        if str(g["kind"][i]) == "move":
            out.append((t, "move", (float(g["width"][i]), float(g["speed"][i]))))
        else:
            out.append((t, "grasp", tuple(float(g[c][i]) for c in ("width", "speed", "force", "eps_inner", "eps_outer"))))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default="outputs/real/replay_v1")
    ap.add_argument("--out", default="outputs/real/replay_v1_slow")
    ap.add_argument("--q1-min", type=float, default=-2.80)
    ap.add_argument("--q1-max", type=float, default=-1.66)
    ap.add_argument("--only", default=None)
    a = ap.parse_args(argv)
    from recovla.common import code_version
    from recovla.expert import generate as G

    src = ROOT / a.src
    out = ROOT / a.out
    if out.exists() and any(out.iterdir()):
        print(f"書き出し先 {out} が空でない")
        return 2
    out.mkdir(parents=True, exist_ok=True)
    man = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    G._init_worker(False, "v3")
    rig = G._RIG
    fk = X.FK(rig.motion)
    res_man = {"schema": "recovery_vla.real_replay/1", "variant": "slow_0.5x", "time_scale": SCALE,
               "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
               "code_version": code_version.code_version(), "git_head": X.git_head(),
               "source": {"dir": a.src, "manifest_git_head": man.get("git_head")},
               "q1_shift_rad": X.Q1_SHIFT, "q1_range_rad": [a.q1_min, a.q1_max], "dt_s": X.DT,
               "ee_convention": man["ee_convention"], "limits": man["limits"],
               "sim_response": "open loop: same layout and start pose, the scaled q_cmd fed 2 per 500 Hz physics step into "
                               "WorldRig.apply_joint_commands/physics_step (the generation's servo path); gripper commands at "
                               "the scaled times; no cubes (contacts off, gravity compensated); no scripted re-decision",
               "ds_view": None, "trials": []}
    try:
        for t in man["trials"]:
            if not t.get("exported") or (a.only and t["name"] not in a.only.split(",")):
                continue
            name = t["name"]
            r = read_csv(src / t["csv"])
            q_cmd = np.stack([r[f"q_cmd{j}"] for j in range(7)], axis=1)
            start_real = np.asarray(t["start_q_real"], float)
            ev1 = events_from_csv(src / t["gripper_csv"], 1.0)
            # 1.0 倍の再現の確かめ（立方体あり＝元と同じ）と、参考の立方体なし
            rep = {}
            for cubes in (True, False):
                o = open_loop(rig, t["seed"], t["layout_kind"], to_sim(q_cmd), ev1, cubes)
                q_o = X.to_real(np.array([m_[1] for m_ in o["meas"]]))
                w_o = np.array([m_[3] for m_ in o["meas"]])
                idx = np.arange(1, len(q_cmd), 2)                  # 物理の 1 ステップの後＝奇数の行の時刻
                q_ref = np.stack([r[f"q_sim{j}"] for j in range(7)], axis=1)[idx]
                e_o = np.array([fk(q)[0] for q in q_o])
                e_ref = np.stack([r[f"ee_sim_{c}"] for c in "xyz"], axis=1)[idx]
                rep["with_cubes" if cubes else "no_cubes"] = {
                    "q_max_abs_diff_rad": float(np.abs(q_o - q_ref).max()),
                    "ee_max_abs_diff_m": float(np.linalg.norm(e_o - e_ref, axis=1).max()),
                    "width_max_abs_diff_m": float(np.abs(w_o - r["width_sim"][idx]).max()),
                    "start_q_same": bool(np.allclose(X.to_real(o["start_q"]), start_real, atol=1e-9))}
            # 遅い版
            q_slow = stretch(q_cmd, start_real, SCALE)
            ev2 = events_from_csv(src / t["gripper_csv"], SCALE)
            o = open_loop(rig, t["seed"], t["layout_kind"], to_sim(q_slow), ev2, False)
            n_main = int(round((len(q_cmd) - round(t["tail_s"] / X.DT)) * SCALE))
            t_rec = t.get("t_record_start")
            res = {"cmds": to_sim(q_slow), "meas": o["meas"], "n_main": n_main, "t0": o["t0"],
                   "summary": {"kind": t["kind"], "t_record_start": None if t_rec is None else t_rec * SCALE}}
            rows, cmds_r, ee_cmd = X.build_rows(res, fk)
            grows = [[f"{e[0]:.3f}", e[1]] + ([X.fmt(e[2][0]), X.fmt(e[2][1]), "", "", ""] if e[1] == "move"
                                                 else [X.fmt(v) for v in e[2]]) for e in ev2]
            chk = X.check_stream(cmds_r, start_real, [float(g[2]) for g in grows], (a.q1_min, a.q1_max))
            entry = {k: t[k] for k in ("name", "kind", "seed", "color", "retry", "instruction", "layout_kind", "start_pose",
                                       "start_q_real", "start_ee_real")}
            entry.update({"source_csv_sha256": t["csv_sha256"], "reproduction_1x": rep, "check": chk,
                          "stream_s": round(len(cmds_r) * X.DT, 3), "tail_s": round((len(cmds_r) - n_main) * X.DT, 3),
                          "t_record_start": None if t_rec is None else t_rec * SCALE,
                          "ee_cart_cmd": X.cart_ratios(ee_cmd),
                          "ee_z_cmd_range_m": [round(float(ee_cmd[:, 2].min()), 4), round(float(ee_cmd[:, 2].max()), 4)],
                          "gripper_events": len(grows), "gripper_events_sent": o["events_sent"],
                          "exported": bool(chk["ok"])})
            if chk["ok"]:
                entry["csv"], entry["gripper_csv"] = f"{name}.csv", f"{name}_gripper.csv"
                entry["csv_sha256"] = X.write_csv(out / entry["csv"], rows, X.COLS)
                entry["gripper_csv_sha256"] = X.write_csv(out / entry["gripper_csv"], grows, X.GRIP_COLS)
            else:
                entry["reasons"] = chk["reasons"]
            res_man["trials"].append(entry)
            print(f"{name}: {'書き出し' if chk['ok'] else '除外'} 流す {entry['stream_s']} s、比 {chk['ratio_max_all']}、"
                  f"再現 q {rep['with_cubes']['q_max_abs_diff_rad']:.2e} rad・手先 {rep['with_cubes']['ee_max_abs_diff_m']:.2e} m"
                  f"（立方体なし {rep['no_cubes']['ee_max_abs_diff_m']:.2e} m）", flush=True)
    finally:
        rig.close()
    res_man["n_exported"] = sum(1 for e in res_man["trials"] if e.get("exported"))
    (out / "manifest.json").write_text(json.dumps(res_man, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"書き出し {res_man['n_exported']}/{len(res_man['trials'])} 本 → {out}")
    return 0 if res_man["n_exported"] == len(res_man["trials"]) else 1


if __name__ == "__main__":
    sys.exit(main())
