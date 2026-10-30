"""腕のサーボを実機の追従に合わせる（試験 A8 で生成プログラムの指令を再生した記録から）: シミュレーションに実機と同じ関節の
指令を開ループで流し、測った関節角を実機の記録（arm.csv の q）と同じ時刻で比べ、ずれが最小になる kp・kv・指令の遅延を探す。

    python scripts/real/fit_servo.py --logs <実機の logs> --fit n_20000_red_r0,... --grid --out outputs/real/fit_servo/fit.json
    python scripts/real/fit_servo.py --logs <実機の logs> --runs all --params '{"scale":20,"ratio":0.005,"delay":1}' --out ...

- 世界は合わせる前の生成と同じ（expert/generate.py の rig "v3"）。口は export_replay_slow.open_loop（WorldRig.apply_joint_commands →
  physics_step）。立方体は無し（実機と同じ）。グリッパーの指令は記録の時刻に出す。
- 変えるのは harness/arm_real.py の ArmServo だけ: kp = scale × 合わせる前の kp（4500・4500・3500・3500・2000・2000・2000）、
  kv = ratio × kp（合わせる前は 0.1 s）、vel_ff = ff（指令の速さの先送り）、cmd_delay_steps = delay（物理の 1 ステップ = 2 ms）。
  scale は 5 を超えると物理の 1 ステップ 2 ms で発散したので、探す範囲は 1〜4。
- 比べる値: 関節ごとのずれの RMS・最大 [rad]、手先（O_T_EE の決まり）の位置の差の中央値・最大 [m]、実機とシミュレーションの
  手先の遅延（指令に対する）。目的の値は、関節のずれの二乗平均（7 関節・全本）。
- CPU だけ。生成・学習はしない。
"""
import argparse
import json
import pathlib
import re
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import export_replay as X  # noqa: E402
import export_replay_slow as S  # noqa: E402
from compare_replay import estimate_lag  # noqa: E402
from real_to_ds import read_csv  # noqa: E402

KP0 = np.array([4500.0, 4500.0, 3500.0, 3500.0, 2000.0, 2000.0, 2000.0])
REPLAY = {"normal": ROOT / "outputs/real/replay_v1", "slow": ROOT / "outputs/real/replay_v1_slow"}
RUN_RE = re.compile(r"a8_(slow|normal)_(.+_r\d)_(\d{8}_\d{6})$")


def servo_of(p: dict):
    from recovla.harness.arm_real import ArmServo
    if p.get("base"):
        return ArmServo()
    kp = KP0 * float(p.get("scale", 1.0))
    return ArmServo(tuple(kp), tuple(kp * float(p.get("ratio", 0.1))), float(p.get("ff", 0.0)), int(p.get("delay", 0)),
                    bool(p.get("gc", False)))


def find_runs(logs: pathlib.Path, skip=("183447",)) -> list:
    """実機の記録のフォルダ（a8_<slow|normal>_<名前>_<時刻>）。同じ名前が複数あれば最後のもの。止まった初回は除く。"""
    out = {}
    for d in sorted(logs.iterdir()):
        m = RUN_RE.match(d.name)
        if m and (d / "arm.csv").is_file() and not any(s in d.name for s in skip):
            out[(m.group(1), m.group(2))] = d
    return [{"variant": v, "name": n, "dir": d} for (v, n), d in sorted(out.items())]


class Case:
    """1 本: 指令（シミュレーションの関節角）、グリッパーの指令、実機の記録（物理の 1 ステップの時刻に補間）。"""

    def __init__(self, run: dict, manifests: dict):
        v, name = run["variant"], run["name"]
        man = manifests[v]
        self.entry = next(t for t in man["trials"] if t["name"] == name)
        rp = read_csv(REPLAY[v] / f"{name}.csv")
        self.q_cmd_real = np.stack([np.asarray(rp[f"q_cmd{j}"], float) for j in range(7)], axis=1)
        self.ee_cmd = np.stack([np.asarray(rp[f"ee_cmd_{c}"], float) for c in "xyz"], axis=1)
        self.t_rp = np.asarray(rp["t"], float)
        self.events = S.events_from_csv(REPLAY[v] / f"{name}_gripper.csv", 1.0)
        a = read_csv(run["dir"] / "arm.csv")
        t = np.asarray(a["t"], float)
        n = len(self.q_cmd_real) // 2
        self.t_meas = (np.arange(n) + 1) * 2 * X.DT
        keep = (self.t_meas >= t[0]) & (self.t_meas <= t[-1])
        self.keep = keep
        self.q_real = np.stack([np.interp(self.t_meas[keep], t, np.asarray(a[f"q{j}"], float)) for j in range(7)], axis=1)
        self.ee_real = np.stack([np.interp(self.t_meas[keep], t, np.asarray(a[f"O_T_EE{i}"], float)) for i in (12, 13, 14)], axis=1)
        self.ev_t = [float(e[0]) for e in self.events if e[1] == "grasp" or float(e[2][0]) >= 0.07]
        self.run, self.variant, self.name = run, v, name


def simulate(rig, case: Case, servo) -> np.ndarray:
    from recovla.harness import arm_real
    saved = arm_real.install(rig, servo)
    if servo.gravity_comp:
        # 再生は制限層を通さず指令をそのまま流すので、置き直す先は流す最初の指令（実機が止まっていた開始姿勢）
        q_first = S.to_sim(case.q_cmd_real[0])
        reset_gc = rig.reset

        def reset(layout):
            reset_gc(layout)
            arm_real.place_arm(rig, q_first)
        rig.reset = reset
    try:
        o = S.open_loop(rig, case.entry["seed"], case.entry["layout_kind"], S.to_sim(case.q_cmd_real), case.events, False)
    finally:
        arm_real.restore(rig, saved)
    return X.to_real(np.array([m[1] for m in o["meas"]])), np.array([m[3] for m in o["meas"]])


def score(case: Case, q_sim, fk) -> dict:
    qs = q_sim[case.keep]
    e = qs - case.q_real
    ee_s = np.array([fk(q)[0] for q in qs])
    d = np.linalg.norm(ee_s - case.ee_real, axis=1)
    t = case.t_meas[case.keep]
    lag_s = estimate_lag(t, ee_s, case.t_rp, case.ee_cmd)
    lag_r = estimate_lag(t, case.ee_real, case.t_rp, case.ee_cmd)
    d_ev = [float(np.interp(x, t, d)) for x in case.ev_t if t[0] <= x <= t[-1]]
    return {"joint_rms_rad": [float(v) for v in np.sqrt(np.mean(e ** 2, axis=0))],
            "joint_max_rad": [float(v) for v in np.abs(e).max(axis=0)],
            "sse": float(np.sum(e ** 2)), "n": int(e.size),
            "ee_median_m": float(np.median(d)), "ee_max_m": float(d.max()), "ee_events_max_m": max(d_ev) if d_ev else None,
            "lag_real_s": lag_r[0], "lag_sim_s": lag_s[0], "lag_diff_s": abs(lag_r[0] - lag_s[0])}


def evaluate(rig, fk, cases, p) -> dict:
    servo = servo_of(p)
    per = {}
    sse = n = 0
    for c in cases:
        q_sim, _ = simulate(rig, c, servo)
        s = score(c, q_sim, fk)
        per[f"{c.variant}/{c.name}"] = s
        sse += s["sse"]
        n += s["n"]
    return {"params": p, "rms_all_rad": float(np.sqrt(sse / n)), "runs": per}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--logs", required=True)
    ap.add_argument("--runs", default="all", help="all か、variant/名前 をカンマで（例 normal/n_20000_red_r0）")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--params", default=None, help="JSON の 1 組か、その配列（@ファイル名 でファイルから）")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from recovla.expert import generate as G
    runs = find_runs(pathlib.Path(a.logs))
    if a.runs != "all":
        want = set(a.runs.split(","))
        runs = [r for r in runs if f"{r['variant']}/{r['name']}" in want]
    manifests = {v: json.loads((REPLAY[v] / "manifest.json").read_text(encoding="utf-8")) for v in REPLAY}
    cases = [Case(r, manifests) for r in runs]
    if a.grid:
        grid = [{"base": True}, {"gc": True}] + [{"gc": True, "scale": s, "ratio": r, "ff": f, "delay": d} for s in (1, 2, 4)
                                                 for r in (0.05, 0.1) for f in (0.9, 1.0) for d in (0, 1, 2, 3, 4)]
    else:
        p = json.loads(pathlib.Path(a.params[1:]).read_text(encoding="utf-8") if a.params.startswith("@") else a.params)
        grid = p if isinstance(p, list) else [p]
    G._init_worker(False, "v3")
    rig = G._RIG
    fk = X.FK(rig.motion)
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res = {"runs": [f"{c.variant}/{c.name}" for c in cases], "logs": str(a.logs), "results": []}
    try:
        for p in grid:
            t0 = time.perf_counter()
            r = evaluate(rig, fk, cases, p)
            res["results"].append(r)
            med = np.median([v["ee_median_m"] for v in r["runs"].values()])
            mx = max(v["ee_max_m"] for v in r["runs"].values())
            lag = np.median([v["lag_sim_s"] for v in r["runs"].values()])
            print(f"{json.dumps(p)}: 関節 RMS {r['rms_all_rad'] * 1e3:.3f} mrad、手先 中央値 {med * 1e3:.2f}・最大 {mx * 1e3:.2f} mm、"
                  f"遅れ {lag * 1e3:.1f} ms（{time.perf_counter() - t0:.0f} s）", flush=True)
            out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    finally:
        rig.close()
    best = min((r for r in res["results"] if not r["params"].get("base")), key=lambda r: r["rms_all_rad"], default=None)
    res["best"] = best["params"] if best else None
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("最良:", res["best"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
