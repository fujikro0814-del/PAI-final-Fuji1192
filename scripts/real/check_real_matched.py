"""実機の Panda で測った応答に合わせたシミュレーション（v3_real、docs/real/v3_real.md）の確かめ。CPU だけ。方策の推論・学習はしない。

    python scripts/real/check_real_matched.py arm --logs C:/PAI/pandatest_data_20261011/logs --replay-root C:/PAI/wt_real/outputs/real \\
        --out outputs/real/v3_real/arm_check.json [--runs normal/n_20000_red_r0,...]
    python scripts/real/check_real_matched.py width --seed 58705 --out outputs/real/v3_real/width_check.json

arm:   実機の試験 A8 で再生した生成プログラムの指令（scripts/real/fit_servo.py の Case・score を使う）を、合わせる前の生成の
       世界（v3）と v3_real の生成の世界（腕は harness/real_matched.py が入れたサーボ）に開ループで流し、手先の遅延（指令に対する）と
       実機との差を比べる。
width: 生成プログラムの通常のデータを 1 本記録して（描画なし）、記録の状態の指の幅（data.npz の fingers）の更新の間隔・遅延・
       真値との差を、v3 と v3_real で比べる。真値は物理の 1 ステップごとに控える。
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


def _rig(world: str):
    from recovla.expert import generate as G
    from recovla.harness import real_matched as V
    if world == "v3":
        G._init_worker(False, "v3")
        return G._RIG
    return V.V3RealGenRig(V.load_config(world, generation=True))


# ------------------------------------------------------------------------------ 腕
def _simulate(rig, case, place: bool):
    import export_replay as X
    import export_replay_slow as S
    from recovla.harness import arm_real
    q_first = S.to_sim(case.q_cmd_real[0])
    old = rig.__dict__.get("reset")
    if place:                                        # 再生は制限層を通さないので、流す最初の指令に腕を置く
        inner = rig.reset

        def reset(layout):
            inner(layout)
            arm_real.place_arm(rig, q_first)
        rig.reset = reset
    try:
        o = S.open_loop(rig, case.entry["seed"], case.entry["layout_kind"], S.to_sim(case.q_cmd_real), case.events, False)
    finally:
        if place:
            if old is None:
                rig.__dict__.pop("reset", None)
            else:
                rig.reset = old
    return X.to_real(np.array([m[1] for m in o["meas"]]))


def cmd_arm(a) -> int:
    import export_replay as X
    import fit_servo as F
    root = pathlib.Path(a.replay_root)
    F.REPLAY = {"normal": root / "replay_v1", "slow": root / "replay_v1_slow"}      # 読むだけ
    runs = F.find_runs(pathlib.Path(a.logs))
    if a.runs:
        want = set(a.runs.split(","))
        runs = [r for r in runs if f"{r['variant']}/{r['name']}" in want]
    manifests = {v: json.loads((F.REPLAY[v] / "manifest.json").read_text(encoding="utf-8")) for v in F.REPLAY}
    cases = [F.Case(r, manifests) for r in runs]
    res = {"logs": str(a.logs), "replay_root": str(root), "runs": [f"{c.variant}/{c.name}" for c in cases], "worlds": {}}
    for world in ("v3", "v3_real"):
        rig = _rig(world)
        t0 = time.perf_counter()
        try:
            fk = X.FK(rig.motion)
            per = {}
            for c in cases:
                q_sim = _simulate(rig, c, place=(world == "v3_real"))
                per[f"{c.variant}/{c.name}"] = F.score(c, q_sim, fk)
        finally:
            rig.close()
        lag_s = [v["lag_sim_s"] for v in per.values()]
        lag_r = [v["lag_real_s"] for v in per.values()]
        sse = sum(v["sse"] for v in per.values())
        n = sum(v["n"] for v in per.values())
        res["worlds"][world] = {
            "lag_sim_ms": {"median": round(1e3 * float(np.median(lag_s)), 2), "min": round(1e3 * min(lag_s), 2),
                           "max": round(1e3 * max(lag_s), 2)},
            "lag_real_ms": {"median": round(1e3 * float(np.median(lag_r)), 2), "min": round(1e3 * min(lag_r), 2),
                            "max": round(1e3 * max(lag_r), 2)},
            "lag_diff_ms_max": round(1e3 * max(v["lag_diff_s"] for v in per.values()), 2),
            "joint_rms_mrad": round(1e3 * float(np.sqrt(sse / n)), 3),
            "ee_median_mm": round(1e3 * float(np.median([v["ee_median_m"] for v in per.values()])), 3),
            "ee_max_mm": round(1e3 * max(v["ee_max_m"] for v in per.values()), 3),
            "lag_pass_10ms": sum(v["lag_diff_s"] <= 0.010 for v in per.values()), "n": len(per),
            "wall_s": round(time.perf_counter() - t0, 1), "runs": per}
        w = res["worlds"][world]
        print(f"{world}: 手先の遅れ（指令に対する）中央値 {w['lag_sim_ms']['median']} ms（{w['lag_sim_ms']['min']}〜"
              f"{w['lag_sim_ms']['max']}）、実機 {w['lag_real_ms']['median']} ms、遅れの差 ≤ 10 ms {w['lag_pass_10ms']}/{w['n']}、"
              f"関節 RMS {w['joint_rms_mrad']} mrad、手先の差 中央値 {w['ee_median_mm']} mm・最大 {w['ee_max_mm']} mm", flush=True)
    out = pathlib.Path(a.out) if pathlib.Path(a.out).is_absolute() else ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {out}")
    return 0


# ------------------------------------------------------------------------------ 幅
def _episode(world: str, seed: int, run_dir: pathlib.Path) -> dict:
    """通常のデータを 1 本記録し、物理の 1 ステップごとの真の幅と、記録の状態の幅を返す。"""
    from recovla.expert import generate as G
    from recovla.expert import generate_real as GR
    from recovla.sim import scene
    rig = _rig(world)
    true_log = []
    try:
        inner = rig.physics_step

        def physics_step(on_step=None):
            inner(on_step)
            true_log.append((float(rig.data.time), float(rig.data.qpos[rig.finger_qadr].sum())))
        rig.physics_step = physics_step
        lay = scene.sample_layout(seed)
        color = lay.table_colors[seed % len(lay.table_colors)]
        run = GR.run_attempt if world != "v3" else G.run_attempt
        r = run(rig, G.EpisodeSpec(seed, color, kind="n"), 0, run_dir / world, False)
        samples = list(getattr(rig.suite, "samples", []))
    finally:
        rig.close()
    data = np.load(pathlib.Path(r["path"]) / "data.npz")
    return {"summary": r, "true": np.array(true_log), "t": np.asarray(data["sim_time"], float),
            "w": np.asarray(data["fingers"], float).sum(axis=1), "samples": samples}


def _width_stats(e: dict) -> dict:
    t, w, true = e["t"], e["w"], e["true"]
    ch = np.where(np.abs(np.diff(w)) > 1e-12)[0] + 1                    # 記録の幅が変わったフレーム
    moving = np.abs(np.gradient(np.interp(t, true[:, 0], true[:, 1]), t)) > 0.01          # 指が動いている間のフレーム
    errs = {}
    for tau in np.arange(0.0, 0.301, 0.002):
        ref = np.interp(t - tau, true[:, 0], true[:, 1])
        errs[round(float(tau), 3)] = float(np.mean(np.abs(w[moving] - ref[moving]))) if moving.any() else None
    best = min((k for k in errs if errs[k] is not None), key=lambda k: errs[k]) if moving.any() else None
    now = np.interp(t, true[:, 0], true[:, 1])
    out = {"frames": int(len(t)), "record_hz": round(1.0 / float(np.median(np.diff(t))), 2),
           "state_changes": int(len(ch)), "duration_s": round(float(t[-1] - t[0]), 2),
           "state_change_rate_hz": round(len(ch) / float(t[-1] - t[0]), 2),
           "max_abs_diff_to_true_now_mm": round(1e3 * float(np.abs(w - now).max()), 3),
           "max_abs_diff_to_true_now_still_mm": round(1e3 * float(np.abs(w - now)[~moving].max()), 3),
           "mean_abs_diff_while_moving_mm": round(1e3 * float(np.abs(w - now)[moving].mean()), 3) if moving.any() else None,
           "effective_lag_s_while_moving": best, "moving_frames": int(moving.sum())}
    s = e["samples"]
    if s:
        ta = np.array([x[0] for x in s])
        src = np.array([x[2] for x in s])
        ws = np.array([x[1] for x in s])
        d = np.diff(ta)
        ref = np.interp(src, true[:, 0], true[:, 1]) if len(true) else ws
        k = ta > 0                                                   # 時刻 0 の値（prime）は止まった場面
        out["sensor"] = {"samples": int(len(s)), "interval_ms": {"median": round(1e3 * float(np.median(d)), 2),
                                                                 "min": round(1e3 * float(d.min()), 2), "max": round(1e3 * float(d.max()), 2)},
                         "rate_hz": round(1.0 / float(np.median(d)), 3),
                         "delay_ms": {"median": round(1e3 * float(np.median((ta - src)[k])), 2),
                                      "max": round(1e3 * float((ta - src)[k].max()), 2)},
                         "reported_minus_true_at_source_max_mm": round(1e3 * float(np.abs(ws - ref)[k].max()), 5)}
        # 記録の各フレームの幅 = そのフレームの時刻までに届いた最新の値
        idx = np.searchsorted(ta, t + 1e-9, side="right") - 1
        out["state_equals_latest_sample"] = bool(np.array_equal(w, ws[idx]))
    return out


def cmd_width(a) -> int:
    out = pathlib.Path(a.out) if pathlib.Path(a.out).is_absolute() else ROOT / a.out
    run_dir = out.parent / f"width_check_gen_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True)
    res = {"seed": a.seed, "run_dir": str(run_dir), "worlds": {}}
    for world in ("v3", "v3_real"):
        e = _episode(world, a.seed, run_dir)
        st = _width_stats(e)
        st["episode"] = {k: e["summary"][k] for k in ("name", "success", "duration_s", "n_frames", "phase_first_t")}
        res["worlds"][world] = st
        print(f"{world}: 記録の幅が変わる頻度 {st['state_change_rate_hz']} Hz（こま {st['record_hz']} Hz）、"
              f"動いている間の実効の遅れ {st['effective_lag_s_while_moving']} s、止まっている間の真値との差の最大 "
              f"{st['max_abs_diff_to_true_now_still_mm']} mm" + (f"、センサ {st['sensor']}" if "sensor" in st else ""), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"→ {out}")
    return 0


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:                                # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("arm")
    s.add_argument("--logs", required=True)
    s.add_argument("--replay-root", required=True)
    s.add_argument("--runs", default=None)
    s.add_argument("--out", required=True)
    s = sub.add_parser("width")
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"arm": cmd_arm, "width": cmd_width}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
