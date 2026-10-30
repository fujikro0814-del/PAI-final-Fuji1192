"""実機の Panda で測った応答に合わせたシミュレーション（v3_real、docs/real/v3_real.md）で学習データを生成する。CPU だけ
（描画は生成と同じ）。学習・方策の推論はしない。

    python scripts/real/gen_real_matched.py gen --world v3_real --out outputs/gen/V3R_x --spec n:58710 A:58711 C:58712 [--workers 6] [--no-render]
    python scripts/real/gen_real_matched.py smoke --world v3_real --seed 58710 [--b-slip --b-seed 58815]   # n・A・C（B 滑り）を 1 本ずつ → 変換 → 照合
    python scripts/real/gen_real_matched.py slip-rate --seeds 58720:40 [--switch arm_servo=off ...] --out outputs/real/v3_real/slip_rate.json
    python scripts/real/gen_real_matched.py torque-rate --spec n:58900 A:58920 ... --out outputs/real/contact_std/tau_on.json   # 腕の力の変化

--switch 名前=on|off で入り・切りを上書きする（configs/v3_real.yaml の v3_real.switches。例 --switch b_slip=on）。
--spec は <種類>:<配置のシード>[:<色>]（色を省くと机上の色からシードの番号で 1 つ）。生成の本体は src/recovla/expert/generate_real.py
（＝ expert/generate.py の generate を v3_real の世界で回す）。シードは学習用の範囲（100000 未満）から選ぶ。
変換と照合は recovla.data.convert（引数 --target-cue --cue-thresholds runtime_v2 --cue-without-flag）。
"""
import argparse
import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from recovla.common import config  # noqa: E402

CUE_ARGS = ["--target-cue", "--cue-thresholds", "runtime_v2", "--cue-without-flag"]   # 合わせる前の学習データ（outputs/f/data_v3.json）と同じ


def parse_switches(items) -> dict:
    out = {}
    for s in items or ():
        k, _, v = s.partition("=")
        if v not in ("on", "off"):
            raise SystemExit(f"--switch {s}: 名前=on|off")
        out[k] = v == "on"
    return out


def parse_spec(s: str):
    from recovla.expert import generate as G
    from recovla.sim import scene
    parts = s.split(":")
    kind, seed = parts[0], int(parts[1])
    lay = scene.sample_layout(seed)
    color = parts[2] if len(parts) > 2 else lay.table_colors[seed % len(lay.table_colors)]
    return G.EpisodeSpec(seed, color, kind=kind)       # 配置の種類は配置の乱数列から決まる


def convert_and_verify(ep_dirs: list, out_ds: pathlib.Path, name: str, log: pathlib.Path) -> dict:
    t0 = time.perf_counter()
    with open(log, "w", encoding="utf-8") as f:
        c1 = subprocess.run([sys.executable, "-m", "recovla.data.convert", *[str(p) for p in ep_dirs], "--out", str(out_ds),
                             "--name", name, *CUE_ARGS], stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
        c2 = subprocess.run([sys.executable, "-m", "recovla.data.convert", "--verify", str(out_ds)],
                            stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
    text = log.read_text(encoding="utf-8", errors="replace")
    return {"convert_exit": c1.returncode, "verify_exit": c2.returncode, "verify_pass": "[verify] PASS" in text,
            "verify_fails": [ln.strip() for ln in text.splitlines() if ln.strip().startswith("FAIL")],
            "wall_s": round(time.perf_counter() - t0, 1), "log": str(log)}


def cmd_gen(a) -> int:
    from recovla.expert import generate_real as GR
    specs = [parse_spec(s) for s in a.spec]
    res = GR.generate(specs, config.path(a.out), world=a.world, workers=a.workers, render=not a.no_render,
                      switches=parse_switches(a.switch) or None)
    for r in res:
        print(f"{r['kind']} {r['layout_seed']} {r['color']}: 成功 {r['success']}（作り直し {r['retries']}、"
              f"{r['worker_wall_s']} s）", flush=True)
    return 0 if all(r["success"] for r in res) else 1


def cmd_smoke(a) -> int:
    """n・A・C（--b-slip なら B の滑りも）をシード seed から 1 つずつ、描画ありで作り、変換と照合を通す。"""
    from recovla.expert import generate_real as GR
    from recovla.expert import inject_real
    from recovla.harness import real_matched as V
    import numpy as np
    kinds = ["n", "A", "C"] + (["B"] if a.b_slip else [])
    specs = [parse_spec(f"{k}:{a.b_seed if (k == 'B' and a.b_seed is not None) else a.seed + i}") for i, k in enumerate(kinds)]
    sw = parse_switches(a.switch)
    if a.b_slip:
        sw["b_slip"] = True
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = config.path(a.out) / f"smoke_{a.world}_{stamp}"
    run = out / "gen"
    t0 = time.perf_counter()
    res = GR.generate(specs, run, world=a.world, workers=a.workers, render=True, max_retry=a.max_retry, switches=sw or None)
    gen_wall = time.perf_counter() - t0
    cfg = V.load_config(a.world, generation=True, switches=sw)
    rows, eps = [], []
    for r in res:
        last = r["attempts"][-1]
        row = {"kind": r["kind"], "seed": r["layout_seed"], "color": r["color"], "success": r["success"],
               "retries": r["retries"], "worker_wall_s": r["worker_wall_s"],
               "attempts": [{"name": x["name"], "success": x["success"], "failure": x["failure"], "wall_s": x["wall_s"],
                             "duration_s": x["duration_s"], "n_frames": x["n_frames"],
                             "inject_status": (x.get("inject") or {}).get("status"),
                             "inject_reason": (x.get("inject") or {}).get("reason")} for x in r["attempts"]]}
        if r["success"]:
            ep = pathlib.Path(last["path"])
            eps.append(ep)
            meta = json.loads((ep / "meta.json").read_text(encoding="utf-8"))
            data = dict(np.load(ep / "data.npz"))
            row["world_in_meta"] = (meta.get("world") or {}).get("world")
            w = data["fingers"].sum(axis=1)
            row["state_width_changes_per_s"] = round(float((np.abs(np.diff(w)) > 1e-12).sum()) / max(meta["duration_s"], 1e-9), 2)
            if r["kind"] in ("A", "C") or (r["kind"] == "B" and not sw.get("b_slip")):
                from recovla.expert import inject
                row["start_condition"] = inject.start_condition(data, meta, cfg)
            elif r["kind"] == "B":
                row["start_condition"] = inject_real.start_condition(data, meta, cfg)
        rows.append(row)
        print(f"{row['kind']} {row['seed']} {row['color']}: 成功 {row['success']}、試み {len(row['attempts'])}、"
              f"{row['worker_wall_s']} s", flush=True)
    conv = None
    if eps:
        conv = convert_and_verify(eps, out / "dataset", f"V3R_smoke_{stamp}", out / "convert.log")
        print(f"変換 {conv['convert_exit']}・照合 {'PASS' if conv['verify_pass'] else 'FAIL'}（{conv['wall_s']} s）", flush=True)
    summary = {"world": a.world, "switches": cfg["v3_real"]["switches"], "seeds": [s.layout_seed for s in specs],
               "generation_wall_s": round(gen_wall, 1), "workers": a.workers, "rows": rows, "convert": conv}
    (out / "smoke.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"→ {out / 'smoke.json'}")
    ok = all(r["success"] for r in rows) and conv is not None and conv["verify_pass"]
    return 0 if ok else 1


def cmd_slip_rate(a) -> int:
    """B の滑らせる型を、描画なし・やり直しなし（retry 0）で回し、結果を数える（保存しない）。"""
    from recovla.expert import generate_real as GR
    from recovla.harness import real_matched as V
    base, n = (int(x) for x in a.seeds.split(":"))
    sw = parse_switches(a.switch)
    sw["b_slip"] = True
    cfg = V.load_config(a.world, generation=True, switches=sw)
    if a.overlap:                                    # 試しだけ: 重なりの範囲を上書きする（--overlap 0.003 か 0.002,0.004）
        lo, _, hi = a.overlap.partition(",")
        cfg["v3_real"]["b_slip"]["overlap_m"] = [float(lo), float(hi or lo)]
    if a.yaw_max is not None:
        cfg["v3_real"]["b_slip"]["yaw_max_deg"] = float(a.yaw_max)
    rig = V.V3RealGenRig(cfg)
    rows = []
    t0 = time.perf_counter()
    try:
        for seed in range(base, base + n):
            spec = parse_spec(f"B:{seed}")
            w0 = time.perf_counter()
            r = GR.run_attempt(rig, spec, 0, None, False)
            r.pop("_frames", None)
            inj = r["inject"]
            info = inj["info"]
            rows.append({"seed": seed, "color": spec.color, "yaw_deg": r["target_yaw_deg"], "status": inj["status"],
                         "reason": inj["reason"], "params": inj["params"], "overlap_at_close_m": info.get("overlap_at_close_m"),
                         "slip_t": info.get("slip_t"), "t_lifted": info.get("t_lifted"),
                         "phase_before_slip": info.get("phase_before_slip"), "hand_at_slip": info.get("hand_at_slip"),
                         "landing_fail": info.get("landing_fail"), "recovered": r["success"], "failure": r["failure"],
                         "wall_s": round(time.perf_counter() - w0, 2)})
            x = rows[-1]
            print(f"{seed} {spec.color} yaw {x['yaw_deg']:+6.1f} 重なり {x['params']['overlap_m'] * 1e3:.1f} mm: {x['status']} "
                  f"{x['reason'] or ''} 滑り {x['slip_t']} 立て直し {x['recovered']}", flush=True)
    finally:
        rig.close()
    lifted = [x for x in rows if x["t_lifted"] is not None or x["reason"] == "no_slip"]
    slipped = [x for x in rows if x["slip_t"] is not None]
    summary = {"world": a.world, "switches": cfg["v3_real"]["switches"], "b_slip": cfg["v3_real"]["b_slip"], "seeds": [base, base + n - 1],
               "n": len(rows), "lifted": len(lifted), "slipped": len(slipped),
               "slip_rate_of_lifted": round(len(slipped) / len(lifted), 3) if lifted else None,
               "confirmed": sum(x["status"] == "confirmed" for x in rows),
               "by_status": _count(rows), "wall_s": round(time.perf_counter() - t0, 1), "rows": rows}
    out = config.path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"持ち上げ {summary['lifted']}、滑り {summary['slipped']}（{summary['slip_rate_of_lifted']}）、確定 {summary['confirmed']} → {out}")
    return 0


def cmd_torque_rate(a) -> int:
    """腕のサーボの力の変化の速さ [Nm/s] を、描画なし・やり直しなし（retry 0）で測る（保存しない）。
    作動器の力（actuator_force）の物理の 1 ステップごとの差の、関節の最大。
    閉じる指令の間（掴んで運ぶ間）と、試み全体の最大を出す。sensor_v1 の基準は 1000 Nm/s。"""
    from recovla.expert import generate_real as GR
    from recovla.harness import real_matched as V
    import numpy as np
    sw = parse_switches(a.switch)
    cfg = V.load_config(a.world, generation=True, switches=sw or None)
    rig = V.V3RealGenRig(cfg)
    acts = [int(x[0]) for x in rig.controller.arm]
    log = []
    step = rig.physics_step

    def physics_step(on_step=None):
        step(on_step)
        log.append((float(rig.data.time), bool(rig.closed), rig.data.actuator_force[acts].copy()))
    rig.physics_step = physics_step
    rows = []
    try:
        for s in a.spec:
            spec = parse_spec(s)
            log.clear()
            r = GR.run_attempt(rig, spec, 0, None, False)
            t = np.array([x[0] for x in log])
            closed = np.array([x[1] for x in log])
            tau = np.array([x[2] for x in log])
            ok = np.diff(t) > 0                                  # reset の切れ目をまたがない
            rate = (np.abs(np.diff(tau, axis=0)) / np.where(ok, np.diff(t), 1.0)[:, None]).max(axis=1) * ok
            both = closed[1:] & closed[:-1]
            rows.append({"spec": s, "success": r["success"], "failure": r["failure"],
                         "rate_max_closed_nms": round(float(rate[both].max()), 1) if both.any() else None,
                         "rate_max_all_nms": round(float(rate.max()), 1)})
            x = rows[-1]
            print(f"{s}: 成功 {x['success']}  閉じている間 {x['rate_max_closed_nms']}  全体 {x['rate_max_all_nms']} Nm/s", flush=True)
    finally:
        rig.close()
    vals = [x["rate_max_closed_nms"] for x in rows if x["rate_max_closed_nms"] is not None]
    summary = {"world": a.world, "switches": cfg["v3_real"]["switches"], "n": len(rows),
               "rate_max_closed_nms": max(vals) if vals else None,
               "rate_max_closed_median_nms": float(np.median(vals)) if vals else None,
               "rate_max_all_nms": max(x["rate_max_all_nms"] for x in rows), "rows": rows}
    out = config.path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"閉じている間の最大 {summary['rate_max_closed_nms']}（中央値 {summary['rate_max_closed_median_nms']}）、"
          f"全体の最大 {summary['rate_max_all_nms']} Nm/s → {out}")
    return 0


def _count(rows) -> dict:
    out = {}
    for x in rows:
        k = x["status"] + (("/" + x["reason"]) if x["reason"] else "")
        out[k] = out.get(k, 0) + 1
    return out


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:                                # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("gen", "smoke", "slip-rate", "torque-rate"):
        s = sub.add_parser(name)
        s.add_argument("--world", default="v3_real", choices=("v3_real",))
        s.add_argument("--switch", action="append", default=[], help="名前=on|off（何度でも）")
    s = sub.choices["gen"]
    s.add_argument("--out", required=True)
    s.add_argument("--spec", nargs="+", required=True, help="<種類>:<配置の種>[:<色>]")
    s.add_argument("--workers", type=int, default=1)
    s.add_argument("--no-render", action="store_true")
    s = sub.choices["smoke"]
    s.add_argument("--seed", type=int, required=True, help="n・A・C・B に seed から順に 1 つずつ")
    s.add_argument("--b-slip", action="store_true")
    s.add_argument("--b-seed", type=int, default=None, help="B の配置の種（省略時は seed + 3）。滑りは向き 7° 以内の配置だけ")
    s.add_argument("--workers", type=int, default=1)
    s.add_argument("--max-retry", type=int, default=None)
    s.add_argument("--out", default="outputs/real/v3_real")
    s = sub.choices["slip-rate"]
    s.add_argument("--seeds", required=True, help="<先頭>:<数>")
    s.add_argument("--overlap", default=None, help="試しだけ: 重なり [m]（1 つか 下限,上限）")
    s.add_argument("--yaw-max", type=float, default=None, help="試しだけ: 向きの上限 [deg]")
    s.add_argument("--out", required=True)
    s = sub.choices["torque-rate"]
    s.add_argument("--spec", nargs="+", required=True, help="<種類>:<配置の種>[:<色>]")
    s.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"gen": cmd_gen, "smoke": cmd_smoke, "slip-rate": cmd_slip_rate, "torque-rate": cmd_torque_rate}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
