"""生成プログラムの終盤の近づきを遅くした設定（configs/expert_v3.yaml）で試しにデータを作り、基準を満たすかを確かめる。

scripts/95_s3_round2.py が、実行のしかたの判断の読み出し（round0_choice）の部品として読む。

    .venv\\Scripts\\python.exe scripts\\check_slow_approach.py trial-gen [--workers 3]    # 学習用のシード 58300〜58329 で 30 本
    .venv\\Scripts\\python.exe scripts\\check_slow_approach.py trial-check --run outputs\\gen\\S3_trial_v3_<日時>

基準（回す前に固定した）: 先行の中央値 ≤ 20 mm（> 25 mm なら止める）、生成プログラムの成功 ≥ 29/30、G3 の違反 0、
所要時間の中央値 ≤ 20 s。
先行の測り方: 閉じる前で、手先（hand 原点）から目標の立方体までの水平の残りが 25〜120 mm、指先が立方体より
60 mm 以上高いフレームの、(x_des − 手先) の水平成分を「手先 → 立方体」の向きに射影した値 [mm]。エピソードごとの中央値の、
エピソードについての中央値。比べる基準として、以前の生成（F_data_v2）の最初の 30 本も同じ定義で測る。
G3: 生成した行動（10 Hz の x_des の差とグリッパ）を実行系の動きの層（Motion＝IK・制限）で再生し、指令の上限の違反を数える。
結果: docs/results/s3_round1_trial.json。
"""
import argparse
import json
import pathlib
import time

import numpy as np

from recovla.common import config

CFG = config.load()
OUT = config.path(CFG["paths"]["outputs"])
GEN = OUT / "gen"
RES = config.ROOT / "docs" / "results"
COLORS = ["red", "green", "blue"]
TRIAL_SEEDS = range(58300, 58330)
REF_RUN = GEN / "F_data_v2_20260929-205251"
LEAD_MAX, LEAD_STOP, SUCC_MIN, DUR_MAX = 20.0, 25.0, 29, 20.0
REM_MM, DZ_MM = (25.0, 120.0), 60.0


def episode_lead(p: pathlib.Path):
    meta = json.loads((p / "meta.json").read_text(encoding="utf-8"))
    z = np.load(p / "data.npz")
    ti = COLORS.index(meta["target"])
    closed = z["gripper_closed"].astype(bool)
    k = int(np.where(closed)[0][0]) if closed.any() else len(closed)
    cube, ee, xdes, tip = z["cube_pos"][:k, ti], z["ee_pos"][:k], z["x_des"][:k], z["fingertip"][:k]
    rel = cube[:, :2] - ee[:, :2]
    rem = np.linalg.norm(rel, axis=1) * 1e3
    u = rel / np.maximum(np.linalg.norm(rel, axis=1, keepdims=True), 1e-9)
    lead = np.sum((xdes[:, :2] - ee[:, :2]) * u, axis=1) * 1e3
    m = (rem >= REM_MM[0]) & (rem <= REM_MM[1]) & ((tip[:, 2] - cube[:, 2]) * 1e3 > DZ_MM)
    return (float(np.median(lead[m])) if m.any() else None), meta


def cmd_trial_gen(a) -> None:
    from recovla.expert import generate as G
    specs = [G.EpisodeSpec(s, COLORS[s % 3], "empty", "n") for s in TRIAL_SEEDS]
    run = GEN / f"S3_trial_v3_{time.strftime('%Y%m%d-%H%M%S')}"
    res = G.generate(specs, run, workers=a.workers, render=True, rig_kind="v3")
    print(f"[s3] {run} success {sum(bool(r.get('success')) for r in res)}/{len(res)}", flush=True)


def replay_g3(paths) -> list:
    """生成した行動を実行系の動きの層で再生し、G3（指令の上限）の違反を数える。"""
    from recovla.data import convert as CV
    from recovla.harness.loop import run_policy_trial
    from recovla.harness.sensors import SensorSuite
    from recovla.harness.world import WorldRig
    from recovla.runtime.motion import Motion
    from recovla.runtime.runner import PolicyRuntime
    from recovla.sim import scene

    class ReplayRuntime(PolicyRuntime):
        actions = None

        def _action_boundary(self) -> None:
            k, t = self.k, self.io.now()
            if k < len(self.actions):
                act, held = np.asarray(self.actions[k], float), False
            else:
                act, held = np.array([0, 0, 0, 0, 0, 0, 1.0 if self.closed else -1.0]), True
            self._apply(act)
            self.log_act.append((k, t, None if held else 0, held, act.copy()))

    class _NoPolicy:
        def start_trial(self, seed):
            pass

        def reset_cue(self):
            pass

    cfg2 = config.load_v2()
    margin = float(cfg2["actuation"]["limiter_margin"])
    world = WorldRig(render=False, cfg=cfg2)
    suite = SensorSuite(world.model, cfg2)
    rows = []
    for p in paths:
        meta, data = CV.load_raw(p)
        arr = CV.episode_arrays(meta, data)
        L = meta["layout"]
        lay = scene.sample_layout(int(meta["layout_seed"]), L["kind"], start=L["start"])

        def make(io, setup):
            rt = ReplayRuntime(io, setup, _NoPolicy(), mode="naive", motion=Motion(setup, margin=margin))
            rt.actions = arr["action"]
            return rt
        tl = max(float(cfg2["eval"]["time_limit_s"]), float(meta["duration_s"]) + 5.0)
        m, arrays, _ = run_policy_trial(world, suite, make, lay, meta["target"], int(meta["sensor_seed"]), time_limit_s=tl, cfg=cfg2)
        n = min(len(arrays["ee_pos"]), len(data["ee_pos"]), 2 * len(arr["action"]) + 1)
        d = np.linalg.norm(arrays["ee_pos"][:n] - data["ee_pos"][:n], axis=1)
        rows.append({"episode": p.name, "replay_success": bool(m["success"]), "ee_rms_mm": float(np.sqrt(np.mean(d ** 2)) * 1e3),
                     "g3_violations": int(m["audit"]["g3"]["total_violations"])})
        print(f"[g3] {p.name} replay {rows[-1]['replay_success']} ee_rms {rows[-1]['ee_rms_mm']:.2f} mm "
              f"g3 {rows[-1]['g3_violations']}", flush=True)
    suite.close()
    return rows


def cmd_trial_check(a) -> None:
    run = config.path(a.run)
    eps = sorted(p for p in run.iterdir() if (p / "meta.json").is_file())
    gen = [json.loads(line) for line in (run / "generation.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    ok = sum(bool(r.get("success")) for r in gen)
    leads, durs, saved = [], [], []
    for p in eps:
        lead, meta = episode_lead(p)
        if not meta.get("success"):
            continue
        saved.append(p)
        durs.append(float(meta["duration_s"]))
        if lead is not None:
            leads.append(lead)
    ref = []
    for p in sorted(REF_RUN.iterdir()):
        if p.name.startswith("n_") and (p / "meta.json").is_file() and len(ref) < 30:
            lead, meta = episode_lead(p)
            if meta.get("success") and lead is not None:
                ref.append(lead)
    g3 = replay_g3(saved)
    lead_med = float(np.median(leads)) if leads else None
    gates = {"lead_median_le_20mm": lead_med is not None and lead_med <= LEAD_MAX,
             "script_success_ge_29": ok >= SUCC_MIN,
             "g3_violations_0": all(r["g3_violations"] == 0 for r in g3),
             "duration_median_le_20s": float(np.median(durs)) <= DUR_MAX}
    res = {"what": "段階 3 の 1 周目（案 1）の試しの生成の関門（0126 の 4・0128）", "run": str(run.relative_to(config.ROOT)),
           "seeds": [TRIAL_SEEDS.start, TRIAL_SEEDS.stop - 1], "lead_definition": __doc__.split("先行の測り方（0128）: ")[1].split("G3:")[0].strip(),
           "script_success": [ok, len(gen)], "lead_mm": {"median": lead_med, "per_episode": leads, "n": len(leads)},
           "reference_v2_lead_mm": {"run": str(REF_RUN.relative_to(config.ROOT)), "median": float(np.median(ref)), "n": len(ref)},
           "duration_s": {"median": float(np.median(durs)), "p95": float(np.percentile(durs, 95))},
           "g3_replay": g3, "gates": gates, "stop_before_training": lead_med is None or lead_med > LEAD_STOP,
           "pass": all(gates.values()), "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "s3_round1_trial.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("script_success", "gates", "stop_before_training", "pass")} |
                     {"lead_median": lead_med, "ref_v2_lead_median": res["reference_v2_lead_mm"]["median"],
                      "duration_median": res["duration_s"]["median"]}, ensure_ascii=False, indent=1))


# ---- 検証: この設定のデータで学習したモデル（v3）と前のモデル（v2）を比べる。
#      実行のしかたと安全フィルタは choose_exec_mode.py の判断（docs/results/s3_round0_*.json）に従う ----
EXP_V = "V3R1"
MODELS_V = {"R1v2": "R1v2_20000", "N1v2": "N1v2", "R1v3": "R1v3", "N1v3": "N1v3"}
TRIALS_V = {"nat": ("natural:199230:33", None), "P1a": ("induced:199130:33", "P1"), "P1b": ("induced:199730:66", "P1")}
OFFSET_GAIN_MM = 15.0      # 下り始めのずれの中央値が R1v2 より 15 mm 以上 0 に近づく
R_REC_DROP = 0.10          # R の P1 の立て直し／実際に失敗した数が R1v2 より 10 ポイント以上下がらない


def round0_choice() -> dict:
    a = json.loads((RES / "s3_round0_a.json").read_text(encoding="utf-8"))
    pb = RES / "s3_round0_b.json"
    if a["candidate"] == "N10":
        arm = "N10"
    elif pb.is_file():
        arm = json.loads(pb.read_text(encoding="utf-8"))["final_arm"]
    else:
        raise SystemExit("0 周目の段 B の判断（s3_round0_b.json）がまだない")
    return {"arm": arm, "safety": bool(a["safety_keep_tentative"])}


def cmd_run_verify(a) -> None:
    r0 = _mod92()
    ch = round0_choice()
    jobs = {}
    for name, model in MODELS_V.items():
        for part, (trials, induce) in TRIALS_V.items():
            jobs[f"{name}_{part}"] = r0.run_args(EXP_V, f"{name}_{part}", model, trials, ch["arm"], ch["safety"], induce=induce)
    r0.run_pool(EXP_V, jobs, a.parallel)


def _mod92():
    import importlib.util
    spec = importlib.util.spec_from_file_location("r0", config.ROOT / "scripts" / "choose_exec_mode.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def descent_offsets(d: pathlib.Path) -> list:
    """下り始め（最初に閉じる前で、指先が立方体より 130 mm 以上高い最後のフレーム）の、指先 − 立方体の水平の差を、ロボットの
    根元から立方体への向きに射影した値 [mm]（負＝手前）。閉じなかった試行は数えない。"""
    out = []
    for f in sorted(d.glob("trial_*.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        z = np.load(f.with_suffix(".npz"))
        ti = COLORS.index(m["target"].split(">")[0])
        closed = z["gripper_closed"].astype(bool)
        if not closed.any():
            continue
        k = int(np.where(closed)[0][0])
        tip, cube = z["fingertip"][:k], z["cube_pos"][:k, ti]
        idx = np.where((tip[:, 2] - cube[:, 2]) * 1e3 >= 130)[0]
        if not idx.size:
            continue
        j = int(idx[-1])
        u = cube[j, :2] / np.linalg.norm(cube[j, :2])
        out.append(float((tip[j, :2] - cube[j, :2]) @ u * 1e3))
    return out


def cmd_decide_verify(a) -> None:
    from recovla.eval import gate
    from recovla.eval import report as R
    r0 = _mod92()
    v = config.path(CFG["paths"]["outputs"]) / "v2eval" / EXP_V
    dirs = [v / f"{n}_{p}" for n in MODELS_V for p in TRIALS_V]
    gsum = gate.require(dirs, "段階 3 の 1 周目の検証（0126・0129）")
    rows = {d.name: {(r["seed"], r["target"]): r for r in R.collect([d])} for d in dirs}

    def p1(name):
        return {**rows[f"{name}_P1a"], **rows[f"{name}_P1b"]}
    nat = {n: sum(bool(r["success"]) for r in rows[f"{n}_nat"].values()) for n in MODELS_V}
    offs = {n: descent_offsets(v / f"{n}_nat") for n in MODELS_V}
    med = {n: (float(np.median(offs[n])) if offs[n] else None) for n in MODELS_V}
    rec = {}
    for n in MODELS_V:
        est = [r for r in p1(n).values() if r["induce_established"]]
        rec[n] = {"established": len(est), "recovered": sum(bool(r["recovered"]) for r in est),
                  "rate": (sum(bool(r["recovered"]) for r in est) / len(est)) if est else None}
    e3 = {ver: r0.e3_stats(p1(f"R1{ver}"), p1(f"N1{ver}")) for ver in ("v2", "v3")}
    ok_offset = med["R1v3"] is not None and med["R1v2"] is not None and abs(med["R1v3"]) <= abs(med["R1v2"]) - OFFSET_GAIN_MM
    ok_nat = nat["R1v3"] >= nat["R1v2"]
    rb, rc = rec["R1v2"]["rate"], rec["R1v3"]["rate"]
    ok_rec = rb is None or (rc is not None and rc > rb - R_REC_DROP)
    adopt = ok_offset and ok_nat and ok_rec
    res = {"what": "段階 3 の 1 周目（案 1）の検証と判定（0126 の 4・0129）", "experiment": EXP_V, "trials": TRIALS_V,
           "models": MODELS_V, "round0": round0_choice(), "natural_success": nat, "descent_offset_mm_median": med,
           "descent_offset_n": {n: len(offs[n]) for n in offs}, "p1_recovery_R": rec, "e3_like": e3,
           "rules": {"offset": f"|中央値(R1v3)| ≤ |中央値(R1v2)| − {OFFSET_GAIN_MM} mm",
                     "natural": "R1v3 の自然の成功 ≥ R1v2", "p1": f"R1v3 の P1 の復帰／成立 > R1v2 − {int(R_REC_DROP * 100)} ポイント",
                     "gate": "関所の違反 0"},
           "ok_offset": ok_offset, "ok_natural": ok_nat, "ok_p1": ok_rec, "adopt": adopt,
           "next": "2 周目は R1v3・N1v3 を土台に案 3" if adopt else "2 周目は R1v2・N1v2 を土台に案 3（0126 の 4）",
           "g_audit": gsum, "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    (RES / "s3_round1_verify.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("natural_success", "descent_offset_mm_median", "p1_recovery_R", "e3_like",
                                          "ok_offset", "ok_natural", "ok_p1", "adopt")}, ensure_ascii=False, indent=1, default=float))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("trial-gen")
    p.add_argument("--workers", type=int, default=3)
    p = sub.add_parser("trial-check")
    p.add_argument("--run", required=True)
    p = sub.add_parser("run-verify")
    p.add_argument("--parallel", type=int, default=3)
    sub.add_parser("decide-verify")
    a = ap.parse_args(argv)
    {"trial-gen": cmd_trial_gen, "trial-check": cmd_trial_check, "run-verify": cmd_run_verify,
     "decide-verify": cmd_decide_verify}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
