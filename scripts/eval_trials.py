"""目標書 v2 の実行系で、方策の試行を回す（1 つのサブタスク。失敗注入 P1〜P3 も）。記録は真値と実行系で分ける。

    .venv\\Scripts\\python.exe scripts\\eval_trials.py run --experiment V2DEV --condition R2_nat --model R2 --trials natural:59050:1
        [--mode naive|sync|rtc] [--induce P1] [--no-limiter]

試行の並びは eval_subtask_direct.py と同じ（natural・induced・selection）。出力: outputs/v2eval/<実験>/<条件>/trial_NNNN.{json,npz}
（json に成否・監査・センサの試行ごとの値、npz に真値のこま）と runtime_NNNN.json（実行系の記録: 推論・行動・計算の時間・知覚）。
"""
import argparse
import importlib.util
import json
import os
import pathlib
import time

import numpy as np

from recovla.common import config

CFG = config.load_v2()
os.environ["HF_HOME"] = str(config.path(CFG["paths"]["models_home"]))
os.environ["HF_HUB_OFFLINE"] = "1"
OUT = config.path(CFG["paths"]["outputs"]) / "v2eval"
CKPT = {
    "R1": "outputs/train/train_R1_20260926-153959_20260926-153959/checkpoints/030000/pretrained_model",
    "N1": "outputs/train/train_N1_20260926-191928_20260926-191928/checkpoints/030000/pretrained_model",
    "R2": "outputs/train/train_R2_20260927-145256_20260927-145256/checkpoints/010000/pretrained_model",
    "R1plus": "outputs/train/train_R1plus_20260927-160208_20260927-160208/checkpoints/010000/pretrained_model",
}
V2_TRAIN = {"R1v2": "train_R1v2_20260929-230928_20260929-230928", "N1v2": "train_N1v2_20260930-060552_20260930-060552"}
for _m, _d in V2_TRAIN.items():
    for _s in (20000, 30000):
        CKPT[f"{_m}_{_s}"] = f"outputs/train/{_d}/checkpoints/{_s:06d}/pretrained_model"
# 保存点の選択の結果。G3 の直しの後に新しい検証用の種（199640〜）で同じ決まりのまま選び直した（0124、
# docs/results/ckpt_decision_v2_*_g3fix.json）: R1v2 は 2 万手（11 対 8）、N1v2 は同数（6 対 6）なので 3 万手。
# 直す前の選択（ckpt_decision_v2_*.json、どちらも 2 万手）は G3 の違反を含んでいたので使わない
CKPT["R1v2"] = CKPT["R1v2_20000"]
CKPT["N1v2"] = CKPT["N1v2_30000"]
# 段階 3（0126）: 学習した組は 2 万手の保存点を使う（保存点の選択はしない）。学習の出力のうち最新のもの
for _m in ("R1v3", "N1v3", "R2v3", "N2v3"):
    _runs = sorted(config.path(CFG["paths"]["train_output"]).glob(f"train_{_m}_2*"))
    if _runs and (_runs[-1] / "checkpoints" / "020000" / "pretrained_model").is_dir():
        CKPT[_m] = str((_runs[-1] / "checkpoints" / "020000" / "pretrained_model").relative_to(config.ROOT))


def trial_list(spec: str) -> list:
    s = importlib.util.spec_from_file_location("r41", config.ROOT / "scripts" / "eval_subtask_direct.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m.trial_list(spec)


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


ABLATIONS = ("nocolor", "wrist60", "nolatency", "nocalib", "noholes", "fixedinf", "freeze", "fastgrip", "strongforce")


def ablate(cfg: dict, names) -> dict:
    """診断だけに使う（評価では使わない）: センサの模型の要素を 1 つずつ外す。"""
    import copy
    c = copy.deepcopy(cfg)
    s = c["sensor"]
    for n in names:
        if n == "nocolor":
            s["color_noise"].update(read_sigma=0.0, shot_gain=0.0, exposure_gain=[1.0, 1.0], white_balance=[1.0, 1.0])
        elif n == "wrist60":
            s["cameras"]["wrist"]["color"] = {"width": 256, "height": 256, "fovy": 60.0, "out": 256}
        elif n == "nolatency":
            s["latency_ms"] = [0.0, 0.0]
        elif n == "nocalib":
            for k in ("overhead", "wrist"):
                s["calibration"][k] = {"trans_sigma": 0.0, "rot_sigma_deg": 0.0}
            s["calibration"]["table"] = {"z_sigma": 0.0, "tilt_sigma_deg": 0.0}
        elif n == "noholes":
            s["depth_noise"]["hole_area"] = 0.0
        elif n == "fixedinf":                     # 推論時間のばらつきを外す（latency_v1 の中央値に固定）
            from recovla.harness.robot_io import LATENCY
            q = LATENCY["kinds"]["policy"]["quantiles_s"]
            c["runtime_v2"].setdefault("diag_fixed_latency", {})["policy"] = float(np.interp(0.5, np.linspace(0, 1, len(q)), q))
        elif n == "freeze":                       # 世界の停止に相当: 推論と知覚の計算の時間を 0 にする
            c["runtime_v2"].setdefault("diag_fixed_latency", {}).update({"policy": 0.0, "perception": 0.0})
        elif n == "fastgrip":                     # グリッパの速さを公称の上限に（0.08 → 0.1 m/s。ハンドの模型はこれを超える指令を拒む）
            c["actuation"]["gripper_speed"] = 0.1
        elif n == "strongforce":                  # 把持力を公称の上限 70 N に
            c["actuation"]["grasp_force"] = 70.0
        else:
            raise SystemExit(f"--ablate {n}: one of {ABLATIONS}")
    return c


def cmd_run(a) -> None:
    global CFG
    if a.ablate:
        CFG = ablate(CFG, a.ablate.split(","))
    from recovla.eval import induce as I
    from recovla.harness.loop import run_policy_trial
    from recovla.harness.sensors import SensorSuite
    from recovla.harness.world import WorldRig
    from recovla.runtime import cue as C
    from recovla.runtime.motion import Motion
    from recovla.runtime.policy import SensorPolicy
    from recovla.runtime.perception import Params, Perception
    from recovla.runtime.runner import PolicyRuntime, disable_rtc_for, enable_rtc_for
    from recovla.runtime.safety import PerceptionSafetyFilter
    out = OUT / a.experiment / a.condition
    if out.exists() and any(out.glob("trial_*.json")):
        raise SystemExit(f"{out} already has trials")
    out.mkdir(parents=True, exist_ok=True)
    world = WorldRig(render=False, cfg=CFG, gravcomp=False if a.diag_no_gravcomp else None)
    suite = SensorSuite(world.model, CFG)
    rt_cfg, act = CFG["runtime"], CFG["actuation"]
    exec_interval = int(a.exec_interval or rt_cfg["exec_interval"])      # 段階 3 の 0 周目で選び直す（0126 の 2）
    cache = {}

    def make_runtime(io, setup):
        if "pol" not in cache:
            mo = Motion(setup)
            cache["fk_motion"] = mo
            cache["pol"] = SensorPolicy(config.path(CKPT[a.model]), setup, mo.hand_pose)
            if a.mode == "rtc":
                enable_rtc_for(cache["pol"], int(rt_cfg["rtc_guidance_horizon"]), rt_cfg["rtc_schedule"],
                               float(rt_cfg["rtc_max_guidance_weight"]))
            else:
                disable_rtc_for(cache["pol"])
        pol = cache["pol"]
        pol.setup = setup
        if pol.cue is not None:                         # 試行ごとの較正誤差（信じている値）で作り直す
            pol.cue = C.TargetCue(C.calibration_from_setup(setup.cameras["overhead"]), pol.cue.thr,
                                  setup.table_z + 0.5 * setup.cube_size, setup.cue_fallback_xy)
        rtv = CFG["runtime_v2"]
        per = Perception(setup, Params.from_config(rtv["perception"]), C.Thresholds.from_dict(config.color_detect(CFG)))
        sf = None
        if not a.no_safety:
            sf = PerceptionSafetyFilter(setup, CFG["safety_filter"], float(rtv["safety_extra_margin_m"] or 0.0))
        gate = dict(rtv.get("gripper_gate") or {})
        if a.grip_gate:
            gate["enabled"] = True
        return PolicyRuntime(io, setup, pol, perception=per, safety=sf, checks=rtv["checks"], gripper_gate=gate,
                             tip_offset=float(CFG["sim"]["fingertip_offset"]), mode=a.mode, s=exec_interval, d_init=int(rt_cfg["delay_steps"]),
                             rtc_horizon=int(rt_cfg["rtc_guidance_horizon"]),
                             motion=Motion(setup, limiter_enabled=not a.no_limiter, margin=float(act["limiter_margin"]),
                                           ik_on=a.diag_ik, xcmd_leash_m=a.xcmd_leash or rtv.get("xcmd_leash_m"),
                                           cart_margin=a.cart_margin or rtv.get("cart_margin")))

    rows = []
    t0 = time.perf_counter()
    for i, (seed, lay, tgt) in enumerate(trial_list(a.trials)):
        ind = I.Inducer(a.induce, seed, lay, tgt, world) if a.induce else None
        w0 = time.perf_counter()
        meta, arrays, rlog = run_policy_trial(world, suite, make_runtime, lay, tgt, seed, inducer=ind, cfg=CFG)
        meta.update({"trial": i, "experiment": a.experiment, "condition": a.condition, "model": {"name": a.model}, "ablate": a.ablate,
                     "runtime": {"mode": a.mode, "exec_interval": exec_interval, "delay_steps": "sampled",
                                 "safety_filter": not a.no_safety,
                                 "gripper_gate": bool(a.grip_gate or (CFG["runtime_v2"].get("gripper_gate") or {}).get("enabled")),
                                 "xcmd_leash_m": a.xcmd_leash or CFG["runtime_v2"].get("xcmd_leash_m"),
                                 "cart_margin": a.cart_margin or CFG["runtime_v2"].get("cart_margin")},
                     "mode": a.mode, "limiter": not a.no_limiter, "safety": not a.no_safety, "wall_s": round(time.perf_counter() - w0, 2)})
        np.savez(out / f"trial_{i:04d}.npz", **arrays)
        (out / f"trial_{i:04d}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=_json_default), encoding="utf-8")
        (out / f"runtime_{i:04d}.json").write_text(json.dumps(rlog, ensure_ascii=False, default=_json_default), encoding="utf-8")
        rows.append({"trial": i, "seed": seed, "target": tgt, "success": meta["success"],
                     "established": meta["induce"].get("established")})
        g = meta["audit"]
        print(f"[v2] {a.condition} {i:3d} seed {seed} {tgt:5s} success {meta['success']} est {meta['induce'].get('established')} "
              f"stops {g['g2']['world_stops']} early {g['g2']['early_use']} g3 {g['g3']['total_violations']} "
              f"wall {meta['wall_s']}", flush=True)
    suite.close()
    (out / "run.json").write_text(json.dumps({"experiment": a.experiment, "condition": a.condition, "model": a.model,
                                               "mode": a.mode, "exec_interval": exec_interval, "safety": not a.no_safety,
                                               "trials": a.trials, "induce": a.induce, "n": len(rows),
                                               "successes": sum(r["success"] for r in rows),
                                               "wall_s": round(time.perf_counter() - t0, 1),
                                               "written": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
    _gate_mark(out)


def _gate_mark(out) -> None:
    from recovla.eval import gate
    s = gate.mark(out)
    print(f"[gate] {out.name}: {'G を満たす' if s['met'] else 'G を満たさない'}（G1 {s['g1_trials']} 本・G2 {s['g2_trials']} 本・"
          f"G3 {s['g3_trials']} 本 / {s['trials']} 本）", flush=True)


def cmd_task(a) -> None:
    """複数手順（E7 の 3 個の連続タスク）。配置は旧版と同じく空の箱・既定の開始姿勢（scene.sample_layout(種, "empty", "home")）。"""
    from recovla.harness.sensors import SensorSuite
    from recovla.harness.task_loop import run_task_trial
    from recovla.harness.world import WorldRig
    from recovla.planner import decompose as D
    from recovla.runtime import cue as C
    from recovla.runtime.executor import TaskRuntime
    from recovla.runtime.judge import JudgeV2
    from recovla.runtime.motion import Motion
    from recovla.runtime.perception import Params, Perception
    from recovla.runtime.policy import SensorPolicy
    from recovla.runtime.runner import PolicyRuntime, disable_rtc_for
    from recovla.runtime.safety import PerceptionSafetyFilter
    from recovla.sim import scene
    out = OUT / a.experiment / a.condition
    if out.exists() and any(out.glob("run_*.json")):
        raise SystemExit(f"{out} already has runs")
    out.mkdir(parents=True, exist_ok=True)
    world = WorldRig(render=False, cfg=CFG)
    suite = SensorSuite(world.model, CFG)
    rt_cfg, act, rtv = CFG["runtime"], CFG["actuation"], CFG["runtime_v2"]
    thr = C.Thresholds.from_dict(config.color_detect(CFG))
    cache = {}

    def make(io, setup):
        if "pol" not in cache:
            cache["pol"] = SensorPolicy(config.path(CKPT[a.model]), setup, Motion(setup).hand_pose)
            disable_rtc_for(cache["pol"])
        pol = cache["pol"]
        pol.setup = setup
        if pol.cue is not None:
            pol.cue = C.TargetCue(C.calibration_from_setup(setup.cameras["overhead"]), pol.cue.thr,
                                  setup.table_z + 0.5 * setup.cube_size, setup.cue_fallback_xy)
        per = Perception(setup, Params.from_config(rtv["perception"]), thr)
        sf = None if a.no_safety else PerceptionSafetyFilter(setup, CFG["safety_filter"], float(rtv["safety_extra_margin_m"]))
        prt = PolicyRuntime(io, setup, pol, perception=per, safety=sf, checks=rtv["checks"], gripper_gate=rtv.get("gripper_gate"),
                            tip_offset=float(CFG["sim"]["fingertip_offset"]), mode="naive",
                            s=int(a.exec_interval or rt_cfg["exec_interval"]),          # 段階 3 は 6 行（0142）
                            d_init=int(rt_cfg["delay_steps"]), motion=Motion(setup, margin=float(act["limiter_margin"]),
                                                                             xcmd_leash_m=rtv.get("xcmd_leash_m"),
                                                                             cart_margin=rtv.get("cart_margin")))
        judge = JudgeV2(setup, per, thr, rtv["judge"])
        e = CFG["expert"]
        mp = {"gain": float(e["gain_per_s"]), "xy_max": float(e["speed_ref"]["xy"]), "z_max": float(e["speed_ref"]["z"]),
              "z_tol": float(e["move_tol_m"]), "tol": float(e["phase"]["retreat_tol_m"])}
        return TaskRuntime(io, setup, prt, judge, CFG["planner"], mp, D.decompose, CFG["convert"]["instruction"])

    base, n = (int(x) for x in a.trials.split(":"))
    rows = []
    for i, seed in enumerate(range(base, base + n)):
        lay = scene.sample_layout(seed, "empty", start="home")
        w0 = time.perf_counter()
        meta, arrays, rlog = run_task_trial(world, suite, make, lay, a.text, seed, cfg=CFG)
        meta.update({"run": i, "model": a.model, "wall_s": round(time.perf_counter() - w0, 1)})
        np.savez(out / f"run_{i:04d}.npz", **arrays)
        (out / f"run_{i:04d}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=_json_default), encoding="utf-8")
        (out / f"run_{i:04d}_runtime.json").write_text(json.dumps(rlog, ensure_ascii=False, default=_json_default), encoding="utf-8")
        rows.append({"seed": seed, "all_three": meta["all_three_in_box"]})
        g = meta["audit"]
        print(f"[v2task] {i} seed {seed} plan {(meta['plan'] or {}).get('steps')} final {meta['final_in_box']} "
              f"stopped {bool(meta['stopped'])} g1 {g['g1']['violations']} g2 {g['g2']['world_stops']}/{g['g2']['early_use']} "
              f"g3 {g['g3']['total_violations']} wall {meta['wall_s']}", flush=True)
    suite.close()
    (out / "run.json").write_text(json.dumps({"n": len(rows), "all_three": sum(r["all_three"] for r in rows),
                                               "text": a.text, "model": a.model, "trials": a.trials}, ensure_ascii=False,
                                              indent=1), encoding="utf-8")
    _gate_mark(out)


def cmd_e6(a) -> None:
    """E6（v2）: 旧版（20_k1.py e6）と同じ手順を、センサの模型の画像・測った状態・テーブル面で直した較正の手がかりで行う。
    配置ごとに試行の始め（時刻 0 の先読みのこま）の観測から、指示と手がかりを一緒に（cue_only なら手がかりだけ）差し替えて
    塊を 5 回引き、1 回目の塊の行き先の多数決が差し替えた色かを見る。"""
    import collections
    from recovla.common import seeds
    from recovla.common.seeds import COLORS
    from recovla.eval.swap_test import chunk_target, majority
    from recovla.harness import setup as HS
    from recovla.harness.sensors import SensorSuite
    from recovla.harness.world import WorldRig
    from recovla.runtime import cue as C
    from recovla.runtime.motion import Motion
    from recovla.runtime.perception import Params, Perception
    from recovla.runtime.policy import SensorPolicy
    from recovla.runtime.runner import disable_rtc_for
    from recovla.runtime.types import JointState
    from recovla.sim import scene
    world = WorldRig(render=False, cfg=CFG)
    suite = SensorSuite(world.model, CFG)
    rtv = CFG["runtime_v2"]
    thr = C.Thresholds.from_dict(config.color_detect(CFG))
    pol, mo = None, None
    rows = []
    base, n = map(int, a.trials.split(":"))
    for seed in range(base, base + n):
        lay = scene.sample_layout(seed, "empty", start="home")
        world.reset(lay)
        setup = suite.start_trial(seed, world.data, HS.nominal_setup(CFG))
        suite.prime(world.data)
        if pol is None:
            mo = Motion(setup)
            pol = SensorPolicy(config.path(CKPT[a.model]), setup, mo.hand_pose)
            disable_rtc_for(pol)
        sf = suite.sense(world.data, world.hand, world._last_cmd, True)
        per = Perception(setup, Params.from_config(rtv["perception"]), thr)
        for i in range(10):
            per.record_joints(JointState(-0.05 + 0.005 * i, sf.joints.q, sf.joints.dq))
        per.record_joints(sf.joints)
        per.check_table(sf.cameras["overhead"], sf.gripper.width)
        R, t = per.camera_pose("overhead", sf.t, sf.gripper.width)
        pol.setup = setup
        pol.cue = C.TargetCue(C.calibration_from_setup(setup.cameras["overhead"]), thr, setup.table_z + 0.5 * setup.cube_size,
                              setup.cue_fallback_xy)
        pol.set_cue_pose(R, t)
        x_des = mo.hand_pose(sf.joints.q)[0]
        cubes_xy = {c: world.data.xpos[world.cube_ids[i]][:2].tolist() for i, c in enumerate(COLORS)}
        pol.start_trial(seed)
        for ci, color in enumerate(COLORS):
            instr_color = COLORS[(ci + 1) % len(COLORS)] if a.cue_mode == "cue_only" else color
            samples = []
            for j in range(5):
                gen = pol.torch.Generator().manual_seed(seeds.torch_seed(seeds.seed_sequence(seed, "noise", ci, j)))
                pol.reset_cue()
                obs = pol.observe(sf, CFG["convert"]["instruction"].format(color=instr_color), cue_color=color)
                ch = pol.infer(obs, gen)
                samples.append(chunk_target(ch, x_des, cubes_xy))
            maj = majority([s["nearest"] for s in samples])
            rows.append({"seed": seed, "instruction": color, "instruction_text_color": instr_color, "majority": maj,
                         "correct_majority": maj == color, "follows_instruction_text_majority": maj == instr_color,
                         "correct_samples": sum(s["nearest"] == color for s in samples),
                         "votes": dict(collections.Counter(s["nearest"] for s in samples)),
                         "cue": None if pol.last_cue is None else [float(v) for v in pol.last_cue]})
        print(f"[e6] seed {seed} {[r['majority'] for r in rows[-3:]]}", flush=True)
    suite.close()
    nrow = len(rows)
    res = {"model": a.model, "trials": a.trials, "cue_mode": a.cue_mode, "pairs": nrow,
           "accuracy_majority": sum(r["correct_majority"] for r in rows) / nrow,
           "accuracy_samples": sum(r["correct_samples"] for r in rows) / (5 * nrow),
           "follows_instruction_text_majority": sum(r["follows_instruction_text_majority"] for r in rows) / nrow,
           "rows": rows, "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    out = OUT / a.experiment
    out.mkdir(parents=True, exist_ok=True)
    (out / f"e6_{a.condition}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=_json_default), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, ensure_ascii=False))


def cmd_decide_ckpt(a) -> None:
    """段階 2 の保存点の選択（0117 で了承）。検証用の種の同じ配置で 2 万手と 3 万手を比べ、成功数の多い方、同数なら 3 万手
    （旧版の eval_subtask_direct.py decide-ckpt と同じ決まり）。"""
    from recovla.eval import gate
    out = {}
    gsum = gate.require([OUT / a.experiment / f"{a.model}_{s}" for s in (20000, 30000)], f"保存点の選択（{a.model}）")
    for step in (20000, 30000):
        d = OUT / a.experiment / f"{a.model}_{step}"
        rs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("trial_*.json"))]
        out[step] = {"n": len(rs), "successes": sum(bool(r["success"]) for r in rs), "seeds": sorted({r["seed"] for r in rs})}
    chosen = 20000 if out[20000]["successes"] > out[30000]["successes"] else 30000
    res = {"model": a.model, "rule": "成功数の多い方、同数なら 30000 手（旧版と同じ決まり）", "candidates": out,
           "same_seeds": out[20000]["seeds"] == out[30000]["seeds"], "chosen_step": chosen,
           "checkpoint": CKPT[f"{a.model}_{chosen}"], "experiment": a.experiment, "g_audit": gsum,
           "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    dst = config.ROOT / "docs" / "results"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / f"ckpt_decision_v2_{a.model}{a.suffix}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--experiment", required=True)
    p.add_argument("--condition", required=True)
    p.add_argument("--model", required=True, choices=sorted(CKPT))
    p.add_argument("--trials", required=True)
    p.add_argument("--mode", default="naive", choices=("naive", "sync", "rtc"))
    p.add_argument("--exec-interval", type=int, default=None, help="塊から実行する行数 s（省略時は設定の値 10）。段階 3（0126）")
    p.add_argument("--induce", default=None)
    p.add_argument("--no-limiter", action="store_true")
    p.add_argument("--no-safety", action="store_true", help="安全フィルタを切る（E5 の比べる側）")
    p.add_argument("--diag-ik", default="commanded", choices=("commanded", "measured"), help="診断だけ")
    p.add_argument("--diag-no-gravcomp", action="store_true", help="診断だけ")
    p.add_argument("--ablate", default=None, help="診断だけ: " + ",".join(ABLATIONS))
    p.add_argument("--grip-gate", action="store_true", help="グリッパのためらいの幅と最短の保持時間を入れる（0121 の B3）")
    p.add_argument("--xcmd-leash", type=float, default=None, help="参照位置の綱 [m]（0121 の B1）。省略時は設定の値")
    p.add_argument("--cart-margin", type=float, default=None, help="手先の上限の余裕（既定 0.95）。省略時は設定の値")
    p = sub.add_parser("task")
    p.add_argument("--experiment", required=True)
    p.add_argument("--condition", required=True)
    p.add_argument("--model", default="R2", choices=sorted(CKPT))
    p.add_argument("--trials", required=True, help="<種の先頭>:<数>")
    p.add_argument("--text", default="全部片付けて")
    p.add_argument("--no-safety", action="store_true")
    p.add_argument("--exec-interval", type=int, default=None, help="塊の実行の行数（既定は configs の値）")
    p = sub.add_parser("e6")
    p.add_argument("--experiment", required=True)
    p.add_argument("--condition", required=True)
    p.add_argument("--model", default="R2", choices=sorted(CKPT))
    p.add_argument("--trials", required=True, help="<種の先頭>:<配置の数>")
    p.add_argument("--cue-mode", default="both", choices=("both", "cue_only"))
    p = sub.add_parser("decide-ckpt")
    p.add_argument("--experiment", default="V2SEL")
    p.add_argument("--model", required=True, choices=sorted(V2_TRAIN))
    p.add_argument("--suffix", default="", help="結果のファイル名の後ろ（回し直しで前の判定を上書きしない）")
    a = ap.parse_args(argv)
    {"run": cmd_run, "task": cmd_task, "e6": cmd_e6, "decide-ckpt": cmd_decide_ckpt}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
