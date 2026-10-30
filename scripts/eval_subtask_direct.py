"""評価の試行を条件ごとに回し、結果の表と図を出す（1 つのサブタスク、シミュレーションの状態を直接使う以前の実行系）。

試行の並びを作る trial_list は、scripts/eval_trials.py などが部品として読む。

    .venv\\Scripts\\python.exe scripts\\eval_subtask_direct.py run --experiment EXP --condition COND --checkpoint CKPT
        --mode rtc --s 10 --d 2 --trials natural:194000:33 [--induce P2] [--videos 3]
    .venv\\Scripts\\python.exe scripts\\eval_subtask_direct.py report --experiment EXP [--pair A:B]
    .venv\\Scripts\\python.exe scripts\\eval_subtask_direct.py dcal --checkpoint CKPT [--n 60]      # 推論の遅延 d を決める

試行の並び（--trials）:
  natural:<先頭のシード>:<配置の数>   空の箱・既定の開始姿勢の配置それぞれで、机の上の 3 色を目標に
  induced:<先頭のシード>:<数>          シードごとに配置と目標を 1 つ（choose_targets）。わざと起こす失敗と一緒に使う
  selection:<先頭のシード>:<数>        空の箱・既定の開始姿勢、シードごとに目標を 1 つ（チェックポイントの選択など）
出力: outputs/eval/<実験>/<条件>/trial_NNNN.{json,npz}（と先頭の mp4）、集計は outputs/results/<実験>/。
学習と同時に回さない（推論の時間が遅く出て d の測定がずれる）。
"""
import argparse
import json
import math
import os
import pathlib
import time

import numpy as np

from recovla.common import config

CFG = config.load()
# lerobot・huggingface_hub を import する前に決める（ネットワークなし）
os.environ["HF_HOME"] = str(config.path(CFG["paths"]["models_home"]))
os.environ["HF_HUB_OFFLINE"] = "1"
EVAL_OUT = config.path(CFG["paths"]["outputs"]) / "eval"
RES_OUT = config.path(CFG["paths"]["outputs"]) / "results"


def trial_list(spec: str) -> list:
    """[(シード, 配置, 目標の色)]。"""
    from recovla.eval import scene_trial as T
    from recovla.sim import scene
    kind, base, n = spec.split(":")
    base, n = int(base), int(n)
    if kind == "natural":
        out = []
        for s in range(base, base + n):
            lay = scene.sample_layout(s, "empty", start="home")
            out += [(s, lay, c) for c in lay.table_colors]
        return out
    seeds_ = list(range(base, base + n))
    if kind == "induced":
        lays = [scene.sample_layout(s) for s in seeds_]
    elif kind == "selection":
        lays = [scene.sample_layout(s, "empty", start="home") for s in seeds_]
    else:
        raise SystemExit(f"--trials {spec!r}: kind must be natural, induced or selection")
    return list(zip(seeds_, lays, T.choose_targets(seeds_, lays)))


def _set_num_steps(pol, n) -> None:
    """流れの積分の刻み数（SmolVLA の num_steps、既定 10）を推論だけ変える。"""
    if n is None:
        return
    from recovla.policy.runner import _base_policy
    _base_policy(pol.policy).config.num_steps = int(n)


def cmd_run(a) -> None:
    from recovla.eval import induce as I
    from recovla.eval import scene_trial as T
    from recovla.eval.closed_loop import write_mp4
    from recovla.policy.runner import RuntimeConfig, SceneRunner
    from recovla.policy.scene_policy import ScenePolicy
    from recovla.sim.rig import SimRig
    out = EVAL_OUT / a.experiment / a.condition
    if out.exists() and any(out.glob("trial_*.json")):
        raise SystemExit(f"{out} already has trials (use a new --condition)")
    pol = ScenePolicy(pathlib.Path(a.checkpoint))
    _set_num_steps(pol, a.num_steps)
    rt = RuntimeConfig(a.mode, a.s, a.d if a.mode != "sync" else None,
                       execution_horizon=a.horizon or int(CFG["runtime"]["rtc_guidance_horizon"]))
    runner = SceneRunner(pol, rt, {"schedule": a.schedule or CFG["runtime"]["rtc_schedule"],
                                   "max_guidance_weight": CFG["runtime"]["rtc_max_guidance_weight"]})
    trials = trial_list(a.trials)
    rig = SimRig(render=True)
    if a.safety != "config":                         # 安全フィルタの入切。既定は configs の safety_filter.enabled
        rig.safety.enabled = a.safety == "on"
    rows = []
    t0 = time.perf_counter()
    try:
        for i, (seed, lay, tgt) in enumerate(trials):
            runner.start_trial(seed)
            ind = I.Inducer(a.induce, seed, lay, tgt, rig) if a.induce else None
            meta, arr, video = T.run_trial(rig, lay, tgt, runner, {
                "trial": i, "seed": seed, "experiment": a.experiment, "condition": a.condition,
                "model": {"name": a.model or a.condition, "checkpoint": str(a.checkpoint)},
                "runtime": runner.runtime_record()}, inducer=ind)
            meta["input_check"] = pol.input_check
            T.write_trial(out, i, meta, arr)
            if i < a.videos:
                write_mp4(out / f"trial_{i:04d}_raw.mp4", video, 20)
            rows.append({"trial": i, "seed": seed, "target": tgt, "success": meta["success"],
                         "induce": meta["induce"]["kind"], "established": meta["induce"]["established"]})
            print(f"[g] {a.condition} {i:3d} seed {seed} {tgt:5s} success {meta['success']} "
                  f"induce {meta['induce']['kind']} fired {meta['induce']['fired']} est {meta['induce']['established']}",
                  flush=True)
    finally:
        rig.close()
    (out / "run.json").write_text(json.dumps({
        "experiment": a.experiment, "condition": a.condition, "checkpoint": str(a.checkpoint), "trials_spec": a.trials,
        "induce": a.induce, "runtime": {**runner.runtime_record(), "safety_filter": rig.safety.enabled}, "n": len(rows),
        "successes": sum(r["success"] for r in rows), "wall_s": round(time.perf_counter() - t0, 1),
        "written": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False, indent=2), encoding="utf-8")


def cmd_report(a) -> None:
    from recovla.eval import report
    base = EVAL_OUT / a.experiment
    dirs = [d for d in sorted(base.iterdir()) if d.is_dir() and any(d.glob("trial_*.json"))]
    rows = report.collect(dirs)
    pairs = [tuple(p.split(":")) for p in (a.pair or [])]
    info = report.write_all(RES_OUT / a.experiment, rows, a.experiment, pairs)
    print(json.dumps(info, ensure_ascii=False, indent=1))


def cmd_dcal(a) -> None:
    """推論の遅延 d（runtime.delay_rule）を決める。RTC ありの推論の時間を n 回測り（最初の 1 回を除く）、
    d = ceil(p95 / 0.1 s)。選択用のシードの配置で、方策を閉ループで回して測る。"""
    from recovla.eval import scene_trial as T
    from recovla.policy.runner import RuntimeConfig, SceneRunner
    from recovla.policy.scene_policy import ScenePolicy
    from recovla.sim.rig import SimRig
    rule = CFG["runtime"]["delay_rule"]
    pol = ScenePolicy(pathlib.Path(a.checkpoint))
    _set_num_steps(pol, a.num_steps)
    rt = RuntimeConfig("rtc", int(CFG["runtime"]["exec_interval"]), 1,
                       execution_horizon=int(CFG["runtime"]["rtc_guidance_horizon"]))
    runner = SceneRunner(pol, rt, {"schedule": CFG["runtime"]["rtc_schedule"],
                                   "max_guidance_weight": CFG["runtime"]["rtc_max_guidance_weight"]})
    walls, parts, trials_run = [], [], 0
    rig = SimRig(render=True)
    try:
        for seed, lay, tgt in trial_list(f"selection:{a.seed}:{a.max_trials}"):
            runner.start_trial(seed)
            T.run_trial(rig, lay, tgt, runner, {"trial": trials_run, "seed": seed, "experiment": "dcal",
                                                "condition": "dcal"}, render=True)
            trials_run += 1
            inf = [e for e in runner.trace()["inference"] if e["wall_s"] is not None]
            walls += [e["wall_s"] for e in inf]
            parts += [e["wall_breakdown_s"] for e in inf]
            if len(walls) - 1 >= a.n:
                break
    finally:
        rig.close()
    used_seeds = [a.seed, a.seed + trials_run - 1]
    w = np.array(walls[1:])                         # 最初の 1 回（ウォームアップ）を除く
    p95 = float(np.percentile(w, float(rule["percentile"])))
    d = int(math.ceil(p95 / float(rule["action_dt_s"])))
    res = {"n": int(w.size), "trials": trials_run, "seeds_used": used_seeds, "wall_mean_s": float(w.mean()), "wall_p95_s": p95,
           "wall_max_s": float(w.max()),
           "breakdown_mean_s": {k: float(np.mean([p[k] for p in parts[1:]])) for k in ("preprocess", "policy")},
           "d": d, "stop": d > int(rule["max_d"]), "rule": rule,
           "checkpoint": str(a.checkpoint), "tf32": pol.config.get("tf32"),
           "num_steps": runner.runtime_record()["num_steps"],
           "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    RES_OUT.mkdir(parents=True, exist_ok=True)
    out = RES_OUT / f"dcal{a.tag}.json"
    if a.tag and out.exists():
        raise SystemExit(f"{out} already exists")
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=1))


def cmd_induce_script(a) -> None:
    """わざと起こす失敗の下見。方策の代わりに生成プログラムを動かし（GPU なし）、種類ごとに n 回、発動と実際に失敗した割合を数える。
    方策で回すときは run --induce。"""
    import collections
    import numpy as np
    from recovla.common import seeds
    from recovla.eval import induce as I
    from recovla.eval import scene_trial as T
    from recovla.expert import script as S
    from recovla.sim.rig import SimRig

    class ScriptPolicy:
        def __init__(self, rig, target, seed):
            self.rig, self.target = rig, target
            self.ex = S.Expert(S.sample_params(seeds.script_rng(seed, target, 0)), T.ACTION_DT)

        def __call__(self, k, frame, raw, task):
            cmd = self.ex.act(self.rig.truth(self.target))
            out = np.zeros(7)
            out[:3] = cmd.vel * T.ACTION_DT
            out[6] = 1.0 if bool(self.rig.controller.gripper_closed) != bool(cmd.press) else -1.0
            return out

    rig = SimRig(render=False)
    res = {}
    try:
        for kind in I.KINDS:
            rows = []
            for seed, lay, tgt in trial_list(f"induced:{a.seed}:{a.n}"):
                ind = I.Inducer(kind, seed, lay, tgt, rig)
                meta, arr, _ = T.run_trial(rig, lay, tgt, ScriptPolicy(rig, tgt, seed),
                                           {"trial": 0, "seed": seed, "experiment": "induce_script", "condition": kind},
                                           render=False, inducer=ind)
                r = meta["induce"]
                land = (r.get("info") or {}).get("landing") or {}
                clear = land.get("clearance_m") or {}
                near = sorted(k for k, v in clear.items() if v < float(CFG["inject"]["landing"]["min_clearance_m"]))
                ti = ("red", "green", "blue").index(tgt)
                rows.append({"seed": seed, "fired": r["fired"], "established": r["established"], "reason": r["reason"],
                             "success": meta["success"], "t_fire": r["t_fire"], "layout_kind": lay.kind,
                             "landing_fail": land.get("landing_fail"), "too_close_to": near,
                             "tilt_deg": land.get("tilt_deg"),
                             "final_target_speed": float(np.linalg.norm(arr["cube_linvel"][-1, ti])),
                             "final_target_z": float(arr["cube_pos"][-1, ti, 2]), "info": r.get("info")})
            n = len(rows)
            res[kind] = {"n": n, "fired": sum(r["fired"] for r in rows), "established": sum(r["established"] for r in rows),
                         "success": sum(r["success"] for r in rows),
                         "reasons": dict(collections.Counter(r["reason"] for r in rows if not r["established"])),
                         "establish_rate": sum(r["established"] for r in rows) / n, "rows": rows}
            print(kind, {k: v for k, v in res[kind].items() if k != "rows"}, flush=True)
    finally:
        rig.close()
    RES_OUT.mkdir(parents=True, exist_ok=True)
    (RES_OUT / "induce_script.json").write_text(json.dumps({
        "note": "台本（真値を読む Expert）を方策の代わりにした誘発の下見。種は Step G の検査の帯", "seed": a.seed,
        "induction_min_rate": CFG["eval"]["induction_min_rate"], "results": res,
        "written": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False, indent=2), encoding="utf-8")


# ------------------------------------------------ 判断の決まり（結果を見る前に固定した）

RTC_COMBOS = [(10, "LINEAR"), (10, "EXP"), (40, "LINEAR"), (40, "EXP")]
RTC_SUCCESS_WITHIN = 3          # 最良の組の成功数から 3 回以内（30 回中）
RTC_SEAM_SAME_REL = 0.10        # 行動列の境界での速度の跳びの中央値が最小の 10% 以内なら「同じ」
RTC_DEFAULT = (10, "EXP")       # 同点のときに近い方を採る既定


def rtc_condition(h: int, sched: str) -> str:
    return f"E{h}_{sched}"


def cmd_decide_rtc(a) -> None:
    """RTC の設定（ガイドの長さ 10・40 × 重みの減り方 LINEAR・EXP）を、outputs/eval/RTC2x2/<条件>/ の試行から選ぶ。"""
    from recovla.eval import report
    base = EVAL_OUT / "RTC2x2"
    rows = {}
    for h, sc in RTC_COMBOS:
        c = rtc_condition(h, sc)
        rows[c] = report.collect([base / c])
    stat = {}
    for (h, sc) in RTC_COMBOS:
        c = rtc_condition(h, sc)
        rs = rows[c]
        est = [r for r in rs if r["induce_established"]]
        seam = [r["seam_jump_mean"] for r in rs if r["seam_jump_mean"] is not None and not math.isnan(r["seam_jump_mean"])]
        react = [r["reaction_time_s"] for r in est if r["reaction_time_s"] is not None and not math.isnan(r["reaction_time_s"])]
        stat[c] = {"horizon": h, "schedule": sc, "n": len(rs), "successes": sum(bool(r["success"]) for r in rs),
                   "established": len(est), "establish_rate": len(est) / len(rs) if rs else None,
                   "seam_jump_median": float(np.median(seam)) if seam else float("nan"),
                   "reaction_time_median_s": float(np.median(react)) if react else float("nan"),
                   "seeds": sorted({r["seed"] for r in rs})}
    seeds_sets = {tuple(v["seeds"]) for v in stat.values()}
    best = max(v["successes"] for v in stat.values())
    step1 = [c for c, v in stat.items() if v["successes"] >= best - RTC_SUCCESS_WITHIN]
    smin = min(stat[c]["seam_jump_median"] for c in step1)
    step2 = [c for c in step1 if stat[c]["seam_jump_median"] <= smin * (1 + RTC_SEAM_SAME_REL)]
    if len(step2) > 1:
        rmin = min(stat[c]["reaction_time_median_s"] for c in step2)
        step3 = [c for c in step2 if stat[c]["reaction_time_median_s"] == rmin]
    else:
        step3 = step2

    def closeness(c):
        v = stat[c]
        return (v["horizon"] != RTC_DEFAULT[0], v["schedule"] != RTC_DEFAULT[1])
    chosen = sorted(step3, key=closeness)[0]
    res = {"rule": {"success_within": RTC_SUCCESS_WITHIN, "seam_same_rel": RTC_SEAM_SAME_REL, "default": RTC_DEFAULT,
                    "success_denominator": "P2 が不成立の試行も含めた全試行（0063 の 2）", "source": "0062・0063"},
           "same_seeds_all_combos": len(seeds_sets) == 1, "stats": stat, "step1_success": step1,
           "step2_seam": step2, "step3_reaction": step3, "chosen": chosen,
           "chosen_horizon": stat[chosen]["horizon"], "chosen_schedule": stat[chosen]["schedule"],
           "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    RES_OUT.mkdir(parents=True, exist_ok=True)
    (RES_OUT / "rtc_decision.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "stats"}, ensure_ascii=False, indent=1))


def cmd_decide_ckpt(a) -> None:
    """チェックポイントの選択。成功数の多い方、同数なら後のチェックポイント。
    R1・N1 は 20000 と 30000 ステップ、R2・R1+（R1plus）は 5000 と 10000 ステップを、同じ配置・同じ設定で比べる。"""
    from recovla.eval import report
    base = EVAL_OUT / f"select_{a.model}"
    early, late = (20000, 30000) if a.model in ("R1", "N1") else (5000, 10000)
    out = {}
    for step in (early, late):
        rs = report.collect([base / f"{a.model}_{step}"])
        out[step] = {"n": len(rs), "successes": sum(bool(r["success"]) for r in rs),
                     "seeds": sorted({r["seed"] for r in rs}),
                     "runtime": sorted({(r["mode"], r["d"], r["safety_filter"]) for r in rs})}
    chosen = early if out[early]["successes"] > out[late]["successes"] else late
    res = {"model": a.model, "rule": f"成功数の多い方、同数なら {late} 手（手順書 Step H の 2・0062・0063・0070 の 3）",
           "candidates": out, "same_seeds": out[early]["seeds"] == out[late]["seeds"], "chosen_step": chosen,
           "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    RES_OUT.mkdir(parents=True, exist_ok=True)
    (RES_OUT / f"ckpt_decision_{a.model}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("run")
    s.add_argument("--experiment", required=True)
    s.add_argument("--condition", required=True)
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--model", default=None)
    s.add_argument("--mode", choices=["sync", "naive", "rtc"], default=CFG["runtime"]["mode"])
    s.add_argument("--s", type=int, default=int(CFG["runtime"]["exec_interval"]))
    s.add_argument("--d", type=int, default=CFG["runtime"]["delay_steps"])
    s.add_argument("--horizon", type=int, default=None)
    s.add_argument("--schedule", default=None)
    s.add_argument("--trials", required=True)
    s.add_argument("--induce", choices=["P1", "P2", "P3"], default=None)
    s.add_argument("--videos", type=int, default=3)
    s.add_argument("--num-steps", type=int, default=None, help="流れの積分の刻み数（既定は保存点の設定＝10）")
    s.add_argument("--safety", choices=["config", "on", "off"], default="config", help="安全フィルタの入切（0080）")

    s = sub.add_parser("report")
    s.add_argument("--experiment", required=True)
    s.add_argument("--pair", action="append", help="条件 a:b（同じ種で対にする）")
    s = sub.add_parser("dcal")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--n", type=int, default=100, help="最初の 1 回を除いた回数（0062・0063: 100 回以上）")
    s.add_argument("--seed", type=int, default=198600, help="0062・0063 の割り当て（198600〜、使った範囲を記録）")
    s.add_argument("--max-trials", type=int, default=100)
    s.add_argument("--num-steps", type=int, default=None)
    s.add_argument("--tag", default="", help="結果を dcal<tag>.json に書く（元の dcal.json を上書きしない）")
    sub.add_parser("decide-rtc")
    s = sub.add_parser("decide-ckpt")
    s.add_argument("--model", choices=["R1", "N1", "R2", "R1plus"], required=True)
    s = sub.add_parser("induce-script")
    s.add_argument("--seed", type=int, default=198100, help="Step G の検査の帯 198000〜198999")
    s.add_argument("--n", type=int, default=20)
    a = ap.parse_args(argv)
    {"run": cmd_run, "report": cmd_report, "dcal": cmd_dcal, "induce-script": cmd_induce_script,
     "decide-rtc": cmd_decide_rtc, "decide-ckpt": cmd_decide_ckpt}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
