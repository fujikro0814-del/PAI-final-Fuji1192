"""VLA（SmolVLA）の学習の設定を書いて学習を起動する。結果は outputs/h/<項目>.json。

scripts/train_seed.py が、学習の手数と設定の作り方の部品として読む。

    .venv\\Scripts\\python.exe scripts\\train_vla.py train R1v3 [--smoke]   # --smoke は 1000 ステップで最後まで通す
    .venv\\Scripts\\python.exe scripts\\train_vla.py decide                 # 目標位置の入力のずらしの採否

名前の R は立て直しデータを含む学習データ、N は成功データだけ。状態は目標位置の入力つきの 17 次元。
学習のシードは R・N で同じ（configs の train.seed）。目標位置の入力のずらしは、採否（outputs/h/cue_aug_decision.json）が
「採る」のときだけ入れる。
"""
import argparse
import json
import pathlib
import time

from recovla.common import config

CFG = config.load()
OUT = config.path(CFG["paths"]["outputs"]) / "h"


def write(name: str, obj: dict) -> pathlib.Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.json"
    p.write_text(json.dumps({"check": name, "written": time.strftime("%Y-%m-%d %H:%M:%S"), **obj},
                            ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[h] wrote {p}", flush=True)
    return p


STAGE3_STEPS = 20000          # v3 の学習の手数（R・N とも）。学習率の予定は configs のまま
STAGE3_SAVE = 5000
STAGE3_ROUND2_DATA = "data_v3r2.json"   # R2v3・N2v3 の学習データ（scripts/95_s3_round2.py data が作る）


def cmd_train(a) -> None:
    """学習の設定を書いて起動する。名前の末尾 v2・v3 で学習データ（outputs/f/data_*.json）を選ぶ。
    v3 は行動列の読み出しを速くする fast_query を使う（学習の値は同じ）。"""
    from recovla.policy import train_launcher as tl
    ver = a.name[-2:] if a.name[-2:] in ("v2", "v3") else ""
    base = a.name[:-2] if ver else a.name
    data_file = STAGE3_ROUND2_DATA if a.name in ("R2v3", "N2v3") else (f"data_{ver}.json" if ver else "data_cue.json")
    d = json.loads((config.path(CFG["paths"]["outputs"]) / "f" / data_file).read_text(encoding="utf-8"))
    dec_path = OUT / "cue_aug_decision.json"
    if not dec_path.is_file():
        raise SystemExit(f"{dec_path} がない（K1 での手がかりのずらしの採否を先に決める＝掲示板 0054）")
    dec = json.loads(dec_path.read_text(encoding="utf-8"))
    if ver == "v3":
        full_steps, save = STAGE3_STEPS, STAGE3_SAVE
    else:
        rc = CFG["train"]["runs"][base]
        full_steps, save = int(rc["steps"]), int(rc["save_freq"])
    steps = 1000 if a.smoke else full_steps
    stage = {"v2": "Stage 2", "v3": "Stage 3"}.get(ver, "Step H")
    cfg = {"dataset": str(config.path(d["datasets"][base]["dataset"])), "train_scope": CFG["train"]["scope"],
           "batch_size": int(CFG["train"]["batch_size"]), "steps": steps,
           "save_freq": steps if a.smoke else save, "seed": int(CFG["train"]["seed"]),
           "log_freq": int(CFG["train"]["log_freq"]),
           "num_workers": int(a.num_workers if a.num_workers is not None else CFG["train"]["num_workers"]),
           "note": f"{stage} {a.name}{' smoke' if a.smoke else ''}: {steps} steps on "
                   f"{d['datasets'][base]['dataset']} (target cue, cue_augment {'on' if dec['adopt'] else 'off'} per board 0054)"}
    if ver == "v3":
        cfg["fast_query"] = True
    if dec["adopt"]:
        cfg["cue_augment"] = dict(CFG["train"]["cue_augment"])
    tag = f"{a.name}{'_smoke' if a.smoke else ''}"
    cfg_path = OUT / f"train_{tag}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    OUT.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    t0 = time.time()
    code = tl.run(cfg_path)
    runs = sorted(config.path(CFG["paths"]["train_output"]).glob(f"{cfg_path.stem}_*"))
    rec = json.loads((runs[-1] / tl.RUN_RECORD).read_text(encoding="utf-8")) if runs else {}
    write(f"train_{tag}", {"config": str(cfg_path.relative_to(config.ROOT)), "exit": code,
                           "wall_s": round(time.time() - t0, 1), "cue_augment": dec["adopt"],
                           "run_dir": str(runs[-1].relative_to(config.ROOT)) if runs else None,
                           "log_summary": rec.get("log_summary"), "command": rec.get("command"),
                           "checkpoints": sorted(p.name for p in (runs[-1] / "checkpoints").iterdir()
                                                 if p.is_dir()) if runs else None})
    if code != 0:
        raise SystemExit(f"training {tag} exited with {code}")          # 後に続く学習を止める


# ------------------------------------------------------------------ R2・R1+: R1 の 3 万ステップから続けて学習

R1_DATASET = "outputs/datasets/R1cue_20260926-140506"           # R1 の学習データ（統計量の出所）
R1_30K = "outputs/train/train_R1_20260926-153959_20260926-153959/checkpoints/030000/pretrained_model"
SECOND = OUT / "data_second.json"


def cmd_statscopy(a) -> None:
    """学習用の写し <src>_statsR1 を作る。data・images はハードリンク、meta は写して stats.json だけ R1 のものに替える
    （正規化の統計量を R1 に揃えるため）。"""
    import hashlib
    import os
    import shutil
    src = config.path(a.src).resolve()
    r1 = config.path(R1_DATASET).resolve()
    dst = src.with_name(f"{src.name}_statsR1")
    if dst.exists():
        raise SystemExit(f"{dst} already exists")
    n = 0
    for p in src.rglob("*"):
        q = dst / p.relative_to(src)
        if p.is_dir():
            q.mkdir(parents=True, exist_ok=True)
        elif p.relative_to(src).parts[0] == "meta":
            q.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, q)
        else:
            q.parent.mkdir(parents=True, exist_ok=True)
            os.link(p, q)
            n += 1
    shutil.copy2(r1 / "meta" / "stats.json", dst / "meta" / "stats.json")
    conv = json.loads((dst / "meta" / "conversion.json").read_text(encoding="utf-8"))
    sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
    conv["stats_source"] = {"dataset": R1_DATASET, "stats_json_sha256": sha(r1 / "meta" / "stats.json"),
                            "own_stats_json_sha256": sha(src / "meta" / "stats.json"),
                            "note": "B_提案書 §10: 統計量を R1 のものに固定するための学習用の写し（data・images は元と同じ中身）"}
    (dst / "meta" / "conversion.json").write_text(json.dumps(conv, ensure_ascii=False, indent=2), encoding="utf-8")
    d = json.loads(SECOND.read_text(encoding="utf-8")) if SECOND.is_file() else {"datasets": {}}
    d["datasets"][a.name] = {"source": str(src.relative_to(config.ROOT)), "dataset": str(dst.relative_to(config.ROOT)),
                             "linked_files": n, "stats_source": conv["stats_source"]}
    write("data_second", d)


def cmd_train2(a) -> None:
    """R2・R1+ の学習（R1 の 3 万ステップから 1 万ステップ、学習率の予定は configs の train.runs.second_round）。"""
    from recovla.policy import train_launcher as tl
    d = json.loads(SECOND.read_text(encoding="utf-8"))["datasets"][a.name]
    dec = json.loads((OUT / "cue_aug_decision.json").read_text(encoding="utf-8"))
    rc = CFG["train"]["runs"]["second_round"]
    steps = 50 if a.smoke else int(rc["steps"])
    cfg = {"dataset": str(config.path(d["dataset"])), "train_scope": CFG["train"]["scope"],
           "batch_size": int(CFG["train"]["batch_size"]), "steps": steps,
           "save_freq": steps if a.smoke else int(rc["save_freq"]), "seed": int(CFG["train"]["seed"]),
           "log_freq": 10 if a.smoke else int(CFG["train"]["log_freq"]), "num_workers": int(CFG["train"]["num_workers"]),
           "init_policy": str(config.path(R1_30K)),
           "lr_schedule": {"peak": float(rc["optimizer_lr"]), "warmup": int(rc["warmup_steps"]),
                           "decay_steps": int(rc["decay_steps"]), "decay_lr": float(rc["decay_lr"])},
           "note": f"Step H {a.name}{' smoke' if a.smoke else ''}: {steps} steps from R1 30000 on {d['dataset']} "
                   f"(stats of R1, B_提案書 §10; cue_augment {'on' if dec['adopt'] else 'off'} as R1)"}
    if dec["adopt"]:
        cfg["cue_augment"] = dict(CFG["train"]["cue_augment"])
    tag = f"{a.name}{'_smoke' if a.smoke else ''}"
    cfg_path = OUT / f"train_{tag}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    t0 = time.time()
    code = tl.run(cfg_path)
    runs = sorted(config.path(CFG["paths"]["train_output"]).glob(f"{cfg_path.stem}_*"))
    rec = json.loads((runs[-1] / tl.RUN_RECORD).read_text(encoding="utf-8")) if runs else {}
    write(f"train_{tag}", {"config": str(cfg_path.relative_to(config.ROOT)), "exit": code,
                           "wall_s": round(time.time() - t0, 1), "run_dir": str(runs[-1].relative_to(config.ROOT)) if runs else None,
                           "log_summary": rec.get("log_summary"), "command": rec.get("command"),
                           "stats_check_pass": (rec.get("stats_check") or {}).get("pass"),
                           "code_version": rec.get("code_version", {}).get("git_commit"),
                           "git_dirty": rec.get("code_version", {}).get("git_dirty")})
    if code != 0 or not (rec.get("stats_check") or {}).get("pass"):
        raise SystemExit(f"training {tag}: exit {code}, stats_check {(rec.get('stats_check') or {}).get('pass')}")


RULE = {"none_min_success": 29, "noise2_success_more_than": 21, "noise2_along_median_less_than_m": 0.0144}


def cmd_decide(a) -> None:
    """目標位置の入力のずらしの採否。決まり（RULE）は結果を見る前に固定した。"""
    import numpy as np
    k1 = config.path(CFG["paths"]["outputs"]) / "k1"
    L = lambda n: json.loads((k1 / f"{n}.json").read_text(encoding="utf-8"))
    res = {}
    for label, before, after in (("none", "closed_cue", "closed_cueaug"), ("1cm", "closed_cue_noise1cm", "closed_cueaug_noise1cm"),
                                 ("2cm", "closed_cue_noise2cm", "closed_cueaug_noise2cm")):
        row = {}
        for side, name in (("without_aug", before), ("with_aug", after)):
            c = L(name)
            al = [r["grasp_err_along_offset_m"] for r in c["rows"] if r.get("grasp_err_along_offset_m") is not None]
            row[side] = {"successes": sum(r["success"] for r in c["rows"]), "trials": len(c["rows"]),
                         "lifted": c["lifted_any"], "wrong": c["lifted_wrong"], "contact": c["contact_trials"],
                         "along_offset_median_m": float(np.median(al)) if al else None, "checkpoint": c["checkpoint"]}
        res[label] = row
    w = {k: v["with_aug"] for k, v in res.items()}
    checks = {"none_success_ge_29": w["none"]["successes"] >= RULE["none_min_success"],
              "noise2_success_gt_21": w["2cm"]["successes"] > RULE["noise2_success_more_than"],
              "noise2_along_median_lt_1.44cm": (w["2cm"]["along_offset_median_m"] is not None
                                                and w["2cm"]["along_offset_median_m"] < RULE["noise2_along_median_less_than_m"])}
    write("cue_aug_decision", {"rule": RULE, "rule_source": "決裁 0054 の 3（結果を見る前に固定）", "results": res,
                               "checks": checks, "adopt": all(checks.values())})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("decide")
    s = sub.add_parser("train")
    s.add_argument("name", choices=["R1", "N1", "R1v2", "N1v2", "R1v3", "N1v3", "R2v3", "N2v3"],
                   help="R1v2・N1v2: 段階 2（outputs/f/data_v2.json）。R1v3・N1v3: 段階 3 の 1 周目（data_v3.json、0126）。"
                        "R2v3・N2v3: 段階 3 の 2 周目（data_v3r2.json、0139）")
    s.add_argument("--num-workers", type=int, default=None,
                   help="データの読み手の数（既定は設定の値）。値は変わらない（試料の並びは主プロセスの sampler が決める）。"
                        "評価と同時に回すときの主記憶の不足を避ける（0135）")
    s.add_argument("--smoke", action="store_true", help="1000 手で最後まで通す")
    s = sub.add_parser("statscopy")
    s.add_argument("name", choices=["R2", "R1plus"])
    s.add_argument("--src", required=True, help="変換したデータセット（R1+ は R1 のもの）")
    s = sub.add_parser("train2")
    s.add_argument("name", choices=["R2", "R1plus"])
    s.add_argument("--smoke", action="store_true", help="50 手で保存と統計量の一致まで通す")
    a = ap.parse_args(argv)
    {"train": cmd_train, "decide": cmd_decide, "statscopy": cmd_statscopy, "train2": cmd_train2}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
