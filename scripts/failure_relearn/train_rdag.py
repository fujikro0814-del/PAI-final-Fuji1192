"""自分の失敗から学び直したモデル Rdag の学習の入口（docs/api/r1_protocol.md 4・4.5 節）。
Rreal の起動の設定（その学習の train_launch_config.json）から、データセット・手数・note・学習率の形だけを替える。
シード 1000・バッチ 32・保存 5000 ステップごと・num_workers・fast_query・学習する範囲（行動エキスパートと状態の射影）は Rreal と同じ
（違えば止める）。

使い方（既定は dry-run。学習を始めるのは --start のときだけ。GPU を使う。前景で 1 本だけ）:
    python scripts/failure_relearn/train_rdag.py                                   # outputs/r1/data.json のデータで、渡す設定と Rreal との違いを出す
    python scripts/failure_relearn/train_rdag.py --dataset outputs/datasets/R1pilot_<日時> --dry-run   # data.json の前の確かめ（試しのデータで）
    python scripts/failure_relearn/train_rdag.py --start
    python scripts/failure_relearn/train_rdag.py --post-check outputs/r1/train/<実行名>

手数の決まり（4.5 節、学習の前に固定）: 手数 = ceil(フレーム数 × 20000 / 50659 / 1000) × 1000（Rreal の 12.6 周にそろえる）。
最後のチェックポイントを使う（チェックポイントを選ばない）。
学習率の形（--lr-schedule、既定 shape）: Rreal は 20,000 ステップ < 減衰の手数 25,000 なので、LeRobot が予熱 800 ステップ・減衰 20,000 ステップに
縮めて、最後の手で下限 2.5e-6 に着く。shape は同じ形（予熱 = 手数の 4%、減衰 = 手数）を lr_schedule で渡す。
default は渡さない（予熱 1,000・減衰 25,000 のまま。手数が 25,000 を超えると残りは下限の学習率で回る）。
書くもの（outputs/r1/ 以下だけ）: train_wrap/dryrun_Rdag.json、--start は train_cfg/・train/<実行名>/・train_logs/・train_wrap/train_Rdag.json。
"""
import argparse
import contextlib
import hashlib
import io
import json
import math
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from recovla.common import config  # noqa: E402

R1 = ROOT / "outputs" / "r1"
WRAP = R1 / "train_wrap"
TRAIN_CFG_DIR = R1 / "train_cfg"
TRAIN_OUT = R1 / "train"
TRAIN_LOGS = R1 / "train_logs"
DATA = R1 / "data.json"
RREAL_RUN = "outputs/real/regen/train/train_Rreals1000_20261014-082146_20261014-082146"
RREAL_FRAMES, RREAL_STEPS, BATCH = 50659, 20000, 32
RREAL_PEAK_LR, RREAL_DECAY_LR, RREAL_WARMUP_FRAC = 1e-4, 2.5e-6, 800 / 20000
ALLOWED = {"/dataset", "/steps", "/note", "/lr_schedule"}
NAME = "Rdag"


def sha256_file(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def deep_diff(a, b, prefix=""):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append((f"{prefix}/{k}", "<なし>", b[k]))
            elif k not in b:
                out.append((f"{prefix}/{k}", a[k], "<なし>"))
            else:
                out += deep_diff(a[k], b[k], f"{prefix}/{k}")
    elif a != b:
        out.append((prefix, a, b))
    return out


def steps_for(frames: int) -> int:
    return int(math.ceil(frames * RREAL_STEPS / RREAL_FRAMES / 1000.0) * 1000)


def base_config() -> dict:
    p = config.path(RREAL_RUN) / "train_launch_config.json"
    if not p.is_file():
        raise SystemExit(f"Rreal の起動の設定がない: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def check_rreal_schedule() -> dict:
    """Rreal の train_config.json の学習率の値が、この入口の前提（1e-4・2.5e-6・予熱 1000・減衰 25000）と同じか。"""
    tc = json.loads((config.path(RREAL_RUN) / "checkpoints" / "020000" / "pretrained_model" / "train_config.json")
                    .read_text(encoding="utf-8"))
    pol = tc["policy"]
    got = {"peak": pol["optimizer_lr"], "decay_lr": pol["scheduler_decay_lr"], "warmup": pol["scheduler_warmup_steps"],
           "decay_steps": pol["scheduler_decay_steps"], "steps": tc["steps"]}
    want = {"peak": RREAL_PEAK_LR, "decay_lr": RREAL_DECAY_LR, "warmup": 1000, "decay_steps": 25000, "steps": RREAL_STEPS}
    if got != want:
        raise SystemExit(f"Rreal の学習率の設定が前提と違う: {got}")
    return got


def dataset_choice(a) -> dict:
    if a.dataset:
        ds = config.path(a.dataset)
        if a.start:
            raise SystemExit("--dataset は dry-run の確かめだけ（本番は outputs/r1/data.json のデータ）")
        info = json.loads((ds / "meta" / "info.json").read_text(encoding="utf-8"))
        return {"dataset": str(pathlib.Path(a.dataset)).replace("\\", "/"), "frames": int(info["total_frames"]),
                "from": "--dataset（dry-run の確かめ）"}
    if not DATA.is_file():
        raise SystemExit(f"{DATA} がない（先に scripts/failure_relearn/failure_data.py data）")
    d = json.loads(DATA.read_text(encoding="utf-8"))
    if not d.get("pass"):
        raise SystemExit("outputs/r1/data.json の確かめが通っていない（学習しない）")
    ds = d["dataset"]
    if not (ds["convert_exit"] == 0 and ds["verify_exit"] == 0 and ds["verify_pass"]):
        raise SystemExit("データセットの変換・照合が通っていない")
    mp = ROOT / ds["manifest"]
    if sha256_file(mp) != ds["manifest_sha256"]:
        raise SystemExit("マニフェストが変換したときと違う")
    conv = json.loads((config.path(ds["dataset"]) / "meta" / "conversion.json").read_text(encoding="utf-8"))
    if conv.get("manifest") != json.loads(mp.read_text(encoding="utf-8")):
        raise SystemExit("conversion.json のマニフェストの写しがファイルと一致しない")
    return {"dataset": ds["dataset"], "frames": int(ds["frames"]), "from": "outputs/r1/data.json",
            "manifest": ds["manifest"], "manifest_sha256": ds["manifest_sha256"], "recovery_share": d.get("recovery_share")}


def build_cfg(ds: dict, lr: str, smoke: bool) -> tuple:
    base = base_config()
    cfg = json.loads(json.dumps(base))
    steps = 1000 if smoke else steps_for(ds["frames"])
    cfg["dataset"] = str(config.path(ds["dataset"]))
    cfg["steps"] = steps
    if smoke:
        cfg["save_freq"] = steps
    if lr == "shape":
        cfg["lr_schedule"] = {"peak": RREAL_PEAK_LR, "warmup": int(round(RREAL_WARMUP_FRAC * steps)), "decay_steps": steps,
                              "decay_lr": RREAL_DECAY_LR}
    cfg["note"] = (f"R1 {NAME}{' smoke' if smoke else ''} (seed {cfg['seed']}): {steps} steps on {ds['dataset']} "
                   f"({ds['frames']} frames, epochs {steps * BATCH / ds['frames']:.2f} = Rreal {RREAL_STEPS * BATCH / RREAL_FRAMES:.2f}); "
                   f"= Rreal settings, lr {lr}; docs/api/r1_protocol.md 4.5")
    diffs = deep_diff(base, cfg)
    allowed = ALLOWED | ({"/save_freq"} if smoke else set())
    bad = [x for x in diffs if x[0] not in allowed]
    if bad:
        raise SystemExit(f"Rreal の設定と、データ・手数・note・学習率の形のほかでも違う: {bad}")
    return cfg, base, diffs


def launcher_dry_run(cfg: dict) -> tuple:
    from recovla.policy import train_launcher as tl
    d = WRAP / "dryrun_cfg"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"train_{NAME}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = tl.run(p, output_root=TRAIN_OUT, log_root=TRAIN_LOGS, dry_run=True)
    text = buf.getvalue()
    if code != 0 or " command:" not in text:
        raise SystemExit(f"起動器の dry-run が失敗: exit {code}\n{text}")
    cmd = [ln.strip() for ln in text.split(" command:\n", 1)[1].splitlines() if ln.strip()]
    return p, cmd, text


def post_check(run_dir: pathlib.Path, steps_expected: int = None) -> dict:
    rd = lambda d, r: json.loads((d / r).read_text(encoding="utf-8"))   # noqa: E731
    run = rd(run_dir, "train_run.json")
    cks = sorted(p.name for p in (run_dir / "checkpoints").iterdir() if p.is_dir() and p.name.isdigit())
    last = cks[-1] if cks else None
    tc = rd(run_dir, f"checkpoints/{last}/pretrained_model/train_config.json") if last else {}
    old = rd(config.path(RREAL_RUN), "train_run.json")
    keys = ("log_points", "loss_first", "loss_last", "loss_min", "loss_mean_last_10pct", "gpu_mem_allocated_max_gib")
    return {"run": str(run_dir), "exit_code": run.get("exit_code"), "checkpoints": cks, "final_checkpoint": last,
            "final_is_steps": last is not None and int(last) == int(tc.get("steps", -1)),
            "policy_for_eval": str(run_dir / "checkpoints" / last / "pretrained_model") if last else None,
            "loss_summary": {k: {"Rreal": old["log_summary"].get(k), NAME: run["log_summary"].get(k)} for k in keys},
            "note": "最後の保存点を使う（選ばない）。loss は人が見る"}


def main(argv=None) -> int:
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default=None, help="dry-run の確かめだけ: data.json の代わりのデータセット")
    ap.add_argument("--lr-schedule", default="shape", choices=("shape", "default"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--start", action="store_true")
    ap.add_argument("--post-check", metavar="RUN_DIR")
    a = ap.parse_args(argv)
    if a.post_check:
        res = post_check(config.path(a.post_check))
        WRAP.mkdir(parents=True, exist_ok=True)
        (WRAP / f"postcheck_{pathlib.Path(a.post_check).name}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(res, ensure_ascii=False, indent=1))
        return 0 if res["exit_code"] == 0 and res["final_is_steps"] else 1
    if a.start and a.dry_run:
        ap.error("--start と --dry-run は同時に使えない")
    sched = check_rreal_schedule()
    ds = dataset_choice(a)
    cfg, base, diffs = build_cfg(ds, a.lr_schedule, a.smoke)
    print(f"データ {ds['dataset']}（{ds['frames']} こま、{ds['from']}）→ 手数 {cfg['steps']}"
          f"（{cfg['steps'] * BATCH / ds['frames']:.2f} 周、Rreal {RREAL_STEPS * BATCH / RREAL_FRAMES:.2f} 周）")
    print("Rreal の設定との違い:")
    for p_, o, n in diffs:
        print(f"  {p_}: {o!r} -> {n!r}")
    tag = f"{NAME}s{cfg['seed']}{'_smoke' if a.smoke else ''}"
    if a.start:
        spec = __import__("importlib.util").util.spec_from_file_location("s4_seed_for_r1", ROOT / "scripts" / "train_seed.py")
        sw = __import__("importlib.util").util.module_from_spec(spec)
        spec.loader.exec_module(sw)
        problems = sw.precheck()
        if problems:
            print("起動しない:\n  " + "\n  ".join(problems))
            return 3
        from recovla.policy import train_launcher as tl
        TRAIN_CFG_DIR.mkdir(parents=True, exist_ok=True)
        cfg_path = TRAIN_CFG_DIR / f"train_{tag}_{time.strftime('%Y%m%d-%H%M%S')}.json"
        cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        t0 = time.time()
        code = tl.run(cfg_path, output_root=TRAIN_OUT, log_root=TRAIN_LOGS)
        runs = sorted(TRAIN_OUT.glob(f"{cfg_path.stem}_*"))
        WRAP.mkdir(parents=True, exist_ok=True)
        (WRAP / f"train_{tag}.json").write_text(json.dumps(
            {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "name": NAME, "config": str(cfg_path.relative_to(ROOT)), "exit": code,
             "wall_s": round(time.time() - t0, 1), "dataset": ds, "diffs_from_rreal": diffs, "rreal_schedule": sched,
             "run_dir": str(runs[-1].relative_to(ROOT)) if runs else None}, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8")
        return 0 if code == 0 else 2
    cfg_path, cmd, _ = launcher_dry_run(cfg)
    out = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "name": NAME, "dataset": ds, "launcher_config": cfg, "rreal_config": base,
           "diffs_from_rreal": diffs, "rreal_schedule": sched, "lr_schedule_mode": a.lr_schedule, "command": cmd,
           "hours_estimate": round(cfg["steps"] / RREAL_STEPS * 2.38, 1), "dry_run_config": str(cfg_path.relative_to(ROOT))}
    WRAP.mkdir(parents=True, exist_ok=True)
    (WRAP / f"dryrun_{NAME}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("起動器の引数:")
    for c in cmd:
        print(f"  {c}")
    print(f"学習の時間の見込み: 約 {out['hours_estimate']} 時間（Rreal の 2 時間 23 分 × 手数の比）")
    print(f"[r1] wrote {WRAP / f'dryrun_{NAME}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
