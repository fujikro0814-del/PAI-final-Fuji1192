"""置く位置の入力を足したモデル Rplace の学習（docs/api/r2_protocol.md 6.4 節）。Rdag の学習の設定から、データセット
（置く位置の入力を足して変換し直した同じ 670 本、scripts/place_relearn/place_data.py）と説明の文だけを替える。
40,000 ステップ・学習率の形・シード 1000・バッチ 32・5,000 ステップごとの保存・num_workers・fast_query・学習する範囲
（行動エキスパートと状態の射影）は Rdag と同じ（違えば止める）。最後のチェックポイントを使う（選ばない）。

    .venv\\Scripts\\python.exe scripts\\place_relearn\\train_place.py --dataset outputs\\datasets\\Rplace_<時刻>            # 確かめだけ
    .venv\\Scripts\\python.exe scripts\\place_relearn\\train_place.py --dataset outputs\\datasets\\Rplace_<時刻> --start
    .venv\\Scripts\\python.exe scripts\\place_relearn\\train_place.py --post-check outputs\\r2c\\train_place\\<実行名>
"""
import argparse
import contextlib
import io
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from recovla.common import config  # noqa: E402

OUT = ROOT / "outputs" / "r2c"
TRAIN_CFG_DIR, TRAIN_OUT, TRAIN_LOGS, WRAP = OUT / "train_place_cfg", OUT / "train_place", OUT / "train_place_logs", OUT / "train_place_wrap"
RDAG_RUN = pathlib.Path(r"C:\PAI\wt_r1\outputs\r1\train\train_Rdags1000_20261021-155742_20261021-155742")
RDAG_FRAMES = 99402
ALLOWED = {"/dataset", "/note"}
NAME = "Rplace"


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


def check_dataset(ds: pathlib.Path, smoke: bool = False) -> dict:
    """変換と照合が通り、置く位置の入力があり、フレーム数が Rdag と同じか（試しの実行のデータでは本数を見ない）。"""
    conv = json.loads((ds / "meta" / "conversion.json").read_text(encoding="utf-8"))
    info = json.loads((ds / "meta" / "info.json").read_text(encoding="utf-8"))
    log = ds.parent / f"convert_{ds.name}.log"
    text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
    problems = []
    if "place_cue" not in conv:
        problems.append("conversion.json に place_cue が無い")
    if not smoke and (int(info["total_frames"]) != RDAG_FRAMES or int(info["total_episodes"]) != 670):
        problems.append(f"こま・本数が Rdag と違う: {info['total_frames']} こま、{info['total_episodes']} 本")
    if "[verify] PASS" not in text:
        problems.append(f"照合の PASS が {log} に無い")
    if problems:
        raise SystemExit("データセットの確かめが通らない:\n  " + "\n  ".join(problems))
    return {"dataset": str(ds), "frames": int(info["total_frames"]), "episodes": int(info["total_episodes"]),
            "state": conv["state"], "manifest_name": (conv.get("manifest") or {}).get("name")}


def build_cfg(ds: dict, smoke: bool) -> tuple:
    base = json.loads((RDAG_RUN / "train_launch_config.json").read_text(encoding="utf-8"))
    cfg = json.loads(json.dumps(base))
    cfg["dataset"] = ds["dataset"]
    cfg["note"] = (f"R2 stage P {NAME}{' smoke' if smoke else ''} (seed {cfg['seed']}): {cfg['steps']} steps on {ds['dataset']} "
                   f"({ds['frames']} frames, the Rdag episodes with the place cue, state {len(ds['state'])}); "
                   f"= Rdag settings; docs/api/r2_protocol.md 6.4")
    if smoke:
        cfg.update(steps=1000, save_freq=1000, lr_schedule=dict(base["lr_schedule"], warmup=40, decay_steps=1000))
    diffs = deep_diff(base, cfg)
    allowed = ALLOWED | ({"/steps", "/save_freq", "/lr_schedule/warmup", "/lr_schedule/decay_steps"} if smoke else set())
    bad = [x for x in diffs if x[0] not in allowed]
    if bad:
        raise SystemExit(f"Rdag の設定と、データ・note のほかでも違う: {bad}")
    return cfg, diffs


def post_check(run_dir: pathlib.Path) -> dict:
    rd = lambda d, r: json.loads((d / r).read_text(encoding="utf-8"))   # noqa: E731
    run = rd(run_dir, "train_run.json")
    cks = sorted(p.name for p in (run_dir / "checkpoints").iterdir() if p.is_dir() and p.name.isdigit())
    last = cks[-1] if cks else None
    tc = rd(run_dir, f"checkpoints/{last}/pretrained_model/train_config.json") if last else {}
    old = rd(RDAG_RUN, "train_run.json")
    keys = ("log_points", "loss_first", "loss_last", "loss_min", "loss_mean_last_10pct", "gpu_mem_allocated_max_gib")
    return {"run": str(run_dir), "exit_code": run.get("exit_code"), "checkpoints": cks, "final_checkpoint": last,
            "final_is_steps": last is not None and int(last) == int(tc.get("steps", -1)),
            "policy_for_eval": str(run_dir / "checkpoints" / last / "pretrained_model") if last else None,
            "loss_summary": {k: {"Rdag": old["log_summary"].get(k), NAME: run["log_summary"].get(k)} for k in keys},
            "note": "最後の保存点を使う（選ばない）。loss は人が見る"}


def main(argv=None) -> int:
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--smoke", action="store_true", help="1,000 手だけ（読み込みと保存の確かめ。評価に使わない）")
    ap.add_argument("--start", action="store_true")
    ap.add_argument("--post-check", metavar="RUN_DIR")
    a = ap.parse_args(argv)
    if a.post_check:
        res = post_check(config.path(a.post_check))
        WRAP.mkdir(parents=True, exist_ok=True)
        (WRAP / f"postcheck_{pathlib.Path(a.post_check).name}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                                                              encoding="utf-8")
        print(json.dumps(res, ensure_ascii=False, indent=1))
        return 0 if res["exit_code"] == 0 and res["final_is_steps"] else 1
    if not a.dataset:
        ap.error("--dataset が要る")
    ds = check_dataset(config.path(a.dataset), a.smoke)
    cfg, diffs = build_cfg(ds, a.smoke)
    print(f"データ {ds['dataset']}（{ds['frames']} こま、状態 {len(ds['state'])}）→ 手数 {cfg['steps']}")
    print("Rdag の設定との違い:")
    for p_, o, n in diffs:
        print(f"  {p_}: {o!r} -> {n!r}")
    from recovla.policy import train_launcher as tl
    tag = f"{NAME}s{cfg['seed']}{'_smoke' if a.smoke else ''}"
    TRAIN_CFG_DIR.mkdir(parents=True, exist_ok=True)
    cfg_path = TRAIN_CFG_DIR / f"train_{tag}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    if not a.start:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = tl.run(cfg_path, output_root=TRAIN_OUT, log_root=TRAIN_LOGS, dry_run=True)
        print(buf.getvalue())
        return code
    spec = __import__("importlib.util").util.spec_from_file_location("s4_seed_for_r2p", ROOT / "scripts" / "train_seed.py")
    sw = __import__("importlib.util").util.module_from_spec(spec)
    spec.loader.exec_module(sw)
    problems = sw.precheck()
    if problems:
        print("起動しない:\n  " + "\n  ".join(problems))
        return 3
    t0 = time.time()
    code = tl.run(cfg_path, output_root=TRAIN_OUT, log_root=TRAIN_LOGS)
    runs = sorted(TRAIN_OUT.glob(f"{cfg_path.stem}_*"))
    WRAP.mkdir(parents=True, exist_ok=True)
    (WRAP / f"train_{tag}.json").write_text(json.dumps(
        {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "name": NAME, "config": str(cfg_path), "exit": code,
         "wall_s": round(time.time() - t0, 1), "dataset": ds, "diffs_from_rdag": diffs,
         "run_dir": str(runs[-1]) if runs else None}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return 0 if code == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
