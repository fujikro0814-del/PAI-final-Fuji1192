"""自分の成功も足したモデル R2a の学習の入口（docs/api/r2_protocol.md 5 節）。Rdag の最後のチェックポイントから続けて学習する。
Rdag の起動の設定（その学習の train_launch_config.json）から、データセット・手数・note・学習率の形・init_policy だけを替える。
シード 1000・バッチ 32・保存 5000 ステップごと・num_workers・fast_query・学習する範囲は Rdag と同じ（違えば止める）。
続けて学習するときの決まり: 最適化は新しく始め、学習率は予熱 200 ステップ → 最大 3e-5 → 手数の終わりで 2.5e-6。データセットの
統計量は Rdag のものに置き換えた写し（scripts/success_relearn/selfroll_data.py data が作る）を使い、学習の後に起動器が正規化の値を Rdag とビットで
照らす（stats_check）。

使い方（既定は dry-run。学習を始めるのは --start のときだけ。GPU を使う。前景で 1 本だけ）:
    python scripts/success_relearn/train_r2a.py                          # outputs/r2/data.json のデータで、渡す設定と Rdag との違いを出す
    python scripts/success_relearn/train_r2a.py --smoke --start          # 50 ステップの試し（データと続きの学習が動くかだけ）
    python scripts/success_relearn/train_r2a.py --start
    python scripts/success_relearn/train_r2a.py --post-check outputs/r2/train/<実行名>

手数（学習の前に固定）: 10,000 ステップ。最後のチェックポイントを使う（チェックポイントを選ばない）。
書くもの（outputs/r2/ 以下だけ）: train_wrap/dryrun_R2a.json、--start は train_cfg/・train/<実行名>/・train_logs/・train_wrap/train_R2a.json。
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
sys.path.insert(0, str(ROOT / "scripts" / "failure_relearn"))

import train_rdag as TR  # noqa: E402
from recovla.common import config  # noqa: E402

R2 = ROOT / "outputs" / "r2"
WRAP = R2 / "train_wrap"
TRAIN_CFG_DIR = R2 / "train_cfg"
TRAIN_OUT = R2 / "train"
TRAIN_LOGS = R2 / "train_logs"
DATA = R2 / "data.json"
RDAG_HOME = pathlib.Path(r"C:\PAI\wt_r1")                    # Rdag を学習した作業場所（読むだけ）
RDAG_RUN = "outputs/r1/train/train_Rdags1000_20261021-155742_20261021-155742"
RDAG_CKPT = RDAG_RUN + "/checkpoints/040000/pretrained_model"
RDAG_SHA256 = "93f74cc86e81a00b6c8610b8fbb5c576e10fa536abda7db646a960e8f0c60233"
STEPS, SMOKE_STEPS = 10000, 50
LR = {"peak": 3.0e-5, "warmup": 200, "decay_steps": STEPS, "decay_lr": 2.5e-6}
ALLOWED = {"/dataset", "/steps", "/note", "/lr_schedule", "/init_policy"}
NAME = "R2a"


def rdag_path(rel: str) -> pathlib.Path:
    """この作業場所にあればそれ、無ければ RDAG_HOME の同じ場所。"""
    p = ROOT / rel
    return p if p.exists() else RDAG_HOME / rel


def base_config() -> dict:
    p = rdag_path(RDAG_RUN) / "train_launch_config.json"
    if not p.is_file():
        raise SystemExit(f"Rdag の起動の設定がない: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def check_init() -> str:
    ck = rdag_path(RDAG_CKPT)
    sha = TR.sha256_file(ck / "model.safetensors")
    if sha != RDAG_SHA256:
        raise SystemExit(f"Rdag の保存点の SHA-256 が記録と違う: {sha}")
    return str(ck)


def dataset_choice() -> dict:
    if not DATA.is_file():
        raise SystemExit(f"{DATA} がない（先に scripts/success_relearn/selfroll_data.py data）")
    d = json.loads(DATA.read_text(encoding="utf-8"))
    if not d.get("pass"):
        raise SystemExit("outputs/r2/data.json の確かめが通っていない（学習しない）")
    ds = d["dataset"]
    if TR.sha256_file(ROOT / ds["manifest"]) != ds["manifest_sha256"]:
        raise SystemExit("マニフェストが変換したときと違う")
    tds = config.path(d["train_dataset"])
    conv = json.loads((tds / "meta" / "conversion.json").read_text(encoding="utf-8"))
    src = conv.get("stats_source") or {}
    if src.get("stats_json_sha256") != TR.sha256_file(tds / "meta" / "stats.json"):
        raise SystemExit("学習用の写しの stats.json が Rdag の統計量と一致しない")
    return {"dataset": d["train_dataset"], "frames": int(ds["frames"]), "converted": ds["dataset"],
            "manifest": ds["manifest"], "manifest_sha256": ds["manifest_sha256"], "self_used": d.get("self_used"),
            "self_share": d.get("self_share"), "stats_source": src}


def build_cfg(ds: dict, init: str, smoke: bool) -> tuple:
    base = base_config()
    cfg = json.loads(json.dumps(base))
    steps = SMOKE_STEPS if smoke else STEPS
    cfg["dataset"] = str(config.path(ds["dataset"]))
    cfg["steps"] = steps
    cfg["init_policy"] = init
    cfg["lr_schedule"] = dict(LR, decay_steps=steps)
    if smoke:
        cfg["save_freq"] = steps
        cfg["log_freq"] = 10
    cfg["note"] = (f"R2 {NAME}{' smoke' if smoke else ''} (seed {cfg['seed']}): {steps} steps from Rdag 040000 on {ds['dataset']} "
                   f"({ds['frames']} frames, self successes {ds.get('self_used')}, stats of Rdag); docs/api/r2_protocol.md 5")
    diffs = TR.deep_diff(base, cfg)
    allowed = ALLOWED | ({"/save_freq", "/log_freq"} if smoke else set())
    # deep_diff は末端の道（/lr_schedule/peak など）を返すので、決めた項目の下も許す
    bad = [x for x in diffs if not any(x[0] == a or x[0].startswith(a + "/") for a in allowed)]
    if bad:
        raise SystemExit(f"Rdag の設定と、決めた項目のほかでも違う: {bad}")
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
    return p, cmd


def post_check(run_dir: pathlib.Path) -> dict:
    rd = lambda d, r: json.loads((d / r).read_text(encoding="utf-8"))   # noqa: E731
    run = rd(run_dir, "train_run.json")
    cks = sorted(p.name for p in (run_dir / "checkpoints").iterdir() if p.is_dir() and p.name.isdigit())
    last = cks[-1] if cks else None
    tc = rd(run_dir, f"checkpoints/{last}/pretrained_model/train_config.json") if last else {}
    old = rd(rdag_path(RDAG_RUN), "train_run.json")
    keys = ("log_points", "loss_first", "loss_last", "loss_min", "loss_mean_last_10pct", "gpu_mem_allocated_max_gib")
    return {"run": str(run_dir), "exit_code": run.get("exit_code"), "checkpoints": cks, "final_checkpoint": last,
            "final_is_steps": last is not None and int(last) == int(tc.get("steps", -1)),
            "stats_check_pass": (run.get("stats_check") or {}).get("pass"),
            "policy_for_eval": str(run_dir / "checkpoints" / last / "pretrained_model") if last else None,
            "loss_summary": {k: {"Rdag": old["log_summary"].get(k), NAME: run["log_summary"].get(k)} for k in keys},
            "note": "最後の保存点を使う（選ばない）。loss は人が見る"}


def main(argv=None) -> int:
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--start", action="store_true")
    ap.add_argument("--post-check", metavar="RUN_DIR")
    a = ap.parse_args(argv)
    if a.post_check:
        res = post_check(config.path(a.post_check))
        WRAP.mkdir(parents=True, exist_ok=True)
        (WRAP / f"postcheck_{pathlib.Path(a.post_check).name}.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(res, ensure_ascii=False, indent=1))
        return 0 if res["exit_code"] == 0 and res["final_is_steps"] and res["stats_check_pass"] else 1
    init = check_init()
    ds = dataset_choice()
    cfg, base, diffs = build_cfg(ds, init, a.smoke)
    print(f"データ {ds['dataset']}（{ds['frames']} こま、自分の成功 {ds.get('self_used')} 本・{ds.get('self_share')}）→ "
          f"Rdag 040000 から {cfg['steps']} 手（{cfg['steps'] * TR.BATCH / ds['frames']:.2f} 周）")
    print("Rdag の設定との違い:")
    for p_, o, n in diffs:
        print(f"  {p_}: {o!r} -> {n!r}")
    tag = f"{NAME}s{cfg['seed']}{'_smoke' if a.smoke else ''}"
    if a.start:
        spec = __import__("importlib.util").util.spec_from_file_location("s4_seed_for_r2", ROOT / "scripts" / "train_seed.py")
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
             "wall_s": round(time.time() - t0, 1), "dataset": ds, "diffs_from_rdag": diffs,
             "run_dir": str(runs[-1].relative_to(ROOT)) if runs else None}, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8")
        return 0 if code == 0 else 2
    cfg_path, cmd = launcher_dry_run(cfg)
    out = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "name": NAME, "dataset": ds, "launcher_config": cfg, "rdag_config": base,
           "diffs_from_rdag": diffs, "command": cmd, "hours_estimate": round(cfg["steps"] / 40000 * 4.75, 1),
           "dry_run_config": str(cfg_path.relative_to(ROOT))}
    WRAP.mkdir(parents=True, exist_ok=True)
    (WRAP / f"dryrun_{NAME}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("起動器の引数:")
    for c in cmd:
        print(f"  {c}")
    print(f"学習の時間の見込み: 約 {out['hours_estimate']} 時間（Rdag の 4 時間 45 分 × 手数の比）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
