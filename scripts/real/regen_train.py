"""実機の Panda で測った応答に合わせたシミュレーションで作り直したデータ（docs/real/prereg_regen_v1.md 第 5 節・変更 1）の
学習の入口: Rreal（成功データ＋立て直しデータ）・Nreal（成功データだけ）。
R1v3・N1v3 の起動の設定（train_seed.py の build_cfg）から、データセットと note だけを替える。シードは 1000、手数・バッチ・
保存・num_workers・fast_query・cue_augment は R1v3・N1v3 と同じ（違えば止める）。train_vla.py・train_seed.py・configs は
書き換えない。

使い方（作業場所 C:\\PAI\\recovery_vla。既定は --dry-run 相当。学習を始めるのは --start のときだけ）:
    .venv\\Scripts\\python.exe scripts\\real\\regen_train.py Rreal                 # 渡す設定と、R1v3 との違いの一覧を出す（学習しない）
    .venv\\Scripts\\python.exe scripts\\real\\regen_train.py Nreal --dry-run
    .venv\\Scripts\\python.exe scripts\\real\\regen_train.py Rreal --start [--smoke]   # GPU を使う。前景で。1 本ずつ
    .venv\\Scripts\\python.exe scripts\\real\\regen_train.py Rreal --post-check outputs\\real\\regen\\train\\<実行名>   # 終わった後（CPU）

名前と元: Rreal ← R1v3（立て直しデータあり）、Nreal ← N1v3（立て直しデータなし）。データは outputs\\real\\regen\\data.json の
datasets.R・N（regen_data.py gen-data が書く。学習の前の確かめ (2)（gate2）を通っていなければ止める）。
違いの検査（毎回）: 元の設定との違いが dataset・note の外にあれば止める。データセットの conversion.json に埋まった
マニフェストが data.json のマニフェストのファイルと同じで、SHA-256 も同じことも確かめる。
書くもの（outputs\\real\\regen\\ 以下だけ）: dry-run は train_wrap\\dryrun_<名前>.json、--start は train_cfg\\・train\\<実行名>\\・
train_logs\\・train_wrap\\train_<名前>.json、--post-check は train_wrap\\postcheck_<実行名>.json。
"""
import argparse
import contextlib
import hashlib
import importlib.util
import json
import pathlib
import sys
import time

from recovla.common import config

ROOT = config.ROOT
REGEN = ROOT / "outputs" / "real" / "regen"
WRAP = REGEN / "train_wrap"
TRAIN_CFG_DIR = REGEN / "train_cfg"
TRAIN_OUT = REGEN / "train"
TRAIN_LOGS = REGEN / "train_logs"
DATA = REGEN / "data.json"
NAMES = {"Rreal": ("R1v3", "R"), "Nreal": ("N1v3", "N")}   # 名前: (元の学習, data.json の datasets の鍵)
SEED = 1000                                              # 事前登録 第 5 節: R1v3・N1v3 と同じシード
ALLOWED_DIFF = {"/dataset", "/note"}                     # 元（R1v3・N1v3）の起動器の設定から変えてよい項目
TRAIN_CONFIG_ALLOWED = {"/output_dir", "/job_name", "/dataset/root", "/dataset/repo_id"}


def load_seedwrap():
    spec = importlib.util.spec_from_file_location("s4_train_seed_for_regen", ROOT / "scripts" / "train_seed.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sha256_file(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def regen_dataset(name: str, data: dict) -> dict:
    """data.json から使うデータセット。確かめ (2)（gate2）と、変換と照合が通っていなければ止める。"""
    if not (data.get("gate2") or {}).get("pass"):
        raise SystemExit("outputs\\real\\regen\\data.json のデータの関門 (2) が通っていない（学習しない）")
    ds = (data.get("datasets") or {}).get(NAMES[name][1])
    if not ds or not (ds.get("convert_exit") == 0 and ds.get("verify_exit") == 0 and ds.get("verify_pass")):
        raise SystemExit(f"{name} のデータセットの変換・照合が通っていない")
    return ds


def build_cfg(name: str, smoke: bool, num_workers: int, sw, h40, data: dict) -> tuple:
    """(起動器の設定, 元の設定, 違いの一覧, データセットの情報, 目標位置の入力のずらしの判定)。違いが dataset・note の外にあれば止める。"""
    base = NAMES[name][0]
    ref, _, dec = sw.build_cfg(base, sw.BASE_SEED, smoke, num_workers, config.load(), h40)
    cfg, _, _ = sw.build_cfg(base, SEED, smoke, num_workers, config.load(), h40)
    ds = regen_dataset(name, data)
    cfg["dataset"] = str(config.path(ds["dataset"]))
    cfg["note"] = (f"Regen {name}{' smoke' if smoke else ''} (seed {SEED}): {cfg['steps']} steps on {ds['dataset']} "
                   f"(= {base} settings; data regenerated in world v3_real, docs/real/prereg_regen_v1.md + amend1)")
    diffs = sw.deep_diff(ref, cfg)
    bad = [d for d in diffs if d[0] not in ALLOWED_DIFF]
    if bad:
        raise SystemExit(f"{name} の設定が {base} と、データ・note のほかでも違う: {bad}")
    return cfg, ref, diffs, ds, dec


def manifest_identity(ds: dict) -> dict:
    mpath = ROOT / ds["manifest"]
    m = json.loads(mpath.read_text(encoding="utf-8"))
    conv = json.loads((config.path(ds["dataset"]) / "meta" / "conversion.json").read_text(encoding="utf-8"))
    sha = sha256_file(mpath)
    return {"manifest": ds["manifest"], "manifest_sha256": sha, "sha256_as_built": ds.get("manifest_sha256"),
            "same_sha_as_built": sha == ds.get("manifest_sha256"), "entries": len(m["entries"]),
            "conversion_json_manifest_equals_file": conv.get("manifest") == m}


def post_check(new_dir: pathlib.Path, name: str, sw) -> dict:
    """終わった学習を元（R1v3・N1v3）と比べる。train_config.json の違いが出力先・データセットだけか、終了コード、
    チェックポイント 4 つ、loss の要約（閾値は事前に決めていない。人が見る）。"""
    old_dir = config.path(sw.EXISTING_RUN[NAMES[name][0]])
    rd = lambda d, rel: json.loads((d / rel).read_text(encoding="utf-8"))   # noqa: E731
    tc = "checkpoints/020000/pretrained_model/train_config.json"
    diffs = sw.deep_diff(rd(old_dir, tc), rd(new_dir, tc))
    bad = [d for d in diffs if d[0] not in TRAIN_CONFIG_ALLOWED]
    old_run, new_run = rd(old_dir, "train_run.json"), rd(new_dir, "train_run.json")
    keys = ("log_points", "loss_first", "loss_last", "loss_min", "loss_mean_last_10pct", "gpu_mem_allocated_max_gib")
    cks = sorted(p.name for p in (new_dir / "checkpoints").iterdir() if p.is_dir() and p.name.isdigit())
    root = str((rd(new_dir, tc).get("dataset") or {}).get("root", ""))
    want = "R_v3r_" if NAMES[name][1] == "R" else "N_v3r_"
    if want not in pathlib.Path(root).name:                 # データセットは名前（R_v3r・N_v3r）のもの
        bad.append(("/dataset/root", None, root))
    return {"existing_run": sw.EXISTING_RUN[NAMES[name][0]], "new_run": str(new_dir), "train_config_diffs": diffs,
            "train_config_unexpected": bad, "train_config_ok": not bad, "exit_code": new_run.get("exit_code"),
            "checkpoints": cks, "checkpoints_ok": cks == ["005000", "010000", "015000", "020000"],
            "loss_summary": {k: {"existing": old_run["log_summary"].get(k), "new": new_run["log_summary"].get(k)} for k in keys},
            "judgement_note": "loss が元と同程度かは数字を見て人が決める（閾値は事前に決めていない）"}


def main(argv=None) -> int:
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("name", choices=sorted(NAMES))
    ap.add_argument("--num-workers", type=int, default=3, help="既定 3（R1v3・N1v3 の実際の起動値と同じ）")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--start", action="store_true", help="学習を起動する（GPU を使う。前景で）")
    ap.add_argument("--post-check", metavar="RUN_DIR")
    a = ap.parse_args(argv)
    sw = load_seedwrap()
    sw.TRAIN_OUT, sw.TRAIN_LOGS = TRAIN_OUT, TRAIN_LOGS     # 起動器の dry-run が出す出力先も outputs/real/regen の下にする
    if a.post_check:
        res = post_check(config.path(a.post_check), a.name, sw)
        WRAP.mkdir(parents=True, exist_ok=True)
        (WRAP / f"postcheck_{pathlib.Path(a.post_check).name}.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(json.dumps({k: res[k] for k in ("train_config_ok", "exit_code", "checkpoints_ok", "loss_summary")}, ensure_ascii=False, indent=1))
        return 0 if res["train_config_ok"] and res["exit_code"] == 0 and res["checkpoints_ok"] else 1
    if a.start and a.dry_run:
        ap.error("--start と --dry-run は同時に使えない")
    h40 = sw.load_h40()
    data = json.loads(DATA.read_text(encoding="utf-8"))
    cfg, ref, diffs, ds, dec = build_cfg(a.name, a.smoke, a.num_workers, sw, h40, data)
    man = manifest_identity(ds)
    if not (man["same_sha_as_built"] and man["conversion_json_manifest_equals_file"]):
        raise SystemExit(f"マニフェストが変換したときと違う、または conversion.json の写しと一致しない: {man}")
    tag = f"{a.name}s{SEED}{'_smoke' if a.smoke else ''}"
    print("元（" + NAMES[a.name][0] + "）の設定との違い:")
    for p_, o, n in diffs:
        print(f"  {p_}: {o!r} -> {n!r}")
    if a.start:
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
            {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "name": a.name, "base": NAMES[a.name][0], "seed": SEED,
             "config": str(cfg_path.relative_to(ROOT)), "exit": code, "wall_s": round(time.time() - t0, 1), "manifest": man,
             "diffs_from_base": diffs, "run_dir": str(runs[-1].relative_to(ROOT)) if runs else None},
            ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return 0 if code == 0 else 2
    cfg_path, cmd, info, _ = sw.launcher_dry_run(cfg, tag, WRAP / "dryrun_cfg")
    out = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "name": a.name, "base": NAMES[a.name][0], "seed": SEED,
           "launcher_config": cfg, "base_config": ref, "diffs_from_base": diffs, "command": cmd, "manifest": man,
           "cue_augment": dec["adopt"], "output_dir_planned": info["output"]}
    WRAP.mkdir(parents=True, exist_ok=True)
    (WRAP / f"dryrun_{a.name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"マニフェスト {man['manifest']} sha256 {man['manifest_sha256']}（{man['entries']} 本）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
