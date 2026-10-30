"""R1v3・N1v3 の学習を、データと手順をそのままに学習のシード（train.seed）だけ変えて起動する包み。

scripts/real/regen_train.py・train_cup_plate.py、scripts/failure_relearn/train_rdag.py・scripts/success_relearn/train_r2a.py が、学習の設定の
組み立て（build_cfg）・起動器の dry-run・起動前の確認の部品として読む。

使い方（既定は --dry-run 相当。実際に学習を始めるのは --start のときだけ）:
    .venv\\Scripts\\python.exe scripts\\train_seed.py R1v3 --seed 1001                   # 渡す設定を出す（学習しない）
    .venv\\Scripts\\python.exe scripts\\train_seed.py R1v3 --seed 1001 --dry-run --resolve
                                                      # + LeRobot が組み立てる train_config.json 相当も CPU だけで出す
    .venv\\Scripts\\python.exe scripts\\train_seed.py N1v3 --seed 1002 --compare outputs\\train\\train_N1v3_20261005-202158_20261005-202158
                                                      # 既存の学習（シード 1000）との差分一覧を出す（--resolve を含む）
    .venv\\Scripts\\python.exe scripts\\train_seed.py R1v3 --post-check outputs\\s4\\train\\<実行名>      # 学習が終わった後の確認（CPU・読み取りだけ）
    .venv\\Scripts\\python.exe scripts\\train_seed.py --check-vs-40h                   # シード 1000 で train_vla.py と同じ設定になるか
    .venv\\Scripts\\python.exe scripts\\train_seed.py R1v3 --seed 1001 --start [--smoke]  # GPU を使う

選べるもの: 名前 R1v3・N1v3、シード 1001・1002（configs\\s4_seed_<シード>.yaml があるものだけ）。--smoke は 1000 ステップ。
--num-workers の既定は 3（R1v3・N1v3 を学習したときの値。default.yaml の 6 ではない）。

読むもの: configs\\default.yaml と configs\\s4_seed_<シード>.yaml（config.load の重ね方）、outputs\\f\\data_v3.json（データセットとマニフェストの出所）、
          outputs\\h\\cue_aug_decision.json（目標位置の入力のずらしの採否）、outputs\\manifests\\<名前>_*.json、outputs\\datasets\\<名前>_*\\meta、
          --compare の相手の train_launch_config.json・train_run.json・checkpoints\\020000\\pretrained_model\\train_config.json
書くもの（outputs\\s4\\ 以下だけ）:
          dry-run・compare : outputs\\s4\\seed_wrap\\dryrun_<名前>_s<シード>.json、compare_<名前>_s<シード>.json、_resolve_*.json
          --start          : outputs\\s4\\train_cfg\\train_<名前>s<シード>_<日時>.json（起動器に渡す設定）、
                             outputs\\s4\\train\\<実行名>\\（学習の出力。チェックポイントなど）、outputs\\s4\\train_logs\\<実行名>.log、
                             outputs\\s4\\seed_wrap\\train_<名前>s<シード>.json（実行の記録）
          --check-vs-40h   : outputs\\s4\\seed_wrap\\check_vs_40h.json
学習の起動（--start）の前に、空き GPU メモリ 12 GiB 以上・ほかの lerobot 学習が走っていないことを確かめる。起動は前景で、切り離さない。

学習の設定（データ・20000 ステップ・バッチ 32・5000 ステップごとの保存・num_workers 3）は R1v3・N1v3 と同じ。
この包みは評価を含まない（loss が R1v3 と同程度かは train_run.json の log_summary で見る）。
"""
import argparse
import contextlib
import copy
import datetime as dt
import hashlib
import importlib.util
import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time

from recovla.common import config

ROOT = config.ROOT
S4 = ROOT / "outputs" / "s4"
WRAP_OUT = S4 / "seed_wrap"
TRAIN_CFG_DIR = S4 / "train_cfg"
TRAIN_OUT = S4 / "train"
TRAIN_LOGS = S4 / "train_logs"
NAMES = ("R1v3", "N1v3")
BASE_SEED = 1000                      # 既存の R1v3・N1v3 のシード（default.yaml の train.seed）
DEFAULT_WORKERS = 3                   # R1v3・N1v3 を学習したときの値（train_run.json の command で確かめた）
MIN_FREE_GIB = 12.0
EXISTING_RUN = {"R1v3": "outputs/train/train_R1v3_20261005-180404_20261005-180404",
                "N1v3": "outputs/train/train_N1v3_20261005-202158_20261005-202158"}

# 差分の分類。seed 以外で学習の結果に影響しない項目は、出力先・実行名だけ（note は起動器が記録に写す自由記述で、学習に渡らない）
SEED_KEYS = {"seed", "--seed"}
PATH_KEYS = {"output_dir", "job_name", "--output_dir", "--job_name"}
NOTE_KEYS = {"note"}


def say(msg="") -> None:
    print(msg, flush=True)


def load_h40():
    """scripts\\train_vla.py をモジュールとして読む（読み込みでは何も書かない。定数と、シード 1000 の照合に使う）。"""
    spec = importlib.util.spec_from_file_location("s4_h40", ROOT / "scripts" / "train_vla.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256_file(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def deep_diff(a, b, prefix=""):
    """入れ子の辞書・配列の違いを [(パス, 旧, 新)] で返す。"""
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


# ------------------------------------------------------------------ 設定の重ね
def load_cfg(seed: int) -> dict:
    """シード 1000 は照合用の default.yaml のまま。それ以外は configs\\s4_seed_<シード>.yaml を重ね、
    default.yaml との違いが train.seed だけであることを確かめる。"""
    if seed == BASE_SEED:
        return config.load()
    name = f"s4_seed_{seed}"
    if not (config.CONFIG_DIR / f"{name}.yaml").is_file():
        raise SystemExit(f"configs\\{name}.yaml がない。種は {sorted(p.stem[8:] for p in config.CONFIG_DIR.glob('s4_seed_*.yaml'))} から選ぶ")
    cfg = config.load(name)
    diffs = deep_diff(config.load(), cfg)
    if [d[0] for d in diffs] != ["/train/seed"] or cfg["train"]["seed"] != seed:
        raise SystemExit(f"configs\\{name}.yaml が train.seed 以外も変えている: {diffs}")
    return cfg


def build_cfg(name: str, seed: int, smoke: bool, num_workers: int, cfg_all: dict, h40) -> tuple:
    """train_vla.py の cmd_train(R1v3・N1v3) が作る起動器の設定と同じものを作る。違うのは seed の出所（重ねた設定）と note の末尾だけ。
    --check-vs-40h がシード 1000 で train_vla.py の出力と一致することを確かめる。"""
    train = cfg_all["train"]
    ver = "v3"
    base = name[:-2]
    data = json.loads((config.path(cfg_all["paths"]["outputs"]) / "f" / f"data_{ver}.json").read_text(encoding="utf-8"))
    dec = json.loads((config.path(cfg_all["paths"]["outputs"]) / "h" / "cue_aug_decision.json").read_text(encoding="utf-8"))
    full_steps, save = h40.STAGE3_STEPS, h40.STAGE3_SAVE
    steps = 1000 if smoke else full_steps
    ds = data["datasets"][base]
    cfg = {"dataset": str(config.path(ds["dataset"])), "train_scope": train["scope"],
           "batch_size": int(train["batch_size"]), "steps": steps,
           "save_freq": steps if smoke else save, "seed": int(train["seed"]),
           "log_freq": int(train["log_freq"]), "num_workers": int(num_workers),
           "note": f"Stage 3 {name}{' smoke' if smoke else ''}: {steps} steps on "
                   f"{ds['dataset']} (target cue, cue_augment {'on' if dec['adopt'] else 'off'} per board 0054)"}
    cfg["fast_query"] = True
    if dec["adopt"]:
        cfg["cue_augment"] = dict(train["cue_augment"])
    return cfg, ds, dec


def manifest_info(name: str, ds: dict) -> dict:
    """マニフェストが R1v3・N1v3 のものと同一のファイルを指すこと: 出所、SHA-256、データセットの conversion.json に埋まった写しとの一致。"""
    mpath = config.path(ds["manifest"])
    m = json.loads(mpath.read_text(encoding="utf-8"))
    conv = json.loads((config.path(ds["dataset"]) / "meta" / "conversion.json").read_text(encoding="utf-8"))
    return {"manifest": str(mpath.relative_to(ROOT)), "manifest_sha256": sha256_file(mpath), "manifest_entries": len(m["entries"]),
            "conversion_json_manifest_equals_file": conv.get("manifest") == m,
            "dataset": ds["dataset"], "dataset_episodes": ds.get("episodes"), "dataset_frames": ds.get("frames")}


# ------------------------------------------------------------------ 起動器の dry-run
def launcher_dry_run(cfg: dict, tag: str, cfg_dir: pathlib.Path):
    """設定を JSON に書き、train_launcher.run(dry_run=True) を呼んで、検査を通したうえで lerobot 側へ渡す引数列を取る。"""
    from recovla.policy import train_launcher as tl
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / f"train_{tag}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = tl.run(cfg_path, output_root=TRAIN_OUT, log_root=TRAIN_LOGS, dry_run=True)
    text = buf.getvalue()
    if code != 0 or " command:" not in text:
        raise SystemExit(f"起動器の dry-run が失敗: exit {code}\n{text}")
    cmd = [ln.strip() for ln in text.split(" command:\n", 1)[1].splitlines() if ln.strip()]
    info = {k: next((ln.split(None, 1)[1].strip() for ln in text.splitlines() if ln.startswith(f" {k}")), None)
            for k in ("output", "log")}
    return cfg_path, cmd, info, text


def split_command(cmd: list) -> tuple:
    """[trainer..., '--', lerobot 引数...] を (前置き, {キー: 値}) にする。値のない引数はそのまま。"""
    i = cmd.index("--") if "--" in cmd else 0
    args = {}
    for a in cmd[i + 1:]:
        k, _, v = a.partition("=")
        args[k] = v if _ else True
    return cmd[:i + 1], args


# ------------------------------------------------------------------ LeRobot が組み立てる train_config.json 相当（CPU だけ）
def resolve_worker(cmd_json: str, out_json: str) -> None:
    """子プロセス側。GPU を隠し（CUDA_VISIBLE_DEVICES を空に）、lerobot-train と同じ構文解析 + validate だけを行って train_config.json の中身を書く。"""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    cmd = json.loads(pathlib.Path(cmd_json).read_text(encoding="utf-8"))
    sys.argv = ["resolve"] + cmd
    import draccus
    from lerobot.configs import parser
    from lerobot.configs.train import TrainPipelineConfig
    import lerobot.policies.smolvla.configuration_smolvla  # noqa: F401  方策の型を登録する（lerobot-train も同じ）

    @parser.wrap()
    def run(cfg: TrainPipelineConfig):
        cfg.validate()
        with open(out_json, "w", encoding="utf-8") as fh:
            with draccus.config_type("json"):
                draccus.dump(cfg, fh, indent=4)

    run()


def resolve_train_config(cmd: list, tag: str) -> dict:
    _, args = split_command(cmd)
    lerobot_args = [f"{k}={v}" if v is not True else k for k, v in args.items()]
    WRAP_OUT.mkdir(parents=True, exist_ok=True)
    cmd_json, out_json = WRAP_OUT / f"_resolve_{tag}_args.json", WRAP_OUT / f"_resolve_{tag}.json"
    cmd_json.write_text(json.dumps(lerobot_args, ensure_ascii=False), encoding="utf-8")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HOME=str(config.path(config.load()["paths"]["models_home"])),
               HF_HUB_OFFLINE="1", PYTHONIOENCODING="utf-8")
    p = subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve()), "--_resolve-worker", str(cmd_json), str(out_json)],
                       env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0 or not out_json.is_file():
        raise SystemExit(f"train_config の解決に失敗: exit {p.returncode}\n{p.stderr[-2000:]}")
    return json.loads(out_json.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ 差分の分類
def classify(layer: str, diffs: list) -> list:
    rows = []
    for path, old, new in diffs:
        key, top = path.rsplit("/", 1)[-1], path.count("/") == 1          # 入れ子の中の同名キーは許さない
        if top and key in SEED_KEYS:
            kind = "seed（意図した違い）"
        elif top and key in PATH_KEYS:
            kind = "出力先・実行名（結果に影響しない）"
        elif top and key in NOTE_KEYS:
            kind = "note（自由記述。学習に渡らない）"
        else:
            kind = "想定外"
        rows.append({"layer": layer, "path": path, "old": old, "new": new, "kind": kind})
    return rows


def compare(name: str, seed: int, cfg: dict, cmd: list, resolved: dict, old_dir: pathlib.Path) -> dict:
    old_launch = json.loads((old_dir / "train_launch_config.json").read_text(encoding="utf-8"))
    old_run = json.loads((old_dir / "train_run.json").read_text(encoding="utf-8"))
    old_tc = json.loads((old_dir / "checkpoints" / "020000" / "pretrained_model" / "train_config.json").read_text(encoding="utf-8"))
    _, old_args = split_command(old_run["command"])
    _, new_args = split_command(cmd)
    old_pre = old_run["command"][:old_run["command"].index("--")]
    new_pre = cmd[:cmd.index("--")]
    rows = []
    rows += classify("起動器の設定（train_launch_config.json）", deep_diff(old_launch, cfg))
    rows += classify("lerobot へ渡す引数（train_run.json の command）", deep_diff(old_args, new_args))
    rows += classify("起動前置き（python -m recovla.policy.train_wrapped ...）", deep_diff({"prefix": old_pre}, {"prefix": new_pre}))
    rows += classify("train_config.json", deep_diff(old_tc, resolved))
    unexpected = [r for r in rows if r["kind"] == "想定外"]
    code = {}
    try:
        from recovla.policy import train_launcher as tl
        code = {"launcher_sha256_old": old_run.get("launcher_sha256"), "launcher_sha256_now": sha256_file(tl.__file__),
                "launcher_same": old_run.get("launcher_sha256") == sha256_file(tl.__file__)}
        fs = (old_run.get("code_version") or {}).get("files_sha256") or {}
        for rel in ("src/recovla/policy/train_launcher.py", "src/recovla/policy/train_wrapped.py", "src/recovla/data/convert.py",
                    "configs/default.yaml"):
            if rel in fs:
                code[f"{rel}_same_as_recorded"] = fs[rel] == sha256_file(ROOT / rel)
    except Exception as e:  # noqa: BLE001
        code = {"error": repr(e)}
    return {"existing_run": str(old_dir.relative_to(ROOT)) if old_dir.is_relative_to(ROOT) else str(old_dir),
            "new": {"name": name, "seed": seed}, "rows": rows, "unexpected": len(unexpected),
            "only_seed_and_output_names": not unexpected, "code_identity": code}


def print_rows(rows: list) -> None:
    layer = None
    for r in rows:
        if r["layer"] != layer:
            layer = r["layer"]
            say(f"\n[{layer}]")
        say(f"  {r['path']}: {r['old']!r} -> {r['new']!r}   ← {r['kind']}")
    if not rows:
        say("  （違いなし）")


# ------------------------------------------------------------------ 起動前の確認
def precheck() -> list:
    problems = []
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True,
                             text=True, timeout=20).stdout.split()
        free_gib = float(out[0]) / 1024
        if free_gib < MIN_FREE_GIB:
            problems.append(f"空き GPU メモリ {free_gib:.1f} GiB < {MIN_FREE_GIB} GiB")
        say(f"空き GPU メモリ {free_gib:.1f} GiB")
    except Exception as e:  # noqa: BLE001
        problems.append(f"nvidia-smi で空きメモリを確かめられない: {e!r}")
    try:
        import psutil
        me = os.getpid()
        busy = []
        for p in psutil.process_iter(["pid", "cmdline"]):
            cl = " ".join(p.info["cmdline"] or [])
            if p.info["pid"] != me and ("train_wrapped" in cl or "lerobot-train" in cl or "lerobot_train" in cl):
                busy.append(p.info["pid"])
        if busy:
            problems.append(f"ほかの lerobot 学習が走っている（pid {busy}）。学習は 1 本ずつ")
    except ImportError:
        problems.append("psutil がなく、ほかの学習が走っていないか確かめられない")
    return problems


# ------------------------------------------------------------------ 照合: シード 1000 で train_vla.py と同じ設定か
def check_vs_40h(h40) -> dict:
    """train_vla.py の cmd_train を、起動器の run だけ差し替えて（起動せず設定の JSON を拾う）走らせ、この包みがシード 1000 で作る設定と比べる。"""
    from recovla.policy import train_launcher as tl
    res = {}
    WRAP_OUT.mkdir(parents=True, exist_ok=True)
    real_run, real_out = tl.run, h40.OUT
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="_check40h_", dir=WRAP_OUT))
    (tmp / "cue_aug_decision.json").write_bytes((real_out / "cue_aug_decision.json").read_bytes())
    captured = {}
    tl.run = lambda cfg_path, *a, **k: captured.update(path=cfg_path) or 0
    h40.OUT = tmp
    try:
        for name in NAMES:
            for smoke in (False, True):
                for workers in (3,):
                    captured.clear()
                    h40.cmd_train(argparse.Namespace(name=name, smoke=smoke, num_workers=workers))
                    theirs = json.loads(pathlib.Path(captured["path"]).read_text(encoding="utf-8"))
                    mine, _, _ = build_cfg(name, BASE_SEED, smoke, workers, config.load(), h40)
                    diffs = deep_diff(theirs, mine)
                    res[f"{name}{'_smoke' if smoke else ''}"] = {"identical": not diffs, "diffs": diffs, "cfg": mine}
    finally:
        tl.run, h40.OUT = real_run, real_out
    return {"all_identical": all(v["identical"] for v in res.values()), "cases": res, "temp_dir": str(tmp.relative_to(ROOT))}


# ------------------------------------------------------------------ 学習が終わった後の確認（CPU・読み取りだけ）
def post_check(new_dir: pathlib.Path, name: str) -> dict:
    """終わった学習（--start の出力）を、同じ名前の既存の学習（シード 1000）と比べる: train_config.json の違いが seed と出力先・実行名だけか、
    データセットの指紋とマニフェストが同じか、終了コード 0・チェックポイント 4 つ・loss の要約が R1v3・N1v3 と同程度か。"""
    old_dir = config.path(EXISTING_RUN[name])
    rd = lambda d, rel: json.loads((d / rel).read_text(encoding="utf-8"))
    tc = "checkpoints/020000/pretrained_model/train_config.json"
    rows = classify("train_config.json", deep_diff(rd(old_dir, tc), rd(new_dir, tc)))
    old_run, new_run = rd(old_dir, "train_run.json"), rd(new_dir, "train_run.json")
    lo, ln = old_run["log_summary"], new_run["log_summary"]
    keys = ("log_points", "loss_first", "loss_last", "loss_min", "loss_mean_last_10pct", "gpu_mem_allocated_max_gib")
    cks = sorted(p.name for p in (new_dir / "checkpoints").iterdir() if p.is_dir() and p.name.isdigit())
    return {"existing_run": EXISTING_RUN[name], "new_run": str(new_dir), "train_config_rows": rows,
            "train_config_only_seed_and_names": not [r for r in rows if r["kind"] == "想定外"],
            "same_dataset_fingerprint": old_run["dataset"]["fingerprint"] == new_run["dataset"]["fingerprint"],
            "exit_code": new_run.get("exit_code"), "checkpoints": cks, "checkpoints_ok": cks == ["005000", "010000", "015000", "020000"],
            "loss_summary": {k: {"existing": lo.get(k), "new": ln.get(k)} for k in keys},
            "gpu_over_limit": ln.get("gpu_over_limit"),
            "judgement_note": "loss が R1v3・N1v3 と同程度かは、上の数字を見て人が決める（閾値は事前に決めていない）"}


# ------------------------------------------------------------------ 本体
def main(argv=None) -> int:
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("name", nargs="?", choices=NAMES)
    ap.add_argument("--seed", type=int, help="1001 か 1002（configs\\s4_seed_<種>.yaml がある種）")
    ap.add_argument("--num-workers", type=int, default=DEFAULT_WORKERS, help=f"データの読み手の数（既定 {DEFAULT_WORKERS}。値は結果を変えない）")
    ap.add_argument("--smoke", action="store_true", help="1000 手で最後まで通す（--start のときは GPU を使う）")
    ap.add_argument("--dry-run", action="store_true", help="学習を始めず、渡す設定を出す（既定の動作）")
    ap.add_argument("--resolve", action="store_true", help="LeRobot が組み立てる train_config.json 相当も出す（CPU だけ。数分かかる）")
    ap.add_argument("--compare", metavar="RUN_DIR", help="既存の学習の出力フォルダと差分一覧を出す（--resolve を含む）")
    ap.add_argument("--start", action="store_true", help="学習を起動する（運用役だけ。GPU を使う。前景で、切り離さない）")
    ap.add_argument("--check-vs-40h", action="store_true", help="種 1000 で train_vla.py の設定と一致するかを確かめる")
    ap.add_argument("--post-check", metavar="RUN_DIR", help="終わった学習（outputs\\s4\\train\\...）を既存の同名の学習と比べる（名前が要る。--seed は不要）")
    ap.add_argument("--_resolve-worker", nargs=2, metavar=("CMD_JSON", "OUT_JSON"), help=argparse.SUPPRESS)
    a = ap.parse_args(argv)

    if a._resolve_worker:
        resolve_worker(*a._resolve_worker)
        return 0
    h40 = load_h40()
    if a.check_vs_40h:
        res = check_vs_40h(h40)
        out = WRAP_OUT / "check_vs_40h.json"
        out.write_text(json.dumps({"written": time.strftime("%Y-%m-%d %H:%M:%S"), **res}, ensure_ascii=False, indent=2), encoding="utf-8")
        for k, v in res["cases"].items():
            say(f"{k}: {'一致' if v['identical'] else '不一致 ' + str(v['diffs'])}")
        say(f"\n全部一致: {res['all_identical']}   記録 {out.relative_to(ROOT)}")
        return 0 if res["all_identical"] else 1

    if a.post_check:
        if not a.name:
            ap.error("--post-check には名前（R1v3・N1v3）が要る")
        res = post_check(config.path(a.post_check), a.name)
        print_rows(res["train_config_rows"])
        for k in ("train_config_only_seed_and_names", "same_dataset_fingerprint", "exit_code", "checkpoints", "checkpoints_ok", "gpu_over_limit"):
            say(f"{k}: {res[k]}")
        for k, v in res["loss_summary"].items():
            say(f"  {k}: 既存 {v['existing']}  新 {v['new']}")
        out = WRAP_OUT / f"postcheck_{pathlib.Path(a.post_check).name}.json"
        WRAP_OUT.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        say(f"記録 {out.relative_to(ROOT)}")
        return 0 if res["train_config_only_seed_and_names"] and res["same_dataset_fingerprint"] and res["exit_code"] == 0 and res["checkpoints_ok"] else 1
    if not a.name or a.seed is None:
        ap.error("名前（R1v3・N1v3）と --seed が要る")
    if a.seed == BASE_SEED:
        ap.error(f"種 {BASE_SEED} は既存の R1v3・N1v3。別の種（configs\\s4_seed_<種>.yaml があるもの）を選ぶ")
    if a.start and a.dry_run:
        ap.error("--start と --dry-run は同時に使えない")
    cfg_all = load_cfg(a.seed)
    cfg, ds, dec = build_cfg(a.name, a.seed, a.smoke, a.num_workers, cfg_all, h40)
    cfg["note"] += f"; S4 seed replication (train.seed={a.seed}, configs/s4_seed_{a.seed}.yaml over default.yaml)"
    tag = f"{a.name}s{a.seed}{'_smoke' if a.smoke else ''}"
    man = manifest_info(a.name, ds)
    if not man["conversion_json_manifest_equals_file"]:
        raise SystemExit("データセットの conversion.json に埋まったマニフェストが、マニフェストのファイルと一致しない")

    if a.start:
        problems = precheck()
        if problems:
            say("起動しない:\n  " + "\n  ".join(problems))
            return 3
        from recovla.policy import train_launcher as tl
        TRAIN_CFG_DIR.mkdir(parents=True, exist_ok=True)
        cfg_path = TRAIN_CFG_DIR / f"train_{tag}_{time.strftime('%Y%m%d-%H%M%S')}.json"
        cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        say(f"設定 {cfg_path.relative_to(ROOT)}\nマニフェスト {man['manifest']} sha256 {man['manifest_sha256']}")
        t0 = time.time()
        code = tl.run(cfg_path, output_root=TRAIN_OUT, log_root=TRAIN_LOGS)
        runs = sorted(TRAIN_OUT.glob(f"{cfg_path.stem}_*"))
        rec = json.loads((runs[-1] / tl.RUN_RECORD).read_text(encoding="utf-8")) if runs and (runs[-1] / tl.RUN_RECORD).is_file() else {}
        WRAP_OUT.mkdir(parents=True, exist_ok=True)
        (WRAP_OUT / f"train_{tag}.json").write_text(json.dumps(
            {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "name": a.name, "seed": a.seed, "config": str(cfg_path.relative_to(ROOT)),
             "exit": code, "wall_s": round(time.time() - t0, 1), "manifest": man, "cue_augment": dec["adopt"],
             "run_dir": str(runs[-1].relative_to(ROOT)) if runs else None, "log_summary": rec.get("log_summary"),
             "command": rec.get("command"),
             "checkpoints": sorted(p.name for p in (runs[-1] / "checkpoints").iterdir() if p.is_dir()) if runs and (runs[-1] / "checkpoints").is_dir() else None},
            ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        if code != 0:
            raise SystemExit(f"training {tag} exited with {code}")
        return 0

    # dry-run（既定）
    cfg_path, cmd, info, text = launcher_dry_run(cfg, tag, WRAP_OUT / "dryrun_cfg")
    say("\n".join(ln if len(ln) < 200 else ln[:80] + " ...（生データの番号の列は省略）" for ln in text.split(" command:")[0].splitlines()))
    say(" command:\n  " + "\n  ".join(cmd))
    say(f"\nマニフェスト {man['manifest']}\n  sha256 {man['manifest_sha256']}  エントリ {man['manifest_entries']}  "
        f"conversion.json の写しと一致 {man['conversion_json_manifest_equals_file']}")
    out = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "name": a.name, "seed": a.seed, "overlay": f"configs/s4_seed_{a.seed}.yaml",
           "overlay_only_changes": "train.seed", "launcher_config": cfg, "launcher_config_file": str(cfg_path.relative_to(ROOT)),
           "command": cmd, "output_dir_planned": info["output"], "log_planned": info["log"], "manifest": man, "cue_augment": dec["adopt"]}
    resolved = None
    if a.resolve or a.compare:
        say("\nLeRobot の train_config.json 相当を組み立てる（CPU だけ）...")
        resolved = resolve_train_config(cmd, tag)
        out["resolved_train_config_file"] = f"outputs/s4/seed_wrap/_resolve_{tag}.json"
        out["resolved_seed"] = resolved.get("seed")
    WRAP_OUT.mkdir(parents=True, exist_ok=True)
    (WRAP_OUT / f"dryrun_{a.name}_s{a.seed}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    if a.compare:
        old_dir = config.path(a.compare)
        cmp_ = compare(a.name, a.seed, cfg, cmd, resolved, old_dir)
        cmp_["manifest_new"] = man
        old_run = json.loads((old_dir / "train_run.json").read_text(encoding="utf-8"))
        cmp_["manifest_old_dataset_root"] = old_run["dataset"]["root"]
        cmp_["same_dataset"] = str(pathlib.Path(old_run["dataset"]["root"]).resolve()) == str(config.path(ds["dataset"]).resolve())
        cmp_["dataset_fingerprint_old"] = old_run["dataset"]["fingerprint"]
        from recovla.policy import train_launcher as tl
        cmp_["dataset_fingerprint_now"] = tl.dataset_fingerprint(config.path(ds["dataset"]))["fingerprint"]
        print_rows(cmp_["rows"])
        say(f"\n想定外の違い: {cmp_['unexpected']} 件   seed と出力先・実行名・note だけ: {cmp_['only_seed_and_output_names']}")
        say(f"同じデータセット: {cmp_['same_dataset']}   データセットの指紋が既存と同じ: {cmp_['dataset_fingerprint_old'] == cmp_['dataset_fingerprint_now']}")
        say(f"コードの同一性: {cmp_['code_identity']}")
        (WRAP_OUT / f"compare_{a.name}_s{a.seed}.json").write_text(json.dumps(cmp_, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        say(f"記録 outputs/s4/seed_wrap/compare_{a.name}_s{a.seed}.json")
        return 0 if cmp_["only_seed_and_output_names"] and cmp_["same_dataset"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
