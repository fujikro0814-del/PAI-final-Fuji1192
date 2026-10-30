"""学習の起動役: 変換したデータセット 1 つに lerobot-train を 1 回、設定の JSON 1 つで回す。

    .venv\\Scripts\\python.exe -m recovla.policy.train_launcher CONFIG.json [--confirm] [--dry-run]
                                                 [--output-root DIR] [--log-root DIR]

起動役が決めて固定するもの（パスは configs/default.yaml の paths から）:
  lerobot-train.exe（LeRobot 0.6.1 に python -m lerobot.scripts.train はない）
  HF_HUB_OFFLINE=1、HF_HOME=configs の paths.models_home
  --policy.path = lerobot/smolvla_libero のローカルのスナップショット（Windows では repo id が通らない）
  --dataset.repo_id=local/<フォルダ名> ＋ --dataset.root（repo_id だけだと HF_HOME を探す）
  --rename_map を両視点に（片方だけの対応表だと黙って画像 1 枚で学習する）
  --policy.push_to_hub=false（スナップショットの設定は作者のリポジトリへ push して終了コード 1 になる）
  --wandb.enable=false、--policy.device=cuda
  回ごとに新しい --output_dir（lerobot-train は既存のフォルダを拒む）。画面の出力はログに保存
  <output_dir>\\conversion.json = データセットの meta\\conversion.json の写し（評価時に vla_observation が確かめる）。
    lerobot-train が <output_dir> を作った直後に写す
  <output_dir>\\train_run.json = 設定・コマンド・データセットの指紋・時刻・終了コード・コードの版と、ログの要約
    （実際に読んだフレーム数・エピソード数、損失の統計、GPU メモリの最大と 14 GiB の上限）
  <output_dir>\\loss.csv・loss.png = ステップごとの学習の損失（lerobot は 1000 以上のステップを "1K" のように
    丸めて表示するので、log_freq を固定して行の順から数える）

設定ごとに明示するもの: 学習の範囲（行動エキスパートだけか全体か）、バッチの大きさ、ステップ数、使うエピソードの選び方
（マニフェストから変換したデータセットか、extra_args の --dataset.episodes / --dataset.eval_split）。

設定の JSON（知らない項目は拒む）:
  dataset      convert で作った LeRobot データセットのフォルダ（meta/conversion.json がある）     必須
  train_scope  "expert"（行動エキスパートだけ、画像の符号化器は固定）か "full"                    必須
  batch_size   整数。14 GiB で測った上限（expert 62、full 12）を超えると拒む
               （"allow_batch_over_14gib": true なら通す）                                         必須
  steps        整数                                                                                必須
  save_freq    整数（既定 = steps）
  seed         整数（既定 1000）
  log_freq     整数（既定 50）
  num_workers  整数（既定は lerobot のもの）
  extra_args   lerobot-train に足す "--key=value" の列（起動役が決める項目は拒む）
  note         自由な文。train_run.json に写す
  lora         {"layers": [8, ..., 15], "modules": ["q_proj", "v_proj"], "r": 16, "alpha": 32}: SmolVLM2 の
               その文章の層に LoRA。行動エキスパートと射影は全部学習する（--peft.full_training_modules）
  first_frames_weight  {"frames": 20, "weight": 5}: 各エピソードの最初の N フレームを W 倍の頻度で引く
  cue_augment  {"prob": 0.5, "max_m": 0.02}: 各学習サンプルの目標位置の入力 (x, y) を確率 prob で、
               一様な向きに U(0, max_m) だけずらす。行動は変えない
  fast_query   true: 行動列を行ごとでなく列ごとの表から読む（値は同じで速い）
  lora・first_frames_weight・cue_augment・fast_query のどれかがあれば、lerobot-train を recovla.policy.train_wrapped
  経由で回す。保存のたびに、決まった入力に対するメモリ上の方策の出力（recovla_reference.pt）も書く。
  init_policy  smolvla_libero のスナップショットの代わりに続きから学習する、学習済みの pretrained_model のフォルダ
               （学び直しで使う）。最適化器は新しく始める（--resume なし）。学習の後、保存したすべての checkpoint の
               正規化・逆正規化の値を init_policy のものと完全で比べる（observation.state.* と action.* が一致する
               こと。データセットの meta/stats.json を元のモデルのデータセットの統計に差し替えておく）。
               結果は train_run.json の "stats_check"
  lr_schedule  {"peak": 3e-5, "warmup": 200, "decay_steps": 10000, "decay_lr": 2.5e-6}: --policy.optimizer_lr、
               --policy.scheduler_warmup_steps、--policy.scheduler_decay_steps、--policy.scheduler_decay_lr
"""
import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import shutil
import socket
import subprocess
import sys
import tempfile

from recovla.common import code_version, config
from recovla.data import vla_image_spec as spec
from recovla.data import vla_observation

_CFG = config.load()

LAUNCHER_VERSION = 4         # 2: log_freq・ログの要約・loss.csv/png・コードの版
                              # 3: lora・first_frames_weight・train_wrapped
                              # 4: init_policy・lr_schedule・stats_check（fast_query は値を変えないので版は上げない）
DEFAULT_LOG_FREQ = int(_CFG["train"]["log_freq"])
# lerobot のログの mem_gb は torch.cuda.max_memory_allocated / 1024**3（確保したテンソルの GiB。
# CUDA のキャッシュの確保はこれより少し多い）
GPU_LIMIT_GIB = float(_CFG["train"]["gpu_limit_gib"])
LOSS_CSV = "loss.csv"
LOSS_PNG = "loss.png"
HF_HOME = config.path(_CFG["paths"]["models_home"])
OUTPUT_ROOT = config.path(_CFG["paths"]["train_output"])
LOG_ROOT = config.path(_CFG["paths"]["train_logs"])
POLICY_SNAPSHOT = config.path(_CFG["paths"]["policy_snapshot"])   # lerobot/smolvla_libero
RUN_RECORD = "train_run.json"
CONFIG_COPY = "train_launch_config.json"
# smolvla_libero のカメラの枠 ← 変換が書いたデータセットの項目名
RENAME_MAP = {spec.IMAGE_KEYS["overhead"]: "observation.images.camera1",
              spec.IMAGE_KEYS["wrist"]: "observation.images.camera2"}
if RENAME_MAP != dict(_CFG["train"]["rename_map"]):
    raise ValueError(f"configs train.rename_map {_CFG['train']['rename_map']} != {RENAME_MAP}")
SCOPE_FLAGS = {"expert": ["--policy.train_expert_only=true", "--policy.freeze_vision_encoder=true"],
               "full": ["--policy.train_expert_only=false", "--policy.freeze_vision_encoder=false"]}
MAX_BATCH_14GIB = {"expert": 62, "full": 12}                 # 1 プロセス 14 GiB に収まると測ったバッチの上限
REQUIRED = ("dataset", "train_scope", "batch_size", "steps")
OPTIONAL = ("save_freq", "seed", "log_freq", "num_workers", "extra_args", "note", "allow_batch_over_14gib",
            "lora", "first_frames_weight", "cue_augment", "init_policy", "lr_schedule", "fast_query")
OWNED = ("--policy.path", "--policy.push_to_hub", "--policy.repo_id", "--policy.device",
         "--policy.train_expert_only", "--policy.freeze_vision_encoder", "--dataset.root",
         "--dataset.repo_id", "--rename_map", "--output_dir", "--job_name", "--batch_size", "--steps",
         "--save_freq", "--seed", "--log_freq", "--num_workers", "--wandb.enable", "--resume", "--config_path",
         "--peft.target_modules", "--peft.full_training_modules", "--peft.method_type", "--peft.r",
         "--peft.lora_alpha", "--peft.init_type", "--policy.optimizer_lr", "--policy.scheduler_warmup_steps",
         "--policy.scheduler_decay_steps", "--policy.scheduler_decay_lr")
NORMALIZER_FILES = ("policy_preprocessor_step_5_normalizer_processor.safetensors",
                    "policy_postprocessor_step_0_unnormalizer_processor.safetensors")
LR_KEYS = {"peak": "--policy.optimizer_lr", "warmup": "--policy.scheduler_warmup_steps",
           "decay_steps": "--policy.scheduler_decay_steps", "decay_lr": "--policy.scheduler_decay_lr"}
VLM_TEXT_LAYER = r"model\.vlm_with_expert\.vlm\.model\.text_model\.layers\.({layers})\.self_attn\.({modules})"
NUM_VLM_TEXT_LAYERS = 16             # SmolVLA は SmolVLM2 の文章の層を 16 に減らしている
# LoRA のときも全部学習するもの: 行動エキスパートと、SmolVLA で新しく足された射影
FULL_TRAINING_MODULES = ["lm_expert", "state_proj", "action_in_proj", "action_out_proj", "action_time_mlp_in",
                         "action_time_mlp_out"]


def lora_flags(lora: dict) -> list:
    layers = [int(v) for v in lora["layers"]]
    target = VLM_TEXT_LAYER.format(layers="|".join(map(str, layers)), modules="|".join(lora["modules"]))
    return ["--peft.method_type=LORA", f"--peft.target_modules={target}",
            f"--peft.full_training_modules={json.dumps(FULL_TRAINING_MODULES)}",
            f"--peft.r={int(lora['r'])}", f"--peft.lora_alpha={int(lora['alpha'])}"]


def check_lora(path, lora) -> None:
    if not isinstance(lora, dict) or set(lora) != {"layers", "modules", "r", "alpha"}:
        raise LaunchError(f"{path}: lora must be {{layers, modules, r, alpha}}, got {lora!r}")
    if not lora["layers"] or not all(isinstance(v, int) and 0 <= v < NUM_VLM_TEXT_LAYERS for v in lora["layers"]):
        raise LaunchError(f"{path}: lora.layers must be text layer numbers 0..{NUM_VLM_TEXT_LAYERS - 1}")
    if not lora["modules"] or not all(m in ("q_proj", "k_proj", "v_proj", "o_proj") for m in lora["modules"]):
        raise LaunchError(f"{path}: lora.modules must be among q_proj, k_proj, v_proj, o_proj")
    for k in ("r", "alpha"):
        if not isinstance(lora[k], int) or isinstance(lora[k], bool) or lora[k] < 1:
            raise LaunchError(f"{path}: lora.{k} must be a positive integer")


def check_first_frames_weight(path, w) -> None:
    if not isinstance(w, dict) or set(w) != {"frames", "weight"}:
        raise LaunchError(f"{path}: first_frames_weight must be {{frames, weight}}, got {w!r}")
    if not isinstance(w["frames"], int) or w["frames"] < 1 or not isinstance(w["weight"], (int, float)) or w["weight"] <= 0:
        raise LaunchError(f"{path}: first_frames_weight needs frames >= 1 and weight > 0")


def check_cue_augment(path, c) -> None:
    if (not isinstance(c, dict) or set(c) != {"prob", "max_m"} or not 0.0 < float(c["prob"]) <= 1.0
            or not 0.0 < float(c["max_m"]) <= 0.1):
        raise LaunchError(f"{path}: cue_augment must be {{prob in (0, 1], max_m in (0, 0.1]}}, got {c!r}")


def wrapped(cfg) -> bool:
    return "lora" in cfg or "first_frames_weight" in cfg or "cue_augment" in cfg or cfg.get("fast_query") is True


def wrapper_prefix(cfg, python=None) -> list:
    """python -m recovla.policy.train_wrapped [...] --（この後に lerobot-train の引数が続く）。"""
    out = [str(python or sys.executable), "-m", "recovla.policy.train_wrapped", "--reference"]
    w = cfg.get("first_frames_weight")
    if w:
        out += [f"--first-frames={w['frames']}", f"--first-weight={w['weight']}"]
    c = cfg.get("cue_augment")
    if c:
        out += [f"--cue-aug-prob={c['prob']}", f"--cue-aug-max-m={c['max_m']}"]
    if cfg.get("fast_query") is True:
        out.append("--fast-query")
    return out + ["--"]


class LaunchError(ValueError):
    pass


def load_config(path) -> dict:
    path = pathlib.Path(path)
    try:
        cfg = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as e:
        raise LaunchError(f"cannot read config {path}: {e}") from e
    if not isinstance(cfg, dict):
        raise LaunchError(f"{path}: top level must be a JSON object")
    unknown = sorted(set(cfg) - set(REQUIRED) - set(OPTIONAL))
    if unknown:
        raise LaunchError(f"{path}: unknown keys {unknown} (allowed: {list(REQUIRED) + list(OPTIONAL)})")
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise LaunchError(f"{path}: missing required keys {missing}")
    if cfg["train_scope"] not in SCOPE_FLAGS:
        raise LaunchError(f"{path}: train_scope must be one of {list(SCOPE_FLAGS)}, got {cfg['train_scope']!r}")
    for k in ("batch_size", "steps", "save_freq", "seed", "log_freq", "num_workers"):
        if k in cfg and (not isinstance(cfg[k], int) or isinstance(cfg[k], bool) or cfg[k] < (0 if k in ("seed", "num_workers") else 1)):
            raise LaunchError(f"{path}: {k} must be a positive integer, got {cfg[k]!r}")
    limit = MAX_BATCH_14GIB[cfg["train_scope"]]
    if cfg["batch_size"] > limit and cfg.get("allow_batch_over_14gib") is not True:
        raise LaunchError(f"{path}: batch_size {cfg['batch_size']} > {limit}, the largest {cfg['train_scope']} batch "
                          f"measured under 14 GiB; beyond it the GPU spills into system memory and slows down "
                          f"without an error. Set \"allow_batch_over_14gib\": true to run anyway")
    extra = cfg.get("extra_args", [])
    if not isinstance(extra, list) or not all(isinstance(a, str) and a.startswith("--") for a in extra):
        raise LaunchError(f"{path}: extra_args must be a list of \"--key=value\" strings")
    for a in extra:
        if a.split("=", 1)[0] in OWNED:
            raise LaunchError(f"{path}: extra_args may not set {a.split('=', 1)[0]} (set by the launcher or a config key)")
    if "lora" in cfg:
        check_lora(path, cfg["lora"])
    if "first_frames_weight" in cfg:
        check_first_frames_weight(path, cfg["first_frames_weight"])
    if "cue_augment" in cfg:
        check_cue_augment(path, cfg["cue_augment"])
    if "fast_query" in cfg and not isinstance(cfg["fast_query"], bool):
        raise LaunchError(f"{path}: fast_query must be true or false, got {cfg['fast_query']!r}")
    if "lr_schedule" in cfg:
        s = cfg["lr_schedule"]
        if (not isinstance(s, dict) or set(s) != set(LR_KEYS) or not 0 < float(s["peak"]) < 1
                or not 0 <= float(s["decay_lr"]) <= float(s["peak"])
                or not all(isinstance(s[k], int) and s[k] >= 0 for k in ("warmup", "decay_steps"))):
            raise LaunchError(f"{path}: lr_schedule must be {{peak, warmup, decay_steps, decay_lr}}, got {s!r}")
    if "init_policy" in cfg:
        init_policy_dir(cfg)
    return cfg


def init_policy_dir(cfg) -> pathlib.Path:
    """続きから学習する pretrained_model のフォルダ（config.json・重み・正規化の 2 つのファイルがあること）。"""
    p = config.path(cfg["init_policy"]).resolve()
    for f in ("config.json", "model.safetensors", *NORMALIZER_FILES):
        if not (p / f).is_file():
            raise LaunchError(f"init_policy {p}: {f} not found; give a checkpoints/<step>/pretrained_model folder")
    return p


def stats_check(init_dir, output_dir) -> dict:
    """保存したすべての checkpoint の正規化・逆正規化のテンソルを init_dir のものと完全で比べる。
    pass = 両方のファイルで observation.state.* と action.* のテンソルがすべて同じ。"""
    from safetensors.numpy import load_file
    ref = {f: load_file(str(pathlib.Path(init_dir) / f)) for f in NORMALIZER_FILES}
    out = {"init_policy": str(init_dir), "checkpoints": {}}
    for ck in sorted((pathlib.Path(output_dir) / "checkpoints").glob("[0-9]*")):
        row = {}
        for f in NORMALIZER_FILES:
            got = load_file(str(ck / "pretrained_model" / f))
            keys = sorted(k for k in ref[f] if k.startswith(("observation.state.", "action.")))
            diff = [k for k in keys if k not in got or got[k].shape != ref[f][k].shape
                    or got[k].tobytes() != ref[f][k].tobytes()]
            other = [k for k in ref[f] if k not in keys and (k not in got or got[k].tobytes() != ref[f][k].tobytes())]
            row[f] = {"compared": len(keys), "different": diff, "other_keys_different": other}
        row["pass"] = all(not v["different"] for v in row.values())
        out["checkpoints"][ck.name] = row
    out["pass"] = bool(out["checkpoints"]) and all(r["pass"] for r in out["checkpoints"].values())
    return out


def check_dataset(dataset) -> dict:
    """変換の版 2 の LeRobot データセットであることを確かめ、その conversion.json を返す。"""
    root = pathlib.Path(dataset)
    for rel in ("meta/info.json", "meta/stats.json", "meta/conversion.json"):
        if not (root / rel).is_file():
            raise LaunchError(f"{root}: {rel} not found; give a dataset made by convert_to_lerobot.py")
    conversion = json.loads((root / "meta" / "conversion.json").read_text(encoding="utf-8"))
    spec.check_record(conversion, str(root / "meta" / "conversion.json"))
    return conversion


def dataset_fingerprint(dataset) -> dict:
    """データセットの中身の指紋。同じ変換でも parquet のバイト列（HF datasets の指紋）と conversion.json の
    converted_at は変わるので、決まって同じになる部分だけのハッシュを取る。"""
    root = pathlib.Path(dataset)
    conversion = json.loads((root / "meta" / "conversion.json").read_text(encoding="utf-8"))
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    parts = {
        "info_json_sha256": hashlib.sha256((root / "meta" / "info.json").read_bytes()).hexdigest(),
        "stats_json_sha256": hashlib.sha256((root / "meta" / "stats.json").read_bytes()).hexdigest(),
        "conversion_sha256_without_converted_at": hashlib.sha256(json.dumps(
            {k: v for k, v in conversion.items() if k != "converted_at"}, sort_keys=True).encode()).hexdigest(),
    }
    return {"total_episodes": info.get("total_episodes"), "total_frames": info.get("total_frames"),
            "converter_version": conversion.get("converter_version"),
            "raw_episode_ids": [s.get("raw_episode_id") for s in conversion.get("sources", [])], **parts,
            "fingerprint": hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()}


def build_command(trainer, policy_dir, cfg, output_dir, job_name) -> list:
    """trainer = コマンドの頭の列（通常は [lerobot-train.exe]）。"""
    dataset = pathlib.Path(cfg["dataset"]).resolve()
    cmd = [*map(str, trainer), f"--policy.path={policy_dir}", f"--dataset.repo_id=local/{dataset.name}",
           f"--dataset.root={dataset}", f"--rename_map={json.dumps(RENAME_MAP)}",
           *SCOPE_FLAGS[cfg["train_scope"]], f"--batch_size={cfg['batch_size']}", f"--steps={cfg['steps']}",
           f"--save_freq={cfg.get('save_freq', cfg['steps'])}", f"--seed={cfg.get('seed', 1000)}",
           f"--log_freq={cfg.get('log_freq', DEFAULT_LOG_FREQ)}",
           f"--output_dir={output_dir}", f"--job_name={job_name}", "--policy.push_to_hub=false",
           "--policy.device=cuda", "--wandb.enable=false"]
    if "num_workers" in cfg:
        cmd.append(f"--num_workers={cfg['num_workers']}")
    if "lora" in cfg:
        cmd += lora_flags(cfg["lora"])
    if "lr_schedule" in cfg:
        cmd += [f"{flag}={cfg['lr_schedule'][k]}" for k, flag in LR_KEYS.items()]
    return cmd + list(cfg.get("extra_args", []))


def policy_snapshot() -> pathlib.Path:
    """HF_HOME の下の lerobot/smolvla_libero のローカルのスナップショット（Windows では repo id が通らない）。"""
    if not (POLICY_SNAPSHOT / "config.json").is_file():
        raise LaunchError(f"{POLICY_SNAPSHOT}: config.json not found; copy the starting model first "
                          f"(scripts/import_teleop_tool.ps1 -Part assets)")
    return POLICY_SNAPSHOT


def check_symlinks(where) -> None:
    """lerobot-train は checkpoints/last をシンボリックリンクにする。開発者モードがないと最初の保存で初めて落ちるので先に確かめる。"""
    pathlib.Path(where).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=where) as d:
        try:
            os.symlink(pathlib.Path(d), pathlib.Path(d) / "link", target_is_directory=True)
        except OSError as e:
            raise LaunchError(f"cannot create symbolic links ({e}); enable Windows Developer Mode, "
                              f"lerobot-train needs one for checkpoints\\last") from e


def copy_conversion(dataset, output_dir) -> pathlib.Path:
    dst = pathlib.Path(output_dir) / vla_observation.COPIED_CONVERSION
    shutil.copyfile(pathlib.Path(dataset) / "meta" / "conversion.json", dst)
    vla_observation.load_training_spec(output_dir)          # 評価の入口が受け付けることを確かめる
    return dst


def sha256_file(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


_TRAIN_LINE = re.compile(r"step:(\S+) smpl:\S+ ep:\S+ epch:\S+ loss:([-\d.eE+naif]+) .*?mem_gb:([\d.]+)")
_EVAL_LINE = re.compile(r"step (\d+): eval_loss=([-\d.eE+naif]+)")


def parse_log(text: str, log_freq: int) -> dict:
    """lerobot-train の出力から、読んだフレーム数・エピソード数、記録したステップごとの損失、メモリの最大、評価の損失を取る。
    ステップはログの行の順 × log_freq から数える（表示のステップは 999 を超えると丸められる）。"""
    def last_int(pattern):
        found = re.findall(pattern, text)
        return int(found[-1]) if found else None

    rows = []
    for i, m in enumerate(_TRAIN_LINE.finditer(text)):
        printed, loss, mem = m.groups()
        step = (i + 1) * log_freq
        if printed.isdigit() and int(printed) != step:
            raise LaunchError(f"log line {i}: printed step {printed} != {step} (log_freq {log_freq})")
        rows.append({"step": step, "loss": float(loss), "mem_gb": float(mem)})
    losses = [r["loss"] for r in rows]
    n10 = max(1, len(losses) // 10)
    summary = {
        "dataset_num_frames": last_int(r"dataset\.num_frames=(\d+)"),
        "dataset_num_episodes": last_int(r"dataset\.num_episodes=(\d+)"),
        "train_eval_split": (re.findall(r"Train/eval split: ([^\r\n]+)", text) or [None])[-1],
        "log_points": len(rows),
        "last_step_logged": rows[-1]["step"] if rows else None,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_min": min(losses) if losses else None,
        "loss_mean_last_10pct": sum(losses[-n10:]) / n10 if losses else None,
        "loss_mean_previous_10pct": (sum(losses[-2 * n10:-n10]) / n10) if len(losses) >= 2 * n10 else None,
        "gpu_mem_allocated_max_gib": max((r["mem_gb"] for r in rows), default=None),
        "gpu_limit_gib": GPU_LIMIT_GIB,
        "eval_losses": [{"step": int(s), "eval_loss": float(v)} for s, v in _EVAL_LINE.findall(text)],
    }
    summary["gpu_over_limit"] = (summary["gpu_mem_allocated_max_gib"] is not None
                                 and summary["gpu_mem_allocated_max_gib"] > GPU_LIMIT_GIB)
    return {"rows": rows, "summary": summary}


def write_loss_files(rows, output_dir) -> None:
    out = pathlib.Path(output_dir)
    with open(out / LOSS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["step", "loss", "mem_gb"])
        w.writeheader()
        w.writerows(rows)
    if rows and importlib.util.find_spec("PIL") is not None:       # PIL は LeRobot の venv にだけある
        plot_loss([r["step"] for r in rows], [r["loss"] for r in rows], out / LOSS_PNG)


def plot_loss(steps, losses, path, size=(900, 500)) -> None:
    """記録したステップごとの損失（細い線）と、点の 10 % の移動平均（太い線）。PIL だけで描く。"""
    from PIL import Image, ImageDraw
    w, h = size
    left, right, top, bottom = 70, 20, 30, 50
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    y_max = max(losses) * 1.05
    y_min = 0.0
    x_max = max(steps)

    def px(s, v):
        return (left + (w - left - right) * s / x_max, top + (h - top - bottom) * (1 - (v - y_min) / (y_max - y_min)))

    d.rectangle([left, top, w - right, h - bottom], outline="black")
    for k in range(5):
        v = y_min + (y_max - y_min) * k / 4
        y = px(0, v)[1]
        d.line([left - 4, y, left, y], fill="black")
        d.text((5, y - 6), f"{v:.3f}", fill="black")
    for k in range(5):
        s = x_max * k / 4
        x = px(s, 0)[0]
        d.line([x, h - bottom, x, h - bottom + 4], fill="black")
        d.text((x - 15, h - bottom + 8), f"{int(s)}", fill="black")
    d.line([px(s, v) for s, v in zip(steps, losses)], fill=(150, 170, 220), width=1)
    n = max(1, len(losses) // 10)
    mean = [sum(losses[max(0, i - n + 1):i + 1]) / len(losses[max(0, i - n + 1):i + 1]) for i in range(len(losses))]
    d.line([px(s, v) for s, v in zip(steps, mean)], fill=(20, 40, 160), width=3)
    d.text((left, 8), f"training loss (thin: logged every {steps[0]} steps, thick: running mean of {n} points)",
           fill="black")
    d.text((w // 2 - 20, h - 20), "step", fill="black")
    img.save(path)


def run(config_path, output_root=OUTPUT_ROOT, log_root=LOG_ROOT, confirm=False, dry_run=False,
        trainer=None, policy_dir=None, now=None) -> int:
    config_path = pathlib.Path(config_path).resolve()
    cfg = load_config(config_path)
    dataset = pathlib.Path(cfg["dataset"]).resolve()
    check_dataset(dataset)
    fingerprint = dataset_fingerprint(dataset)
    stamp = (now or dt.datetime.now()).strftime("%Y%m%d-%H%M%S")
    run_name = f"{config_path.stem}_{stamp}"
    output_dir = pathlib.Path(output_root).resolve() / run_name
    log_path = pathlib.Path(log_root).resolve() / f"{run_name}.log"
    if output_dir.exists():
        raise LaunchError(f"{output_dir} already exists")
    if trainer is None:
        trainer = wrapper_prefix(cfg) if wrapped(cfg) else [pathlib.Path(sys.executable).with_name("lerobot-train.exe")]
    if not pathlib.Path(trainer[0]).is_file():
        raise LaunchError(f"{trainer[0]} not found; run with the LeRobot venv python")
    policy_dir = policy_dir or (init_policy_dir(cfg) if "init_policy" in cfg else policy_snapshot())
    cmd = build_command(trainer, policy_dir, cfg, output_dir, run_name)
    print(f"\n config   {config_path}\n dataset  {dataset}\n          {fingerprint['total_episodes']} episodes, "
          f"{fingerprint['total_frames']} frames, raw ids {fingerprint['raw_episode_ids']}\n"
          f" scope    {cfg['train_scope']}   batch {cfg['batch_size']}   steps {cfg['steps']}\n"
          f" extra    {cfg.get('extra_args', [])}\n output   {output_dir}\n log      {log_path}\n")
    if dry_run:
        print(" command:\n  " + "\n  ".join(cmd))
        return 0
    check_symlinks(pathlib.Path(output_root))
    if confirm and input(" Start training? (y/n): ").strip()[:1].lower() != "y":
        print(" Cancelled.")
        return 0

    record = {"launcher_version": LAUNCHER_VERSION, "launcher_sha256": sha256_file(__file__),
              "code_version": code_version.code_version(), "run_name": run_name, "host": socket.gethostname(), "config_path": str(config_path),
              "config": cfg, "command": cmd, "dataset": {"root": str(dataset), **fingerprint},
              "policy_path": str(policy_dir), "output_dir": str(output_dir), "log": str(log_path),
              "started_at": dt.datetime.now().isoformat(timespec="seconds")}
    try:
        import lerobot
        record["lerobot_version"] = getattr(lerobot, "__version__", None)
    except ImportError:
        record["lerobot_version"] = None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, HF_HOME=str(HF_HOME), HF_HUB_OFFLINE="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    copied = None
    exit_code = None
    with open(log_path, "wb") as log:
        log.write(("launcher: " + json.dumps(cmd, ensure_ascii=False) + "\n").encode("utf-8"))
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            while True:
                chunk = proc.stdout.read1(4096) if hasattr(proc.stdout, "read1") else proc.stdout.read(4096)
                if not chunk:
                    break
                log.write(chunk)
                log.flush()                                 # 学習中もログを追えるように
                sys.stdout.buffer.write(chunk)
                sys.stdout.flush()
                if copied is None and output_dir.is_dir():
                    copied = copy_conversion(dataset, output_dir)
            exit_code = proc.wait()
        except KeyboardInterrupt:
            proc.wait()
            exit_code = "interrupted"
        finally:
            record["ended_at"] = dt.datetime.now().isoformat(timespec="seconds")
            record["exit_code"] = exit_code
            where = output_dir if output_dir.is_dir() else log_path.parent
            log.flush()                                     # 開いたままなので、読み戻す前に書き出す
            try:
                parsed = parse_log(log_path.read_text(encoding="utf-8", errors="replace"),
                                   cfg.get("log_freq", DEFAULT_LOG_FREQ))
                record["log_summary"] = parsed["summary"]
            except (OSError, LaunchError) as e:
                parsed = None
                record["log_summary"] = {"error": repr(e)}
            if output_dir.is_dir():
                if copied is None:
                    copy_conversion(dataset, output_dir)
                shutil.copyfile(config_path, output_dir / CONFIG_COPY)
                if parsed is not None:
                    write_loss_files(parsed["rows"], output_dir)
                if "init_policy" in cfg:
                    try:
                        record["stats_check"] = stats_check(policy_dir, output_dir)
                    except Exception as e:                    # noqa: BLE001  失敗しても記録は書く
                        record["stats_check"] = {"pass": False, "error": repr(e)}
            name = RUN_RECORD if where == output_dir else f"{run_name}.{RUN_RECORD}"
            (where / name).write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    s = record.get("log_summary", {})
    print(f"\n exit code {exit_code}\n output    {output_dir}\n log       {log_path}\n"
          f" read      {s.get('dataset_num_frames')} frames, {s.get('dataset_num_episodes')} episodes\n"
          f" loss      first {s.get('loss_first')}  last {s.get('loss_last')}  "
          f"mean last 10% {s.get('loss_mean_last_10pct')}  previous 10% {s.get('loss_mean_previous_10pct')}\n"
          f" gpu       max allocated {s.get('gpu_mem_allocated_max_gib')} GiB"
          f"{'  OVER 14 GiB' if s.get('gpu_over_limit') else ''}")
    return exit_code if isinstance(exit_code, int) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--confirm", action="store_true", help="ask y/n before starting")
    ap.add_argument("--dry-run", action="store_true", help="check the config and print the command only")
    ap.add_argument("--output-root", default=str(OUTPUT_ROOT))
    ap.add_argument("--log-root", default=str(LOG_ROOT))
    a = ap.parse_args(argv)
    try:
        return run(a.config, a.output_root, a.log_root, confirm=a.confirm, dry_run=a.dry_run)
    except (LaunchError, spec.ImageSpecError) as e:
        print(f"\n ERROR: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
