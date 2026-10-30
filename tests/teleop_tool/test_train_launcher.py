"""学習の起動器（recovla.policy.train_launcher）の確かめ。

GPU も LeRobot も要らない: データは起動器が読む meta だけの偽物、lerobot-train は同じように振る舞う小さなスクリプト
（既にある出力先を断る、checkpoints/<step>/pretrained_model を作る、表示する、決めた終了コードで終わる）に差し替える。
"""
import importlib.util
import json
import pathlib
import sys

import pytest

from recovla.data import vla_image_spec as spec
from recovla.data import vla_observation
from recovla.policy import train_launcher as tl

FAKE_TRAINER = r'''
import pathlib, sys, time
args = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
out = pathlib.Path(args["output_dir"])
if out.exists():
    print("FileExistsError"); sys.exit(1)
print("INFO ot_train.py:442 dataset.num_frames=30 (30)", flush=True)
print("INFO ot_train.py:443 dataset.num_episodes=2", flush=True)
f = int(args["log_freq"])
for i, loss in enumerate((1.5, 0.9)):
    print(f"Training: 50%|##| INFO ot_train.py:641 step:{(i + 1) * f} smpl:4 ep:0 epch:0.01 loss:{loss} grdn:3.1 "
          f"lr:1e-05 updt_s:0.2 data_s:0.3 smp/s:8 mem_gb:2.09 losses_after_forward:1.0", flush=True)
time.sleep(0.2)
(out / "checkpoints" / "000001" / "pretrained_model").mkdir(parents=True)
print("output dir created", flush=True)
time.sleep(0.2)
print("End of training", flush=True)
sys.exit(int(pathlib.Path(__file__).with_suffix(".rc").read_text()))
'''


def conversion(**over):
    c = {"converter_version": 2, "converted_at": "2026-09-17T00:00:00", **spec.spec_record(),
         "sources": [{"episode_index": 0, "raw_episode_id": 7}, {"episode_index": 1, "raw_episode_id": 9}]}
    c.update(over)
    return c


def make_dataset(tmp_path, conv=None, name="ds"):
    root = tmp_path / name
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"total_episodes": 2, "total_frames": 30}), encoding="utf-8")
    (root / "meta" / "stats.json").write_text("{}", encoding="utf-8")
    (root / "meta" / "conversion.json").write_text(json.dumps(conv or conversion()), encoding="utf-8")
    return root


def make_config(tmp_path, dataset, name="run_a", **over):
    cfg = {"dataset": str(dataset), "train_scope": "expert", "batch_size": 4, "steps": 2, **over}
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return p


def fake_trainer(tmp_path, rc=0):
    script = tmp_path / "fake_trainer.py"
    script.write_text(FAKE_TRAINER, encoding="utf-8")
    script.with_suffix(".rc").write_text(str(rc), encoding="utf-8")
    return [sys.executable, str(script)]


# --- 設定 --------------------------------------------------------------------------------------------

def test_config_accepts_minimal(tmp_path):
    cfg = tl.load_config(make_config(tmp_path, tmp_path / "ds"))
    assert cfg["train_scope"] == "expert"


@pytest.mark.parametrize("over, words", [
    ({"batchsize": 4}, "unknown keys"),
    ({"train_scope": "all"}, "train_scope"),
    ({"steps": 0}, "steps"),
    ({"batch_size": True}, "batch_size"),
    ({"batch_size": 63}, "14 GiB"),
    ({"train_scope": "full", "batch_size": 13}, "14 GiB"),
    ({"extra_args": "--dataset.episodes=[1]"}, "extra_args"),
    ({"extra_args": ["--policy.push_to_hub=true"]}, "--policy.push_to_hub"),
    ({"extra_args": ["--output_dir=x"]}, "--output_dir"),
    ({"extra_args": ["--rename_map={}"]}, "--rename_map"),
    ({"extra_args": ["--log_freq=10"]}, "--log_freq"),
    ({"log_freq": 0}, "log_freq"),
])
def test_config_rejects(tmp_path, over, words):
    with pytest.raises(tl.LaunchError, match=words.replace("[", r"\[")):
        tl.load_config(make_config(tmp_path, tmp_path / "ds", **over))


def test_config_missing_required(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"dataset": "x", "train_scope": "expert", "steps": 2}), encoding="utf-8")
    with pytest.raises(tl.LaunchError, match="batch_size"):
        tl.load_config(p)


def test_config_batch_over_limit_allowed_explicitly(tmp_path):
    cfg = tl.load_config(make_config(tmp_path, tmp_path / "ds", batch_size=64, allow_batch_over_14gib=True))
    assert cfg["batch_size"] == 64


# --- コマンド ----------------------------------------------------------------------------------------

def test_command_fixes_what_the_survey_found(tmp_path):
    cfg = tl.load_config(make_config(tmp_path, tmp_path / "my_ds", extra_args=["--dataset.episodes=[0,1]"]))
    cmd = tl.build_command(["lerobot-train.exe"], "SNAP", cfg, tmp_path / "out", "job")
    args = dict(a.split("=", 1) for a in cmd[1:])
    assert cmd[0] == "lerobot-train.exe"
    assert args["--policy.path"] == "SNAP"
    assert args["--dataset.repo_id"] == "local/my_ds"
    assert args["--dataset.root"] == str((tmp_path / "my_ds").resolve())
    assert json.loads(args["--rename_map"]) == {"observation.images.image": "observation.images.camera1",
                                                "observation.images.image2": "observation.images.camera2"}
    assert args["--policy.push_to_hub"] == "false"
    assert args["--wandb.enable"] == "false"
    assert args["--policy.train_expert_only"] == "true" and args["--policy.freeze_vision_encoder"] == "true"
    assert args["--save_freq"] == "2"                        # 既定は steps と同じ
    assert args["--dataset.episodes"] == "[0,1]"             # 起動器は決めず、そのまま渡す
    assert len(args) == len(cmd) - 1                         # 同じ鍵を 2 度渡さない


def test_command_full_scope(tmp_path):
    cfg = tl.load_config(make_config(tmp_path, tmp_path / "ds", train_scope="full", batch_size=8))
    cmd = tl.build_command(["t"], "S", cfg, tmp_path / "o", "j")
    assert "--policy.train_expert_only=false" in cmd and "--policy.freeze_vision_encoder=false" in cmd


def test_rename_map_covers_every_dataset_image_key():
    assert set(tl.RENAME_MAP) == set(spec.IMAGE_KEYS.values())


# --- データ ------------------------------------------------------------------------------------------

def test_dataset_checks(tmp_path):
    ds = make_dataset(tmp_path)
    tl.check_dataset(ds)
    (ds / "meta" / "conversion.json").unlink()
    with pytest.raises(tl.LaunchError, match="conversion.json"):
        tl.check_dataset(ds)


def test_dataset_version1_refused(tmp_path):
    v1 = {"converter_version": 1, "images": {}}
    with pytest.raises(spec.ImageSpecError):
        tl.check_dataset(make_dataset(tmp_path, v1))


def test_fingerprint_ignores_converted_at_only(tmp_path):
    a = tl.dataset_fingerprint(make_dataset(tmp_path, conversion(converted_at="t1"), "a"))
    b = tl.dataset_fingerprint(make_dataset(tmp_path, conversion(converted_at="t2"), "b"))
    c = tl.dataset_fingerprint(make_dataset(tmp_path, conversion(sources=[{"episode_index": 0, "raw_episode_id": 8}]), "c"))
    assert a["fingerprint"] == b["fingerprint"] != c["fingerprint"]
    assert a["raw_episode_ids"] == [7, 9]


# --- 偽の lerobot-train で3 個の連続タスクの実行 -------------------------------------------------------------------

def launch(tmp_path, rc=0, **cfg_over):
    ds = make_dataset(tmp_path)
    cfg = make_config(tmp_path, ds, **cfg_over)
    code = tl.run(cfg, tmp_path / "ckpt", tmp_path / "logs", trainer=fake_trainer(tmp_path, rc), policy_dir="SNAP")
    outs = list((tmp_path / "ckpt").glob("run_a_*"))
    return code, outs


def test_run_copies_conversion_and_writes_record(tmp_path, capsys):
    code, outs = launch(tmp_path)
    assert code == 0 and len(outs) == 1
    out = outs[0]
    ckpt = out / "checkpoints" / "000001" / "pretrained_model"
    assert vla_observation.load_training_spec(ckpt)["image_spec_version"] == spec.SPEC_VERSION
    rec = json.loads((out / tl.RUN_RECORD).read_text(encoding="utf-8"))
    assert rec["exit_code"] == 0 and rec["dataset"]["raw_episode_ids"] == [7, 9]
    assert rec["config"]["batch_size"] == 4 and rec["policy_path"] == "SNAP"
    assert (out / tl.CONFIG_COPY).is_file()
    log = pathlib.Path(rec["log"]).read_text(encoding="utf-8")
    assert "End of training" in log and "End of training" in capsys.readouterr().out
    s = rec["log_summary"]
    assert s["dataset_num_frames"] == 30 and s["dataset_num_episodes"] == 2
    assert s["loss_first"] == 1.5 and s["loss_last"] == 0.9 and s["last_step_logged"] == 100
    assert s["gpu_mem_allocated_max_gib"] == 2.09 and s["gpu_over_limit"] is False
    assert rec["code_version"]["code_sha256"] and "files_sha256" in rec["code_version"]
    assert (out / tl.LOSS_CSV).read_text(encoding="utf-8").splitlines()[1:] == ["50,1.5,2.09", "100,0.9,2.09"]
    if importlib.util.find_spec("PIL"):
        assert (out / tl.LOSS_PNG).stat().st_size > 0


def _line(step, loss, mem=2.0):
    return (f"\rTraining:  5%|| INFO 2026 ot_train.py:641 step:{step} smpl:32 ep:1 epch:0.5 loss:{loss} grdn:1.0 "
            f"lr:1e-04 updt_s:0.3 data_s:0.1 smp/s:90 mem_gb:{mem} losses_after_forward:1.0\n")


def test_parse_log_steps_from_order_even_when_printed_rounded():
    text = "dataset.num_frames=404 (404)\ndataset.num_episodes=2\n" + "".join(
        _line(f"{(i + 1) * 100}" if (i + 1) * 100 < 1000 else f"{(i + 1) * 100 // 1000}K", 1.0 / (i + 1),
              mem=3.0 + i) for i in range(20)) + "INFO step 2000: eval_loss=0.25\n"
    p = tl.parse_log(text, 100)
    assert [r["step"] for r in p["rows"]][-3:] == [1800, 1900, 2000]
    s = p["summary"]
    assert s["log_points"] == 20 and s["dataset_num_frames"] == 404
    assert s["loss_mean_last_10pct"] == pytest.approx((1 / 19 + 1 / 20) / 2)
    assert s["loss_mean_previous_10pct"] == pytest.approx((1 / 17 + 1 / 18) / 2)
    assert s["gpu_mem_allocated_max_gib"] == 22.0 and s["gpu_over_limit"] is True
    assert s["eval_losses"] == [{"step": 2000, "eval_loss": 0.25}]


def test_parse_log_refuses_wrong_log_freq():
    with pytest.raises(tl.LaunchError, match="log_freq"):
        tl.parse_log(_line(50, 1.0) + _line(100, 0.9), 100)


def test_run_failure_keeps_exit_code_and_record(tmp_path):
    code, outs = launch(tmp_path, rc=3)
    assert code == 3
    rec = json.loads((outs[0] / tl.RUN_RECORD).read_text(encoding="utf-8"))
    assert rec["exit_code"] == 3
    assert (outs[0] / vla_observation.COPIED_CONVERSION).is_file()


def test_dry_run_creates_nothing(tmp_path):
    ds = make_dataset(tmp_path)
    code = tl.run(make_config(tmp_path, ds), tmp_path / "ckpt", tmp_path / "logs", dry_run=True,
                  trainer=fake_trainer(tmp_path), policy_dir="SNAP")
    assert code == 0 and not (tmp_path / "ckpt").exists() and not (tmp_path / "logs").exists()


def test_main_reports_config_errors(tmp_path, capsys):
    p = make_config(tmp_path, tmp_path / "missing_ds")
    assert tl.main([str(p), "--dry-run"]) == 2
    assert "ERROR" in capsys.readouterr().out


# --- 学習済みの方策からの学習し直し（init_policy・lr_schedule・stats_check） ----------------------------------

def _policy_dir(tmp_path, name, state_mean=0.0):
    import numpy as np
    from safetensors.numpy import save_file
    d = tmp_path / name
    d.mkdir(parents=True)
    for f in ("config.json", "model.safetensors"):
        (d / f).write_text("{}", encoding="utf-8")
    for f in tl.NORMALIZER_FILES:
        save_file({"observation.state.mean": np.full(17, state_mean, np.float32), "action.std": np.ones(7, np.float32),
                   "index.max": np.full(1, 5.0 if name == "init" else 9.0, np.float32)}, str(d / f))
    return d


def test_lr_schedule_flags_and_owned(tmp_path):
    s = {"peak": 3e-5, "warmup": 200, "decay_steps": 10000, "decay_lr": 2.5e-6}
    cfg = tl.load_config(make_config(tmp_path, tmp_path / "ds", lr_schedule=s))
    args = dict(a.split("=", 1) for a in tl.build_command(["t"], "S", cfg, tmp_path / "o", "j")[1:])
    assert (args["--policy.optimizer_lr"], args["--policy.scheduler_warmup_steps"],
            args["--policy.scheduler_decay_steps"], args["--policy.scheduler_decay_lr"]) == ("3e-05", "200", "10000", "2.5e-06")
    with pytest.raises(tl.LaunchError, match="lr_schedule"):
        tl.load_config(make_config(tmp_path, tmp_path / "ds", name="b", lr_schedule={**s, "decay_lr": 1.0}))
    with pytest.raises(tl.LaunchError, match="--policy.optimizer_lr"):
        tl.load_config(make_config(tmp_path, tmp_path / "ds", name="c", extra_args=["--policy.optimizer_lr=1"]))


def test_init_policy_must_be_a_trained_folder(tmp_path):
    with pytest.raises(tl.LaunchError, match="init_policy"):
        tl.load_config(make_config(tmp_path, tmp_path / "ds", init_policy=str(tmp_path / "nothing")))
    p = _policy_dir(tmp_path, "init")
    assert tl.load_config(make_config(tmp_path, tmp_path / "ds", name="b", init_policy=str(p)))["init_policy"] == str(p)


def test_stats_check_bitwise(tmp_path):
    init = _policy_dir(tmp_path, "init")
    out = tmp_path / "out"
    for step, mean in (("005000", 0.0), ("010000", 1e-7)):
        _policy_dir(out / "checkpoints" / step, "pretrained_model", mean)
    r = tl.stats_check(init, out)
    assert r["checkpoints"]["005000"]["pass"] and not r["checkpoints"]["010000"]["pass"] and not r["pass"]
    row = r["checkpoints"]["005000"][tl.NORMALIZER_FILES[0]]
    assert row["compared"] == 2 and row["other_keys_different"] == ["index.max"]    # 状態・行動以外は記録だけ