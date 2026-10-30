"""評価の本体 scripts/eval_resumable.py の環境の記録と、環境の食い違いでの停止の確かめ（CPU だけ。GPU・シミュレーション・
nvidia-smi を使わない）。

使い方: .venv\\Scripts\\python.exe -m pytest -q tests\\test_eval_resumable_env.py -p no:cacheprovider
読むもの: scripts/eval_resumable.py（importlib）。書くもの: pytest の一時フォルダだけ。
eval_resumable.py が読み込む eval_trials・ops_tools は偽物に差し替え、read_env も差し替える（ドライバの版を「読み替える」）。
"""
import importlib.util
import json
import pathlib
import sys
import types

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("s4_resume_t", ROOT / "scripts" / "eval_resumable.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["s4_resume_t"] = m
    spec.loader.exec_module(m)
    return m


OLD = {"driver": "560.94", "torch": "2.11.0+cu126", "torch_cuda": "12.6", "os_build": "26300.1", "git_head": "a" * 40}
NEW = dict(OLD, driver="610.88")


def _fake_ops():
    return types.SimpleNamespace(
        LIVE_STATUS={"starting", "loading", "running", "quiet_wait", "memory_wait"},
        _alive=lambda pid, ct=None: False,
        memory_gb=lambda: {"phys_free_gb": 50.0, "commit_free_gb": 50.0},
        window_state=lambda now, w: {"in_window": False, "minutes_to_window_end": 0},
        _now=lambda: None, _git=lambda *a: "", _reg_value=lambda p, n: None)


def _fake_82(out_root: pathlib.Path):
    trials = [(44400, None, "red"), (44401, None, "green")]
    return types.SimpleNamespace(
        CFG={"planner": {"retry": 1, "step_timeout_s": 30.0}, "eval": {"time_limit_s": 30.0}, "runtime": {"exec_interval": 6}},
        OUT=out_root, CKPT={"R1v3": "x"}, trial_list=lambda s: trials, ablate=lambda c, x: c,
        _json_default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o), _gate_mark=lambda out: None)


def _write_complete(d: pathlib.Path, i: int, seed: int, target: str, env):
    d.mkdir(parents=True, exist_ok=True)
    meta = {"success": True, "seed": seed, "target": target, "trial": i, "audit": {}, "time_limit_s": 60.0,
            "induce": {"kind": None, "established": False},
            "time_limits": {"time_limit_s": 60.0}}
    if env is not None:
        meta["env"] = env
    np.savez(d / f"trial_{i:04d}.npz", cube_pos=np.zeros((2, 3, 3)))
    (d / f"trial_{i:04d}.json").write_text(json.dumps(meta), encoding="utf-8")
    (d / f"runtime_{i:04d}.json").write_text("{}", encoding="utf-8")


def _setup(mod, monkeypatch, tmp_path, rec_env, cur_env):
    out_root = tmp_path / "v2eval"
    d = out_root / "S4T" / "C1"
    _write_complete(d, 0, 44400, "red", rec_env)
    monkeypatch.setattr(mod, "load_ops", _fake_ops)
    monkeypatch.setattr(mod, "load_82", lambda allow: _fake_82(out_root))
    monkeypatch.setattr(mod, "read_env", lambda ops: dict(cur_env, read="t", source="test"))
    monkeypatch.setattr(mod, "S4", tmp_path / "s4")
    return d


ARGV = ["run", "--experiment", "S4T", "--condition", "C1", "--model", "R1v3", "--trials", "natural:44400:2",
        "--exec-interval", "6", "--no-safety", "--ignore-quiet"]


def test_parse_nvidia_smi_driver(mod):
    assert mod.parse_nvidia_smi_driver("610.88\n") == "610.88"
    assert mod.parse_nvidia_smi_driver("  560.94  \r\n560.94\r\n") == "560.94"
    assert mod.parse_nvidia_smi_driver("NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.") is None
    assert mod.parse_nvidia_smi_driver("") is None


def test_env_conflicts_rules(mod):
    st = [(True, "ok", {"env": OLD}), (True, "ok", {}), (False, "missing", None)]
    c = mod.env_conflicts(NEW, st, {"sessions": [{"env": OLD}]})
    assert c["records"] == {0: {"driver": ("560.94", "610.88")}}
    assert c["unknown_records"] == [1]
    assert c["previous_session"] == {"driver": ("560.94", "610.88")}
    assert mod.env_conflicts(OLD, [(True, "ok", {"env": OLD})], {"sessions": [{"env": OLD}]}) == {}
    # git の HEAD の違いでは止めない（記録だけ）
    assert mod.env_conflicts(dict(OLD, git_head="b" * 40), [(True, "ok", {"env": OLD})], {"sessions": []}) == {}
    # 前の回で今の環境への切り替えを受け入れていれば、古い環境の記録は分けてあるので止めない
    log = {"sessions": [{"env": OLD}, {"env": NEW, "env_segment": {"accepted": "--accept-env-change"}}]}
    assert mod.env_conflicts(NEW, [(True, "ok", {"env": OLD})], log) == {}
    segs = mod.env_segments([(0, {"env": OLD}), (1, {"env": NEW}), (2, {"env": NEW})])
    assert [s["trials"] for s in segs] == [[0], [1, 2]]


def test_driver_change_stops_with_exit_3(mod, monkeypatch, tmp_path):
    d = _setup(mod, monkeypatch, tmp_path, OLD, NEW)
    before = {p.name: p.read_bytes() for p in d.iterdir()}
    assert mod.main(ARGV + ["--dry-run"]) == 3
    assert mod.main(ARGV) == 3
    assert not (d / "resume_log.json").exists() and not (d / "progress.json").exists()
    assert {p.name: p.read_bytes() for p in d.iterdir()} == before          # 何も書かず、何も動かさない


def test_unknown_env_record_stops_with_exit_3(mod, monkeypatch, tmp_path):
    _setup(mod, monkeypatch, tmp_path, None, NEW)
    assert mod.main(ARGV) == 3


def test_unreadable_driver_stops_with_exit_3(mod, monkeypatch, tmp_path):
    _setup(mod, monkeypatch, tmp_path, NEW, dict(NEW, driver=None))
    assert mod.main(ARGV) == 3


def test_same_env_dry_run_passes(mod, monkeypatch, tmp_path):
    _setup(mod, monkeypatch, tmp_path, NEW, NEW)
    assert mod.main(ARGV + ["--dry-run"]) == 0


def test_accept_env_change_writes_segment(mod, monkeypatch, tmp_path):
    d = _setup(mod, monkeypatch, tmp_path, OLD, NEW)
    (d / "STOP").write_text("", encoding="utf-8")                          # 回す前に止める（シミュレーションを動かさない）
    assert mod.main(ARGV + ["--accept-env-change"]) == 1
    log = json.loads((d / "resume_log.json").read_text(encoding="utf-8"))
    s = log["sessions"][-1]
    assert s["env"]["driver"] == "610.88"
    assert s["env_segment"]["conflicts"]["records"] == {"0": {"driver": ["560.94", "610.88"]}}
    assert s["env_segment"]["first_trial"] == 1
    prog = json.loads((d / "progress.json").read_text(encoding="utf-8"))
    assert prog["env"]["driver"] == "610.88" and prog["status"] == "stopped"
    # 次の回は、受け入れ済みの新しい環境のままなら止まらない（dry-run で 0）
    (d / "STOP").unlink()
    assert mod.main(ARGV + ["--dry-run"]) == 0
