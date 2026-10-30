"""開発の目標の文書と版のタグの照合と、G1（実行系は真値を読まない）の境界の検査。"""
import importlib.util
import pathlib

import pytest

from recovla.common import goals

ROOT = pathlib.Path(__file__).resolve().parents[1]
# 目標の文書・版のタグ・G1 の既知の一覧は開発用のリポジトリにだけある（提出用のリポジトリでは飛ばす）
needs_governance = pytest.mark.skipif(not (ROOT / goals.GOALS_PATH).is_file(), reason="目標書と版のタグがないリポジトリ")


def _g1():
    spec = importlib.util.spec_from_file_location("check_real_boundary", ROOT / "scripts" / "check_real_boundary.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@needs_governance
def test_goals_fingerprint_matches_tag():
    fp = goals.fingerprint()
    assert fp["version"] >= 2 and fp["tag"] == f"goals-v{fp['version']}" and len(fp["sha256"]) == 64


@needs_governance
def test_goals_rejects_edited_copy(tmp_path, monkeypatch):
    real = (ROOT / goals.GOALS_PATH).read_bytes()
    monkeypatch.setattr(goals, "_blob_at", lambda root, tag: real)
    monkeypatch.setattr(goals, "_tags", lambda root: ["goals-v2"])
    (tmp_path / "docs").mkdir()
    (tmp_path / goals.GOALS_PATH).write_bytes(real + b"\n")
    with pytest.raises(goals.GoalsMismatch):
        goals.fingerprint(tmp_path)
    (tmp_path / goals.GOALS_PATH).write_bytes(real)
    assert goals.fingerprint(tmp_path)["version"] == 2
    monkeypatch.setattr(goals, "_tags", lambda root: ["goals-v99"])    # 変更履歴に v99 の行がない
    with pytest.raises(goals.GoalsMismatch):
        goals.fingerprint(tmp_path)


def test_g1_scan_finds_truth_access():
    src = ("from recovla.sim import rig as R\n"
           "import recovla.expert.script\n"
           "def f(rig, frame, cfg):\n"
           "    a = rig.data.qpos[3]\n"
           "    b = frame['cube_pos']\n"
           "    c = cfg['scene']['box']\n"
           "    return rig.controller.gripper_closed\n")
    hits = _g1().scan(src)
    for k in ("import:recovla.sim.rig", "import:recovla.expert.script", "attr:qpos", "key:cube_pos", "key:box",
              "attr:controller", "attr:gripper_closed", "name:rig"):
        assert hits[k] >= 1, k


def test_g1_scan_allows_sensor_only_code():
    src = ("def f(sensor, setup):\n"
           "    return sensor.q, sensor.gripper_width, setup.box_size\n")
    assert not _g1().scan(src)


@needs_governance
def test_g1_no_new_violations_in_work_tree():
    assert _g1().main([]) == 0


def test_g1_runtime_v2_strict_scan():
    g = _g1()
    bad = ("from recovla.sim import rig\nimport recovla.eval.induce\n"
           "def f(self):\n    x = self.io.world.data\n    return truth['cube_pos']\n")
    hits = g.scan_runtime(bad, "x.py")
    for k in ("import:recovla.sim.rig", "import:recovla.eval.induce", "io:world", "key:cube_pos", "name:truth"):
        assert hits[k] >= 1, k
    ok = "from recovla.sim import control\nimport numpy as np\ndef f(self):\n    return self.io.sense(cameras=False).joints.q\n"
    assert not g.scan_runtime(ok, "y.py")


def test_reachability_audit_finds_planted_reference():
    from recovla.harness.audit import reachable_forbidden

    class World:
        pass

    class Boundary:
        def __init__(self, w):
            self.w = w

    class Runtime:
        pass

    w = World()
    rt = Runtime()
    rt.io = Boundary(w)                                       # 境界の中は辿らない
    rt.cb = (lambda: 1)
    assert reachable_forbidden(rt, {id(w)}, (), (Boundary,))["violations"] == []
    holder = {"x": [w]}
    rt.cb = (lambda: holder)                                  # 閉包を通って届く
    assert reachable_forbidden(rt, {id(w)}, (), (Boundary,))["violations"] == ["World"]
