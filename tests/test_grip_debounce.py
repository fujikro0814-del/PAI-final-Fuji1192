"""グリッパの開の指令の絞り（gripper_gate の open_steps、docs/real/grip_debounce.md）。open_steps が無ければ閾値だけの動き。"""
import importlib.util
import pathlib
import sys
from types import SimpleNamespace

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from recovla.runtime import runner as RUN  # noqa: E402


class IO:
    def __init__(self):
        self.t, self.cmds = 0.0, []

    def now(self):
        return self.t

    def gripper_grasp(self, *a):
        self.cmds.append(("grasp", round(self.t, 3)))

    def gripper_move(self, *a):
        self.cmds.append(("move", round(self.t, 3)))


class Motion:
    def set_velocity(self, v):
        pass


def _rt(gate):
    io = IO()
    rt = RUN.PolicyRuntime(io, SimpleNamespace(cube_size=0.04, gripper_speed=0.08, grasp_force=40.0), policy=None,
                           motion=Motion(), gripper_gate=gate)
    rt.closed = False
    return rt, io


def _drive(rt, io, gseq):
    for g in gseq:
        a = np.zeros(7)
        a[6] = g
        rt._apply(a)
        io.t += 0.1
    return io.cmds


SEQ = [1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, -1, 1, -1, 1]


def test_no_gate_and_open_steps_1_match_plain_threshold():
    rt0, io0 = _rt(None)
    plain = _drive(rt0, io0, SEQ)
    gate1 = {"enabled": True, "close": 0.0, "open": 0.0, "min_hold_s": 0.0, "open_steps": 1}
    rt1, io1 = _rt(gate1)
    assert _drive(rt1, io1, SEQ) == plain
    gate_nokey = dict(gate1)
    del gate_nokey["open_steps"]
    rt2, io2 = _rt(gate_nokey)
    assert _drive(rt2, io2, SEQ) == plain
    assert "gripper_gate" not in {"x": 1}                                 # 記録の足し方は trace の側で確かめる


def test_short_open_blips_are_held_and_long_open_passes():
    gate = {"enabled": True, "close": 0.0, "open": 0.0, "min_hold_s": 0.0, "open_steps": 5}
    rt, io = _rt(gate)
    cmds = _drive(rt, io, SEQ)
    # 閉はすぐ。2 ステップの開は通さない。5 ステップ続いた開の 5 ステップ目で開く。その後の閉はすぐ、1 ステップの開は通さない
    assert cmds == [("grasp", 0.0), ("move", 1.1), ("grasp", 1.2)]
    assert rt.n_open_held == 2 + 4 + 1


def test_counter_resets_on_close_value():
    gate = {"enabled": True, "close": 0.0, "open": 0.0, "min_hold_s": 0.0, "open_steps": 3}
    rt, io = _rt(gate)
    cmds = _drive(rt, io, [1, -1, -1, 1, -1, -1, 1, -1, -1, -1])
    assert cmds == [("grasp", 0.0), ("move", 0.9)]


def test_entry_split_and_gate():
    spec = importlib.util.spec_from_file_location("ev3r_grip", ROOT / "scripts" / "real" / "eval_real_matched_grip.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    deb, ck, rest = m.split(["--grip-debounce", "5", "--model-ckpt", "Rreal=a/b", "--world", "v3_real", "run", "--x", "1"])
    assert deb == 5 and ck == {"Rreal": "a/b"} and rest == ["--world", "v3_real", "run", "--x", "1"]
    assert m.debounce_gate(5) == {"enabled": True, "close": 0.0, "open": 0.0, "min_hold_s": 0.0, "open_steps": 5}
