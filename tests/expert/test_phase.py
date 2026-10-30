"""生成プログラムの段階の判定 phase_of（純関数）と生成プログラムの単体検査（描画なしで通る）。

作った真値で段階の表を確かめ、描画なしの 1 本の生成で、箱の上で放した直後に settle が現れ descend が現れないこと
を確かめる。
"""
import dataclasses

import numpy as np
import pytest

from recovla.expert import generate as G
from recovla.expert.script import Phase, PhaseParams, Truth, phase_of
from recovla.sim import frames

PP = PhaseParams.from_config()
BOX = np.array([0.45, 0.25, 0.0])


def truth(**kw) -> Truth:
    base = dict(t=0.0, target=0,
                cube_pos=np.array([[0.40, -0.05, frames.CUBE_REST_Z], [0.50, 0.0, frames.CUBE_REST_Z],
                                   [0.45, -0.12, frames.CUBE_REST_Z]]),
                cube_quat=np.tile([1.0, 0, 0, 0], (3, 1)), cube_linvel=np.zeros((3, 3)),
                fingers=np.array([0.04, 0.04]), finger_vel=np.zeros(2), gripper_closed=False,
                hand_pos=np.array([0.307, 0.0, 0.355]), hand_vel=np.zeros(3),
                fingertip=np.array([0.307, 0.0, 0.252]), x_cmd=np.array([0.307, 0.0, 0.35]), box=BOX)
    base.update(kw)
    return Truth(**base)


def with_target(t: Truth, pos, vel=(0, 0, 0)) -> Truth:
    cp, cv = t.cube_pos.copy(), t.cube_linvel.copy()
    cp[0], cv[0] = pos, vel
    return dataclasses.replace(t, cube_pos=cp, cube_linvel=cv)


def test_approach_descend_close():
    assert phase_of(truth(), 0.0, PP) == Phase.approach
    ft = np.array([0.40, -0.05, 0.20])
    assert phase_of(truth(fingertip=ft, x_cmd=np.array([0.39, -0.05, 0.25])), 0.0, PP) == Phase.descend
    assert phase_of(truth(fingertip=ft, x_cmd=np.array([0.39, -0.05, PP.grasp_z])), 0.0, PP) == Phase.close


def test_lift_carry_release_reopen():
    tip = np.array([0.40, -0.05, 0.022])
    closed = dict(gripper_closed=True, fingers=np.array([0.02, 0.02]), fingertip=tip)
    assert phase_of(truth(**closed), 0.0, PP) == Phase.lift
    up = with_target(truth(**{**closed, "fingertip": np.array([0.40, -0.05, 0.08])}), [0.40, -0.05, 0.08])
    assert phase_of(up, 0.0, PP) == Phase.carry
    rel = dataclasses.replace(with_target(up, [0.48, 0.28, 0.075]), fingertip=np.array([0.48, 0.28, 0.077]),
                              x_cmd=np.array([0.475, 0.283, PP.release_z]))
    assert phase_of(rel, 0.0, PP) == Phase.release
    empty = truth(gripper_closed=True, fingers=np.array([0.001, 0.001]), fingertip=tip)   # 何も掴んでいない
    assert phase_of(empty, 0.0, PP) == Phase.reopen


def test_settle_right_after_release_over_the_box_not_descend():
    """放した直後: 開いていて、目標は指の間（宙）にあり、指先は目標の真上。descend ではなく settle。"""
    t = with_target(truth(fingertip=np.array([0.48, 0.28, 0.077]), x_cmd=np.array([0.475, 0.283, PP.release_z])),
                    [0.48, 0.28, 0.075])
    assert phase_of(t, 0.0, PP) == Phase.settle
    falling = with_target(t, [0.48, 0.28, 0.05], vel=(0, 0, -0.5))
    assert phase_of(falling, 0.0, PP) == Phase.settle
    landed = with_target(t, [0.48, 0.28, frames.BOX_FLOOR_Z + frames.CUBE_HALF])
    assert phase_of(landed, 0.0, PP) == Phase.settle                   # 箱の中で、まだ静止の時間が足りない
    assert phase_of(landed, PP.rest_hold_s, PP) == Phase.retreat
    at_retreat = dataclasses.replace(landed, x_cmd=PP.retreat_pose.copy())
    assert phase_of(at_retreat, PP.rest_hold_s, PP) == Phase.done


def test_settle_when_target_rolls_on_the_table():
    t = with_target(truth(), [0.40, -0.05, frames.CUBE_REST_Z], vel=(0.05, 0, 0))
    assert phase_of(t, 0.0, PP) == Phase.settle


def test_phase_of_is_pure():
    t = truth()
    before = {k: np.array(v, copy=True) for k, v in dataclasses.asdict(t).items()}
    phase_of(t, 0.1, PP)
    assert all(np.array_equal(before[k], v) for k, v in dataclasses.asdict(t).items())


@pytest.fixture(scope="module")
def rig():
    from recovla.sim.rig import SimRig
    r = SimRig(render=False)
    yield r
    r.close()


def test_episode_phase_sequence_and_settle_after_release(rig):
    a = G.run_attempt(rig, G.EpisodeSpec(50000, "red", "empty"), 0, None, False)
    assert a["success"] and a["failure"] is None
    ph = np.array([f["phase"] for f in a["_frames"]])
    first = {Phase(p).name: int(np.argmax(ph == p)) for p in np.unique(ph)}
    order = ["approach", "descend", "close", "lift", "carry", "release", "settle", "retreat", "done"]
    assert [p for p in order if p in first] == sorted(first, key=first.get)
    after = ph[first["release"]:]
    assert Phase.settle in after and Phase.descend not in after


def test_generation_is_deterministic(rig):
    from recovla.sim import scene
    color = scene.sample_layout(50001, "prefilled_1").table_colors[0]
    spec = G.EpisodeSpec(50001, color, "prefilled_1")
    a = G.run_attempt(rig, spec, 0, None, False)
    b = G.run_attempt(rig, spec, 0, None, False)
    for k in ("cube_pos", "x_des", "joints", "min_dist", "phase"):
        assert np.array_equal(np.array([f[k] for f in a["_frames"]]), np.array([f[k] for f in b["_frames"]])), k
