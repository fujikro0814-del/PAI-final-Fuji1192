"""記録の再生（recovla.record.replay）の確かめ: 描画なしで記録した 1 本が、物理の 1 ステップごとの再生でビット一致し、
20 Hz の行動からの再生で 5 mm 以内に収まり、行動の定義の取り違えは見つかる。

記録は ScriptPad → 積分器 → 制御器の入口と記録器（EpisodeWriter・FrameSampler）で作る
（開始時に 1 フレーム、25 物理ステップごとに 1 フレーム、保存時に打ち切り）。
"""
import mujoco
import numpy as np
import pytest

from recovla.record import recorder, replay
from recovla.sim import control, render
from recovla.sim.device import ScriptPad, pad_state


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(control.SCENE_PATH)


@pytest.fixture(scope="module")
def renderer(model):
    r = mujoco.Renderer(model, control.IMAGE_SIZE, control.IMAGE_SIZE)
    yield r
    render.close_renderer(r)


@pytest.fixture(scope="module")
def episode(model, renderer, tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    data = mujoco.MjData(model)
    controller = control.make_collect_controller(model, data)
    pad = ScriptPad()
    integrator = control.make_integrator(pad, model, controller)
    integrator.start()
    start = control.settle_start_state(model)
    placement = recorder.training_placements()[4]
    control.reset_episode(model, data, controller, integrator, start, placement)
    sampler = recorder.FrameSampler(model, controller, renderer, control.CAMERAS)
    every = control.RECORD_EVERY
    timestep = float(model.opt.timestep)
    writer = recorder.EpisodeWriter(raw, 0, "2026-09-25", control.CAMERAS, every, every * timestep)
    grip = controller.gripper_indices[0]
    writer.add_step(controller.desired_pos, float(data.ctrl[grip]))
    writer.add_frame(*sampler.capture(data, 0))
    step = {"n": 0}

    def run(steps, **pad_kw):
        pad.state = pad_state(**pad_kw)
        integrator.refresh()
        for _ in range(steps):
            controller.update(integrator)
            mujoco.mj_step(model, data)
            step["n"] += 1
            writer.add_step(controller.desired_pos, float(data.ctrl[grip]))
            if step["n"] % every == 0:
                writer.add_frame(*sampler.capture(data, step["n"]))

    run(300, vel=np.array([0.20, 0.10, 0.0]))     # 加速して動く
    run(137)                                       # 止まる（部分の境目でない所）
    run(400, vel=np.array([0.0, 0.0, -0.10]))     # 下りる
    run(1, button_grip=True)                       # 閉じる
    run(212)
    run(350, vel=np.array([0.0, 0.20, 0.10]))     # 持ち上げて動く
    run(1, button_grip=True)                       # 開く
    run(240)
    n_keep = step["n"] // every + 1
    last = writer.frames[n_keep - 1]
    box = data.xpos[model.body(recorder.BOX_BODY).id]
    meta = {"episode_id": 0, "placement_id": placement.placement_id, "placement_seed": placement.seed,
            "cube_init": {"x": placement.x, "y": placement.y, "z": recorder.CUBE_HALF, "yaw": placement.yaw},
            "success": recorder.cube_in_box(last["cube_pos"], box), "record_every": every,
            "record_hz": round(1.0 / (every * timestep)), "duration_s": (n_keep - 1) * every * timestep,
            "cameras": {"names": list(control.CAMERAS), "width": control.IMAGE_SIZE, "height": control.IMAGE_SIZE}}
    extra = {"start_qpos": start.qpos, "start_ctrl": start.ctrl, "start_q_des": start.q_des}
    path = writer.finalize(n_keep, meta, extra)
    return replay.load_episode(path)


@pytest.mark.render
def test_step_replay_is_bit_exact(episode, renderer, model):
    rep = replay.replay(episode, "step", renderer, model)
    r = replay.compare(episode, rep)
    assert r["bit_exact_state"] and r["gripper_mismatch_frames"] == 0
    assert r["x_des_err_max_m"] == 0.0
    assert r["img_overhead_max_diff"] <= replay.IMAGE_TOL
    assert r["img_wrist_max_diff"] <= replay.IMAGE_TOL


@pytest.mark.render
def test_window_actions_reproduce_trajectory(episode, renderer, model):
    assert episode.data["gripper_closed"].any()          # 記録の中で閉じている
    rep = replay.replay(episode, "window", renderer, model)
    r = replay.compare(episode, rep, images=False)
    assert r["ee_err_max_m"] < replay.WINDOW_TOL_M
    assert r["x_des_err_max_m"] < 1e-12                   # フレームの時点で x_des に一致
    assert r["gripper_mismatch_frames"] == 0


@pytest.mark.render
@pytest.mark.parametrize("mode", ["last_step", "shifted"])
def test_wrong_action_definitions_are_detected(episode, renderer, model, mode):
    rep = replay.replay(episode, mode, renderer, model)
    r = replay.compare(episode, rep, images=False)
    assert r["ee_err_max_m"] >= replay.WINDOW_TOL_M


@pytest.mark.render
def test_episode_directory_is_not_modified(episode, renderer, model):
    before = {p: p.stat().st_mtime_ns for p in episode.path.rglob("*")}
    replay.replay(episode, "window", renderer, model)
    assert {p: p.stat().st_mtime_ns for p in episode.path.rglob("*")} == before


def test_scripted_pad_presses_once_per_toggle(model):
    data = mujoco.MjData(model)
    controller = control.make_collect_controller(model, data)
    pad = replay.ScriptedPad(controller)
    pad.reset(controller.target_pos)
    controller.gripper_closed = False
    controller.prev_grip = False
    pad.closed_target = True
    controller.update(pad)                  # 押した瞬間の 1 回で閉じる
    assert controller.gripper_closed is True
    for _ in range(5):                      # 目標の状態になった: もう押さない
        controller.update(pad)
    assert controller.gripper_closed is True
    pad.closed_target = False
    controller.update(pad)
    assert controller.gripper_closed is False
