"""自分の成功で学び直すデータの収集（試して採用しなかった方法）: 方策を評価と同じ実行系で走らせ、始めから記録し、
自分で成功した試行だけを保存する（生成プログラムへの引き継ぎはしない）。

    rig = V.V3RealGenRig(V.load_config("v3_real", generation=True))
    out = run_r2_selfroll(rig, make_runtime, seed, run_dir)     # make_runtime(io, setup) -> PolicyRuntime

走らせ方は handover_real.run_r1_handover の前半と同じ（実行系の tick → 指令を世界へ → 物理の 1 ステップ → センサのモデル、
遅延は robot_io.fixed_latency、サブタスクは 1 つ "put the {color} cube in the box"、配置は失敗の出やすい状態を多めに）。
指令の差し替え（action_filter）はしない。方策が出したものをそのまま実行する。
記録: 時刻 0 から。フレーム・行動の定義は生成プログラムの記録と同じ（x_des = 実行系の x_cmd、グリッパ = 実行系の開閉）。
記録のフレームのためにセンサのモデルをもう一度読むので、方策が受け取る雑音の列は記録しない走りと同じにはならない
（記録したものは、その走りで実際に起きたこと）。
立て直しの印: 失敗の判定（handover_real.Watch の miss・drop・misplace）を、引き継がずに数えるだけ（events）。
  1 つでもあれば recovered（失敗してから自分で立て直した成功）。
成功: 行動の境目で、目標が箱の中で静止し、開いている。そこから TAIL_S だけ方策を続け（離れる動きも記録する）、
  最後にもう一度確かめる。
捨てる: 時間切れ（POLICY_LIMIT_S、timeout）、目標でない立方体を持ち上げた（wrong_cube）、始めに箱の中にあった立方体が
  箱の外に出た（box_disturbed）、目標が箱の中で傾いた（tilted_in_box）、成功の後に箱から出た（lost_after_success）、
  監査の違反（g1_g2: 実行系が真の状態に届いた・世界が止まった・観測を早く使った、g3: 指令が Panda の上限を超えた）、
  配置を作れないシード（layout_failed、走らせない）。
"""
import time

import numpy as np

from recovla.common.seeds import COLORS
from recovla.eval import scene_trial as T
from recovla.expert import script as S
from recovla.expert.handover_real import STEPS_PER_ACTION, Watch, _own_motion, rollout_layout, yaw_rel_deg
from recovla.harness.loop import g1_audit
from recovla.harness.robot_io import SimRobotIO, fixed_latency
from recovla.record import episode as E
from recovla.record import snapshot
from recovla.sim import frames
from recovla.sim.rig import quiet

KIND = "S"
POLICY_LIMIT_S = 30.0          # 評価の 1 手順と同じ
TAIL_S = 1.0                   # 成功の後も方策を続けて記録する長さ


def run_r2_selfroll(rig, make_runtime, seed: int, run_dir=None, model_name: str = None, eval_cfg: dict = None) -> dict:
    """方策を最後まで走らせて記録し、自分で成功したら保存する。eval_cfg は実行系の設定（遅延の値を読む。None なら rig.cfg）。"""
    cfg = rig.cfg
    try:
        lay, color, lay_info = rollout_layout(int(seed), cfg)
    except RuntimeError as e:          # 配置を作れないシードは捨てて記録する
        return {"name": f"{KIND}_{int(seed)}", "seed": int(seed), "color": "-", "model": model_name,
                "rollout_layout": {"stratum": "-", "error": str(e)}, "saved": False, "discard": "layout_failed",
                "success": False, "recovered": False, "events": [], "closes": [], "t_success": None, "t_end": 0.0,
                "wall_s": 0.0}
    ti = COLORS.index(color)
    pp = S.PhaseParams.from_config(cfg)
    rig.motion = _own_motion(rig)
    rig.reset(lay)
    rtv = (eval_cfg or cfg).get("runtime_v2", {})
    io = SimRobotIO(rig, rig.suite, int(seed), fixed_latency=fixed_latency(rtv))
    rt = make_runtime(io, rig.setup_believed)
    rig.motion = rt.motion
    rt.start(T.instruction(color), int(seed))
    box0 = set(rig.cubes_in_box())
    dt_b = STEPS_PER_ACTION * rig.timestep
    watch = Watch(pp, dt_b)
    events, closes = [], []

    every = rig.record_every
    assert rig.step % (every * int(cfg["sim"]["stride"])) == 0
    start = snapshot.capture(rig.model, rig.data, rig.controller, rig.integrator, rig.step)
    rig.meter.reset_window()
    name = f"{KIND}_{int(seed)}_{color}_r0"
    writer = E.SceneEpisodeWriter(run_dir, name, rig.cameras) if run_dir else None
    frames_n = [0]
    rest = {"s": 0.0}

    def capture():
        tr = rig.truth(color, rig.data)
        rest["s"] = rest["s"] + every * rig.timestep if tr.target_speed < pp.rest_speed else 0.0
        f, im = E.capture_frame(rig, color, rest["s"], pp, writer is not None)
        f = rig.postprocess_frame(f)
        frames_n[0] += 1
        if writer is not None:
            writer.add_frame(f, im)

    def add_step():
        if writer is not None:
            writer.add_step(rt.motion.x_cmd, float(rig.data.ctrl[rig.grip_act]))

    rig.closed = bool(rt.closed)
    add_step()
    capture()
    wall0 = time.perf_counter()
    dt = rig.timestep
    stops = 0
    audits = [g1_audit(rt, rig, rig.suite)]
    discard, t_success, t_end = None, None, None
    with quiet():
        while True:
            if rig.step % STEPS_PER_ACTION == 0 and rig.step > 0:
                tr = rig.truth(color, rig.data)
                done = frames.in_box(tr.target_pos, tr.box) and tr.target_speed < pp.rest_speed and not rt.closed
                if t_success is None:
                    if done:
                        t_success = float(tr.t)
                    elif tr.t >= POLICY_LIMIT_S - 1e-9:
                        discard = "timeout"
                    else:
                        watch.update(tr, bool(rt.closed))
                        if watch.discard:
                            discard = watch.discard
                        elif watch.trigger:
                            events.append({"trigger": watch.trigger, "detail": watch.detail, "t": round(float(tr.t), 3)})
                            closes += watch.closes
                            watch = Watch(pp, dt_b)              # 引き継がずに数え続ける（次の閉じから見直す）
                            watch.prev_closed = bool(rt.closed)
                elif tr.t >= t_success + TAIL_S - 1e-9:
                    t_end = float(tr.t)
                if discard or t_end is not None:
                    break
            t_before = float(rig.data.time)
            rt.tick()
            rig.closed = bool(rt.closed)
            rig.apply_joint_commands(io.take_commands())
            rig.physics_step(None)
            add_step()
            if rig.step % every == 0:
                capture()
            if abs(float(rig.data.time) - t_before - dt) > 1e-9:
                stops += 1
            if rig.step % 2500 == 0:
                audits.append(g1_audit(rt, rig, rig.suite))
    audits.append(g1_audit(rt, rig, rig.suite))
    tr = rig.truth(color, rig.data)
    ioa = io.audit()
    g12 = {"g1_violations": sum(len(x["violations"]) for x in audits), "g2_world_stops": stops, "g2_early_use": ioa["early_use"]}
    in_box_end = set(rig.cubes_in_box())
    tilt = frames.tilt_deg(tr.cube_quat[ti])
    if discard is None:
        if not (frames.in_box(tr.target_pos, tr.box) and tr.target_speed < pp.rest_speed):
            discard = "lost_after_success"
        elif not box0 <= in_box_end:
            discard = "box_disturbed"
        elif tilt > float(cfg["inject"]["landing"]["max_tilt_deg"]):
            discard = "tilted_in_box"
        elif g12["g1_violations"] or stops or ioa["early_use"]:
            discard = "g1_g2"
    g3 = rig.audit_summary()
    if discard is None and int(g3["total_violations"]):
        discard = "g3"
    saved = discard is None
    summary = {"name": name, "seed": int(seed), "color": color, "rollout_layout": lay_info, "model": model_name,
               "saved": saved, "discard": discard, "success": t_success is not None and discard != "lost_after_success",
               "recovered": bool(events), "events": events, "closes": closes + watch.closes,
               "t_success": None if t_success is None else round(t_success, 3), "t_end": round(float(tr.t), 3),
               "n_frames": frames_n[0], "policy_actions": int(rt.k + 1), "audit_policy": g12, "g3": g3,
               "box_at_start": sorted(box0), "box_at_end": sorted(in_box_end),
               "final_target_pos": [round(float(v), 5) for v in tr.target_pos],
               "final_target_yaw_rel_deg": yaw_rel_deg(tr.cube_quat[ti]), "final_target_tilt_deg": round(float(tilt), 2),
               "gripper_gate": rt.trace().get("gripper_gate"),
               "rules": {"policy_limit_s": POLICY_LIMIT_S, "tail_s": TAIL_S},
               "wall_s": round(time.perf_counter() - wall0, 3)}
    if writer is not None:
        if saved:
            meta = {**summary, "kind": KIND, "instruction": T.instruction(color), "target": color,
                    "layout": lay.to_json(), "layout_seed": int(seed), "pair_id": int(seed), "retry": 0,
                    "layout_kind": lay.kind, "start_pose": lay.start, "duration_s": round(float(tr.t), 3),
                    "record_every": every, "record_dt": every * rig.timestep,
                    "record_hz": round(1.0 / (every * rig.timestep)), "timestep": rig.timestep, "rendered": True,
                    "frame0_image_shared_with_pair": False, "record_start_step": int(start.step)}
            meta.update(rig.episode_meta())
            summary["path"] = str(writer.finalize(meta, start.to_arrays(), every))
        else:
            writer.discard()
    return summary
