"""評価の 1 試行（指示「全部片付けて」のような複数手順のタスク）。世界は物理の周期で必ず進み、
実行器（runtime/executor.py）はその中で呼ばれる。

    meta, arrays, runtime_log = run_task_trial(world, suite, make_task_runtime, layout, text, seed)

評価の側が真の状態で記録するもの: 色ごとの真の成功の時刻（目標が箱の成功の体積で 1 s 静止）、終わりに箱に入っている色、
真の状態のフレーム（20 Hz）。監査 g1〜g3 は harness/loop.py と同じ。

別のタスクの世界（world.place_task を持つ世界。harness/world_cup_plate のカップと皿）は run_place_task_trial へ渡す（配置は
世界が同じシードから引き直す。渡された 3 色の配置は使わない）。記録は同じ形に、物の名前で引ける final_in_place・final_on_detail・
goal_met・task・vla_known を足す（集計が読む all_three_in_box には goal_met、final_in_box には final_in_place を写す）。
"""
import numpy as np

from recovla.api import task_spec as TS
from recovla.api.harness import goal_truth as GT
from recovla.common.seeds import COLORS
from recovla.expert.script import PhaseParams
from recovla.harness.loop import TRUTH_KEYS, g1_audit
from recovla.harness.robot_io import SimRobotIO, fixed_latency
from recovla.harness.setup import nominal_setup
from recovla.record import episode as E
from recovla.sim import frames
from recovla.sim.rig import quiet


def run_task_trial(world, suite, make_task_runtime, layout, text: str, seed: int, cfg: dict = None,
                   time_limit_s: float = 200.0):
    if getattr(world, "place_task", None) is not None:
        return run_place_task_trial(world, suite, make_task_runtime, world.sample_layout(seed, layout.start), text, seed,
                                    cfg, time_limit_s)
    cfg = cfg or world.cfg
    ev = cfg["eval"]
    rest_speed, rest_hold = float(ev["success"]["rest_speed"]), float(ev["success"]["rest_hold_s"])
    pp = PhaseParams.from_config()
    world.reset(layout)
    setup = suite.start_trial(seed, world.data, nominal_setup(cfg))
    suite.prime(world.data)
    rtv = cfg.get("runtime_v2", {})
    io = SimRobotIO(world, suite, seed, fixed_latency=fixed_latency(rtv))
    ex = make_task_runtime(io, setup)
    ex.start(text, seed)
    dt = world.timestep
    st = {"hold": {}, "truth_t": {}}
    truth_log = []

    def capture():
        world.integrator.x_cmd = ex.prt.motion.x_cmd
        world.controller.gripper_closed = bool(ex.prt.closed)
        color = ex.plan["steps"][ex.j] if (ex.plan and 0 <= ex.j < len(ex.plan["steps"])) else COLORS[0]
        f, _ = E.capture_frame(world, color, 0.0, pp, render=False)
        f["x_des"] = ex.prt.motion.x_cmd
        f["gripper_closed"] = bool(ex.prt.closed)
        f["plan_step"] = ex.j
        truth_log.append(f)

    def on_step(r):
        d = r.data
        for i, c in enumerate(COLORS):                    # 手順によらず、色ごとに真の成功の時刻を記録する
            if c in st["truth_t"]:
                continue
            v = r.cube_vadr[i]
            if frames.in_box(d.xpos[r.cube_ids[i]], r.box) and float(np.linalg.norm(d.qvel[v:v + 3])) < rest_speed:
                st["hold"][c] = st["hold"].get(c, 0.0) + dt
                if st["hold"][c] >= rest_hold - 1e-9:
                    st["truth_t"][c] = float(r.step * dt)
            else:
                st["hold"][c] = 0.0
        if r.step % r.record_every == 0:
            capture()

    stops = 0
    audits = [g1_audit(ex, world, suite)]
    capture()
    with quiet():
        while not ex.finished and world.step * dt < time_limit_s - 1e-9:
            t_before = float(world.data.time)
            ex.tick()
            world.apply_joint_commands(io.take_commands())
            world.physics_step(on_step)
            suite.on_physics_step(world.data)
            if abs(float(world.data.time) - t_before - dt) > 1e-9:
                stops += 1
            if world.step % 2500 == 0:
                audits.append(g1_audit(ex, world, suite))
    audits.append(g1_audit(ex, world, suite))
    final_in_box = {c: bool(frames.in_box(world.data.xpos[world.cube_ids[i]], world.box)) for i, c in enumerate(COLORS)}
    rec = ex.record()
    plan_steps = (rec["plan"] or {}).get("steps") or []
    arrays = {k: np.array([f[k] for f in truth_log]) for k in TRUTH_KEYS if k != "target"}
    arrays["plan_step"] = np.array([f["plan_step"] for f in truth_log], dtype=np.int16)
    ioa = io.audit()
    meta = {
        "seed": int(seed), "text": text, "layout": {"kind": layout.kind, "start": layout.start, "seed": layout.seed,
                                                    "table_colors": list(layout.table_colors)},
        "detected": rec["detected"], "plan": rec["plan"], "steps": rec["steps"], "returns": rec["returns"],
        "stopped": rec["stopped"], "startup": rec["startup"], "truth_success_t": st["truth_t"],
        "final_in_box": final_in_box, "all_planned_in_box": bool(plan_steps) and all(final_in_box[c] for c in plan_steps),
        "all_three_in_box": all(final_in_box.values()), "t_end": float(world.data.time), "timed_out": not ex.finished,
        "audit": {"g1": {"checks": len(audits), "violations": sum(len(x["violations"]) for x in audits),
                         "violating_types": sorted({v for x in audits for v in x["violations"]})},
                  "g2": {"world_stops": stops, "early_use": ioa["early_use"], "n_compute": ioa["n_compute"],
                         "fixed_latency": ioa["fixed_latency"], "compute_wall_s": ioa["compute_wall_s"]},
                  "g3": world.audit_summary()},
        "sensor_trial_values": suite.trial_values,
    }
    return meta, arrays, {"executor": {"judge_rows": rec["judge_rows"]}, "runtime": ex.prt.trace(),
                          "latency": ioa["latency_by_kind"], "compute_wall_s": io.wall_by_kind()}


PLACE_TRUTH_KEYS = ("step", "sim_time", "ee_pos", "ee_quat", "fingertip", "fingers", "x_des", "gripper_closed", "obj_pos",
                    "obj_quat", "obj_linvel", "obj_in_place", "contact_robot", "plan_step")


def run_place_task_trial(world, suite, make_task_runtime, layout, text: str, seed: int, cfg: dict = None,
                         time_limit_s: float = 200.0):
    """別のタスクの世界（world.place_task、harness/world_cup_plate）の試行。流れ・監査・計算の時間のモデルは run_task_trial と同じ。
    真の成功の時刻は、物ごとに「目標の場所の上に載って立っている（goal_truth.on_place_truth）かつ速さ < rest_speed」が
    rest_hold_s 続いた時刻。既知の情報は world.setup_info（VLA の側の既知の値）に較正の誤差を足したもの。"""
    cfg = cfg or world.cfg
    spec = world.place_task
    ev = cfg["eval"]
    rest_speed, rest_hold = float(ev["success"]["rest_speed"]), float(ev["success"]["rest_hold_s"])
    world.reset(layout)
    setup = suite.start_trial(seed, world.data, world.setup_info(cfg))
    suite.prime(world.data)
    rtv = cfg.get("runtime_v2", {})
    io = SimRobotIO(world, suite, seed, fixed_latency=fixed_latency(rtv))
    ex = make_task_runtime(io, setup)
    ex.start(text, seed)
    dt = world.timestep
    names = tuple(spec.goal.objects)
    st = {"hold": {}, "truth_t": {}}
    truth_log = []

    def capture():
        s = world.forward_scratch()
        h = world.hand_id
        quat = s.xquat[h].copy()
        if quat[0] < 0.0:
            quat = -quat
        c_robot, _ = world.meter.take_window()
        poses = {n: world.object_pose(n, s) for n in names}
        on = world.in_place_truth(s)
        truth_log.append({"step": world.step, "sim_time": float(s.time), "ee_pos": s.xpos[h].copy(), "ee_quat": quat,
                          "fingertip": frames.fingertip_center(s, h), "fingers": s.qpos[world.finger_qadr].copy(),
                          "x_des": np.asarray(ex.prt.motion.x_cmd, float).copy(), "gripper_closed": bool(ex.prt.closed),
                          "obj_pos": np.array([world.object_center(n, s) for n in names]),
                          "obj_quat": np.array([poses[n][1] for n in names]),
                          "obj_linvel": np.array([poses[n][2] for n in names]),
                          "obj_in_place": np.array([bool(on[n]) for n in names]), "contact_robot": c_robot,
                          "plan_step": ex.j})

    def on_step(r):
        on = None
        for n in names:                                   # 手順によらず、物ごとに真の成功の時刻を記録する
            if n in st["truth_t"]:
                continue
            on = r.in_place_truth() if on is None else on
            v = float(np.linalg.norm(r.object_pose(n)[2]))
            if on[n] and v < rest_speed:
                st["hold"][n] = st["hold"].get(n, 0.0) + dt
                if st["hold"][n] >= rest_hold - 1e-9:
                    st["truth_t"][n] = float(r.step * dt)
            else:
                st["hold"][n] = 0.0
        if r.step % r.record_every == 0:
            capture()

    stops = 0
    audits = [g1_audit(ex, world, suite)]
    capture()
    with quiet():
        while not ex.finished and world.step * dt < time_limit_s - 1e-9:
            t_before = float(world.data.time)
            ex.tick()
            world.apply_joint_commands(io.take_commands())
            world.physics_step(on_step)
            suite.on_physics_step(world.data)
            if abs(float(world.data.time) - t_before - dt) > 1e-9:
                stops += 1
            if world.step % 2500 == 0:
                audits.append(g1_audit(ex, world, suite))
    audits.append(g1_audit(ex, world, suite))
    final_in_place = {n: bool(v) for n, v in world.in_place_truth().items()}
    goal_met = GT.goal_met_truth(spec, final_in_place)
    rec = ex.record()
    plan_steps = (rec["plan"] or {}).get("steps") or []
    arrays = {k: np.array([f[k] for f in truth_log]) for k in PLACE_TRUTH_KEYS if k != "plan_step"}
    arrays["plan_step"] = np.array([f["plan_step"] for f in truth_log], dtype=np.int16)
    ioa = io.audit()
    planned = [TS.step_of_cue(spec, c).obj for c in plan_steps]
    meta = {
        "seed": int(seed), "text": text, "task": spec.name, "layout": dict(layout.to_json(), table_colors=[]),
        "detected": rec["detected"], "plan": rec["plan"], "steps": rec["steps"], "returns": rec["returns"],
        "stopped": rec["stopped"], "startup": rec["startup"], "truth_success_t": st["truth_t"],
        "final_in_place": final_in_place, "final_on_detail": {n: world.on_detail(n) for n in names}, "goal_met": goal_met,
        "final_in_box": dict(final_in_place),                       # 集計用（物の名前で引く。箱は無い）
        "all_planned_in_box": bool(planned) and all(final_in_place[n] for n in planned),
        "all_three_in_box": goal_met,                               # 集計の成功の欄（このタスクでは goal_met）
        "t_end": float(world.data.time), "timed_out": not ex.finished,
        "vla_known": world.vla_known(cfg),
        "audit": {"g1": {"checks": len(audits), "violations": sum(len(x["violations"]) for x in audits),
                         "violating_types": sorted({v for x in audits for v in x["violations"]})},
                  "g2": {"world_stops": stops, "early_use": ioa["early_use"], "n_compute": ioa["n_compute"],
                         "fixed_latency": ioa["fixed_latency"], "compute_wall_s": ioa["compute_wall_s"]},
                  "g3": world.audit_summary()},
        "sensor_trial_values": suite.trial_values,
    }
    return meta, arrays, {"executor": {"judge_rows": rec["judge_rows"]}, "runtime": ex.prt.trace(),
                          "latency": ioa["latency_by_kind"], "compute_wall_s": io.wall_by_kind()}
