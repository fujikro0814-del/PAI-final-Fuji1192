"""自分の失敗から学び直すデータの収集: 学習した方策を評価と同じ実行系で走らせ、失敗が起きたら生成プログラムに
引き継いで、その場から箱に入れるまでを記録する（手順書 docs/api/r1_protocol.md の 2 節）。

    rig = V.V3RealGenRig(V.load_config("v3_real", generation=True))
    out = run_r1_handover(rig, make_runtime, seed, run_dir)     # make_runtime(io, setup) -> PolicyRuntime

前半（記録しない）: 評価と同じ順（実行系の tick → 指令を世界へ → 物理の 1 ステップ → センサのモデル）で方策を走らせる。
サブタスクは 1 つ（"put the {color} cube in the box"）。遅延は評価と同じ（robot_io.fixed_latency）。
引き継ぎの判定はデータ生成の側で真の状態を使う（実行系には入れない）。判定は行動の境目（物理 50 ステップ）ごと:
  (i)   miss      指先が目標から NEAR_M 以内で閉じ、どの立方体も LIFT_M 上がらないまま、
                  指が空で閉じ切った（幅 < min_grip_gap が FINGER_REST_S 続いた）・手だけが HAND_RISE_M 上がった・開き直した。
                  1 回目ですぐ引き継ぐ（閉じたままなら生成プログラムは開いて上がる＝reopen から）
  (ii)  drop      持ち上げた目標（閉じてから LIFT_M 上がった）が、閉じたまま指先から DROP_M 超離れた
        misplace  持ち上げた目標を放して SETTLE_S たっても箱に入っていない
  (iii) pre_release  目標を持ち上げたまま箱の上で、方策が初めて開く値を出したとき、放す位置（目標の中心）から
                  PRE_CUBE_M 以内に箱の中の立方体の足跡がある、または箱の壁まで PRE_WALL_M 未満（置く直前に外す）。
                  どちらも点からの距離。目標の足跡と壁の隙間は記録だけ。
                  その指令は閉のままに替え（action_filter）、次の境目で引き継ぐ
捨てる（前半）: 引き継ぐ前に成功（success_before_handover）、POLICY_LIMIT_S を超えた（timeout_before_handover）、
  目標でない立方体を持ち上げた（wrong_cube）、監査の違反（g1_g2。実行系が真の状態に届いた・世界が止まった・観測を早く使った）。
後半（記録する）: 生成プログラム（引き継ぎ用のシード）が実行系の Motion とハンドをそのまま受け継ぐ。
  記録・失敗の区分・保存の形は generate.run_attempt と同じ。
  目標が手の外（指先から DROP_M 超）で最初に静止したとき状態を確かめ、傾き > landing.max_tilt_deg（tilt）、箱の外寸の上（rim）、
  箱の手前の壁の際（y ≥ NEAR_WALL_Y、near_wall）、作業空間の外（outside_workspace）、他の立方体の上（stacked）なら捨てる。
  一度も掴まずに終わった（trivial）・立て直せなかった（script_*）・指令が Panda の上限を超えた（g3）も捨てる。

方策を回す配置（rollout_layout）は、失敗の出やすい状態を多めにする（集める速さのため）:
  シード % 10 が 0〜2: 箱に 2 個入り（prefilled_2）、3〜5: 目標の向き |yaw| 20〜30°、6〜7: 目標が手前（x < NEAR_X）、
  8〜9: そのまま。2 個入り以外の配置の種類は empty と prefilled_1 を半々（シード // 10 の偶奇）。
"""
import dataclasses
import time

import mujoco
import numpy as np

from recovla.common import seeds
from recovla.common.seeds import COLORS
from recovla.eval import scene_trial as T
from recovla.expert import generate as G
from recovla.expert import script as S
from recovla.harness.loop import HarnessHook, g1_audit
from recovla.harness.robot_io import SimRobotIO, fixed_latency
from recovla.record import episode as E
from recovla.record import snapshot
from recovla.sim import contact, frames, scene
from recovla.sim.rig import quiet

KIND = "H"
STEPS_PER_ACTION = 50
POLICY_LIMIT_S = 25.0          # 前半の上限（評価の 1 手順 30 s より短く）
NEAR_M = 0.06
LIFT_M = 0.015
DROP_M = 0.04
SETTLE_S = 2.0
FINGER_REST_S = 0.3            # 空で閉じ切った: 幅 < min_grip_gap がこの時間（境目で数える）
HAND_RISE_M = 0.03             # 手だけが上がった: 指先が閉じた時点から上がった量
PRE_CUBE_M = 0.025
PRE_WALL_M = 0.01
NEAR_WALL_Y = 0.06             # 配置の範囲の y の上限 0.05 の外（手が箱の手前の壁に当たる）
NEAR_X = 0.40
YAW_BAND_DEG = (20.0, 30.0)
LAYOUT_KEY = 4710              # 配置の偏りの乱数: SeedSequence([シード, 4710])
STRATA = {0: "pf2", 1: "pf2", 2: "pf2", 3: "yaw", 4: "yaw", 5: "yaw", 6: "near", 7: "near", 8: "free", 9: "free"}
TRIGGERS = ("miss", "drop", "misplace", "pre_release")


# ------------------------------------------------------------------------------ 方策を回す配置
def stratum(seed: int) -> str:
    return STRATA[int(seed) % 10]


def rollout_layout(seed: int, cfg: dict) -> tuple:
    """(配置, 目標の色, 偏りの情報)。配置は scene.sample_layout から、目標の立方体だけを層に合わせて引き直す。"""
    st = stratum(seed)
    kind = "prefilled_2" if st == "pf2" else ("empty" if (int(seed) // 10) % 2 == 0 else "prefilled_1")
    lay = scene.sample_layout(int(seed), kind)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), LAYOUT_KEY]))
    colors = lay.table_colors
    color = colors[int(rng.integers(len(colors)))]
    sc = cfg["scene"]
    cubes = dict(lay.cubes)
    x, y, yaw = cubes[color]
    others = [np.asarray(v[:2], float) for c, v in cubes.items() if c != color]
    min_d = float(sc["min_center_dist"])
    if st == "yaw":
        yaw = float(np.radians(rng.uniform(*YAW_BAND_DEG)) * (1.0 if rng.random() < 0.5 else -1.0))
    elif st == "near":
        rx, ry = sc["region"]["x"], sc["region"]["y"]
        for _ in range(1000):
            nx, ny = float(rng.uniform(rx[0], NEAR_X)), float(rng.uniform(*ry))
            if all(np.hypot(nx - o[0], ny - o[1]) >= min_d for o in others):
                x, y = nx, ny
                break
        else:
            raise RuntimeError(f"seed {seed}: 手前の目標を置けない")
    cubes[color] = (float(x), float(y), float(yaw))
    lay = dataclasses.replace(lay, cubes=cubes)
    info = {"stratum": st, "kind": kind, "target_x": round(float(x), 4), "target_y": round(float(y), 4),
            "target_yaw_deg": round(float(np.degrees(yaw)), 2), "prefilled": len(lay.prefilled)}
    return lay, color, info


# ------------------------------------------------------------------------------ 引き継ぎの判定
def yaw_rel_deg(quat) -> float:
    yaw = float(np.degrees(frames.quat_yaw(quat)))
    return round((yaw + 45.0) % 90.0 - 45.0, 2)


def release_check(tr: S.Truth, model) -> dict:
    """放す位置（目標の中心）の良し悪し。箱の中の他の立方体の足跡までの距離と、目標の足跡と壁の隙間。"""
    ti = tr.target
    p = np.asarray(tr.target_pos[:2], float)
    mine = frames.cube_footprint(tr.target_pos, tr.cube_quat[ti])
    cube_d = {}
    for i, c in enumerate(COLORS):
        if i != ti and frames.in_box(tr.cube_pos[i], tr.box):
            fp = frames.cube_footprint(tr.cube_pos[i], tr.cube_quat[i])
            cube_d[c] = frames.polygon_distance(np.array([p]), fp) if not _inside(p, fp) else 0.0
    walls = frames.wall_footprints(model)
    wall_d = {w: (0.0 if _inside(p, fp) else frames.polygon_distance(np.array([p]), fp)) for w, fp in walls.items()}
    gap = {w: frames.polygon_distance(mine, fp) for w, fp in walls.items()}
    bad_cube = any(v < PRE_CUBE_M for v in cube_d.values())
    bad_wall = min(wall_d.values()) < PRE_WALL_M
    return {"bad": bool(bad_cube or bad_wall), "bad_cube": bool(bad_cube), "bad_wall": bool(bad_wall),
            "cube_dist_m": {k: round(v, 4) for k, v in cube_d.items()}, "wall_dist_m": round(min(wall_d.values()), 4),
            "wall_gap_m": round(min(gap.values()), 4)}


def _inside(p, poly) -> bool:
    poly, p = np.asarray(poly, float), np.asarray(p, float)
    n = len(poly)
    s = []
    for i in range(n):
        e, r = poly[(i + 1) % n] - poly[i], p - poly[i]
        s.append(float(e[0] * r[1] - e[1] * r[0]))
    return bool(all(v >= 0 for v in s) or all(v <= 0 for v in s))


class Watch:
    """行動の境目ごとの真の状態から、引き継ぎの時点を決める。"""

    def __init__(self, pp: S.PhaseParams, dt_boundary: float):
        self.pp, self.dt = pp, float(dt_boundary)
        self.prev_closed = False
        self.iv = None                 # 閉じている間の記録
        self.release_t = None          # 持ち上げた目標を放した時刻
        self.trigger, self.detail, self.discard = None, None, None
        self.closes = []

    def update(self, tr: S.Truth, closed: bool) -> None:
        if self.trigger or self.discard:
            return
        ti = tr.target
        if closed and not self.prev_closed:                       # 閉じた
            near = float(np.linalg.norm(tr.fingertip - tr.target_pos)) < NEAR_M
            self.iv = {"t": tr.t, "near": near, "z0": tr.cube_pos[:, 2].copy(), "tip_z0": float(tr.fingertip[2]),
                       "rise": np.zeros(len(COLORS)), "empty_s": 0.0}
            self.release_t = None
            self.closes.append({"t": round(tr.t, 3), "near": near,
                                "tip_to_cube_m": round(float(np.linalg.norm(tr.fingertip - tr.target_pos)), 4)})
        if closed and self.iv is not None:
            iv = self.iv
            iv["rise"] = np.maximum(iv["rise"], tr.cube_pos[:, 2] - iv["z0"])
            others = [i for i in range(len(COLORS)) if i != ti]
            if any(iv["rise"][i] >= LIFT_M for i in others):
                self.discard = "wrong_cube"
                return
            lifted = iv["rise"][ti] >= LIFT_M
            gap = float(np.sum(tr.fingers))
            if lifted and float(np.linalg.norm(tr.target_pos - tr.fingertip)) > DROP_M:
                self._fire("drop", "closed_apart")
                return
            if iv["near"] and not lifted:
                iv["empty_s"] = iv["empty_s"] + self.dt if gap < self.pp.min_grip_gap else 0.0
                if iv["empty_s"] >= FINGER_REST_S - 1e-9:
                    self._fire("miss", "closed_empty")
                    return
                if float(tr.fingertip[2]) - iv["tip_z0"] >= HAND_RISE_M:
                    self._fire("miss", "hand_rose")
                    return
        if not closed and self.prev_closed and self.iv is not None:    # 開いた
            iv = self.iv
            if iv["rise"][ti] >= LIFT_M:
                self.release_t = tr.t
            elif iv["near"]:
                self.prev_closed = closed
                self._fire("miss", "reopened")
                return
            self.iv = None
        if not closed and self.release_t is not None and tr.t - self.release_t >= SETTLE_S - 1e-9:
            if not frames.in_box(tr.target_pos, tr.box):
                self._fire("misplace", "not_in_box_after_release")
                return
            self.release_t = None
        self.prev_closed = closed

    def _fire(self, trig: str, detail: str) -> None:
        self.trigger, self.detail = trig, detail


def state_check(tr: S.Truth, cfg: dict, model) -> str:
    """生成プログラムが立て直す前の、机の上の目標の状態。悪ければ理由、良ければ None。"""
    ti = tr.target
    pos = tr.target_pos
    if frames.tilt_deg(tr.cube_quat[ti]) > float(cfg["inject"]["landing"]["max_tilt_deg"]):
        return "tilt"
    outer = frames.box_outer_half(model)
    if bool(np.all(np.abs(pos[:2] - tr.box[:2]) < outer + frames.CUBE_HALF)):
        return "rim"
    ws = cfg["sim"]["workspace"]
    if not (ws["x"][0] <= pos[0] <= ws["x"][1] and ws["y"][0] <= pos[1] <= ws["y"][1]):
        return "outside_workspace"
    if float(pos[1]) >= NEAR_WALL_Y:
        return "near_wall"
    if float(pos[2]) - frames.CUBE_REST_Z > 0.5 * frames.CUBE_HALF:
        return "stacked"
    return None


def _own_motion(rig):
    if not hasattr(rig, "_own_motion"):
        rig._own_motion = rig.motion
    return rig._own_motion


# ------------------------------------------------------------------------------ 1 本
def run_r1_handover(rig, make_runtime, seed: int, run_dir=None, model_name: str = None, eval_cfg: dict = None) -> dict:
    """方策を走らせ、意図的な失敗なしで起きた失敗で引き継いで記録する。eval_cfg は実行系の設定（遅延の値を読む。None なら rig.cfg）。"""
    cfg = rig.cfg
    lay, color, lay_info = rollout_layout(int(seed), cfg)
    ti = COLORS.index(color)
    pp = S.PhaseParams.from_config(cfg)
    rig.motion = _own_motion(rig)
    rig.reset(lay)
    rtv = (eval_cfg or cfg).get("runtime_v2", {})
    io = SimRobotIO(rig, rig.suite, int(seed), fixed_latency=fixed_latency(rtv))
    rt = make_runtime(io, rig.setup_believed)
    rig.motion = rt.motion
    task = T.instruction(color)
    rt.start(task, int(seed))
    watch = Watch(pp, STEPS_PER_ACTION * rig.timestep)
    st = {"pending": None, "pre_checks": []}

    def act_filter(k, a):
        a = np.asarray(a, float)
        if rt.closed and a[6] <= 0.0 and st["pending"] is None:
            tr = rig.truth(color, rig.data)
            if S.holding(tr, pp) and S.lifted(tr, pp) and frames.over_box_interior(tr.target_pos, tr.box):
                chk = release_check(tr, rig.model)
                st["pre_checks"].append({"k": int(k), "t": round(tr.t, 3), **chk})
                if chk["bad"]:
                    st["pending"] = chk
        if st["pending"] is not None and a[6] <= 0.0:
            a = a.copy()
            a[6] = 1.0                                     # 開く指令を閉のままに替える（次の境目で引き継ぐ）
        return a
    rt.action_filter = HarnessHook(act_filter)

    wall0 = time.perf_counter()
    dt = rig.timestep
    stops = 0
    audits = [g1_audit(rt, rig, rig.suite)]
    trigger, detail, discard = None, None, None
    with quiet():
        while True:
            if rig.step % STEPS_PER_ACTION == 0:
                tr = rig.truth(color, rig.data)
                if rig.step > 0 and frames.in_box(tr.target_pos, tr.box) and tr.target_speed < pp.rest_speed \
                        and not rt.closed:
                    discard = "success_before_handover"
                elif tr.t >= POLICY_LIMIT_S - 1e-9:
                    discard = "timeout_before_handover"
                elif st["pending"] is not None:
                    trigger, detail = "pre_release", "bad_cube" if st["pending"]["bad_cube"] else "bad_wall"
                else:
                    watch.update(tr, bool(rt.closed))
                    trigger, detail, discard = watch.trigger, watch.detail, watch.discard
                if discard or trigger:
                    break
            t_before = float(rig.data.time)
            rt.tick()
            rig.closed = bool(rt.closed)
            rig.apply_joint_commands(io.take_commands())
            rig.physics_step(None)
            if abs(float(rig.data.time) - t_before - dt) > 1e-9:
                stops += 1
            if rig.step % 2500 == 0:
                audits.append(g1_audit(rt, rig, rig.suite))
    audits.append(g1_audit(rt, rig, rig.suite))
    rt.action_filter = None
    tr = rig.truth(color, rig.data)
    ioa = io.audit()
    g12 = {"g1_violations": sum(len(x["violations"]) for x in audits), "g2_world_stops": stops, "g2_early_use": ioa["early_use"]}
    state = {"t": round(float(tr.t), 3), "phase": S.Phase(S.phase_of(tr, 0.0, pp)).name,
             "tip_to_cube_m": round(float(np.linalg.norm(tr.fingertip - tr.target_pos)), 4),
             "tip_z": round(float(tr.fingertip[2]), 4), "hand_z_minus_grasp_z": round(float(tr.hand_pos[2] - pp.grasp_z), 4),
             "finger_gap": round(float(np.sum(tr.fingers)), 4), "gripper_closed": bool(rt.closed),
             "target_pos": [round(float(v), 4) for v in tr.target_pos], "target_yaw_rel_deg": yaw_rel_deg(tr.cube_quat[ti]),
             "target_tilt_deg": round(frames.tilt_deg(tr.cube_quat[ti]), 2),
             "target_moved_m": round(float(np.hypot(*(tr.target_pos[:2] - np.asarray(lay.cubes[color][:2])))), 4),
             "in_box": list(rig.cubes_in_box())}
    hinfo = {"model": model_name, "trigger": trigger, "detail": detail, "t_handover": round(float(tr.t), 3),
             "step_handover": int(rig.step), "state_at_handover": state, "closes": watch.closes,
             "pre_release_checks": st["pre_checks"], "policy_actions": int(rt.k + 1), "audit_policy": g12,
             "gripper_gate": rt.trace().get("gripper_gate"),
             "rules": {"policy_limit_s": POLICY_LIMIT_S, "near_m": NEAR_M, "lift_m": LIFT_M, "drop_m": DROP_M,
                       "settle_s": SETTLE_S, "finger_rest_s": FINGER_REST_S, "hand_rise_m": HAND_RISE_M,
                       "pre_cube_m": PRE_CUBE_M, "pre_wall_m": PRE_WALL_M, "near_wall_y": NEAR_WALL_Y}}
    base = {"name": f"{KIND}_{int(seed)}_{color}_r0", "seed": int(seed), "color": color, "rollout_layout": lay_info,
            "handover": hinfo}
    if discard is None and (g12["g1_violations"] or stops or ioa["early_use"]):
        discard = "g1_g2"
    if discard:
        return {**base, "saved": False, "discard": discard, "g3": rig.audit_summary(),
                "wall_s": round(time.perf_counter() - wall0, 3), "wall_policy_s": round(time.perf_counter() - wall0, 3)}
    wall_policy = time.perf_counter() - wall0
    assert rig.step % (rig.record_every * int(cfg["sim"]["stride"])) == 0
    # ---------------------------------------------------------------- 後半: 生成プログラム（記録する）
    if hasattr(rig, "_t_grip_cmd"):
        rig._t_grip_cmd = float(rt.t_grip_switch)          # 最後の開閉の指令の時刻を受け継ぐ
    rt_dt = rig.steps_per_read * rig.timestep
    hp = S.sample_params(seeds.script_rng(int(seed), color, G.HANDOVER_RETRY_KEY), cfg)
    expert = S.Expert(hp, rt_dt, cfg)
    mask = contact.ContactMeter.obstacle_mask(color, rig.cubes_in_box())
    every = rig.record_every
    frames_log = []
    start = snapshot.capture(rig.model, rig.data, rig.controller, rig.integrator, rig.step)
    rig.meter.reset_window()
    name = base["name"]
    writer = E.SceneEpisodeWriter(run_dir, name, rig.cameras) if run_dir else None

    def capture():
        f, im = E.capture_frame(rig, color, expert.clocks.target_rest_s, pp, writer is not None)
        f = rig.postprocess_frame(f)
        frames_log.append(f)
        if writer is not None:
            writer.add_frame(f, im)

    def on_rec(r):
        if writer is not None:
            writer.add_step(r.controller.desired_pos, float(r.data.ctrl[r.grip_act]))
        if r.step % every == 0:
            capture()

    rig.controller.desired_pos = rig.motion.x_cmd
    if writer is not None:
        writer.add_step(rig.controller.desired_pos, float(rig.data.ctrl[rig.grip_act]))
    capture()
    t0 = float(rig.data.time)
    limit = float(cfg["inject"]["recovery_time_limit_s"])
    first, failure, t_fail, bad_state, checked = {}, None, None, None, None
    with quiet():
        while True:
            tr = rig.truth(color)
            cmd = expert.act(tr)
            ph = cmd.phase
            first.setdefault(ph.name, round(tr.t, 3))
            trt = rig.truth(color, rig.data)
            apart = float(np.linalg.norm(trt.target_pos - trt.fingertip)) > DROP_M      # 手の中にない
            if checked is None and expert.clocks.target_rest_s >= pp.rest_hold_s - 1e-9 and apart \
                    and not frames.in_box(trt.target_pos, trt.box):
                checked = {"t": round(trt.t, 3), "target_pos": [round(float(v), 4) for v in trt.target_pos],
                           "yaw_rel_deg": yaw_rel_deg(trt.cube_quat[ti])}
                bad_state = state_check(trt, cfg, rig.model)
                checked["bad"] = bad_state
                if bad_state:
                    failure = f"state_{bad_state}"
            if failure is None:
                if "release" in first and ph in (S.Phase.approach, S.Phase.descend, S.Phase.close, S.Phase.lift,
                                                 S.Phase.carry, S.Phase.reopen):
                    failure = "place"
                elif "carry" in first and "release" not in first and ph in (S.Phase.settle, S.Phase.reopen):
                    failure = "slip"
                elif tr.t - t0 > limit:
                    failure = "timeout"
            if failure:
                t_fail = round(tr.t, 3)
                break
            if cmd.finished:
                break
            rig.pad_read(cmd.vel, cmd.press, on_rec)
    tr = rig.truth(color)
    success = failure is None and frames.in_box(tr.target_pos, tr.box) and tr.target_speed < pp.rest_speed
    if failure is None and not success:
        failure = "place"
    trivial = success and "close" not in first and "release" not in first
    g3 = rig.audit_summary()
    g3_bad = int(g3["total_violations"])
    saved = bool(success) and not trivial and g3_bad == 0
    if saved:
        disc = None
    elif bad_state:
        disc = f"state_{bad_state}"
    elif not success:
        disc = f"script_{failure}"
    else:
        disc = "trivial" if trivial else "g3"
    cr = np.array([f["contact_robot"] for f in frames_log])[:, mask]
    summary = {**base, "layout": lay.to_json(), "script": hp.to_json(), "saved": saved, "success": bool(success),
               "discard": disc, "failure": failure, "t_failure": t_fail, "t_record_start": round(t0, 3),
               "record_start_step": int(start.step), "recovery_duration_s": round(float(tr.t) - t0, 3),
               "n_frames": len(frames_log), "phase_first_t": first, "state_check": checked,
               "obstacle_mask": mask.tolist(), "g3": g3,
               "contact_obstacle_frames": int(cr.any(axis=1).sum()) if cr.size else 0,
               "final_target_pos": [round(float(v), 5) for v in tr.target_pos],
               "wall_s": round(time.perf_counter() - wall0, 3), "wall_policy_s": round(wall_policy, 3)}
    if writer is not None:
        if saved:
            meta = {**summary, "kind": KIND, "instruction": T.instruction(color), "target": color,
                    "layout_seed": int(seed), "pair_id": int(seed), "retry": 0, "layout_kind": lay.kind,
                    "start_pose": lay.start, "duration_s": round(float(tr.t), 3), "record_every": every,
                    "record_dt": every * rig.timestep, "record_hz": round(1.0 / (every * rig.timestep)),
                    "timestep": rig.timestep, "rendered": True, "frame0_image_shared_with_pair": False}
            meta.update(rig.episode_meta())
            summary["path"] = str(writer.finalize(meta, start.to_arrays(), every))
        else:
            writer.discard()
    return summary
