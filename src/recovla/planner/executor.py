"""手順の実行器。指示を色の列に分け、色ごとに VLA を動かして完了を判定する。

    ex = TaskExecutor(rig, runner)                       # runner = policy.runner.SceneRunner
    meta, arrays, video = ex.run(layout, "全部片付けて", seed)

1. 最初の俯瞰の画像で机上・箱の中の色を判定し（planner.detect）、LLM で色の列に分ける（planner.decompose）
2. 手順ごとに指示文と目標の色を切り替え、残りの行動列を捨て（runner.start_trial。推論の雑音は
   (シード, 手順の番号, やり直しの回数) から作る）、安全フィルタの障害物を作り直す（rig.safety.start_trial）
3. 完了判定（planner.judge）が真になったら次の手順へ。planner.step_timeout_s（30 s）を超えたら、方策の状態を
   初期化して planner.retry（1）回だけやり直す。それでも駄目なら止めて人に知らせる
4. 待機位置へ戻す動き（planner.return_to_retreat.enabled で入切）: 方策を止め、指を開き、真上に rise_z まで
   上げてから水平に待機位置（expert.retreat_pose）へ、生成プログラムと同じ速さの決まりで動かす（ReturnMotion）。
   この間は安全フィルタを切る（安全フィルタは方策の指令にだけ効く）。使うのは次の 2 つ
   (a) やり直しの前: 30 s で終わらなかったら、戻してからやり直す
   (b) 置いた後: 俯瞰で箱の中に目標の色が presence.box_min_pixels 以上・指が開・手（x_des）が待機位置から
       judge.retreat_tol_m より離れている、が trigger_s 続いたら戻し、wait_s の間完了判定を待つ。
       完了しなければ方策に戻す（残りの行動列は捨てる＝runner.reset）
記録（評価用。方策と判定には渡さない）: フレームごとの真値（立方体の向き cube_quat を含む）、手順の番号・やり直しの回数・
完了判定の条件・戻す動きの間か（mode）、手順ごとの真値の成功の時刻（箱の中で 1 s 静止）、戻す動きの記録（meta["returns"]）
"""
import time

import numpy as np

from recovla.common import code_version, config
from recovla.common.seeds import COLORS
from recovla.eval import scene_trial as T
from recovla.expert.script import PhaseParams
from recovla.perception import color as PC
from recovla.planner import decompose as D
from recovla.planner.detect import Regions, wrist_box_mask
from recovla.planner.judge import CompletionJudge
from recovla.record import episode as E
from recovla.sim import frames

_CFG = config.load()
NOISE_KEY = 7100


def step_seed(seed: int, step: int, attempt: int) -> int:
    return int(np.random.SeedSequence([int(seed), NOISE_KEY, int(step), int(attempt)]).generate_state(1)[0])


class ReturnMotion:
    """待機位置へ戻す速度の指令（m/s）。真上に rise_z まで（z の差が expert.move_tol_m 以内まで）上げ、水平に
    expert.retreat_pose へ動かす。速さの決まりは生成プログラム（expert.script.Expert._vel_to）と同じ: 利得 gain_per_s、
    上限 speed_ref。着いた = 水平の段で x_des が待機位置から expert.phase.retreat_tol_m 以内"""

    def __init__(self, x_cmd, cfg: dict = None):
        cfg = cfg or _CFG
        e = cfg["expert"]
        self.goal = np.array(e["retreat_pose"], dtype=float)
        self.rise_z = float(cfg["planner"]["return_to_retreat"]["rise_z"])
        self.xy0 = np.asarray(x_cmd, float)[:2].copy()
        self.gain = float(e["gain_per_s"])
        self.xy_max, self.z_max = float(e["speed_ref"]["xy"]), float(e["speed_ref"]["z"])
        self.z_tol = float(e["move_tol_m"])
        self.tol = float(e["phase"]["retreat_tol_m"])
        self.stage = "rise"

    def velocity(self, x_cmd) -> np.ndarray:
        x = np.asarray(x_cmd, float)
        if self.stage == "rise" and abs(x[2] - self.rise_z) <= self.z_tol:
            self.stage = "move"
        goal = np.r_[self.xy0, self.rise_z] if self.stage == "rise" else self.goal
        v = self.gain * (goal - x)
        n = np.linalg.norm(v[:2])
        if n > self.xy_max:
            v[:2] *= self.xy_max / n
        v[2] = float(np.clip(v[2], -self.z_max, self.z_max))
        return v

    def arrived(self, x_cmd) -> bool:
        return self.stage == "move" and bool(np.linalg.norm(np.asarray(x_cmd, float) - self.goal) <= self.tol)


class TaskExecutor:
    def __init__(self, rig, runner, cfg: dict = None, decompose=None):
        self.rig, self.runner = rig, runner
        self.cfg = cfg or _CFG
        self.regions = Regions(PC.overhead_calibration(rig.model), frames.box_outer_half(rig.model))
        self.judge = CompletionJudge(self.regions, self.cfg)
        self.decompose = decompose or D.decompose
        p = self.cfg["planner"]
        self.step_timeout = float(p["step_timeout_s"])
        self.retries = int(p["retry"])
        self.ret = dict(p["return_to_retreat"])
        e = self.cfg["expert"]
        self.close_settle_s, self.finger_rest_speed = float(e["close_settle_s"]), float(e["finger_rest_speed"])

    def _return_to_retreat(self, kind: str, on_step, capture_state: dict, judge_wait: bool) -> dict:
        """待機位置へ戻す動き。judge_wait なら着いた後 wait_s の間その場で保ち、完了判定を待つ。
        → 記録 {kind, t_begin, t_arrive, t_end, arrived, opened, judged, contact_cube, contact_box, cube_moved_m}"""
        from recovla.sim.contact import COLUMNS
        rig = self.rig
        rig.safety.gate = False
        capture_state["mode"] = 1
        n0 = capture_state["n_frames"]()
        t0 = float(rig.data.time)
        pos0 = rig.data.xpos[rig.cube_ids].copy()
        read_dt = rig.steps_per_read * rig.timestep
        rec = {"kind": kind, "t_begin": t0, "t_arrive": None, "arrived": False, "opened": False, "judged": False}
        opened_wait = 0.0
        if rig.controller.gripper_closed:              # 指を開く（押し始めで切り替わる）。指が止まるまで待つ（上限 1 s）
            rec["opened"] = True
            rig.pad_read(np.zeros(3), True, on_step)
            rest = 0.0
            while opened_wait < 1.0 - 1e-9 and rest < self.close_settle_s - 1e-9:
                rig.pad_read(np.zeros(3), False, on_step)
                opened_wait += read_dt
                fv = float(np.sum(np.abs(rig.data.qvel[rig.finger_vadr])))
                rest = rest + read_dt if fv < self.finger_rest_speed else 0.0
        mot = ReturnMotion(rig.integrator.x_cmd, self.cfg)
        t_limit = float(self.ret["timeout_s"])
        while float(rig.data.time) - t0 < t_limit - 1e-9:
            if mot.arrived(rig.integrator.x_cmd):
                rec.update(arrived=True, t_arrive=float(rig.data.time))
                break
            rig.pad_read(mot.velocity(rig.integrator.x_cmd), False, on_step)
        if judge_wait and rec["arrived"]:
            t1 = float(rig.data.time)
            while capture_state["done_t"] is None and float(rig.data.time) - t1 < float(self.ret["wait_s"]) - 1e-9:
                rig.pad_read(mot.velocity(rig.integrator.x_cmd), False, on_step)
            rec["judged"] = capture_state["done_t"] is not None
        fr = capture_state["frames"][n0:]
        cr = np.array([f["contact_robot"] for f in fr], dtype=bool).reshape(len(fr), -1)
        cube_cols = [i for i, c in enumerate(COLUMNS) if c.startswith("cube_")]
        wall_cols = [i for i, c in enumerate(COLUMNS) if not c.startswith("cube_")]
        rec.update(t_end=float(rig.data.time), contact_cube=bool(cr[:, cube_cols].any()) if len(fr) else False,
                   contact_box=bool(cr[:, wall_cols].any()) if len(fr) else False,
                   cube_moved_m=[float(v) for v in np.linalg.norm(rig.data.xpos[rig.cube_ids] - pos0, axis=1)])
        capture_state["mode"] = 0
        return rec

    def run(self, layout, text: str, seed: int, render_video: bool = False):
        rig = self.rig
        ev = self.cfg["eval"]
        rest_speed, rest_hold = float(ev["success"]["rest_speed"]), float(ev["success"]["rest_hold_s"])
        pp = PhaseParams.from_config()
        rig.reset(layout)
        dt = rig.timestep
        f0, imgs0 = E.capture_frame(rig, COLORS[0], 0.0, pp, True)
        raw0 = dict(zip(rig.cameras, imgs0))
        table = self.regions.table_colors(raw0[PC.OVERHEAD])
        inbox = self.regions.box_colors(raw0[PC.OVERHEAD])
        plan = self.decompose(text, table, inbox)
        frames_log, rows, video, steps_rec, returns = [], [], [], [], []
        state = {"step": -1, "attempt": 0, "color": None, "hold": 0.0, "truth_t": None, "raw": raw0, "done_t": None,
                 "mode": 0, "placed_since": None, "frames": frames_log, "n_frames": lambda: len(frames_log)}
        use_return = bool(self.ret["enabled"])
        judge_tol = float(self.cfg["planner"]["judge"]["retreat_tol_m"])

        def capture(color):
            f, imgs = E.capture_frame(rig, color, 0.0, pp, True)
            state["raw"] = dict(zip(rig.cameras, imgs))
            wm = wrist_box_mask(rig.model, rig.scratch, rig.renderer.width, rig.renderer.height)
            done = self.judge.update(f["sim_time"], state["raw"][PC.OVERHEAD], f["gripper_closed"], f["x_des"],
                                     state["raw"]["wrist"], wm)
            last = self.judge.last
            # (b) の条件（方策が動かしている間だけ数える）: 俯瞰で箱の中に見え・指が開・手が待機位置から離れている
            placed = (state["mode"] == 0 and last["box_pixels"] >= self.regions.box_min and last["gripper_open"]
                      and last["retreat_dist_m"] > judge_tol)
            if not placed:
                state["placed_since"] = None
            elif state["placed_since"] is None:
                state["placed_since"] = float(f["sim_time"])
            frames_log.append(f)
            rows.append({"step": state["step"], "attempt": state["attempt"], "judge_ok": last["ok"],
                         "judge_held_s": last["held_s"], "box_pixels": last["box_pixels"],
                         "retreat_dist_m": last["retreat_dist_m"], "mode": state["mode"]})
            if render_video:
                video.append(np.hstack(imgs))
            if done and state["done_t"] is None:
                state["done_t"] = float(f["sim_time"])
            return f

        def on_step(r):
            ti = COLORS.index(state["color"])
            d = r.data
            pos = d.xpos[r.cube_ids[ti]]
            v = r.cube_vadr[ti]
            speed = float(np.linalg.norm(d.qvel[v:v + 3]))
            if state["truth_t"] is None:
                if frames.in_box(pos, r.box) and speed < rest_speed:
                    state["hold"] += dt
                    if state["hold"] >= rest_hold - 1e-9:
                        state["truth_t"] = float(r.step * dt)
                else:
                    state["hold"] = 0.0
            if r.step % r.record_every == 0:
                capture(state["color"])

        stopped = None
        wall0 = time.perf_counter()
        for j, color in enumerate(plan["steps"]):
            state.update(step=j, color=color, hold=0.0, truth_t=None)
            rec = {"step": j, "color": color, "attempts": [], "t_start": float(rig.data.time)}
            finished = False
            for attempt in range(self.retries + 1):
                state.update(attempt=attempt, done_t=None, placed_since=None)
                self.runner.start_trial(step_seed(seed, j, attempt))
                rig.safety.start_trial(color)
                self.judge.reset(color)
                task = T.instruction(color)
                t_begin = float(rig.data.time)
                f = frames_log[-1] if frames_log else f0
                raw = state["raw"]
                k = 0
                n_ret = len(returns)
                from recovla.sim.rig import quiet
                with quiet():
                    while state["done_t"] is None and float(rig.data.time) - t_begin < self.step_timeout - 1e-9:
                        a = np.asarray(self.runner(k, f, raw, task), dtype=np.float64)
                        rig.safety.gate = True
                        T.execute_action(rig, a, on_step)
                        rig.safety.gate = False
                        f, raw = frames_log[-1], state["raw"]
                        k += 1
                        ps = state["placed_since"]
                        if (use_return and state["done_t"] is None and ps is not None
                                and float(rig.data.time) - ps >= float(self.ret["trigger_s"]) - 1e-9):
                            returns.append({"step": j, "attempt": attempt,
                                            **self._return_to_retreat("placed", on_step, state, judge_wait=True)})
                            state["placed_since"] = None
                            if state["done_t"] is None:
                                self.runner.reset()          # 方策に戻す（残りの行動列は捨てる）
                            f, raw = frames_log[-1], state["raw"]
                    if (use_return and state["done_t"] is None and attempt < self.retries):
                        returns.append({"step": j, "attempt": attempt,
                                        **self._return_to_retreat("retry", on_step, state, judge_wait=False)})
                rig.safety.gate = False
                rec["attempts"].append({"attempt": attempt, "t_begin": t_begin, "t_judge": state["done_t"],
                                        "policy_actions": k, "returns": len(returns) - n_ret})
                if state["done_t"] is not None:
                    finished = True
                    break
            rec.update(judged_complete=finished, t_judge=state["done_t"], t_truth_success=state["truth_t"],
                       truth_in_box_at_end=bool(frames.in_box(rig.data.xpos[rig.cube_ids[COLORS.index(color)]], rig.box)))
            steps_rec.append(rec)
            if not finished:
                stopped = {"step": j, "color": color, "reply": f"{j + 1} 番目の手順（{color}）を {self.retries + 1} 回試して"
                                                               "終えられなかったので、止めました"}
                break
        final_in_box = {c: bool(frames.in_box(rig.data.xpos[rig.cube_ids[i]], rig.box)) for i, c in enumerate(COLORS)}
        planned_ok = bool(plan["steps"]) and all(final_in_box[c] for c in plan["steps"])
        meta = {"seed": int(seed), "text": text, "layout": {"kind": layout.kind, "start": layout.start, "seed": layout.seed,
                                                              "table_colors": list(layout.table_colors)},
                "detected": {"table": table, "box": inbox}, "plan": plan, "steps": steps_rec, "stopped": stopped,
                "final_in_box": final_in_box, "all_planned_in_box": planned_ok,
                "all_three_in_box": all(final_in_box.values()), "t_end": float(rig.data.time),
                "return_to_retreat": self.ret, "returns": returns,
                "runtime": {**self.runner.runtime_record(), "safety_filter": rig.safety.enabled},
                "wall_s": round(time.perf_counter() - wall0, 1), "code_version": code_version.code_version()}
        keys = ("step", "sim_time", "ee_pos", "fingertip", "fingers", "x_des", "gripper_closed", "cube_pos",
                "cube_quat", "cube_in_box", "phase", "contact_robot", "min_dist", "safety_active")
        arrays = {k: np.array([f[k] for f in frames_log]) for k in keys} if frames_log else {}
        for k in ("step", "attempt", "judge_ok", "judge_held_s", "box_pixels", "retreat_dist_m", "mode"):
            arrays["plan_" + k if k in ("step",) else k] = np.array([r[k] for r in rows]) if rows else np.zeros(0)
        return meta, arrays, video
