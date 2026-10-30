"""手順の実行器。指示を LLM で手順（色の列）に分け、手順ごとに方策に指示文を渡し、完了を判定して次へ進む。
世界を止めない tick の状態機械として動く。

    ex = TaskRuntime(io, setup, prt, judge, cfg, decompose)
    ex.start("全部片付けて", seed)
    ex.tick                     # 物理の 1 ステップごと（評価の枠が呼ぶ）
    ex.finished                   # 真なら終わり（全部の手順を終えた、または止まった）

流れ:
  1. 起動時の確かめ（知覚。PolicyRuntime が行う）の後、知覚した机上と箱の中の色から、LLM で手順の色の列に分ける
     （LLM の時間は計算の口で引いた応答時間。その間、腕は保持し、世界は進む）
  2. 手順ごとに指示文と目標位置の入力の色を切り替え（PolicyRuntime.set_task。行動列は持ち越さない、雑音の列は (シード, 手順, やり直し)）
  3. 完了判定（runtime.judge.JudgeV2、20 Hz）が真になったら次の手順へ。step_timeout_s を超えたら、待機位置へ戻してから
     同じ手順を retry 回だけやり直す。それでも駄目なら止めて返答で知らせる
  4. 置いた後の戻す動き: 俯瞰で箱の中に目標の色が見え・指が開・手が待機位置から離れている、が trigger_s 続いたら戻し、
     wait_s の間その場で完了判定をかける。出なければ方策に戻す（前の行動列は持ち越さない）
  待機位置へ戻す動き: 方策を止め（PolicyRuntime.external）、指を開いて（ハンドの move）開き幅が止まるのを待ち、真上に rise_z まで、
  その後水平に待機位置へ（ReturnMotion。自分の指令 x_cmd で動かす）
"""
import numpy as np

NOISE_KEY = 7100


def step_seed(seed: int, step: int, attempt: int) -> int:
    return int(np.random.SeedSequence([int(seed), NOISE_KEY, int(step), int(attempt)]).generate_state(1)[0])


class ReturnMotion:
    """まず真上に rise_z まで、その後水平に待機位置へ。速さは生成プログラムと同じ決まり。"""

    def __init__(self, x_cmd, goal, rise_z, gain, xy_max, z_max, z_tol, tol):
        self.goal = np.asarray(goal, float)
        self.rise_z, self.gain, self.xy_max, self.z_max, self.z_tol, self.tol = rise_z, gain, xy_max, z_max, z_tol, tol
        self.xy0 = np.asarray(x_cmd, float)[:2].copy()
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


class TaskRuntime:
    def __init__(self, io, setup, prt, judge, planner_cfg: dict, motion_params: dict, decompose, instruction_fmt: str):
        """planner_cfg: configs の planner（step_timeout_s・retry・return_to_retreat）。motion_params: 戻す動きの速さの決まり
        （生成プログラムと同じ値。gain・xy_max・z_max・z_tol・tol。入口のスクリプトが configs の expert から作る）。"""
        self.io, self.setup, self.prt, self.judge = io, setup, prt, judge
        self.decompose, self.fmt = decompose, instruction_fmt
        p = planner_cfg
        self.step_timeout, self.retries = float(p["step_timeout_s"]), int(p["retry"])
        self.ret = dict(p["return_to_retreat"])
        self.rm = dict(goal=setup.retreat_pose, rise_z=float(self.ret["rise_z"]), **motion_params)
        self.open_m = float(judge.p["gripper_open_m"])
        self.judge_tol = float(judge.p["retreat_tol_m"])

    def instruction(self, color: str) -> str:
        return self.fmt.format(color=color)

    # ------------------------------------------------------------------ 実行
    def start(self, text: str, seed: int) -> None:
        self.text, self.seed = text, int(seed)
        self.prt.start(self.instruction("red"), step_seed(seed, 0, 0))   # 起動時の確かめ（知覚）。方策はまだ動かさない
        self.prt.external = True
        self.phase, self.n = "startup", 0
        self.plan, self.detected, self.fut = None, None, None
        self.j, self.attempt = -1, 0
        self.steps, self.returns, self.rows = [], [], []
        self.stopped, self.finished = None, False
        self.placed_since, self.done_t = None, None
        self.ret_state = None

    def _hand(self, q) -> np.ndarray:
        return self.prt.motion.hand_pose(q)[0]

    def tick(self) -> None:
        if self.finished:
            self.prt.tick()
            return
        t = self.io.now()
        if self.n % 25 == 0 and self.phase in ("run", "return", "wait_judge"):
            self._judge(t)
        if self.n % 10 == 0:
            self._control(t)
        self.n += 1
        self.prt.tick()

    def _judge(self, t: float) -> None:
        sensor = self.io.sense(cameras=True)
        if not {"overhead", "wrist"} <= set(sensor.cameras):
            return
        done = self.judge.update(sensor, self.prt.wm, self._hand(sensor.joints.q))
        last = self.judge.last
        placed = (self.phase == "run" and last["box_pixels"] >= self.judge.p["box_min_pixels"] and last["gripper_open"]
                  and last["retreat_dist_m"] > self.judge_tol)
        if not placed:
            self.placed_since = None
        elif self.placed_since is None:
            self.placed_since = t
        self.rows.append({"t": t, "step": self.j, "attempt": self.attempt, "phase": self.phase, "ok": last["ok"],
                          "held_s": last["held_s"], "box_pixels": last["box_pixels"], "wrist_in": last["wrist_in_pixels"],
                          "depth_ok": last["depth_ok"], "retreat_dist_m": last["retreat_dist_m"],
                          # 記録だけ（判定は変えない）: 手首の画像の内側の割合と、残りの条件
                          "wrist_frac": last.get("wrist_fraction"), "in_box": last.get("in_box"),
                          "gripper_open": last.get("gripper_open"), "at_retreat": last.get("at_retreat"),
                          "still": last.get("still")})
        if done and self.done_t is None:
            self.done_t = t

    def _control(self, t: float) -> None:
        prt = self.prt
        if self.phase == "startup":
            if prt.stop_reason:
                self._stop(t, f"知覚の失敗（{prt.stop_reason}）で始められませんでした")
            elif prt.ready and prt.wm is not None:
                wm = prt.wm
                table = [c for c, e in sorted(wm.cubes.items()) if e.status in ("seen", "held") and not e.in_box]
                inbox = [c for c, e in sorted(wm.cubes.items()) if e.status in ("seen", "held") and e.in_box]
                self.detected = {"table": table, "box": inbox}
                text = self.text
                self.fut = self.io.compute("llm", lambda: self.decompose(text, table, inbox))
                self.phase = "planning"
        elif self.phase == "planning":
            if self.fut.ready(t):
                self.plan = self.fut.result(t)
                self.fut = None
                if not self.plan.get("steps"):
                    self._finish(t)
                else:
                    self._begin_step(t, 0, 0)
        elif self.phase == "run":
            rec = self.steps[-1]
            if self.done_t is not None:
                self._step_done(t, True)
            elif (bool(self.ret["enabled"]) and self.placed_since is not None
                  and t - self.placed_since >= float(self.ret["trigger_s"]) - 1e-9):
                self._begin_return(t, "placed", judge_wait=True)
            elif t - rec["attempts"][-1]["t_begin"] >= self.step_timeout - 1e-9:
                if self.attempt < self.retries:
                    if bool(self.ret["enabled"]):
                        self._begin_return(t, "retry", judge_wait=False)
                    else:
                        self._begin_step(t, self.j, self.attempt + 1)
                else:
                    self._step_done(t, False)
        elif self.phase in ("return", "wait_judge"):
            self._return_tick(t)

    def _begin_step(self, t: float, j: int, attempt: int) -> None:
        color = self.plan["steps"][j]
        if j != self.j:
            self.steps.append({"step": j, "color": color, "attempts": [], "t_start": t})
        self.j, self.attempt = j, attempt
        self.steps[-1]["attempts"].append({"attempt": attempt, "t_begin": t, "t_judge": None})
        self.prt.set_task(self.instruction(color), step_seed(self.seed, j, attempt))
        self.prt.external = False
        self.judge.reset(color)
        self.done_t, self.placed_since = None, None
        self.phase = "run"

    def _step_done(self, t: float, ok: bool) -> None:
        rec = self.steps[-1]
        rec["attempts"][-1]["t_judge"] = self.done_t
        rec.update(judged_complete=ok, t_judge=self.done_t, t_end=t)
        if not ok:
            j = self.j
            self._stop(t, f"{j + 1} 番目の手順（{rec['color']}）を {self.retries + 1} 回試して終えられなかったので、止めました")
            return
        if self.j + 1 < len(self.plan["steps"]):
            self._begin_step(t, self.j + 1, 0)
        else:
            self._finish(t)

    def _stop(self, t: float, reply: str) -> None:
        self.stopped = {"step": self.j, "t": t, "reply": reply}
        self._finish(t)

    def _finish(self, t: float) -> None:
        self.finished = True
        self.prt.external = True
        self.prt.motion.set_velocity(np.zeros(3))
        self.t_end = t

    # --------------------------------------------------------------- 待機位置へ戻す
    def _begin_return(self, t: float, kind: str, judge_wait: bool) -> None:
        prt = self.prt
        prt.external = True
        prt.motion.set_velocity(np.zeros(3))
        opened = False
        if prt.closed:
            su = self.setup
            self.io.gripper_move(0.08, su.gripper_speed)
            prt.closed = False
            opened = True
        self.ret_state = {"kind": kind, "t_begin": t, "judge_wait": judge_wait, "opened": opened, "stage": "open",
                          "rest": 0.0, "t_arrive": None, "mot": None, "last_w": None, "t_last": t}
        self.phase = "return"

    def _return_tick(self, t: float) -> None:
        prt, rs = self.prt, self.ret_state
        sensor = self.io.sense(cameras=False)
        if rs["stage"] == "open":                              # 開き幅が止まるのを待つ（上限 1 s）
            w = sensor.gripper.width
            dt = t - rs["t_last"]
            if rs["last_w"] is not None and dt > 0:
                rs["rest"] = rs["rest"] + dt if abs(w - rs["last_w"]) / dt < 0.005 and w >= self.open_m else 0.0
            rs["last_w"], rs["t_last"] = w, t
            if not rs["opened"] or rs["rest"] >= 0.3 - 1e-9 or t - rs["t_begin"] >= 1.0 - 1e-9:
                rs["mot"] = ReturnMotion(prt.motion.x_cmd, **self.rm)
                rs["stage"] = "move"
            return
        mot = rs["mot"]
        x = prt.motion.x_cmd
        if rs["stage"] == "move":
            if mot.arrived(x):
                rs["t_arrive"], rs["stage"] = t, "arrived"
                if rs["judge_wait"]:
                    self.phase = "wait_judge"
            elif t - rs["t_begin"] >= float(self.ret["timeout_s"]) - 1e-9:
                rs["stage"] = "arrived"
            prt.motion.set_velocity(mot.velocity(x))
            if rs["stage"] != "arrived":
                return
        if rs["judge_wait"] and self.done_t is None and t - (rs["t_arrive"] or t) < float(self.ret["wait_s"]) - 1e-9:
            prt.motion.set_velocity(mot.velocity(x))
            return
        self.returns.append({"step": self.j, "attempt": self.attempt, "kind": rs["kind"], "t_begin": rs["t_begin"],
                             "t_arrive": rs["t_arrive"], "t_end": t, "arrived": rs["t_arrive"] is not None,
                             "opened": rs["opened"], "judged": self.done_t is not None})
        self.ret_state = None
        if rs["kind"] == "retry":
            self._begin_step(t, self.j, self.attempt + 1)
        elif self.done_t is not None:
            self.phase = "run"
            self._step_done(t, True)
        else:                                                  # 方策に戻す（前の行動列は持ち越さない）
            prt.external = False
            prt.reset_chunks()
            self.placed_since = None
            self.phase = "run"

    def record(self) -> dict:
        return {"text": self.text, "detected": self.detected, "plan": self.plan, "steps": self.steps,
                "returns": self.returns, "stopped": self.stopped, "judge_rows": self.rows,
                "startup": getattr(self.prt, "startup", None)}
