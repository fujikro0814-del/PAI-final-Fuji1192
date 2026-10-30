"""条件 CL（手で書いたスクリプト方策。識別子では CL2）の実行器。runtime/executor.TaskRuntime と
同じ口（start・tick・finished・plan・j・prt・record）を持ち、task_loop と ladder.py の run_one_task がそのまま読める。

    ex = CL2TaskRuntime(io, setup, skill, judge, planner_cfg, decompose, instruction_fmt, spec, rules)
    ex.start("全部片付けて", seed); ex.tick ...

流れ:
  1. 起動: スクリプト方策（api/skills_script_self.CL2Skill）の知覚が机の平面を確かめた後、推定（EstimatedGeometryFinder）で箱らしい物と
     色の物が見えたら始める。startup_wait_s の間に箱らしい物（皿のタスクでは皿らしい物。呼び方は書式の場所の words）が見えなければ
     「箱らしい物が見つからない」、物が 1 つも見えなければ「物が 1 つも見えない」で止める（既知の寸法・置き場所との差では止めない）
  2. 分解: 推定で「机の上」「箱の中」と決めた色の列を、ほかの条件と同じ関数（decompose_haiku）に渡す（io.compute の "llm"）
  3. 手順ごと: set_task（その色の指示文）→ スクリプト方策が自分で「済んだ」（step_done）と言えば次へ。時間の上限はほかの条件と
     同じ予算（configs の planner の step_timeout_s ×（retry ＋ 1））だけで、上限に来たら済まないまま次へ進む。実行器からの
     戻す動き・待機位置への戻し・やり直しの指示はしない（スクリプト方策が毎回決め直す）
  4. 全部の手順の後: 試行の始めの姿勢へ戻し（go_home）、着くか home_wait_s で終える
  JudgeV2 は記録だけ（何も決めない）。試行の成否は評価の真値（final_in_box）で決める。
"""
import numpy as np

from recovla.api import task_spec as TS
from recovla.runtime.executor import step_seed


class CL2TaskRuntime:
    def __init__(self, io, setup, prt, judge, planner_cfg: dict, decompose, instruction_fmt: str, spec, rules):
        self.io, self.setup, self.prt, self.judge = io, setup, prt, judge
        self.decompose, self.fmt, self.spec, self.rules = decompose, instruction_fmt, spec, rules
        p = planner_cfg
        self.step_budget = float(p["step_timeout_s"]) * (int(p["retry"]) + 1)   # ほかの条件と同じ 1 手順の予算（30 s × 2）

    def instruction(self, color: str) -> str:
        return self.fmt.format(color=color)

    def _place_word(self) -> str:
        """目標の場所の呼び方（書式の words の最初。tidy3_cubes は「箱」、cup_plate は「皿」）。"""
        p = self.spec.place(self.spec.goal.place)
        return p.words[0] if p.words else p.name

    def _color(self, name: str) -> str:
        return TS.cue_of(self.spec, TS.StepSpec("pick_place", name, self.spec.goal.place))

    # ------------------------------------------------------------------ run
    def start(self, text: str, seed: int) -> None:
        self.text, self.seed = text, int(seed)
        first = self._color(self.spec.object_names[0])
        self.prt.start(self.instruction(first), step_seed(seed, 0, 0))
        self.prt.external = True                                    # 起動の間はスクリプト方策を動かさない
        self.phase, self.n = "startup", 0
        self.plan, self.detected, self.fut = None, None, None
        self.j, self.attempt = -1, 0
        self.steps, self.returns, self.rows = [], [], []
        self.stopped, self.finished, self.ended = None, False, None
        self.t_ready, self.done_t, self.t_home = None, None, None
        self.cl2 = {"startup": None, "step_ends": []}

    def tick(self) -> None:
        if self.finished:
            self.prt.tick()
            return
        t = self.io.now()
        if self.n % 25 == 0 and self.phase == "run":
            self._judge(t)
        if self.n % 10 == 0:
            self._control(t)
        self.n += 1
        self.prt.tick()

    def _judge(self, t: float) -> None:
        """JudgeV2（既知の寸法を使う完了の判定）は記録だけに使う。進み方には使わない。"""
        if self.judge is None:
            return
        sensor = self.io.sense(cameras=True)
        if not {"overhead", "wrist"} <= set(sensor.cameras):
            return
        hand = self.prt.motion.hand_pose(sensor.joints.q)[0]
        done = self.judge.update(sensor, self.prt.wm, hand)
        last = dict(self.judge.last)
        self.rows.append({"t": t, "step": self.j, "attempt": 0, "phase": self.phase, "ok": last.get("ok"),
                          "held_s": last.get("held_s"), "box_pixels": last.get("box_pixels"),
                          "wrist_in": last.get("wrist_in_pixels"), "depth_ok": last.get("depth_ok"),
                          "retreat_dist_m": last.get("retreat_dist_m"), "record_only": True})
        if done and self.done_t is None:
            self.done_t = t

    def _estimate(self):
        return self.prt.estimate() if self.prt.wm is not None else None

    def _control(self, t: float) -> None:
        prt = self.prt
        if self.phase == "startup":
            if prt.stop_reason:
                self._stop(t, f"知覚の失敗（{prt.stop_reason}）で始められませんでした")
                return
            if not prt.ready:
                return
            if self.t_ready is None:
                self.t_ready = t
            est = self._estimate()
            box_ok = bool(est is not None and any(p.ok for p in est.places.values()))
            seen = {} if est is None else {n: o for n, o in est.objects.items() if o.status in ("seen", "occluded")}
            if not (box_ok and seen) and t - self.t_ready < self.rules.startup_wait_s - 1e-9:
                return
            self.cl2["startup"] = {"t": t, "box_ok": box_ok, "seen": sorted(seen)}
            if not box_ok:
                self._stop(t, f"{self._place_word()}らしい物が見つからないので、始められませんでした")
                return
            if not seen:
                self._stop(t, "物が 1 つも見えないので、始められませんでした")
                return
            place = self.spec.goal.place
            table = sorted(self._color(n) for n, o in seen.items() if not o.in_place.get(place, False))
            inbox = sorted(self._color(n) for n, o in seen.items() if o.in_place.get(place, False))
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
                    self._begin_step(t, 0)
        elif self.phase == "run":
            rec = self.steps[-1]
            if prt.step_done:
                self._step_done(t, True, "self")
            elif t - rec["t_start"] >= self.step_budget - 1e-9:
                self._step_done(t, False, "budget")
        elif self.phase == "home":
            if prt.at_home or t - self.t_home >= self.rules.home_wait_s - 1e-9:
                self.ended = {"kind": "home", "at_home": bool(prt.at_home), "t": t}
                self._finish(t)

    def _begin_step(self, t: float, j: int) -> None:
        color = self.plan["steps"][j]
        self.steps.append({"step": j, "color": color, "attempts": [{"attempt": 0, "t_begin": t, "t_judge": None}],
                           "t_start": t})
        self.j, self.attempt = j, 0
        self.prt.set_task(self.instruction(color), step_seed(self.seed, j, 0))
        self.prt.external = False
        if self.judge is not None:
            self.judge.reset(color)
        self.done_t = None
        self.phase = "run"

    def _step_done(self, t: float, ok: bool, why: str) -> None:
        rec = self.steps[-1]
        rec["attempts"][-1]["t_judge"] = t if ok else None
        rec.update(judged_complete=ok, t_judge=t if ok else None, t_end=t, end=why, judge_v2_t=self.done_t)
        self.cl2["step_ends"].append({"step": self.j, "color": rec["color"], "end": why, "t": t,
                                      "judge_v2_t": self.done_t})
        if self.j + 1 < len(self.plan["steps"]):
            self._begin_step(t, self.j + 1)
        else:
            self.prt.go_home()
            self.t_home = t
            self.phase = "home"

    def _stop(self, t: float, reply: str) -> None:
        self.stopped = {"step": self.j, "t": t, "reply": reply}
        self._finish(t)

    def _finish(self, t: float) -> None:
        self.finished = True
        self.prt.external = True
        self.prt.motion.set_velocity(np.zeros(3))
        self.t_end = t

    def record(self) -> dict:
        return {"text": self.text, "detected": self.detected, "plan": self.plan, "steps": self.steps,
                "returns": self.returns, "stopped": self.stopped, "judge_rows": self.rows,
                "startup": getattr(self.prt, "startup", None), "ended": self.ended, "replans": [], "early_events": [],
                "interventions": {"plan_change": 0, "replan_requests": 0, "llm_calls": 0},
                "cl2": dict(self.cl2, drops=list(self.prt.drops), step_budget_s=self.step_budget)}
