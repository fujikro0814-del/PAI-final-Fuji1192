"""LLM による手順の組み直しの包み: 手順が失敗して実行器（runtime/executor.py の TaskRuntime）が止まる所に、組み直しを挟む。

    ex = ReplanTaskRuntime(io, setup, prt, judge, cfg, motion_params, decompose, fmt, replan=R4.replan_step)
    ex.start("全部片付けて", seed); ex.tick ...      # 使い方は TaskRuntime と同じ
    ex.record["replans"]                             # 組み直しの記録（介入の「計画の変更」に数える欄を含む）

TaskRuntime は書き換えない。変えるのは、手順が step_timeout_s x (retry + 1) で終わらなかったとき（TaskRuntime は止まって
知らせる）の扱いだけ。それ以外（計画・手順の切り替え・完了判定・戻す動き・やり直し）は TaskRuntime のまま。
流れ:
  1. 手順が終わらなかったら、腕をその場で保持し（PolicyRuntime.external、速さ 0）、知覚の要約（WorldModel の色の状態だけ）と
     自分の記録（手順ごとの結果・試みの回数）を、引数で受けた replan 関数に渡す（io.compute の "llm"。応答時間の間、世界は進む）
  2. 応答の手が next・reorder・skip（続ける）なら、待機位置へ戻す動き（kind "replan"。やり直しの前と同じ動き）の後に、
     これから実行する色の並びで次の手順を始める。計画の steps は「実行した手順の色＋これからの並び」に置き換え、元の並びは
     plan["steps_initial"] に残す（task_loop の記録は plan["steps"][j] を手順 j の色として読むので、添字を合わせる）
  3. finish なら終える（ended に報告を残す。stopped は空のまま）。stop（LLM の選択、または規則の検査で退けて選んだ安全な手）なら、
     TaskRuntime と同じ止まり方（その場で止まり、stopped に同じ文を残す）に、日本語の報告を足す
  4. 組み直しの回数が max_replans に達していたら、replan を呼ばずに TaskRuntime と同じに止まる
  - 戻す動きの間に、失敗した手順の完了の判定が遅延して出たら、TaskRuntime と同じく完了として次へ進む（late_complete）。
    そのとき、これからの並びに既に成功した色が残っていれば外す
実行系は真値を見ない: このファイルは runtime の中にあり、scripts/check_real_boundary.py の検査を受ける。
  replan に渡すのは、指示文・計画の色・自分の記録・知覚の WorldModel の色の状態（seen・held・in_hand・lost と in_box）だけ。
  手順を作る側（recovla.planner）は import せず、関数を引数で受ける（decompose と同じ）。
"""
import numpy as np

from recovla.runtime.executor import TaskRuntime

CONTINUE_ACTIONS = ("next", "reorder", "skip")


class ReplanTaskRuntime(TaskRuntime):
    def __init__(self, io, setup, prt, judge, planner_cfg: dict, motion_params: dict, decompose, instruction_fmt: str,
                 replan, max_replans: int = 2):
        super().__init__(io, setup, prt, judge, planner_cfg, motion_params, decompose, instruction_fmt)
        self.replan, self.max_replans = replan, int(max_replans)

    def start(self, text: str, seed: int) -> None:
        super().start(text, seed)
        self.replans, self.ended = [], None
        self._replan_return = False

    # ------------------------------------------------------------ 知覚と記録
    def perception_summary(self) -> dict:
        """知覚の WorldModel の色の状態だけ（起動時の確かめの table・box と同じ決め方（箱の中は in_box の欄）に、手の中・見失いを足す）。"""
        wm = self.prt.wm
        out = {"table": [], "in_box": [], "in_hand": [], "lost": []}
        if wm is None:
            return out
        for c, e in sorted(wm.cubes.items()):
            if e.status in ("seen", "held"):
                out["in_box" if e.in_box else "table"].append(c)
            elif e.status == "in_hand":
                out["in_hand"].append(c)
            elif e.status == "lost":
                out["lost"].append(c)
        return out

    @staticmethod
    def step_result(rec: dict) -> str:
        if rec.get("judged_complete") is True:
            return "success"
        if rec.get("skipped"):
            return "skipped"
        if rec.get("judged_complete") is False:
            return "timeout"
        return "pending"

    def step_summary(self) -> list:
        return [{"color": s["color"], "result": self.step_result(s), "attempts": len(s["attempts"])} for s in self.steps]

    # ------------------------------------------------------------ 差し込み
    def _step_done(self, t: float, ok: bool) -> None:
        if ok:
            if self.replans:
                self._prune_succeeded()
            super()._step_done(t, True)
            return
        if len(self.replans) >= self.max_replans:
            super()._step_done(t, False)                       # TaskRuntime と同じに止まる
            return
        rec = self.steps[-1]
        rec["attempts"][-1]["t_judge"] = self.done_t
        rec.update(judged_complete=False, t_judge=self.done_t, t_end=t)
        self._request_replan(t)

    def _prune_succeeded(self) -> None:
        """組み直しの後に、これからの並びから既に成功した色を外す（遅延して出た完了の判定のとき）。"""
        done = {s["color"] for s in self.steps if s.get("judged_complete") is True} | {self.steps[-1]["color"]}
        head = list(self.plan["steps"][:self.j + 1])
        tail = [c for c in self.plan["steps"][self.j + 1:] if c not in done]
        self.plan = dict(self.plan, steps=head + tail)

    def _request_replan(self, t: float) -> None:
        self.prt.external = True
        self.prt.motion.set_velocity(np.zeros(3))
        per = self.perception_summary()
        failed = self.steps[-1]["color"]
        kw = {"text": self.text, "plan_order": list(self.plan["steps"]), "steps": self.step_summary(), "perception": per,
              "failed_step": int(self.j), "failed_color": failed, "replans_done": len(self.replans)}
        fn = self.replan
        self.fut = self.io.compute("llm", lambda: fn(**kw))
        self.replans.append({"n": len(self.replans), "t_request": t, "t_decided": None, "failed_step": int(self.j),
                             "failed_color": failed, "perception": per, "action": None, "applied": None,
                             "intervention_kind": None, "report": None, "out": None})
        self.phase = "replanning"

    def _control(self, t: float) -> None:
        if self.phase != "replanning":
            super()._control(t)
            return
        if not self.fut.ready(t):
            return
        out = self.fut.result(t)
        self.fut = None
        self._apply_replan(t, out)

    def _apply_replan(self, t: float, out: dict) -> None:
        rec = self.replans[-1]
        decision = dict(out.get("decision") or {})
        action = decision.get("action", "stop")
        order = [c for c in (out.get("next_order") or [])]
        report = str(out.get("report") or "")
        rec.update(t_decided=t, action=action, report=report, out=out)
        if action in ("next", "reorder") and not order:      # 続ける手なのに並びが空（replan_step が先に安全な手へ切り替えるので通常は起きない）
            rec["rejected"] = ["empty_order"]
            report = ""                                       # 「続ける」と書いた報告で止まらない（報告なしで止まる）
            rec["report"] = report
            out = dict(out, fallback=True)
        if action == "skip":
            self.steps[-1]["skipped"] = True
        if action in CONTINUE_ACTIONS and order:
            if "steps_initial" not in self.plan:
                self.plan = dict(self.plan, steps_initial=list(self.plan["steps"]))
            self.plan = dict(self.plan, steps=[s["color"] for s in self.steps] + order)
            rec.update(applied={"kind": "continue", "order": order}, intervention_kind="plan_change")
            if bool(self.ret["enabled"]):
                self._replan_return = True
                self._begin_return(t, "replan", judge_wait=False)
            else:
                self._begin_step(t, len(self.steps), 0)
        elif action in ("finish", "skip"):                   # skip で残りがなければ終える
            rec.update(applied={"kind": "finish"}, intervention_kind="plan_change")
            self.ended = {"kind": "finish", "step": self.j, "t": t, "report": report}
            self._finish(t)
        else:                                                 # stop（LLM の選択、または規則の検査で退けて選んだ安全な手）
            rec.update(applied={"kind": "stop", "fallback": bool(out.get("fallback"))})
            j, color = self.j, self.steps[-1]["color"]
            reply = f"{j + 1} 番目の手順（{color}）を {self.retries + 1} 回試して終えられなかったので、止めました"
            self._stop(t, reply)
            self.stopped["report"] = report
            self.stopped["by"] = "replan_fallback" if out.get("fallback") else "replan"

    def _return_tick(self, t: float) -> None:
        rs = self.ret_state
        mine = self._replan_return and rs is not None and rs["kind"] == "replan"
        late = self.done_t is not None
        super()._return_tick(t)
        if mine and self.ret_state is None:
            self._replan_return = False
            if late:                                          # 戻す間に失敗した手順の完了が出た（TaskRuntime が次へ進めた）
                self.replans[-1]["late_complete"] = True
            else:
                self._begin_step(t, len(self.steps), 0)

    # ------------------------------------------------------------ 記録
    def interventions(self) -> dict:
        """介入の種類別の数。この包みが足すのは「計画の変更」だけ。
        API の呼び出しの数は介入とは別に llm_calls に数える。"""
        calls = sum(int(((r.get("out") or {}).get("llm") or {}).get("calls", 0)) for r in self.replans)
        return {"plan_change": sum(1 for r in self.replans if r["intervention_kind"] == "plan_change"),
                "replan_requests": len(self.replans), "llm_calls": calls}

    def record(self) -> dict:
        d = super().record()
        d.update(replans=self.replans, ended=self.ended, interventions=self.interventions())
        return d


def install(ex, replan, max_replans: int = 2):
    """作った TaskRuntime を、この包みに差し替える（__class__ を替える。持っている値はそのまま）。
    start の前に呼ぶ。返り値は ex。"""
    ex.__class__ = ReplanTaskRuntime
    ex.replan, ex.max_replans = replan, int(max_replans)
    return ex
