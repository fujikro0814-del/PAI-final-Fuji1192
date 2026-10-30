"""LLM による手順の組み直し（executor_replan.ReplanTaskRuntime）に、落下の早い検知を足す包み。手順の途中で「狙いの色が
手の中にあった後、箱の外の机の上に戻った」（運ぶ途中で手から離れた）を知覚で見つけたら、時間切れを待たずに組み直しを頼む。

    ex = install(ex, replan_l2_step, max_replans=2, drop_confirm_s=1.0)

検知（実行系が読めるものだけ）: 知覚の WorldModel の狙いの色の状態。今の試みの中で一度 in_hand になった後、seen・held で
  in_box が偽の状態が drop_confirm_s 続いたら「落とした」（event "dropped"）。試みが変わると数え直す。時間切れの扱い（event
  "timeout"）は ReplanTaskRuntime と同じ。組み直しの上限 max_replans も同じ数を数える（上限に達したら早い検知もしない）。
組み直しの関数は event を受ける（planner/replan_event.replan_step）。受けない関数（replan_base・replan_multi）には event を渡さない。
落としたときの手順の記録: judged_complete 偽、t_end、"early_event"（"dropped"・時刻・状態の列）。試みの記録に "aborted"。
"""
from recovla.runtime.executor_replan import ReplanTaskRuntime

DROP_CONFIRM_S = 1.0


class EarlyReplanTaskRuntime(ReplanTaskRuntime):
    def start(self, text: str, seed: int) -> None:
        super().start(text, seed)
        self.early_events = []
        self._event = "timeout"
        self._watch_reset()

    def _watch_reset(self) -> None:
        self._held_seen, self._drop_since = False, None

    def _begin_step(self, t: float, j: int, attempt: int) -> None:
        super()._begin_step(t, j, attempt)
        self._watch_reset()

    # ------------------------------------------------------------ 早い検知
    def _drop_seen(self, t: float) -> bool:
        wm = self.prt.wm
        if wm is None or not self.steps:
            return False
        e = wm.cubes.get(self.steps[-1]["color"])
        if e is None:
            return False
        if e.status == "in_hand":
            self._held_seen, self._drop_since = True, None
            return False
        if self._held_seen and e.status in ("seen", "held") and not e.in_box:
            if self._drop_since is None:
                self._drop_since = t
            return t - self._drop_since >= self.drop_confirm_s - 1e-9
        self._drop_since = None
        return False

    def _control(self, t: float) -> None:
        if (self.phase == "run" and self.done_t is None and len(self.replans) < self.max_replans and self._drop_seen(t)):
            rec = self.steps[-1]
            ev = {"event": "dropped", "t": t, "t_since": self._drop_since, "step": int(self.j), "attempt": int(self.attempt),
                  "color": rec["color"]}
            self.early_events.append(ev)
            rec["attempts"][-1]["aborted"] = "dropped"
            rec["attempts"][-1]["t_judge"] = None
            rec.update(judged_complete=False, t_judge=None, t_end=t, early_event=ev)
            self._event = "dropped"
            self._request_replan(t)
            self._event = "timeout"
            return
        super()._control(t)

    def _request_replan(self, t: float) -> None:
        if getattr(self.replan, "takes_event", False):
            fn, ev = self.replan, self._event
            base = self.replan
            self.replan = lambda **kw: base(**kw, event=ev)
            try:
                super()._request_replan(t)
            finally:
                self.replan = fn
            self.replans[-1]["event"] = ev
        else:
            super()._request_replan(t)

    def step_summary(self) -> list:
        """落としたで終えた手順の結果は "failed"（replan_base.RESULTS の 1 つ。時間切れの "timeout" と分ける）。"""
        out = super().step_summary()
        for s, o in zip(self.steps, out):
            if s.get("early_event") and o["result"] == "timeout":
                o["result"] = "failed"
        return out

    def record(self) -> dict:
        d = super().record()
        d["early_events"] = self.early_events
        return d


def install(ex, replan, max_replans: int = 2, drop_confirm_s: float = DROP_CONFIRM_S):
    """作った TaskRuntime を、この包みに差し替える（executor_replan.install と同じやり方）。start の前に呼ぶ。"""
    ex.__class__ = EarlyReplanTaskRuntime
    ex.replan, ex.max_replans, ex.drop_confirm_s = replan, int(max_replans), float(drop_confirm_s)
    return ex
