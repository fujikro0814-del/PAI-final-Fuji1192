"""実行系の改良（場面の要約・完了判定・見失い。手順書 docs/api/r1_protocol.md 5 節）を入れた実行器の包み。
api/executor_monitored の包みの上に重ねる。入れるのは install を通したときだけ。

    ex = install(ex, replan_r1_step, monitors, spec, max_replans=2, finder="estimated")

足すこと:
- 場面の要約（perception_summary）: api/scene_summary.refine で、箱の中の立方体の上に重なった色を in_box から
  stacked_in_box へ移し、壁ぎわ（near_box_wall）と「持っていたはずが手の中にない」（not_in_hand_after_hold）を足す。
  「持っていた」は、この試みの中で知覚が手の中（in_hand）と見た色（落下の早い検知と同じ見方）
- 手順の要約（step_summary）に tries_limit（1 回の実行の中の試みの上限 = retry + 1）を足す（planner/replan_improved の入力）
- 次の手 skip_notify: 失敗した色を飛ばして（skip と同じ流れ）残りを続け、終わりに ended の kind を "notify" にして、
  後に回した色・まだ終わっていない色・報告を残す（片付けられる残りを先にやって、最後に人に知らせる）
- 完了判定（fix_judge、既定で入れる）: api/judge_improved.JudgeR1（手首の画像と深度がそろえば俯瞰の画素の閾値を緩める）。
  判定の行に overhead_relaxed を足す
- 見失いの見直し（fix_lost、既定で入れる）: 組み直しを頼む時に、まだ一度も試していない計画の色が
  知覚の要約で lost なら、先に腕を待機位置（俯瞰から何も隠さない所）へ動かし、LOOK_WAIT_S 待ってから要約を取り直して頼む。
  見え直せば机の上の色として候補に入る。まだ lost で、組み直しの後の並びから外れたら、notify に kind "lost" で残し
  （文は replan 関数の lost_notice が作る。物・場所・置き方の言葉はタスクの言葉 wording から）、
  終わりに ended の kind "notify" で知らせる（黙って外さない）。手の中に色があるときは動かさない（取り直しだけ）
LLM の手順の部品（planner）は import しない（関数を引数で受ける）。
"""
import numpy as np

from recovla.api import executor_monitored as XL
from recovla.api import scene_summary as SS
from recovla.runtime.executor import ReturnMotion

LOOK_WAIT_S = 1.0                   # 待機位置に着いてから知覚の見直しを待つ時間（知覚の更新が数回入る長さ）


class R1ReplanTaskRuntime(XL.MonitoredReplanTaskRuntime):
    look_lost = False               # 見失いの見直し（install の fix_lost で入れる）

    def start(self, text: str, seed: int) -> None:
        self.notify = []
        self._held = set()
        self._look, self._look_last = None, None
        super().start(text, seed)

    def _watch_reset(self) -> None:
        super()._watch_reset()
        self._held = set()

    def _drop_seen(self, t: float) -> bool:
        wm = self.prt.wm
        if wm is not None and self.steps:
            c = self.steps[-1]["color"]
            e = wm.cubes.get(c)
            if e is not None and e.status == "in_hand":
                self._held.add(c)
        return super()._drop_seen(t)

    def _tip(self):
        try:
            return self._kin(self.io.sense(cameras=False).joints.q)[1]
        except Exception:                                    # noqa: BLE001  順運動学が読めなければ手のそばは分からない
            return None

    def perception_summary(self) -> dict:
        base = super().perception_summary()
        est = self._estimate() if self.finder == "estimated" else None
        return SS.refine(base, est, self.spec.goal.place, self._held, self._tip())

    def step_summary(self) -> list:
        out = super().step_summary()
        for o in out:
            o["tries_limit"] = int(self.retries) + 1
        return out

    # ------------------------------------------------------------ 見失いの見直し
    def _untried_lost(self, per: dict) -> list:
        """まだ試していない計画の残りの色のうち、知覚の要約で lost の色。"""
        tried = {s["color"] for s in self.steps}
        rest = list(self.plan["steps"][self.j + 1:]) if self.plan else []
        return [c for c in per["lost"] if c in rest and c not in tried]

    def _request_replan(self, t: float) -> None:
        lk, self._look_last = self._look_last, None
        if not self.look_lost:
            super()._request_replan(t)
            return
        if lk is None:
            per = self.perception_summary()
            miss = self._untried_lost(per)
            if miss and not per["in_hand"]:
                self._begin_look(t, miss)
                return
        super()._request_replan(t)
        rec = self.replans[-1]
        rec["untried_lost"] = self._untried_lost(rec["perception"])
        if lk is not None:
            rec["look"] = lk

    def _begin_look(self, t: float, colors) -> None:
        self.prt.external = True
        self.prt.motion.set_velocity(np.zeros(3))
        self._look = {"colors": list(colors), "event": self._event, "t_begin": t, "t_arrive": None, "arrived": False,
                      "mot": ReturnMotion(self.prt.motion.x_cmd, **self.rm)}
        self.phase = "look"

    def _look_tick(self, t: float) -> None:
        lk, x = self._look, self.prt.motion.x_cmd
        mot = lk["mot"]
        self.prt.motion.set_velocity(mot.velocity(x))
        if lk["t_arrive"] is None:
            if mot.arrived(x):
                lk.update(t_arrive=t, arrived=True)
            elif t - lk["t_begin"] >= float(self.ret["timeout_s"]) - 1e-9:
                lk["t_arrive"] = t                           # 着かなくても打ち切って見直す
            return
        if t - lk["t_arrive"] < LOOK_WAIT_S - 1e-9:
            return
        lk = {k: v for k, v in lk.items() if k != "mot"}
        self._look = None
        per = self.perception_summary()
        lk.update(t_end=t, still_lost=[c for c in lk["colors"] if c in per["lost"]])
        lk["seen_again"] = [c for c in lk["colors"] if c not in lk["still_lost"]]
        ev, self._event = self._event, lk["event"]
        self._look_last = lk
        try:
            self._request_replan(t)                          # 取り直した要約で頼む（見直しは 1 回だけ）
        finally:
            self._event = ev

    def _control(self, t: float) -> None:
        if self.phase == "look":
            self._look_tick(t)
            return
        super()._control(t)

    def _note_lost(self, t: float, out: dict) -> None:
        """まだ lost の試していない色が、これからの並びから外れたら知らせに残す。文は組み直しの関数の lost_notice
        （タスクの言葉）で作る（無ければ文なし。色と kind は残る）。"""
        keep = set(out.get("next_order") or [])
        n = len(self.replans) - 1
        fn = getattr(self.replan, "lost_notice", None)
        for c in self.replans[-1].get("untried_lost") or []:
            if c not in keep:
                self.notify.append({"color": c, "t": t, "report": str(fn(c)) if fn is not None else "", "replan": n,
                                    "kind": "lost"})

    def _apply_replan(self, t: float, out: dict) -> None:
        if self.look_lost:
            self._note_lost(t, out)
        d = dict(out.get("decision") or {})
        if d.get("action") != "skip_notify":
            super()._apply_replan(t, out)
            return
        self.notify.append({"color": d.get("color"), "t": t, "report": str(out.get("report") or ""),
                            "replan": len(self.replans) - 1})
        super()._apply_replan(t, dict(out, decision=dict(d, action="skip")))   # 流れは skip と同じ
        self.replans[-1]["action"] = "skip_notify"
        if self.ended is not None and self.ended.get("kind") == "finish":
            self._mark_notify(t)

    def _mark_notify(self, t: float) -> None:
        done = {s["color"] for s in self.steps if s.get("judged_complete") is True}
        left = [c for c in dict.fromkeys(list(self.plan.get("steps_initial") or []) + list(self.plan["steps"])) if c not in done]
        own = [n for n in self.notify if n.get("kind") != "lost"]
        lost = [n["report"] for n in self.notify if n.get("kind") == "lost" and n["report"]]
        report = (own[-1]["report"] if own else "") + "".join(dict.fromkeys(lost))   # 見失いの知らせは後ろに足す
        self.ended = {"kind": "notify", "step": self.j, "t": t, "deferred": [n["color"] for n in self.notify], "left": left,
                      "report": report}

    def _finish(self, t: float) -> None:
        super()._finish(t)
        if self.notify and self.stopped is None and (self.ended is None or self.ended.get("kind") == "finish"):
            self._mark_notify(t)

    def _judge(self, t: float) -> None:
        n = len(self.rows)
        super()._judge(t)
        if len(self.rows) > n:                               # 緩めた俯瞰の閾値で通ったかの印
            self.rows[-1]["overhead_relaxed"] = (self.judge.last or {}).get("overhead_relaxed")

    def record(self) -> dict:
        d = super().record()
        d["notify"] = list(self.notify)
        return d


def install(ex, replan, monitors, spec, max_replans: int = 2, finder="estimated", fix_judge: bool = True,
            fix_lost: bool = True):
    """executor_monitored.install の後でこの包みに差し替える。start の前に呼ぶ。fix_judge なら完了判定を
    api/judge_improved.JudgeR1 に替える。fix_lost なら見失いの見直しを入れる。"""
    ex = XL.install(ex, replan, monitors, spec, max_replans=max_replans, finder=finder)
    ex.__class__ = R1ReplanTaskRuntime
    ex.look_lost = bool(fix_lost)
    if fix_judge:
        from recovla.api import judge_improved as JR
        JR.install(ex.judge)
    return ex
