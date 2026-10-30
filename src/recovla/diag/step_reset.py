"""「全部片付けて」の手順の切り替え（2 番目以降の手順の始め）に、腕を戻す動きと実行系の作り直しを挟む包み（docs/real/step_reset.md）。

diag/return_pose.py の DiagTaskRuntime の EH（home の関節角へ関節空間で戻す）と EX（実行系を作り直す）を、そのまま組み合わせて使う。
どの実行器（TaskRuntime・LLM の組み直しつきの ReplanTaskRuntime など）にも後から被せられる
（クラスを「この混ぜもの ＋ 今のクラス」に差し替える）。

    MODES:
      home          腕だけ home の関節角へ（EH と同じ動き）
      rebuild       実行系だけ作り直す（EX と同じ）
      home_rebuild  home へ戻してから作り直す

切り替えの見つけ方: _begin_step(j, 0) が今の手順 self.j と違う j で呼ばれたとき（前の手順が完了・立て直しで次へ、のどちらでも）。
やり直し（同じ手順の attempt 1）には挟まない（今の実行器の待機位置へ戻す動きのまま）。1 番目の手順の前にも挟まない。
"""
import numpy as np

from recovla.diag import return_pose as E7

MODES = ("home", "rebuild", "home_rebuild")


class StepResetMixin:
    """被せる側。E7.DiagTaskRuntime の戻す動き・作り直しの関数をそのまま借りる。"""

    _begin_goal = E7.DiagTaskRuntime._begin_goal
    _goal_tick = E7.DiagTaskRuntime._goal_tick
    _begin_rebuild = E7.DiagTaskRuntime._begin_rebuild
    _rebuild_now = E7.DiagTaskRuntime._rebuild_now
    _rebuild_tick = E7.DiagTaskRuntime._rebuild_tick

    def step_reset_setup(self, mode: str, new_perception=None) -> None:
        if mode not in MODES:
            raise ValueError(f"mode {mode!r} は {MODES} のどれか")
        self.sr_mode = mode
        self.diag_arm = "EH" if mode != "rebuild" else "EX"
        self.goal_q = np.asarray(E7.HOME_Q, float) if mode != "rebuild" else None
        self.gp = dict(E7.GOAL_PARAMS)
        self.new_perception = new_perception
        self.goal_state = None
        self.sr_stage = None                                        # None → "goal" → "rebuild" → None
        self.sr_log = []

    def _begin_step(self, t: float, j: int, attempt: int) -> None:
        first = not self.steps
        switching = not first and attempt == 0 and j != self.j
        st = self.sr_stage
        if switching and st is None:
            self.sr_log.append({"step": j, "t_begin": t, "mode": self.sr_mode})
            if self.sr_mode == "rebuild":
                self.sr_stage = "rebuild"
                self._begin_rebuild(t, j)
            else:
                self.sr_stage = "goal"
                self._begin_goal(t, j)
            return
        if st == "goal" and self.sr_mode == "home_rebuild":
            self.sr_stage = "rebuild"
            self._begin_rebuild(t, j)
            return
        self.sr_stage = None
        if self.sr_log and self.sr_log[-1].get("t_end") is None and self.sr_log[-1]["step"] == j:
            self.sr_log[-1]["t_end"] = t
        super()._begin_step(t, j, attempt)

    def _control(self, t: float) -> None:
        if self.phase == "goal":
            self._goal_tick(t)
        elif self.phase == "rebuild":
            self._rebuild_tick(t)
        else:
            super()._control(t)

    def record(self) -> dict:
        rec = super().record()
        rec["step_reset"] = {"mode": self.sr_mode, "switches": list(self.sr_log)}
        return rec


def install(ex, mode: str, perception_cls=None):
    """ex（どの実行器でも）に被せる。mode が None なら何もしない。返り値は ex。"""
    if mode is None:
        return ex
    cls = ex.__class__
    ex.__class__ = type(f"StepReset{cls.__name__}", (StepResetMixin, cls), {})
    newp = None
    if mode != "home" and perception_cls is not None:
        def newp(old):
            return perception_cls(old.setup, old.p, old.thr)
    ex.step_reset_setup(mode, newp)
    if mode != "rebuild":
        ex.prt.motion.__class__ = E7.GoalMotion
    return ex
