"""diag/step_reset.py（手順の切り替えに home へ戻す・実行系を作り直すを挟む）の切り替えの順番の確かめ。戻す動きと作り直しの中身は
tests/test_s4_diag_B.py が確かめるので、ここでは呼ぶ順と記録だけを見る。"""
import pytest

from recovla.diag import step_reset as SR


class Base:
    """TaskRuntime の _begin_step・_control・record の形だけの偽物。"""

    def __init__(self):
        self.steps, self.j, self.calls, self.phase = [], -1, [], "run"

    def _begin_step(self, t, j, attempt):
        if j != self.j:
            self.steps.append({"step": j})
        self.j = j
        self.calls.append(("begin", j, attempt))
        self.phase = "run"

    def _control(self, t):
        self.calls.append(("control", self.phase))

    def record(self):
        return {"steps": self.steps}


class Stubbed(SR.StepResetMixin, Base):
    def _begin_goal(self, t, j):
        self.calls.append(("goal", j))
        self.phase = "goal"

    def _goal_tick(self, t):
        self._begin_step(t, j=self.j_next, attempt=0)

    def _begin_rebuild(self, t, j):
        self.calls.append(("rebuild", j))
        self.phase = "rebuild"

    def _rebuild_tick(self, t):
        self._begin_step(t, j=self.j_next, attempt=0)


def make(mode):
    ex = Stubbed()
    ex.step_reset_setup(mode)
    return ex


@pytest.mark.parametrize("mode,between", [("home", ["goal"]), ("rebuild", ["rebuild"]), ("home_rebuild", ["goal", "rebuild"])])
def test_switch_inserts_in_order_and_not_before_first_or_on_retry(mode, between):
    ex = make(mode)
    ex._begin_step(0.0, 0, 0)                     # 1 番目の前には挟まない
    ex._begin_step(1.0, 0, 1)                     # やり直しにも挟まない
    ex.j_next = 1
    ex._begin_step(2.0, 1, 0)                     # 切り替え
    while ex.phase != "run":
        ex._control(3.0)
    kinds = [c[0] for c in ex.calls]
    assert kinds == ["begin", "begin"] + between + ["begin"]
    assert ex.calls[-1] == ("begin", 1, 0) and [s["step"] for s in ex.steps] == [0, 1]
    log = ex.record()["step_reset"]
    assert log["mode"] == mode and len(log["switches"]) == 1
    sw = log["switches"][0]
    assert sw["step"] == 1 and sw["t_begin"] == 2.0 and sw["t_end"] == 3.0


def test_bad_mode_and_install_none():
    with pytest.raises(ValueError):
        make("away")
    obj = object()
    assert SR.install(obj, None) is obj


def test_mixin_keeps_other_phases_to_the_wrapped_class():
    ex = make("home_rebuild")
    ex._control(0.0)
    assert ex.calls == [("control", "run")]
