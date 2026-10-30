"""失敗の検知（api/monitors）を手順の組み直しにつなぐ実行器の包み。組み直しの実行器（runtime/executor_replan_early.EarlyReplanTaskRuntime）
の落下の早い検知（_drop_seen）を、失敗の検知の出来事に替える。組み直しの頼み方・記録・上限は元の実行器のまま
（組み直しの関数は replan_event、event は dropped・timeout）。

    ex = install(ex, replan_l2_step, MonitorSet([GraspDropMonitor]), spec, max_replans=2)

失敗の検知に渡すもの（実行系が読めるものだけ）: io.sense(cameras=False) の SensorFrame（Senses で包む。順運動学は
prt.motion.hand_pose）、知覚の推定（prt.wm を api.perception で写したもの）、グリッパの開閉の指令（prt.closed）と実行器の段。
検知は元の実行器と同じ所（手順の run の間、完了の判定の前、組み直しの上限の前）でだけ動き、試みが変わると start で数え直す。
出来事のうち replan_kinds（既定 dropped）が出たら組み直しを頼む。記録: early_events の各項に "monitor"（出来事の
根拠）、record["monitors"]（MonitorSet.record）。
注: io.sense の呼び出しが増えるので、シミュレーションのセンサの雑音の列は元の実行器とビット一致しない。
"""
from recovla.api import perception as AP
from recovla.api import senses as AS
from recovla.api import task_spec as TS
from recovla.api.monitors import Intent
from recovla.runtime import executor_replan_early as L2

REPLAN_KINDS = ("dropped",)


class MonitoredReplanTaskRuntime(L2.EarlyReplanTaskRuntime):
    def _watch_reset(self) -> None:
        super()._watch_reset()
        self._mon_event = None
        self._mon_step = TS.step_of_cue(self.spec, self.steps[-1]["color"]) if self.steps else None
        if self._mon_step is not None:
            self.monitors.start(self.spec, self._mon_step)

    def _estimate(self):
        """失敗の検知に渡す知覚の推定。finder が "estimated" なら知覚の更新で一緒に作った wm.geometry（GeometryPerception）。"""
        if self.finder == "estimated":
            return getattr(self.prt.wm, "geometry", None)
        return self.finder.estimate(self.prt.wm)

    def _kin(self, q):
        pos, quat = self.prt.motion.hand_pose(q)
        return pos, AS.fingertip(pos, quat, self.prt.tip_offset)

    def _drop_seen(self, t: float) -> bool:
        if self._mon_step is None:
            return False
        if self._mon_event is None:
            senses = AS.Senses.from_sensor_frame(self.io.sense(cameras=False), self._kin)
            est = self._estimate()
            intent = Intent(self._mon_step, bool(self.prt.closed), self.phase)
            for ev in self.monitors.observe(senses, est, intent, t):
                if ev.kind in self.replan_kinds and self._mon_event is None:
                    self._mon_event = ev
        if self._mon_event is None:
            return False
        self._drop_since = float(self._mon_event.evidence.get("t_since") or self._mon_event.t)
        return True

    def _control(self, t: float) -> None:
        n = len(self.early_events)
        super()._control(t)
        if len(self.early_events) > n and self._mon_event is not None:
            self.early_events[-1]["monitor"] = self._mon_event.to_dict()

    def record(self) -> dict:
        d = super().record()
        d["monitors"] = dict(self.monitors.record(), finder=self.finder if isinstance(self.finder, str) else
                             getattr(self.finder, "kind", type(self.finder).__name__))
        return d


def install(ex, replan, monitors, spec, max_replans: int = 2, finder="color", replan_kinds=REPLAN_KINDS):
    """作った TaskRuntime を、この包みに差し替える。start の前に呼ぶ。
    finder: "color"（既知の値の ColorWorldModelFinder。既定）・"estimated"（prt.perception に GeometryPerception を被せて
    wm.geometry を使う。WorldModel の計算は同じ）・ColorWorldModelFinder など estimate(wm) を持つ物。"""
    ex = L2.install(ex, replan, max_replans=max_replans, drop_confirm_s=L2.DROP_CONFIRM_S)
    ex.__class__ = MonitoredReplanTaskRuntime
    ex.monitors, ex.spec = monitors, spec
    if finder == "estimated":
        from recovla.api.estimated import EstimatedGeometryFinder, GeometryPerception
        per = ex.prt.perception
        if per is not None and not hasattr(per, "geom"):
            GeometryPerception.attach(per, EstimatedGeometryFinder.from_spec(spec, per, ex.setup))
        ex.finder = "estimated"
    elif finder == "color":
        ex.finder = AP.ColorWorldModelFinder.from_spec(spec, setup=ex.setup)
    else:
        ex.finder = finder
    ex.replan_kinds = tuple(replan_kinds)
    return ex
