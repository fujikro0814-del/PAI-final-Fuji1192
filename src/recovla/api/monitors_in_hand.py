"""失敗の検知「持っていたはずが手の中にない」（ExpectedInHandMonitor。手順書 docs/api/r1_protocol.md 5 節の 2）。

    m = ExpectedInHandMonitor(ReflexParams, confirm_s=1.0)     # api/monitors.MonitorSet に足して使う（口は同じ）

HeldEstimateDropMonitor（api/monitors）は、落ちた物が机の上で箱の外のときだけ出す。落とした・置き損ねた物が
箱の縁・壁ぎわ・箱の中の立方体の上に止まると、どれでもない（机の上でも箱の外でもない）ので出ず、時間切れまで待つ。
この検知は場所を問わない:
  「持っていた」= GraspLossDetector の holding の間に、狙いの物の推定が手に近い（relations.near_hand）。
  「手の中にない」= その後に、推定が写っていて（seen）、手に近くなく、箱にきちんと入っていない（内側の領域の外、または
    api/scene_summary.stacked の重なり）が confirm_s 続いた。グリッパの開閉は問わない（閉じたまま抜けた・開いて落とした）。
  箱にきちんと入ったら構えを解く（置いた）。出来事は dropped（実行器は手順の組み直しを頼む）。
閾値は推定の大きさ・不確かさからの相対（relations）。confirm_s は 1.0 s（放してから箱の底へ落ちきる時間より長く）。
"""
from recovla.api import relations as REL
from recovla.api import scene_summary as SS
from recovla.api.monitors import Event, _r, _vec
from recovla.runtime.reflex import GraspLossDetector, ReflexParams

CONFIRM_S = 1.0


class ExpectedInHandMonitor:
    name = "expected_in_hand"

    def __init__(self, params: ReflexParams = None, confirm_s: float = CONFIRM_S, near_k: float = REL.NEAR_K,
                 unc_k: float = REL.UNC_K):
        self.p = params or ReflexParams()
        self.p.check()
        self.confirm_s = float(confirm_s)
        self.near_k, self.unc_k = float(near_k), float(unc_k)
        self.det = GraspLossDetector(self.p)
        self.needs = ("gripper", "joints", "camera/overhead")
        self.events, self.log = [], []
        self.task, self.step = None, None
        self._reset()

    def _reset(self) -> None:
        self.det.reset()
        self.last_gt = None
        self.armed, self.t_armed, self.since = False, None, None

    def start(self, task, step) -> None:
        self.task, self.step = task, step
        self._reset()

    def _placed(self, o, est) -> bool:
        """箱にきちんと入った: 内側の領域の中で、重なりでない。"""
        pl = None if est is None else est.places.get(self.step.place)
        if pl is None or not o.in_place.get(self.step.place, False):
            return False
        objs = {n: x for n, x in est.objects.items() if x.status != "lost"}
        return not (pl.region is not None and SS.stacked(self.step.obj, objs, pl))

    def observe(self, senses, est, intent, t: float) -> list:
        if self.step is None:
            return []
        t = float(t)
        gt = senses.time("gripper")
        if gt != self.last_gt:                                    # ハンドの新しい読みで「持っている」を更新
            self.last_gt = gt
            closed = bool(intent.gripper_closed_cmd)
            if self.det.state == "fired" and not closed:
                self.det.reset()
            self.det.update(t, senses.get("gripper").width, closed, senses.hand())
        o = None if est is None else est.objects.get(self.step.obj)
        if o is None:
            return []
        tip = senses.tip()
        near = True if o.near_hand is True else REL.near_hand(o, tip, self.near_k, self.unc_k)
        if self.det.state == "holding" and near and o.status != "lost":
            if not self.armed:
                self.log.append({"kind": "armed", "t": _r(t)})
            self.armed, self.t_armed, self.since = True, t, None
            return []
        if not self.armed:
            return []
        if self._placed(o, est):
            self.armed, self.since = False, None
            self.log.append({"kind": "placed", "t": _r(t)})
            return []
        if not (o.status == "seen" and near is False):
            self.since = None
            return []
        if self.since is None:
            self.since = t
        if t - self.since < self.confirm_s - 1e-9:
            return []
        self.armed = False
        pl = est.places.get(self.step.place)
        ev = Event("dropped", self.step.obj, t, self.name,
                   {"t_since": _r(self.since), "t_armed": _r(self.t_armed), "status": o.status, "pos": _vec(o.pos),
                    "radius_m": _r(REL.radius(o)), "height_m": _r(REL.height(o)),
                    "depth_m": None if pl is None else _r(pl.depth(o.pos)),
                    "in_place": bool(o.in_place.get(self.step.place, False)), "confirm_s": self.confirm_s})
        self.events.append(ev.to_dict())
        return [ev]

    def record(self) -> dict:
        return {"name": self.name, "needs": list(self.needs), "params": self.p.to_dict(), "confirm_s": self.confirm_s,
                "near_k": self.near_k, "unc_k": self.unc_k, "events": list(self.events), "log": list(self.log)}
