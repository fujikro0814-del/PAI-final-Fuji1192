"""失敗の検知（見張り）の口。実行器がいま何をしようとしているか（Intent）とセンサ・知覚の推定から、出来事（Event）を出す。

    ms = MonitorSet([GraspDropMonitor, LegacyInHandDropMonitor], standard_registry("sim"))
    ms.start(task, step)                                  # 手順（試み）の始めごと
    events = ms.observe(senses, est, intent, t)           # 実行器の制御の刻みごと（50 Hz）
    ms.record                                           # 動いた検知・needs がそろわず止めた検知・出来事（時刻・根拠）

Event.kind: dropped（落とした）・missed（空を掴んだ）・misplaced・collision・timeout。

GraspDropMonitor（指の幅による落下の早い検知）:
  「持っている」= runtime/reflex.GraspLossDetector の holding（閉の指令のまま幅が持つ範囲で止まり、手先が上がった）。
  「落とした」= holding の後に GraspLossDetector が発火（幅の低下 collapse・開き open）。camera_confirm なら、さらに狙いの物の推定が
    seen か occluded で、手に近くなく（api/relations.near_hand、物の推定の大きさからの相対）、置き場所の内側の領域から余白以上外
    （relations.outside、余白は推定の不確かさから）が confirm_s 続いたとき。
    confirm_s=None（または camera_confirm 切）ならカメラなし（実機のカメラなし Panda 用）で、発火の時点で出す。
  出さないもの: 意図した放し（閉の指令が開に変わった。GraspLossDetector が release にする）、置き場所の上での発火（置いた）。
  空掴み（発火の理由 miss）は missed として出す。
  引数（ReflexParams・confirm_s・camera_confirm・near_k・unc_k）は全部外から渡す。既定は ReflexParams の既定・confirm_s=1.0。
  「持っている」に使う開閉は、実行系が実際にハンドへ送った指令（Intent。指令の制限を通った後）。
HeldEstimateDropMonitor（知覚の推定による落下の検知。閾値は推定の大きさからの相対）:
  「持っていた」= GraspLossDetector の holding の間に、狙いの物の推定が手に近い（relations.near_hand）。
  「落とした」= その後に、推定が手に近くなく、机の上（relations.on_table）で箱の外（relations.outside）が on_table_s 続いた。
  幅の発火は要らない（方策が一瞬指を開いて落とす形も見つける）。箱の中へ入れた・箱の上で放したは、箱の外にならないので出さない。
LegacyInHandDropMonitor（比べ用）: runtime/executor_replan_early.EarlyReplanTaskRuntime._drop_seen と同じ判定（WorldModel の in_hand の後、
  seen・held で箱の外が confirm_s 続く）を同じ口で包む。
"""
import dataclasses
from typing import Protocol

import numpy as np

from recovla.api import relations as REL
from recovla.api.senses import standard_registry
from recovla.runtime.executor_replan_early import DROP_CONFIRM_S
from recovla.runtime.reflex import GraspLossDetector, ReflexParams

EVENT_KINDS = ("dropped", "missed", "misplaced", "collision", "timeout")
CONFIRM_S = 1.0                           # 落下をカメラで確かめる時間 [s]
ON_TABLE_S = 0.5                          # 机の上で箱の外が続けば落下とする時間 [s]


@dataclasses.dataclass(frozen=True)
class Event:
    kind: str                             # EVENT_KINDS のどれか
    obj: str
    t: float
    monitor: str
    evidence: dict

    def to_dict(self) -> dict:
        return {"kind": self.kind, "obj": self.obj, "t": round(float(self.t), 4), "monitor": self.monitor,
                "evidence": dict(self.evidence)}


@dataclasses.dataclass(frozen=True)
class Intent:
    step: object                          # api.task_spec.StepSpec
    gripper_closed_cmd: bool              # 実行系がハンドに出している開閉（閉なら真）
    phase: str                            # 実行器の段（run・return など）


class Monitor(Protocol):
    name: str
    needs: tuple

    def start(self, task, step) -> None: ...

    def observe(self, senses, est, intent: Intent, t: float) -> list: ...

    def record(self) -> dict: ...


def _r(x, n=4):
    return None if x is None else round(float(x), n)


def _vec(v, n=4):
    return None if v is None else [round(float(x), n) for x in v]


class GraspDropMonitor:
    name = "grasp_drop"

    def __init__(self, params: ReflexParams = None, confirm_s=CONFIRM_S, camera_confirm: bool = True,
                 near_k: float = REL.NEAR_K, unc_k: float = REL.UNC_K):
        self.p = params or ReflexParams()
        self.p.check()
        self.confirm_s = None if confirm_s is None else float(confirm_s)
        self.camera = bool(camera_confirm) and self.confirm_s is not None
        self.near_k, self.unc_k = float(near_k), float(unc_k)
        self.det = GraspLossDetector(self.p)
        self.needs = (("gripper",) + (("joints",) if self.p.min_lift_m > 0 else ())
                      + (("camera/overhead",) if self.camera else ()))
        self.events, self.log = [], []
        self.n_hold = self.n_release = self.n_suppressed = self.n_cancelled = 0
        self.task, self.step = None, None
        self._reset()

    def _reset(self) -> None:
        self.det.reset()
        self.last_gt = None
        self.pending = None

    def start(self, task, step) -> None:
        self.task, self.step = task, step
        self._reset()

    def _event(self, kind: str, t: float, evidence: dict) -> Event:
        ev = Event(kind, self.step.obj, float(t), self.name, evidence)
        self.events.append(ev.to_dict())
        return ev

    def observe(self, senses, est, intent: Intent, t: float) -> list:
        if self.step is None:
            return []
        out = []
        gt = senses.time("gripper")
        if gt != self.last_gt:                                    # ハンドの新しい読みごと（10 Hz）
            self.last_gt = gt
            out += self._reading(senses, est, intent, float(t), senses.get("gripper"), gt)
        if self.pending is not None:
            out += self._confirm(senses, est, float(t))
        return out

    def _reading(self, senses, est, intent, t, g, gt) -> list:
        closed = bool(intent.gripper_closed_cmd)
        if self.det.state == "fired" and not closed:              # 発火の後に開いたら、次の把持から数え直す
            self.det.reset()
        hand = senses.hand()
        ev = self.det.update(t, g.width, closed, hand)
        last = self.det.last
        if last is not None:
            self.log.append(dict(last, step=self.step.obj))
            if last["kind"] == "hold":
                self.n_hold += 1
                if self.pending is not None:                      # 確かめの間に持ち直した: 落としたにしない
                    self.n_cancelled += 1
                    self.log.append({"kind": "cancel", "t": _r(t), "why": "held_again"})
                    self.pending = None
            elif last["kind"] == "release":
                self.n_release += 1
        if ev is None:
            return []
        ev = dict(ev, sample_t=_r(gt), hand=_vec(hand))
        if ev["reason"] == "miss":
            return [self._event("missed", t, {"fire": ev})]
        tip = senses.tip()
        xy = tip if tip is not None else hand
        pl = None if est is None else est.places.get(self.step.place)
        if xy is not None and pl is not None and pl.contains_xy(xy):
            self.n_suppressed += 1                                # 置き場所の上で幅が変わった（置いた）
            self.log.append({"kind": "suppressed", "t": _r(t), "why": "over_place", "fire": ev})
            return []
        if not self.camera:
            return [self._event("dropped", t, {"fire": ev, "t_since": _r(t), "camera": None})]
        self.pending = {"fire": ev, "t_fire": t, "since": None}
        return []

    def _confirm(self, senses, est, t) -> list:
        """カメラの確かめ: 狙いの物の推定が置き場所の外で、seen か occluded（手の中でない）が confirm_s 続く。"""
        pd = self.pending
        o = None if est is None else est.objects.get(self.step.obj)
        if o is not None and o.in_place.get(self.step.place, False):
            self.n_cancelled += 1                                 # 置き場所に入った: 落としたにしない
            self.log.append({"kind": "cancel", "t": _r(t), "why": "in_place"})
            self.pending = None
            return []
        ok = o is not None and o.status in ("seen", "occluded") and o.near_hand is not True
        tip = senses.tip()
        dist = None if (o is None or tip is None) else float(np.linalg.norm(np.asarray(o.pos, float) - tip))
        if ok and REL.near_hand(o, tip, self.near_k, self.unc_k):         # 手に近い（物の推定の大きさからの相対）
            ok = False
        pl = None if est is None else est.places.get(self.step.place)
        if ok and REL.outside(o, pl, self.unc_k) is False:                # 箱の内側の領域から余白以上外でない
            ok = False
        if not ok:
            pd["since"] = None
            return []
        if pd["since"] is None:
            pd["since"] = t
        if t - pd["since"] < self.confirm_s - 1e-9:
            return []
        self.pending = None
        return [self._event("dropped", t, {"fire": pd["fire"], "t_fire": _r(pd["t_fire"]), "t_since": _r(pd["since"]),
                                           "camera": {"status": o.status, "pos": _vec(o.pos), "dist_tip_m": _r(dist),
                                                      "radius_m": _r(REL.radius(o)), "confirm_s": self.confirm_s}})]

    def record(self) -> dict:
        return {"name": self.name, "needs": list(self.needs), "params": self.p.to_dict(), "confirm_s": self.confirm_s,
                "camera_confirm": self.camera, "near_k": self.near_k, "unc_k": self.unc_k, "n_hold": self.n_hold,
                "n_release": self.n_release,
                "n_suppressed": self.n_suppressed, "n_cancelled": self.n_cancelled, "events": list(self.events),
                "log": list(self.log)}


class HeldEstimateDropMonitor:
    name = "held_estimate_drop"

    def __init__(self, params: ReflexParams = None, on_table_s: float = ON_TABLE_S, near_k: float = REL.NEAR_K,
                 unc_k: float = REL.UNC_K, table_frac: float = REL.TABLE_FRAC):
        self.p = params or ReflexParams()
        self.p.check()
        self.on_table_s = float(on_table_s)
        self.near_k, self.unc_k, self.table_frac = float(near_k), float(unc_k), float(table_frac)
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

    def observe(self, senses, est, intent: Intent, t: float) -> list:
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
        tip = senses.tip()
        near = REL.near_hand(o, tip, self.near_k, self.unc_k)
        if o is not None and o.near_hand is True:
            near = True
        if self.det.state == "holding" and near and o.status != "lost":
            if not self.armed:
                self.log.append({"kind": "armed", "t": _r(t)})
            self.armed, self.t_armed, self.since = True, t, None
            return []
        if not self.armed:
            return []
        pl = None if est is None else est.places.get(self.step.place)
        table_z = None if est is None else est.table_z
        ok = (o is not None and o.status in ("seen", "occluded") and near is False
              and REL.on_table(o, table_z, self.table_frac, self.unc_k) is True and REL.outside(o, pl, self.unc_k) is True)
        if not ok:
            self.since = None
            return []
        if self.since is None:
            self.since = t
        if t - self.since < self.on_table_s - 1e-9:
            return []
        self.armed = False
        ev = Event("dropped", self.step.obj, t, self.name,
                   {"t_since": _r(self.since), "t_armed": _r(self.t_armed), "status": o.status, "pos": _vec(o.pos),
                    "radius_m": _r(REL.radius(o)), "height_m": _r(REL.height(o)), "depth_m": _r(pl.depth(o.pos)),
                    "sigma_m": _r(REL.sigma(o)), "on_table_s": self.on_table_s})
        self.events.append(ev.to_dict())
        return [ev]

    def record(self) -> dict:
        return {"name": self.name, "needs": list(self.needs), "params": self.p.to_dict(), "on_table_s": self.on_table_s,
                "near_k": self.near_k, "unc_k": self.unc_k, "table_frac": self.table_frac, "events": list(self.events),
                "log": list(self.log)}


class LegacyInHandDropMonitor:
    """executor_replan_early._drop_seen と同じ判定（比べ用。動きは同じ）。in_hand は推定の occluded かつ near_hand、held は occluded。"""
    name = "legacy_in_hand_drop"
    needs = ("joints", "gripper", "camera/overhead", "camera/wrist")

    def __init__(self, confirm_s: float = DROP_CONFIRM_S):
        self.confirm_s = float(confirm_s)
        self.events = []
        self.task, self.step = None, None
        self._reset()

    def _reset(self) -> None:
        self.held_seen, self.since, self.fired = False, None, False

    def start(self, task, step) -> None:
        self.task, self.step = task, step
        self._reset()

    def observe(self, senses, est, intent: Intent, t: float) -> list:
        if est is None or self.step is None:
            return []
        o = est.objects.get(self.step.obj)
        if o is None:
            return []
        if o.status == "occluded" and o.near_hand is True:                      # in_hand
            self.held_seen, self.since = True, None
            return []
        if self.held_seen and o.status in ("seen", "occluded") and not o.in_place.get(self.step.place, False):
            if self.since is None:
                self.since = float(t)
            if float(t) - self.since >= self.confirm_s - 1e-9 and not self.fired:
                self.fired = True
                # t_since は丸めない（executor_replan_early の記録の値と同じにする）
                ev = Event("dropped", self.step.obj, float(t), self.name, {"t_since": self.since, "status": o.status})
                self.events.append(ev.to_dict())
                return [ev]
            return []
        self.since = None
        return []

    def record(self) -> dict:
        return {"name": self.name, "needs": list(self.needs), "confirm_s": self.confirm_s, "events": list(self.events)}


class MonitorSet:
    """needs がそろう検知だけ動かす。出来事は時刻・根拠つきで残す。"""

    def __init__(self, monitors, registry=None):
        reg = registry if registry is not None else standard_registry("sim")
        self.registry = reg
        self.monitors = list(monitors)
        self.active = [m for m in self.monitors if reg.available(m.needs)]
        self.skipped = [{"name": m.name, "missing": reg.missing(m.needs)} for m in self.monitors if m not in self.active]
        self.events = []
        self.step = None

    def start(self, task, step) -> None:
        self.step = step
        for m in self.active:
            m.start(task, step)

    def observe(self, senses, est, intent: Intent, t: float) -> list:
        out = []
        for m in self.active:
            for ev in m.observe(senses, est, intent, t):
                out.append(ev)
                self.events.append(dict(ev.to_dict(), step=None if self.step is None else dataclasses.asdict(self.step)))
        return out

    def record(self) -> dict:
        return {"target": self.registry.target, "active": [m.name for m in self.active], "skipped": list(self.skipped),
                "events": list(self.events), "monitors": {m.name: m.record() for m in self.active}}
