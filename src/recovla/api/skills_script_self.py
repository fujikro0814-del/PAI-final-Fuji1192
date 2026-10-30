"""手で書いたスクリプト方策（条件 CL。識別子の CL2 はこの版）。

api/skills.ScriptSkill（段階の決まりは api/skills_script）を、物の種類の知識（色の立方体・黄の箱）とロボット自身・較正・
制御の速さのほかは何も借りずに動くようにしたもの。ScriptSkill と違う所:
  - 通る高さ・放す高さ・掴む高さは、高さの地図（api/heightmap）と推定した大きさから決める（approach_height・carry_z・release_z・
    lift_m・retreat_pose を使わない）。戻る先は試行の始めの自分の姿勢
  - 「持っている」は 開き幅 ≥ 推定の短い幅 × hold_frac（min_grip_gap_m を使わない）。持っていたのに持っていない（運ぶ・持ち上げた
    段階で、閉の指令のまま）に変わったら落下として drops に記録する
  - 手順の完了を自分で判定する（step_done）: 物の推定が箱の内側の領域で rest_hold_s 止まり、指が開き、手が物の上から通る高さ
    まで離れた。実行器（api/executor_script）はこれで次へ進む
  - 起動は机の平面の確かめだけで、止まらない（箱の既知の寸法での当てはめ・box_moved などで止めない）。WorldModel と
    JudgeV2 は記録のためだけに作る（何も決めない）
  - 掴む高さは、指先の中心を物の推定の中心の高さへ。ただし指の下端が机＋余白より上、かつ手のひら（手の各部の下端）が物の
    上端＋余白より上（高いカップでは中心より上を掴む。立方体では効かない）。中心より上を掴んだ分（grasp_off）は、持っている
    物の下端・放す高さに足す
  - 置き場所が皿らしい物（kind surface、api/geometry_place.PlateGeom）なら、上面の円の中心から、地図で「持っている物の半径＋
    余白のシード範囲の中が皿の上面の高さだけ」の所を選ぶ（_surface_spot）
倍数は configs/api/cl2.yaml。速さ・ゲイン・時間は configs の expert（制御の値）。作業空間の下限・上限は安全の限界
として setup.workspace から読む。
シミュレーションの真値を読まない（recovla.expert・sim・harness・eval を import せず、既知の寸法の名前を読まない。テストで検査）。
"""
import dataclasses
import enum
import math
from typing import Optional

import numpy as np

from recovla.api import heightmap as HM
from recovla.api import relations as REL
from recovla.api.estimated import GeometryPerception
from recovla.api.skills import ScriptSkill, SkillInfo
from recovla.api.skills_script import Clocks, choose_slot, slot_xy
from recovla.common import config
from recovla.runtime.runner import ACTION_DT

CONFIG = "configs/api/cl2.yaml"
HAND_MAX_WIDTH = 0.08                       # ハンドの開きの上限（ロボット自身の性質。開く指令・手と指の形と同じ値）
CL2_INFO = SkillInfo("script_cl2", ("joints", "gripper", "camera/overhead", "camera/wrist"),
                     "自力の先生（高さの地図・推定の大きさ・自分の完了の判定。物の種類の知識のほかは借りない）")


class CL2Phase(enum.IntEnum):
    approach = 0
    descend = 1
    close = 2
    reopen = 3
    lift = 4
    carry = 5
    release = 6
    settle = 7
    clear = 8                               # 放した後、物の上から通る高さまで離れる（ScriptSkill の retreat の代わり）
    done = 9                                # この手順は済んだ（自分の判定）
    search = 10
    home = 11                               # 試行の始めの姿勢へ戻る


@dataclasses.dataclass(frozen=True)
class CL2Rules:
    # configs の expert（制御の速さ・ゲイン・時間）
    gain: float
    xy_max: float
    z_max: float
    move_tol: float
    still: float
    close_settle_s: float
    finger_rest_speed: float
    rest_speed: float
    rest_hold_s: float
    z_tol: float                            # expert.phase.grasp_z_tol_m（参照の高さに着いた）
    h_tol: float                            # expert.phase.above_tol_m（大きさの分からない物の真上・経路の水平の着き）
    # configs/api/cl2.yaml（推定からの相対の倍数）
    hold_frac: float = 0.5
    open_frac: float = 1.1
    lift_frac: float = 0.25
    above_frac: float = 0.25
    grasp_tol_frac: float = 0.1
    grasp_unc_k: float = 1.0
    far_k: float = 2.0
    airborne_frac: float = 0.25
    clear_frac: float = 0.25
    slot_frac: float = 0.375
    unc_k: float = 2.0
    map_cell_k: float = 2.0
    map_unc_k: float = 3.0
    map_self_k: float = 2.0
    map_below_k: float = 3.0
    map_edge_k: float = 5.0
    grasp_eps_frac: float = 0.125
    startup_wait_s: float = 3.0
    home_wait_s: float = 10.0
    est_speed_window_s: float = 1.0
    still_window_s: float = 0.1
    finger_moved_m: float = 0.002

    @classmethod
    def from_config(cls, expert: dict, phase: dict, cl2: dict = None, **over) -> "CL2Rules":
        """expert・phase は configs の expert とその phase（呼び手が分けて渡す）。cl2 は configs/api/cl2.yaml の cl2（省けば読む）。"""
        e, p = expert, phase
        c = dict(cl2 if cl2 is not None else load_cl2())
        names = {f.name for f in dataclasses.fields(cls)}
        bad = sorted(set(c) - names)
        if bad:
            raise ValueError(f"cl2.yaml に知らない項目 {bad}")
        kw = dict(gain=float(e["gain_per_s"]), xy_max=float(e["speed_ref"]["xy"]), z_max=float(e["speed_ref"]["z"]),
                  move_tol=float(e["move_tol_m"]), still=float(e["hand_still_speed"]), close_settle_s=float(e["close_settle_s"]),
                  finger_rest_speed=float(e["finger_rest_speed"]), rest_speed=float(p["rest_speed"]),
                  rest_hold_s=float(p["rest_hold_s"]), z_tol=float(p["grasp_z_tol_m"]), h_tol=float(p["above_tol_m"]))
        kw.update({k: float(v) for k, v in c.items()})
        kw.update(over)
        return cls(**kw)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def load_cl2() -> dict:
    import yaml
    return yaml.safe_load(config.path(CONFIG).read_text(encoding="utf-8"))["cl2"]


@dataclasses.dataclass(frozen=True)
class CL2Obs:
    """1 回の決め直しの入力（真値なし）。skills_script.Obs と同じ欄に、地図・戻る先・作業空間を足した。"""
    t: float
    target: object
    others: dict
    place: object
    place_name: str
    closed: bool
    width: float
    width_t: float
    hand: np.ndarray
    tip: np.ndarray
    x_cmd: np.ndarray
    table_z: float
    est_t: Optional[float] = None
    hmap: object = None                     # api.heightmap.HeightMap（無ければ None）
    home: Optional[np.ndarray] = None       # 試行の始めの手先の参照（作業空間の中へ）
    ws_z: tuple = (-math.inf, math.inf)     # 作業空間の高さの下限・上限（安全の限界）


@dataclasses.dataclass
class CL2Command:
    vel: np.ndarray
    closed: bool
    phase: CL2Phase
    stage: str = ""                         # 経路の段（rise・move・lower）


def _known(o) -> bool:
    return o is not None and o.status != "lost"


def _foot_r(o) -> Optional[float]:
    """物の足跡の半径: 向きつきの外接箱の、机に沿う 2 軸（上向きにいちばん近い軸を除く）の長方形の対角の半分。"""
    e = getattr(o, "extent", None)
    if e is None:
        return None
    R = np.eye(3) if getattr(o, "R", None) is None else np.asarray(o.R, float)
    k = int(np.argmax(np.abs(R[2, :])))
    return 0.5 * float(np.hypot(*[float(e[i]) for i in range(3) if i != k]))


class CL2Brain:
    """段階を毎回決め直し、その段階の目標へ向かう指令を 1 回分出す（ScriptBrain と同じ考え。高さは地図から）。
    profile_fn(width) → 手と指の形（api/heightmap.Part の列）。width が None なら開いた指。"""

    def __init__(self, rules: CL2Rules, dt: float, profile_fn, slot_order=None, view_from=None, palm_fn=None):
        """palm_fn → 手のひら（指を除く手）の形（Part の列）。None なら掴む高さに手のひらの決まりを足さない。"""
        self.r = rules
        self.dt = float(dt)
        self.profile_fn = profile_fn
        self.palm_fn = palm_fn
        self.slot_order = None if slot_order is None else tuple(int(s) for s in slot_order)
        self.view_from = None if view_from is None else np.asarray(view_from, float)[:2]
        self.drops = []
        self.home_mode = False
        self.reset()

    def reset(self) -> None:
        self.clocks = Clocks()
        self.obj_w = None                   # 狙いの物の推定の短い幅（最後に分かった値）
        self.obj_r = None                   # 同じく半径
        self.obj_rh = None                  # 同じく足跡の半径（机に沿う 2 軸の外接長方形の対角の半分。皿の置き場所に使う）
        self.prev_carry = False             # 前の刻みで、持っていて持ち上がっていた
        self.prev_phase = None
        self.step_done = False
        self.slot_lock = None               # 持っている間の置き場所（世界）
        self.rt_lock = None                 # 「放す」に入った時の放す指先の高さ
        self.stall_s = 0.0                  # 「放す」で、止まったまま放す高さに届かない時間
        self.grasp_off = 0.0                # 閉じたとき、指先の中心が物の推定の中心より上だった分（開いたら 0）

    # ------------------------------------------------------------------ 道具
    def _order(self, place) -> tuple:
        if self.slot_order is not None:
            return self.slot_order
        if self.view_from is None:
            return tuple(range(4))
        return tuple(sorted(range(4), key=lambda s: -float(np.hypot(*(slot_xy(place, s) - self.view_from)))))

    @staticmethod
    def _txy(obs, p) -> np.ndarray:
        return obs.hmap.to_table_xy(p) if obs.hmap is not None else np.asarray(p, float)[:2]

    @staticmethod
    def _tip_h(obs) -> float:
        return float(obs.tip[2]) - obs.table_z

    @staticmethod
    def _drop(obs) -> float:
        return float(obs.hand[2] - obs.tip[2])

    def _hand_z(self, obs, tip_h: float) -> float:
        lo, hi = obs.ws_z
        return float(np.clip(obs.table_z + tip_h + self._drop(obs), lo, hi))

    def _held(self, obs):
        """持っている物の (半径, 指先の中心から下端までの長さ)。分からなければ (None, 0)。"""
        h = self.clocks.obj_h
        if self.obj_r is None or h is None:
            return None, 0.0
        return self.obj_r, (0.5 + self.r.clear_frac) * h + self.grasp_off

    def pass_hand_z(self, obs, a_world, b_world, width=None, holding=False) -> float:
        """a→b（世界の点の水平）を通る手先の参照の高さ。地図が無ければ今の高さ（下りない）。"""
        if obs.hmap is None:
            return float(obs.x_cmd[2])
        hr, hb = self._held(obs) if holding else (None, 0.0)
        need = HM.pass_tip_height(obs.hmap, self._txy(obs, a_world), self._txy(obs, b_world), self.profile_fn(width),
                                  hr, hb, self.r.map_unc_k)
        if need is None:
            return float(obs.x_cmd[2])
        return self._hand_z(obs, need)

    def hand_clear(self, obs) -> bool:
        """手（指・手のひら）が、今の真下の地図（置いた物を含む）から通る高さまで離れたか。"""
        if obs.hmap is None:
            return True
        need = HM.pass_tip_height(obs.hmap, self._txy(obs, obs.tip), self._txy(obs, obs.tip), self.profile_fn(None),
                                  None, 0.0, self.r.map_unc_k)
        return need is None or self._tip_h(obs) >= need - self.r.z_tol

    def grasp_tip_h(self, obs) -> float:
        """掴む指先の中心の高さ（机から）。指先の中心を物の推定の中心の高さへ。ただし (1) 指の下端が机（地図の、物のまわりの
        低い所ではなく机の高さ）＋余白より上、(2) 手のひら（palm_fn の各部の下端）が物の上端（中心 ＋ 見かけの高さの半分）＋余白
        より上。余白はどちらも地図の σ × map_unc_k（上端は上から見えている面なので、下へ延ばした分の不確かさは入れない）。"""
        o = obs.target
        sig = obs.hmap.sigma if obs.hmap else 0.0
        below = HM.finger_below(self.profile_fn(None))
        tip_h = max(float(o.pos[2]) - obs.table_z, below + self.r.map_unc_k * sig)
        h = REL.height(o)
        if self.palm_fn is not None and h is not None:
            top = float(o.pos[2]) - obs.table_z + 0.5 * h
            tip_h = max(tip_h, top + self.r.map_unc_k * sig - min(p.low for p in self.palm_fn()))
        return tip_h

    def grasp_hand_z(self, obs) -> float:
        return self._hand_z(obs, self.grasp_tip_h(obs))

    def release_tip_h(self, obs, slot_world) -> Optional[float]:
        """持っている物の下端が、置き場所の下の地図の高さ（箱の底・底の上の物）＋余裕に来る指先の高さ。開く指が壁に当たらない
        ように、開いた指の下端も置き場所のまわりの地図より上にする。地図か物の大きさが分からなければ None。"""
        h, hr = self.clocks.obj_h, self.obj_r
        if obs.hmap is None or h is None or hr is None:
            return None
        s = self._txy(obs, slot_world)
        fp = obs.hmap.band_max(s, s, hr)
        if fp is None:
            return None
        need = fp + 0.5 * h + self.grasp_off
        prof = self.profile_fn(None)
        for p in prof:
            m = obs.hmap.band_max(s, s, p.radius)
            if m is not None:
                need = max(need, m - p.low)
        return need + self.r.clear_frac * h + self.r.map_unc_k * obs.hmap.sigma

    def above_tol(self, o) -> float:
        w = REL.width_min(o) if o is not None else None
        return self.r.h_tol if w is None else self.r.above_frac * w + self.r.unc_k * REL.sigma(o)

    def grasp_tol(self, o) -> float:
        w = REL.width_min(o) if o is not None else None
        return self.r.move_tol if w is None else self.r.grasp_tol_frac * w + self.r.grasp_unc_k * REL.sigma(o)

    def holding(self, obs) -> bool:
        """閉じていて、開き幅の読み ≥ 推定の短い幅 × hold_frac、かつ写っている物が指先から far_k × 半径 ＋ 余白 より遠くない。"""
        if not obs.closed or self.obj_w is None or float(obs.width) < self.r.hold_frac * self.obj_w:
            return False
        o = obs.target
        rad = REL.radius(o) if o is not None else None
        if o is not None and o.status == "seen" and o.near_hand is not True and rad is not None:
            if float(np.linalg.norm(np.asarray(o.pos, float) - obs.tip)) > self.r.far_k * rad + self.r.unc_k * REL.sigma(o):
                return False
        return True

    def lifted(self, obs) -> bool:
        z0, h = self.clocks.z_close, self.clocks.obj_h
        if z0 is None or h is None:
            return False
        return float(obs.hand[2]) - z0 >= self.r.lift_frac * h

    def airborne(self, obs) -> bool:
        o = obs.target
        h = REL.height(o) if o is not None else None
        if not _known(o) or h is None:
            return False
        return float(o.pos[2]) - (obs.table_z + 0.5 * h) >= self.r.airborne_frac * h + self.r.unc_k * REL.sigma(o)

    def opened(self, obs) -> bool:
        return (not obs.closed) and self.obj_w is not None and float(obs.width) >= self.r.open_frac * self.obj_w

    def _slot(self, obs):
        """置き場所（世界）。持っている間は、最初に決めた所を変えない（推定の揺れで置き場所が跳ばないように）。"""
        if self.slot_lock is not None:
            return self.slot_lock
        pl = obs.place
        if pl is None or not pl.ok or pl.region is None:
            return None
        if REL.place_kind(pl) == "surface":
            s = self._surface_spot(obs)
        else:
            s = slot_xy(pl, choose_slot(obs, self.r, self._order(pl)))
            s = self._free_slot(obs, np.array([s[0], s[1], float(pl.region.center[2])]))
        if self.holding(obs) and self.lifted(obs):
            self.slot_lock = s
        return s

    def _free_slot(self, obs, s) -> np.ndarray:
        """置き場所 s を、地図で「持っている物の半径＋余白のシード範囲の中が箱の底の高さだけ」（壁にも置いた物にも載らない）になる
        所まで、箱の中心へ寄せる。底の高さは箱の中の升の低い方（band_low）。地図か物の大きさが分からなければ s のまま。"""
        hm, h, hr, pl = obs.hmap, self.clocks.obj_h, self.obj_r, obs.place
        if hm is None or h is None or hr is None:
            return s
        c = np.array([pl.region.center[0], pl.region.center[1], s[2]])
        inner = float(np.min(pl.region.inner))
        floor = hm.band_low(self._txy(obs, c), inner)
        if floor is None:
            return s
        reach = hr + hm.cell + self.r.unc_k * REL.sigma(obs.target)
        tol = self.r.clear_frac * h + self.r.map_unc_k * hm.sigma
        for t in np.linspace(0.0, 1.0, 9):
            p = s + t * (c - s)
            m = hm.band_max(self._txy(obs, p), self._txy(obs, p), reach)
            if m is not None and m <= floor + tol:
                return p
        return s

    def _surface_spot(self, obs) -> np.ndarray:
        """皿らしい置き場所（PlateGeom）: 上面の円の中心から、中心に近い順（中心、升の幅ごとの輪の 8 方向、輪はシード範囲が上面の円の中に
        収まるまで）に、地図で「持っている物の足跡の半径＋余白のシード範囲の中が皿の上面の高さだけ」になる所（_free_slot と同じ決まり。
        立った円筒は 3 次元の半径より足跡の方がずっと小さいので足跡で見る）。
        上面の高さは中心のまわり（半径の半分）の升の低い方（band_low）。地図か物の大きさが分からなければ中心。"""
        reg = obs.place.region
        c = np.asarray(reg.center, float)
        hm, h, hr = obs.hmap, self.clocks.obj_h, self.obj_rh
        if hm is None or h is None or hr is None:
            return c
        top = hm.band_low(self._txy(obs, c), 0.5 * float(reg.radius))
        if top is None:
            return c
        reach = hr + hm.cell + self.r.unc_k * REL.sigma(obs.target)
        tol = self.r.clear_frac * h + self.r.map_unc_k * hm.sigma
        R = np.asarray(reg.R, float)
        for rr in np.arange(0.0, max(float(reg.radius) - reach, 0.0) + 1e-9, hm.cell):
            for k in range(1 if rr == 0.0 else 8):
                a = 2.0 * math.pi * k / 8.0
                p = c + R @ np.array([rr * math.cos(a), rr * math.sin(a), 0.0])
                m = hm.band_max(self._txy(obs, p), self._txy(obs, p), reach)
                if m is not None and m <= top + tol:
                    return p
        return c

    # ------------------------------------------------------------------ 段階
    def phase_of(self, obs) -> CL2Phase:
        if self.home_mode:
            return CL2Phase.home
        r, c = self.r, self.clocks
        o = obs.target
        known = _known(o)
        inb = bool(known and o.in_place.get(obs.place_name, False))
        rested = c.target_rest_s >= r.rest_hold_s - 1e-9
        hold = self.holding(obs)
        up = hold and self.lifted(obs)
        x = np.asarray(obs.x_cmd, float)
        if not obs.closed and inb and rested:
            return CL2Phase.done if (self.opened(obs) and self.hand_clear(obs)) else CL2Phase.clear
        if not hold and known:
            moving = c.target_speed >= r.rest_speed
            if moving or (inb and not rested) or ((not obs.closed) and self.airborne(obs) and not inb):
                return CL2Phase.settle
        if hold and up:
            s = self._slot(obs)
            if s is not None:
                near = math.hypot(*(np.asarray(obs.tip, float)[:2] - s[:2])) <= self.above_tol(o)
                if near and (self.rt_lock is not None             # 一度「放す」に入ったら、真上にいる間は戻らない
                             or self.stall_s >= r.close_settle_s - 1e-9):   # 下ろす途中で載って止まった
                    return CL2Phase.release
                rt = self.release_tip_h(obs, s)
                if near and rt is not None and self._tip_h(obs) <= rt + r.z_tol:
                    return CL2Phase.release
            return CL2Phase.carry
        if hold:
            return CL2Phase.lift
        if obs.closed:
            return CL2Phase.reopen
        if not known:
            return CL2Phase.search
        above = math.hypot(*(np.asarray(obs.tip, float)[:2] - np.asarray(o.pos, float)[:2])) <= self.above_tol(o)
        if above and x[2] <= self.grasp_hand_z(obs) + r.z_tol:
            return CL2Phase.close
        if above:
            return CL2Phase.descend
        return CL2Phase.approach

    def _vel_to(self, x, goal) -> np.ndarray:
        r = self.r
        v = r.gain * (np.asarray(goal, float) - np.asarray(x, float))
        n = np.linalg.norm(v[:2])
        if n > r.xy_max:
            v[:2] *= r.xy_max / n
        v[2] = float(np.clip(v[2], -r.z_max, r.z_max))
        return v

    def _still(self) -> bool:
        return self.clocks.hand_speed < self.r.still

    def step(self, obs: CL2Obs) -> CL2Command:
        r = self.r
        self.clocks.update(obs, self.dt, r)
        o = obs.target
        if o is not None and o.extent is not None:
            self.obj_w, self.obj_r, self.obj_rh = REL.width_min(o), REL.radius(o), _foot_r(o)
        if not obs.closed:
            self.grasp_off = 0.0
        hold = self.holding(obs)
        carry = hold and self.lifted(obs)
        if not carry:
            self.slot_lock, self.rt_lock = None, None
        ph = self.phase_of(obs)
        if ph not in (CL2Phase.carry, CL2Phase.release):
            self.stall_s = 0.0
        if self.prev_carry and not hold and obs.closed:          # 持ち上げて持っていたのに、閉の指令のまま持っていない
            self.drops.append({"t": round(float(obs.t), 4), "phase_before": None if self.prev_phase is None else
                               CL2Phase(self.prev_phase).name, "width": round(float(obs.width), 5),
                               "obj_w": None if self.obj_w is None else round(self.obj_w, 5),
                               "obj_pos": None if o is None else [round(float(v), 4) for v in o.pos],
                               "on_table": REL.on_table(o, obs.table_z), "in_place": REL.in_place(o, obs.place)})
        self.prev_carry, self.prev_phase = carry, ph
        self.step_done = ph == CL2Phase.done
        x = np.asarray(obs.x_cmd, float)
        zero = np.zeros(3)
        closed = bool(obs.closed)
        if ph == CL2Phase.home:
            home = np.array(obs.home if obs.home is not None else x, float)
            home[2] = float(np.clip(home[2], *obs.ws_z))
            tip_home = home - (obs.hand - obs.tip)
            pz = self.pass_hand_z(obs, obs.tip, tip_home, obs.width if closed else None)
            goal, stage = HM.path_goal(x, home, pz, r.h_tol, r.z_tol)
            return CL2Command(self._vel_to(x, goal), False, ph, stage)
        if ph in (CL2Phase.done, CL2Phase.settle):
            return CL2Command(zero, closed, ph)
        if ph in (CL2Phase.clear, CL2Phase.search):
            if ph == CL2Phase.clear and not self.opened(obs):
                return CL2Command(zero, False, ph)                # まだ指が開き切っていない: 待つ
            pz = self.pass_hand_z(obs, obs.tip, obs.tip)
            return CL2Command(self._vel_to(x, np.array([x[0], x[1], max(pz, x[2])])), closed, ph, "rise")
        if ph == CL2Phase.reopen:
            return CL2Command(np.array([0.0, 0.0, r.z_max]), False, ph, "rise")
        if ph in (CL2Phase.lift, CL2Phase.carry, CL2Phase.release):
            s = self._slot(obs)
            if ph == CL2Phase.lift:
                if self.clocks.finger_rest_s < r.close_settle_s - 1e-9:
                    return CL2Command(zero, closed, ph)
                to = obs.tip if s is None else s
                pz = self.pass_hand_z(obs, obs.tip, to, obs.width, holding=True)
                return CL2Command(self._vel_to(x, np.array([x[0], x[1], max(pz, x[2])])), closed, ph, "rise")
            if s is None:
                return CL2Command(zero, closed, ph)              # 置き場所が分からない: その場で保つ
            off = s[:2] - np.asarray(obs.tip, float)[:2]
            aim = x[:2] + off
            rt = self.release_tip_h(obs, s) if self.rt_lock is None else self.rt_lock
            if ph == CL2Phase.release and self.rt_lock is None and rt is not None:
                self.rt_lock = rt
            gz = self._hand_z(obs, rt) if rt is not None else float(x[2])
            # 置き場所の真上で止まったまま下ろす高さに届かない（持った物か指が縁などに載った）: close_settle_s 続けば放す
            # 指令は届いても、実際の手が止まったまま上に残る（指令と手の差の綱の内側で載っている）ので、手の高さで見る
            stalled = self._still() and np.hypot(*off) <= r.move_tol and float(obs.hand[2]) > gz + r.move_tol
            self.stall_s = self.stall_s + self.dt if stalled else 0.0
            if ph == CL2Phase.release:
                if np.hypot(*off) <= r.move_tol and self._still() and (abs(x[2] - gz) <= r.move_tol
                                                                         or self.stall_s >= r.close_settle_s - 1e-9):
                    return CL2Command(zero, False, ph)            # 開く（着いた、または下ろす途中で載って止まった）
                return CL2Command(self._vel_to(x, np.array([aim[0], aim[1], gz])), closed, ph, "lower")
            pz = self.pass_hand_z(obs, obs.tip, s, obs.width, holding=True)
            goal, stage = HM.path_goal(x, np.array([aim[0], aim[1], gz]), pz, self.above_tol(o) if o else r.h_tol, r.z_tol)
            return CL2Command(self._vel_to(x, goal), closed, ph, stage)
        # 掴みに行く: 指先の中心を物の推定の中心に合わせる
        g = np.asarray(o.pos, float)[:2]
        off = g - np.asarray(obs.tip, float)[:2]
        aim = x[:2] + off
        gz = self.grasp_hand_z(obs)
        if ph == CL2Phase.close:
            if np.hypot(*off) <= self.grasp_tol(o) and abs(x[2] - gz) <= r.move_tol and self._still():
                self.grasp_off = max(0.0, self.grasp_tip_h(obs) - (float(o.pos[2]) - obs.table_z))
                return CL2Command(zero, True, ph)                 # 閉じる
            return CL2Command(self._vel_to(x, np.array([aim[0], aim[1], gz])), closed, ph, "lower")
        pz = self.pass_hand_z(obs, obs.tip, o.pos)
        goal, stage = HM.path_goal(x, np.array([aim[0], aim[1], gz]), pz, self.above_tol(o), r.z_tol)
        return CL2Command(self._vel_to(x, goal), closed, ph, stage)


# ------------------------------------------------------------------------------ 知覚
class CL2Perception(GeometryPerception):
    """GeometryPerception（WorldModel は記録の JudgeV2 のためだけ）に、高さの地図を足したもの。
    update の返り値の WorldModel に heightmap（その時刻の写し）を付ける。持っている物（手のそばの推定）の点は地図から外す。
    io.compute の種類は "perception_cl2"（地図の分を含む計算の時間を測った遅延。configs/runtime_v2.yaml の compute_latency_s）。"""
    compute_kind = "perception_cl2"

    @classmethod
    def attach_cl2(cls, per, finder, hmap: HM.HeightMap, near_k: float = REL.NEAR_K, unc_k: float = REL.UNC_K):
        GeometryPerception.attach(per, finder)
        per.__class__ = cls
        per.hmap, per.near_k, per.unc_k = hmap, float(near_k), float(unc_k)
        return per

    def points(self, frame, width: float, margin: float = None):
        """推定の版と同じ点群に、そのフレームの関節角で置いたロボットの部品の箱（地図から自分の点を外すため）を添える。
        Perception.points の camera_pose が撮った時刻の関節角を selfr に置いた直後なので、その姿勢の箱になる。"""
        P = super().points(frame, width, margin)
        c = self._clouds.get(frame.name) if margin is None else None
        if c is not None:
            c["robot_boxes"] = HM.robot_boxes(self.selfr.m, self.selfr.d)
            if P.get("idx") is not None and getattr(frame, "depth", None) is not None:
                c["edge"] = HM.depth_edges(frame.depth, self.hmap.edge_k * self.hmap.sigma).reshape(-1)[P["idx"]]
        return P

    def update(self, frames: dict, gripper, joints_t: float, hand_tip=None):
        wm = super().update(frames, gripper, joints_t, hand_tip)
        exclude = []
        if getattr(gripper, "is_grasped", False) and wm.geometry is not None:
            held_r, any_r, held_low = None, None, None
            for o in wm.geometry.objects.values():
                r = REL.radius(o)
                if r is None:
                    continue
                reach = self.near_k * r + self.unc_k * REL.sigma(o)
                any_r = reach if any_r is None else max(any_r, reach)
                if o.near_hand:
                    exclude.append((o.pos, reach))
                if hand_tip is not None and (o.near_hand or np.linalg.norm(np.asarray(o.pos) - hand_tip) <= reach):
                    held_r = reach if held_r is None else max(held_r, reach)
                    h = REL.height(o)                                  # 持っている物の推定の下端（高い物は指先から半径より下へ出る。
                    #                                                    推定の中心が遅延して球の外へ出ても柱で外す）
                    if h is not None:
                        low = float(o.pos[2]) - 0.5 * h - self.unc_k * REL.sigma(o)
                        held_low = low if held_low is None else min(held_low, low)
            held_r = held_r if held_r is not None else any_r          # どれを持っているか分からなければ、いちばん大きい物で
            if hand_tip is not None and held_r is not None:
                # 持っている物は指先と一緒に動く: 推定が遅延しても、指先の真上の柱（物の下端より上、半径 + 升の幅）を外す。
                # 持っている物の真上にあるのは、その物かハンドだけ（ほかの物があればぶつかっている）
                tip = np.asarray(hand_tip, float)
                z_min = float(tip[2]) - held_r
                if held_low is not None:
                    z_min = min(z_min, held_low)
                exclude.append((tip, held_r + (self.hmap.cell or 0.0), z_min))
        self.hmap.update(list(self._clouds.values()), float(joints_t), getattr(self.geom, "cam_shift", {}), exclude)
        wm.heightmap = self.hmap.snapshot()
        return wm


# ------------------------------------------------------------------------------ 技能
class CL2Skill(ScriptSkill):
    """ScriptSkill（PolicyRuntime と同じ口）を継ぎ、段階の決まりを CL2Brain に、起動の確かめを机の平面だけに替える。"""
    info = CL2_INFO
    needs = CL2_INFO.needs

    def __init__(self, io, setup, rules: CL2Rules, spec, instruction_fmt: str, perception=None, checks: dict = None,
                 tip_offset: float = 0.1034, motion=None, ctrl_every: int = 10, profile=None, slot_order=None, palm=None):
        """profile・palm: 手と指・手のひらの形を固定で渡す（偽の世界の確かめ用）。profile を渡して palm を渡さなければ、手のひらの
        決まりは使わない。どちらも渡さなければロボットのモデルから作る。"""
        super().__init__(io, setup, rules, spec, instruction_fmt, perception=perception, checks=checks, tip_offset=tip_offset,
                         motion=motion, ctrl_every=ctrl_every, slot_order=slot_order, finder="estimated")
        self.grasp_eps_frac = float(rules.grasp_eps_frac)
        self._profiles = {}
        self._profile = None if profile is None else tuple(profile)
        self._palm = tuple(palm) if palm is not None else None
        cams = getattr(setup, "cameras", None) or {}
        fixed = [c for c in cams.values() if getattr(c, "mount", None) == "world"]
        view = np.asarray(fixed[0].extrinsic.t, float)[:2] if fixed else None
        self.brain = CL2Brain(rules, dt=self.ctrl_every * float(self.motion.dt), profile_fn=self.profile_for,
                              slot_order=slot_order, view_from=view,
                              palm_fn=self.palm_for if (profile is None or palm is not None) else None)
        ws = getattr(setup, "workspace", None) or {}
        self.ws_z = tuple(float(v) for v in ws.get("z", (-math.inf, math.inf)))
        self.home = None
        self.at_home = False
        self.t_home = None

    def profile_for(self, width=None) -> tuple:
        """手と指の形（ロボットのモデルから、開き幅 5 mm 刻みで一度だけ計算する）。width が None なら開いた指（0.08）。"""
        if self._profile is not None:
            return self._profile
        w = 0.08 if width is None else float(np.clip(np.ceil(float(width) / 0.005) * 0.005, 0.0, 0.08))
        if w not in self._profiles:
            self._profiles[w] = HM.gripper_profile(self.motion.model, self.tip_offset, w)
        return self._profiles[w]

    def palm_for(self) -> tuple:
        """手のひら（指を除く手）の形（ロボットのモデルの手の体から、一度だけ計算する）。"""
        if self._palm is None:
            self._palm = HM.gripper_profile(self.motion.model, self.tip_offset, 0.08, bodies=("hand",))
        return self._palm

    # ------------------------------------------------------------------ 試行
    def start(self, task: str, seed: int) -> None:
        super().start(task, seed)
        if self.home is None:
            h = np.asarray(self.motion.x_cmd, float).copy()
            h[2] = float(np.clip(h[2], *self.ws_z))
            self.home = h

    def _begin(self, task: str) -> None:
        super()._begin(task)
        self.brain.home_mode = False
        self.at_home = False

    def go_home(self) -> None:
        """全部の手順の後: 試行の始めの姿勢へ、同じ経路の決まりで戻る。"""
        self.brain.home_mode = True
        self.t_home = float(self.io.now())

    @property
    def step_done(self) -> bool:
        return bool(self.brain.step_done)

    @property
    def drops(self) -> list:
        return self.brain.drops

    def _perception(self, t: float) -> None:
        """PolicyRuntime._perception と同じ流れ（io.compute の遅延・10 Hz）。起動の確かめは机の平面（check_table）と、記録の
        JudgeV2 のための箱の当てはめ（init_box）だけで、どちらの結果でも止めない（推定で始める）。"""
        per = self.perception
        if self.pend_per is not None and self.pend_per.ready(t):
            out = self.pend_per.result(t)
            self.pend_per = None
            if out["kind"] == "startup":
                self.startup = {"table": out["table"], "record_only_box": {k: (np.asarray(v).tolist() if isinstance(v, np.ndarray)
                                                                               else v) for k, v in out["box"].items()}}
                self.ready = True
            else:
                self.wm = out["wm"]
                est = getattr(self.wm, "geometry", None)
                self.log_per.append({"t": t, "t_obs": out["t_obs"], "geometry": None if est is None else {
                    n: {"pos": np.asarray(o.pos).round(4).tolist(), "status": o.status, "in_place": o.in_place}
                    for n, o in est.objects.items()}, "box_ok": None if est is None else bool(est.places and
                                                                                         next(iter(est.places.values())).ok)})
                self._check_target(t)
        if self.pend_per is not None or self.stop_reason:
            return
        sensor = self.io.sense(cameras=True)
        if not {"overhead", "wrist"} <= set(sensor.cameras):
            return
        if not self.ready:
            if not getattr(per, "hist_t", None):             # 関節の記録がまだ無い（最初の区切り）: 次の区切りで確かめる
                return
            fr, w = sensor.cameras["overhead"], sensor.gripper.width

            def startup():
                return {"kind": "startup", "table": per.check_table(fr, w), "box": per.init_box([(fr, w)])}
            self.pend_per = self.io.compute("perception_startup", startup)    # 起動時の 1 回は別の遅延（VLA の条件と同じ種類）
            return
        tip = self._tip(sensor.joints.q)
        t_obs = t

        def work():
            return {"kind": "update", "wm": per.update(sensor.cameras, sensor.gripper, sensor.t, tip), "t_obs": t_obs}
        self.pend_per = self.io.compute(getattr(per, "compute_kind", "perception"), work)   # 高さの地図を含む重い計算の遅延

    def observation(self, sensor) -> CL2Obs:
        o = super().observation(sensor)
        return CL2Obs(**{f.name: getattr(o, f.name) for f in dataclasses.fields(o)},
                      hmap=getattr(self.wm, "heightmap", None), home=self.home, ws_z=self.ws_z)

    def _apply(self, a) -> None:
        """腕の参照の速さ・ハンドの grasp / move。grasp の幅は推定の短い幅（分からなければ閉じない。既知の大きさに戻さない）。"""
        self.motion.set_velocity(a[:3] / ACTION_DT)
        want = bool(a[6] > 0.0)
        if want == self.closed:
            return
        su = self.setup
        if not want:
            self.closed = False
            self.t_grip_switch = self.io.now()
            self.io.gripper_move(0.08, su.gripper_speed)
            return
        o = None if self._obs is None else self._obs.target
        w = None if o is None else REL.width_min(o)
        if w is None:
            return
        w = min(w, HAND_MAX_WIDTH)                                      # ハンドの開きの上限を超える幅は渡さない（ハンドが拒む）
        self.closed = True
        self.t_grip_switch = self.io.now()
        eps = self.grasp_eps_frac * w + self.rules.unc_k * REL.sigma(o)
        self.grasp_log.append({"t": round(float(self.io.now()), 4), "width": round(w, 5), "eps": round(eps, 5)})
        self.io.gripper_grasp(w, su.gripper_speed, su.grasp_force, eps, eps)

    def _script_step(self, sensor) -> None:
        obs = self.observation(sensor)
        self._obs = obs
        cmd = self.brain.step(obs)
        if cmd.phase == CL2Phase.home and float(np.linalg.norm(np.asarray(obs.x_cmd) - self.home)) <= self.rules.h_tol:
            self.at_home = True
        a = np.zeros(7)
        a[:3] = cmd.vel * ACTION_DT
        a[6] = 1.0 if cmd.closed else -1.0
        self._apply(a)
        if cmd.phase != self.last_phase:
            self.log_script.append({"t": round(obs.t, 4), "k": int(self.k), "phase": CL2Phase(cmd.phase).name,
                                    "stage": cmd.stage, "obj": self.step.obj,
                                    "target_status": None if obs.target is None else obs.target.status,
                                    "width": round(obs.width, 5), "closed": bool(cmd.closed)})
            self.last_phase = cmd.phase
        if self.k != self.k_logged:
            self.log_act.append((self.k, obs.t, None, False, a.copy()))
            self.k_logged = self.k

    def trace(self) -> dict:
        out = super().trace()
        out["script"]["skill"] = self.info.name
        out["cl2"] = {"drops": list(self.drops), "home": None if self.home is None else np.round(self.home, 4).tolist(),
                      "at_home": self.at_home, "ws_z": list(self.ws_z)}
        return out


def make_perception(per, spec, setup, rules: CL2Rules):
    """Perception に、推定の版（立方体と箱は EstimatedGeometryFinder、カップ・皿のあるタスクは api/estimated_place の
    PlaceGeometryFinder）と高さの地図を被せる。"""
    from recovla.api.estimated_place import finder_class
    finder = finder_class(spec).from_spec(spec, per, setup)
    return CL2Perception.attach_cl2(per, finder, HM.HeightMap(finder.frame, cell_k=rules.map_cell_k, noise=finder.noise,
                                                              self_k=rules.map_self_k, below_k=rules.map_below_k,
                                                              edge_k=rules.map_edge_k))

