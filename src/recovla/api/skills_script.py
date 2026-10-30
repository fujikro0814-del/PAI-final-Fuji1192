"""手で書いたスクリプト方策の段階の決まり。生成プログラム（expert/script.py）の段階（phase_of）と指令（Expert.act）を、
シミュレーションの真値の代わりに、知覚の推定・測った関節角の順運動学・ハンドの幅・自分の指令だけで書き直したもの
（expert/script.py は真値と場面の座標に依るので、実行系からは import しない）。

    e = cfg["expert"]
    rules = ScriptRules.from_config(e, e["phase"])
    brain = ScriptBrain(rules, dt=0.02)
    cmd = brain.step(obs)          # Obs は 1 回の決め直しの入力。毎回、段階を決め直す（内部に段階を持たない）

数値（速さ・高さの上限と下限・時間）は configs/default.yaml の expert を読む。物と箱の大きさに関わる閾値は、知覚の口が推定した
大きさ・向き・不確かさと箱の内側の領域からの相対で書く（既知の物の寸法・箱の寸法・置き場所の表を使わない）:
  掴む高さ    指先の中心を物の推定の中心の高さへ（config の grasp_z は下限）
  真上        指先と物の水平の差 ≤ above_frac × 物の短い幅 ＋ unc_k × σ
  閉じてよい  同じく ≤ grasp_tol_frac × 物の短い幅 ＋ grasp_unc_k × σ
  持っていない 写っている物が指先から far_k × 物の半径 ＋ unc_k × σ より遠い
  宙にある    物の中心 − （机 ＋ 見かけの高さ × 0.5）≥ airborne_frac × 見かけの高さ ＋ unc_k × σ
  放す高さ    持っている物の下端が、箱の縁の高さ ＋ clear_frac × 見かけの高さ に来る手先の高さ（config の release_z は使わない）
  置き場所    箱の推定の内寸を 2×2 に分けた中心。埋まり = 箱の中の物が slot_frac × 内寸の半分の短い方 以内
物の大きさが分からない推定（extent が None）では、生成プログラムと同じ config の値（above_tol・move_tol・release_z）に戻す。
推定の揺れに合わせた値（*_frac・*_k・*_window_s）は ScriptRules の引数で変えられる。
"""
import dataclasses
import enum
import math
from typing import Optional

import numpy as np

from recovla.api import relations as REL


class Phase(enum.IntEnum):
    """expert/script.Phase と同じ符号に search を足したもの。"""
    approach = 0
    descend = 1
    close = 2
    reopen = 3
    lift = 4
    carry = 5
    release = 6
    settle = 7
    retreat = 8
    done = 9
    search = 10                            # 狙いの物が見えない（見失った）。接近の高さへ上がって待つ


@dataclasses.dataclass(frozen=True)
class ScriptRules:
    # configs の expert.phase（expert/script.PhaseParams と同じ）
    rest_speed: float
    rest_hold_s: float
    grasped_lift_min: float
    min_grip_gap: float
    above_tol: float
    grasp_z: float
    grasp_z_tol: float
    release_z: float
    release_z_tol: float
    retreat_pose: tuple
    retreat_tol: float
    # configs の expert（expert/script.Expert と同じ）
    gain: float
    xy_max: float
    z_max: float
    approach_z: float
    lift_m: float
    carry_z: float
    move_tol: float
    still: float
    close_settle_s: float
    finger_rest_speed: float
    idle_s: float
    final: Optional[dict] = None           # 掴む直前の速さの上限（configs の final_approach。無ければ None）
    # 推定からの相対の閾値の倍数
    above_frac: float = 0.25               # 真上: 物の短い幅に対する割合
    grasp_tol_frac: float = 0.1            # 閉じてよい: 物の短い幅に対する割合
    grasp_unc_k: float = 1.0               # 閉じてよい: 不確かさの倍数
    far_k: float = 2.0                     # 持っていない: 物の半径の倍数
    airborne_frac: float = 0.25            # 宙にある: 見かけの高さに対する割合
    clear_frac: float = 0.25               # 放す高さ: 縁の上の余裕の、見かけの高さに対する割合
    slot_frac: float = 0.375               # 置き場所の埋まり: 内寸の半分の短い方に対する割合
    unc_k: float = 2.0                     # 余白に掛ける不確かさの倍数
    est_speed_window_s: float = 1.0        # 狙いの物の推定の速さを測る窓 [s]（前半と後半の中央値で、推定の揺れ・跳びをならす）
    finger_moved_m: float = 0.002          # 閉じる指令の後に指が動いたとみなす開き幅の変化 [m]（ハンドの読みの性質。物の大きさではない）
    still_window_s: float = 0.1            # 手先の速さを測る窓 [s]（測った関節角の雑音をならす）

    @classmethod
    def from_config(cls, expert: dict, phase: dict, speed_scale: float = 1.0, **over) -> "ScriptRules":
        """expert は configs の expert、phase はその phase（段階の判定の値）。呼び手が分けて渡す（真値を読まない検査が
        "phase" の添字を数えるので、実行系の中で設定の phase を添字で読まない）。"""
        e, p = expert, phase
        fa = e.get("final_approach")
        final = None if not fa else {"radius": float(fa["radius_m"]), "blend": float(fa["blend_m"]),
                                     "xy": float(fa["xy"]) * speed_scale, "z_band": float(fa["z_band_m"]),
                                     "z_blend": float(fa["z_blend_m"]), "z": float(fa["z"]) * speed_scale}
        kw = dict(rest_speed=float(p["rest_speed"]), rest_hold_s=float(p["rest_hold_s"]),
                  grasped_lift_min=float(p["grasped_lift_min_m"]), min_grip_gap=float(p["min_grip_gap_m"]),
                  above_tol=float(p["above_tol_m"]), grasp_z=float(e["grasp_z"]), grasp_z_tol=float(p["grasp_z_tol_m"]),
                  release_z=float(e["release_z"]), release_z_tol=float(p["release_z_tol_m"]),
                  retreat_pose=tuple(float(v) for v in e["retreat_pose"]), retreat_tol=float(p["retreat_tol_m"]),
                  gain=float(e["gain_per_s"]), xy_max=float(e["speed_ref"]["xy"]) * speed_scale,
                  z_max=float(e["speed_ref"]["z"]) * speed_scale, approach_z=float(e["approach_height"]),
                  lift_m=float(e["lift_m"]), carry_z=float(e["carry_z"]), move_tol=float(e["move_tol_m"]),
                  still=float(e["hand_still_speed"]), close_settle_s=float(e["close_settle_s"]),
                  finger_rest_speed=float(e["finger_rest_speed"]), idle_s=float(e["idle_after_place_s"]), final=final)
        kw.update(over)
        return cls(**kw)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class Obs:
    """1 回の決め直しの入力（真値なし）。"""
    t: float
    target: object                         # api.perception.ObjState か None（推定が無い）
    others: dict                           # {名前: ObjState}（狙い以外）
    place: object                          # api.perception.PlaceState か None
    place_name: str
    closed: bool                           # 自分がハンドに出している開閉
    width: float                           # 測った開き幅（2 本の和）
    width_t: float                         # その読みの時刻
    hand: np.ndarray                       # 測った関節角の順運動学の手先（hand の原点）
    tip: np.ndarray                        # 同じく指先の中心
    x_cmd: np.ndarray                      # 自分の手先の参照
    table_z: float                         # 机の高さ（較正の値）
    est_t: Optional[float] = None          # 推定の時刻


@dataclasses.dataclass
class Command:
    vel: np.ndarray                        # (3,) 手先の参照の速さ [m/s]
    closed: bool                           # 出したいハンドの開閉
    phase: Phase
    finished: bool = False


def _known(o) -> bool:
    return o is not None and o.status != "lost"


def _in_hand_mark(o) -> bool:
    return o is not None and o.status == "occluded" and o.near_hand is True


def _tip_drop(obs: Obs) -> float:
    """手先（hand の原点）から指先の中心までの高さ（手先の参照の高さ ＝ 指先の高さ ＋ この値）。"""
    return float(obs.hand[2] - obs.tip[2])


def _sig(o) -> float:
    return REL.sigma(o)


class Clocks:
    """推定・読みの列から数え直せる時間と速さ（expert/script.Clocks の代わり）。dt ごとに update する。"""

    def __init__(self):
        self.target_rest_s = 0.0
        self.finger_rest_s = 0.0
        self.done_s = 0.0
        self.target_speed = 0.0
        self.hand_speed = 0.0
        self._est = []                     # [(推定の時刻, 位置)]
        self._w = None                     # (読みの時刻, 幅)
        self._fv = 0.0
        self._hand = []                    # [(時刻, 手先)]
        self.z_close = None                # 閉じたときの手先の高さ（開いたら None）
        self.obj_h = None                  # 狙いの物の見かけの高さ（最後に分かった値。運ぶ間に推定が隠れても使う）
        self.w_close = None                # 閉じる指令を出したときの開き幅の読み
        self.moved = False                 # 閉じる指令の後に指が動いたか

    def update(self, obs: Obs, dt: float, r: ScriptRules) -> None:
        if not obs.closed:
            self.z_close, self.w_close, self.moved = None, None, False
        else:
            if self.z_close is None:
                self.z_close = float(obs.hand[2])
            if self.w_close is None:
                self.w_close = float(obs.width)
        if obs.closed and not self.moved and abs(float(obs.width) - self.w_close) >= r.finger_moved_m:
            self.moved = True
        o = obs.target
        h = REL.height(o) if o is not None else None
        if h is not None:
            self.obj_h = h
        if not _known(o) or _in_hand_mark(o):
            self._est, self.target_speed = [], 0.0
        elif obs.est_t is not None and (not self._est or obs.est_t != self._est[-1][0]):
            self._est.append((float(obs.est_t), np.asarray(o.pos, float).copy()))
            while len(self._est) > 2 and self._est[-1][0] - self._est[1][0] >= r.est_speed_window_s - 1e-9:
                del self._est[0]
            self.target_speed = _robust_speed(self._est)
        self.target_rest_s = self.target_rest_s + dt if (_known(o) and self.target_speed < r.rest_speed) else 0.0
        if self._w is None or obs.width_t != self._w[0]:
            if self._w is not None and obs.width_t > self._w[0]:
                self._fv = abs(float(obs.width) - self._w[1]) / (float(obs.width_t) - self._w[0])
            self._w = (float(obs.width_t), float(obs.width))
        # 指が止まった: 閉じた後は、指が動いてから止まったときだけ数える（10 Hz の幅の読みは指令の直後まだ開いたままで
        # 速さ 0 に見え、閉じる前に持ち上げて掴み損ねるため）
        settled = self._fv < r.finger_rest_speed and (not obs.closed or self.moved)
        self.finger_rest_s = self.finger_rest_s + dt if settled else 0.0
        self._hand.append((float(obs.t), np.asarray(obs.hand, float).copy()))
        while len(self._hand) > 2 and self._hand[-1][0] - self._hand[1][0] >= r.still_window_s - 1e-9:
            del self._hand[0]
        (t0, h0), (t1, h1) = self._hand[0], self._hand[-1]
        self.hand_speed = float(np.linalg.norm(h1 - h0) / (t1 - t0)) if t1 > t0 else 0.0


def _robust_speed(est) -> float:
    """窓の中の推定 [(時刻, 位置)] の速さ。前半と後半の位置の中央値の差を、時刻の中央値の差で割る（点が 3 つ以下なら両端）。
    1 フレームだけ跳んだ推定（隠れ・見え方の変化で 1 cm ほど）で、止まった物を動いていると判定しないため。"""
    if len(est) < 2:
        return 0.0
    if len(est) <= 3:
        (t0, p0), (t1, p1) = est[0], est[-1]
        return float(np.linalg.norm(p1 - p0) / (t1 - t0)) if t1 > t0 else 0.0
    k = len(est) // 2
    a, b = est[:k], est[-k:]
    ta, tb = np.median([x[0] for x in a]), np.median([x[0] for x in b])
    if tb <= ta:
        return 0.0
    pa, pb = np.median([x[1] for x in a], axis=0), np.median([x[1] for x in b], axis=0)
    return float(np.linalg.norm(pb - pa) / (tb - ta))


# ------------------------------------------------------------------------------ 段階（純関数）
def holding(obs: Obs, r: ScriptRules) -> bool:
    """掴んでいる = 閉じていて、開き幅が min_grip_gap 以上（何かを挟んでいる）。写っている狙いの物が指先から
    far_k × 物の半径 ＋ unc_k × σ より遠ければ持っていない。"""
    if not obs.closed or float(obs.width) < r.min_grip_gap:
        return False
    o = obs.target
    rad = REL.radius(o) if o is not None else None
    if o is not None and o.status == "seen" and o.near_hand is not True and rad is not None:
        if float(np.linalg.norm(np.asarray(o.pos, float) - obs.tip)) > r.far_k * rad + r.unc_k * _sig(o):
            return False
    return True


def lifted(obs: Obs, r: ScriptRules, z_close: float = None) -> bool:
    """持ち上がっている = 手先（順運動学）が閉じたときの高さ z_close（分からなければ grasp_z）より grasped_lift_min 以上高い
    （測った手先は参照の把持の高さより 1 cm ほど上で止まることがあるので、閉じた所から測る）。"""
    z0 = r.grasp_z if z_close is None else float(z_close)
    return bool(float(obs.hand[2]) - z0 >= r.grasped_lift_min)


def airborne_est(obs: Obs, r: ScriptRules) -> bool:
    """推定の中心が、机に置いたときの中心の高さ（机 ＋ 見かけの高さ × 0.5）より airborne_frac × 見かけの高さ ＋ 余白 以上高い。"""
    o = obs.target
    h = REL.height(o) if o is not None else None
    if not _known(o) or h is None:
        return False
    return bool(float(o.pos[2]) - (obs.table_z + 0.5 * h) >= r.airborne_frac * h + r.unc_k * _sig(o))


def grasp_hand_z(obs: Obs, r: ScriptRules) -> float:
    """掴む手先の参照の高さ: 指先の中心を物の推定の中心の高さへ（config の grasp_z を下限）。推定が無ければ grasp_z。"""
    o = obs.target
    if not _known(o):
        return r.grasp_z
    return max(r.grasp_z, float(o.pos[2]) + _tip_drop(obs))


def release_hand_z(obs: Obs, r: ScriptRules, obj_h: float = None) -> float:
    """放す手先の参照の高さ: 持っている物（中心 ≒ 指先）の下端が、箱の縁のいちばん高い所 ＋ clear_frac × 見かけの高さ に来る
    高さ。箱の領域か物の高さが分からなければ config の release_z。"""
    pl = obs.place
    reg = None if pl is None else pl.region
    if reg is None or obj_h is None:
        return r.release_z
    rim_top = float(reg.center[2]) + math.sin(reg.tilt) * float(np.max(reg.outer))
    return rim_top + (0.5 + r.clear_frac) * obj_h + _tip_drop(obs)


def above_tol(obs: Obs, r: ScriptRules) -> float:
    w = REL.width_min(obs.target) if obs.target is not None else None
    return r.above_tol if w is None else r.above_frac * w + r.unc_k * _sig(obs.target)


def grasp_tol(obs: Obs, r: ScriptRules) -> float:
    w = REL.width_min(obs.target) if obs.target is not None else None
    return r.move_tol if w is None else r.grasp_tol_frac * w + r.grasp_unc_k * _sig(obs.target)


def in_place(obs: Obs) -> bool:
    o = obs.target
    return bool(_known(o) and o.in_place.get(obs.place_name, False))


def phase_of(obs: Obs, clocks: Clocks, r: ScriptRules) -> Phase:
    """純関数。上から順に最初に当てはまる段階（expert/script.phase_of と同じ並びで、reopen の後に search）。"""
    closed = bool(obs.closed)
    o = obs.target
    known = _known(o)
    inb = in_place(obs)
    rested = clocks.target_rest_s >= r.rest_hold_s - 1e-9
    hold = holding(obs, r)
    up = hold and lifted(obs, r, clocks.z_close)
    x = np.asarray(obs.x_cmd, float)
    if not closed and inb and rested:
        at_retreat = np.linalg.norm(x - np.asarray(r.retreat_pose)) <= r.retreat_tol
        return Phase.done if at_retreat else Phase.retreat
    if not hold and known:
        moving = clocks.target_speed >= r.rest_speed
        airborne = (not closed) and airborne_est(obs, r) and not inb
        if moving or (inb and not rested) or airborne:
            return Phase.settle
    if hold and up:
        pl = obs.place
        at_release = (pl is not None and pl.contains_xy(x[:2])
                      and x[2] <= release_hand_z(obs, r, clocks.obj_h) + r.release_z_tol)
        return Phase.release if at_release else Phase.carry
    if hold:
        return Phase.lift
    if closed:
        return Phase.reopen
    if not known:
        return Phase.search
    above = math.hypot(*(np.asarray(obs.tip, float)[:2] - np.asarray(o.pos, float)[:2])) <= above_tol(obs, r)
    if above and x[2] <= grasp_hand_z(obs, r) + r.grasp_z_tol:
        return Phase.close
    if above:
        return Phase.descend
    return Phase.approach


N_SLOTS = 4                                # 内寸を 2×2 に分けた置き場所


def slot_xy(place, slot: int) -> np.ndarray:
    """箱の推定の内側の領域を 2×2 に分けた置き場所の中心（水平）。"""
    return np.asarray(place.region.slot(int(slot)), float)[:2]


def choose_slot(obs: Obs, r: ScriptRules, order) -> int:
    """空いている最初の置き場所（箱の中にある他の物の推定の水平位置が、slot_frac × 内寸の半分の短い方 以内なら埋まっている）。"""
    rad = r.slot_frac * float(np.min(obs.place.region.inner))
    others = [np.asarray(p.pos, float)[:2] for p in obs.others.values() if _known(p) and p.in_place.get(obs.place_name, False)]
    for s in order:
        c = slot_xy(obs.place, s)
        if not any(math.hypot(*(p - c)) < rad for p in others):
            return int(s)
    return int(order[0])


class ScriptBrain:
    """段階を毎回決め直し、その段階の目標へ向かう指令を 1 回分出す（expert/script.Expert.act と同じ指令の決まり）。"""

    def __init__(self, rules: ScriptRules, dt: float, grasp_offset=(0.0, 0.0), slot_order=None, view_from=None):
        """view_from: 固定のカメラの水平の位置（較正の値）。slot_order を渡さないとき、置き場所をこのカメラから遠い順に使う
        （手前の壁に隠れず、置いた物を完了の判定でカメラから確かめやすい）。"""
        self.r = rules
        self.dt = float(dt)
        self.grasp_offset = np.asarray(grasp_offset, float)
        self.slot_order = None if slot_order is None else tuple(int(s) for s in slot_order)
        self.view_from = None if view_from is None else np.asarray(view_from, float)[:2]
        self.reset()

    def _order(self, place) -> tuple:
        if self.slot_order is not None:
            return self.slot_order
        if self.view_from is None:
            return tuple(range(N_SLOTS))
        return tuple(sorted(range(N_SLOTS), key=lambda s: -float(np.hypot(*(slot_xy(place, s) - self.view_from)))))

    def reset(self) -> None:
        self.clocks = Clocks()

    @staticmethod
    def _ramp(dist, near, blend, low, high) -> float:
        if dist <= near:
            return low
        if dist >= near + blend:
            return high
        return low + (high - low) * (dist - near) / blend

    def _vel_to(self, x, goal, final_aim=None) -> np.ndarray:
        r = self.r
        x = np.asarray(x, float)
        v = r.gain * (np.asarray(goal, float) - x)
        xy_max, z_max = r.xy_max, r.z_max
        if r.final is not None and final_aim is not None:
            f = r.final
            xy_max = min(xy_max, self._ramp(float(np.hypot(*(np.asarray(final_aim) - x[:2]))), f["radius"], f["blend"],
                                            f["xy"], r.xy_max))
            z_max = min(z_max, self._ramp(float(x[2] - r.grasp_z), f["z_band"], f["z_blend"], f["z"], r.z_max))
        n = np.linalg.norm(v[:2])
        if n > xy_max:
            v[:2] *= xy_max / n
        v[2] = float(np.clip(v[2], -z_max, z_max))
        return v

    def _rise_first(self, x, goal, safe_z: float) -> np.ndarray:
        far = np.hypot(*(np.asarray(goal[:2]) - x[:2])) > self.r.above_tol
        if far and x[2] < safe_z - self.r.grasp_z_tol:
            return np.array([x[0], x[1], safe_z])
        return np.asarray(goal, float)

    def _still(self) -> bool:
        return self.clocks.hand_speed < self.r.still

    def step(self, obs: Obs) -> Command:
        r = self.r
        self.clocks.update(obs, self.dt, r)
        ph = phase_of(obs, self.clocks, r)
        x = np.asarray(obs.x_cmd, float)
        zero = np.zeros(3)
        closed = bool(obs.closed)
        self.clocks.done_s = self.clocks.done_s + self.dt if ph == Phase.done else 0.0
        if ph == Phase.done:
            return Command(zero, closed, ph, finished=self.clocks.done_s >= r.idle_s - 1e-9)
        if ph == Phase.retreat:
            rp = np.asarray(r.retreat_pose, float)
            return Command(self._vel_to(x, self._rise_first(x, rp, min(r.carry_z, rp[2]))), closed, ph)
        if ph == Phase.settle:
            return Command(zero, closed, ph)
        if ph == Phase.search:
            return Command(self._vel_to(x, np.array([x[0], x[1], r.approach_z])), closed, ph)
        if ph in (Phase.carry, Phase.release):
            if obs.place is None or not obs.place.ok or obs.place.region is None:
                return Command(zero, closed, ph)                  # 置き場所が分からない: その場で保つ
            # 持っている物の中心を指先の中心（順運動学）とみなし、置き場所の真上に合わせる
            s = slot_xy(obs.place, choose_slot(obs, r, self._order(obs.place)))
            off = s - np.asarray(obs.tip, float)[:2]
            aim = x[:2] + off
            placed = np.hypot(*off) <= r.move_tol
            rz = release_hand_z(obs, r, self.clocks.obj_h)
            carry_z = max(r.carry_z, rz)
            if ph == Phase.release:
                if placed and abs(x[2] - rz) <= r.move_tol and self._still():
                    return Command(zero, False, ph)                # 開く
                return Command(self._vel_to(x, np.array([aim[0], aim[1], rz])), closed, ph)
            if np.hypot(*off) > r.above_tol:
                goal = self._rise_first(x, np.array([aim[0], aim[1], carry_z]), carry_z)
            else:
                goal = np.array([aim[0], aim[1], rz])
            return Command(self._vel_to(x, goal), closed, ph)
        if ph == Phase.lift:
            if self.clocks.finger_rest_s < r.close_settle_s - 1e-9:
                return Command(zero, closed, ph)
            z0 = r.grasp_z if self.clocks.z_close is None else self.clocks.z_close
            return Command(self._vel_to(x, np.array([x[0], x[1], z0 + r.lift_m])), closed, ph)
        if ph == Phase.reopen:
            return Command(np.array([0.0, 0.0, r.z_max]), False, ph)
        # 指先の中心（順運動学）を把持点（狙いの物の推定の中心＋ずれ）に合わせる
        g = np.asarray(obs.target.pos, float)[:2] + self.grasp_offset
        off = g - np.asarray(obs.tip, float)[:2]
        aim = x[:2] + off
        gz = grasp_hand_z(obs, r)
        if ph == Phase.close:
            if np.hypot(*off) <= grasp_tol(obs, r) and abs(x[2] - gz) <= r.move_tol and self._still():
                return Command(zero, True, ph)                     # 閉じる
            return Command(self._vel_to(x, np.array([aim[0], aim[1], gz]), aim), closed, ph)
        if ph == Phase.descend:
            return Command(self._vel_to(x, np.array([aim[0], aim[1], gz]), aim), closed, ph)
        goal = self._rise_first(x, np.array([aim[0], aim[1], r.approach_z]), r.approach_z)
        return Command(self._vel_to(x, goal, aim), closed, ph)
