"""生成プログラム: シミュレーションの真の状態から段階を判定して続きを行う（学習しないスクリプト方策）。

**内部に段階を持たない。** 入力の読み取りごと（20 ms）に、真の状態から今の段階を `phase_of` で判定し、その段階の
目標へ向かう指令を 1 回分だけ出す。どの状態から始めても続きを行えるので、失敗の後や方策からの引き継ぎの後も
そのまま立て直せる。`phase_of` は純関数で、評価の段階の判定と共用する。

持つ状態は 2 種類だけ:
  - パラメータ（ScriptParams）: 把持点のずれ・速さのばらつき・箱の置き場所の好みの順。乱数列から
    エピソードの開始時に一度だけ引く（引き継いでも同じ値を使える）
  - 時計（Clocks）: 「目標が静止している時間」「指が止まっている時間」「done に入ってからの時間」。真の状態の列から
    数え直せる量で、引き継いだ直後は 0 から数える（長めに待つ側に倒れるだけ）

段階（上から順に最初に当てはまるもの。数値は configs/default.yaml の expert）:

| 段階 | 条件 | 指令 |
|---|---|---|
| done | 開いていて、目標が箱の中で静止し、x_des が待機位置にある | 静止（idle_after_place_s 数えて終わる） |
| retreat | 開いていて、目標が箱の中で静止している | 待機位置へ（低ければ先に真上へ） |
| settle | 掴んでいない（下の lift・carry に当たらない）うえで、目標が動いている、目標が箱の中でまだ静止していない、または開いていて目標が宙にある（放した直後など） | 静止して待つ |
| release | 掴んでいて持ち上がっており、x_des が箱の内寸の上の放す高さにある | 立方体の中心が置き場所の真上（1 mm 以内）に来て手が止まったら開く |
| carry | 掴んでいて持ち上がっている | 運ぶ高さ → 置き場所の上 → 放す高さ（立方体の中心を置き場所に合わせる） |
| lift | 閉じていて、目標が指先の中心の近くにあり、指の開きが立方体を挟んだ幅 | 指が止まってから真上へ |
| reopen | 閉じているが、上に当たらない（掴み損ね・落とした後） | 開いて真上へ |
| close | 開いていて、指先の中心が目標の真上にあり、x_des が把持の高さにある | 指先の中心が把持点（1 mm 以内）に着いて手が止まったら閉じる |
| descend | 開いていて、指先の中心が目標の真上にある | 把持の高さへ下ろす |
| approach | 上のどれでもない | 接近の高さで目標の真上へ（低ければ先に真上へ） |

「掴んでいる」= 閉じていて、目標の中心が指先の中心から drop_dist 以内で、指の開き（2 本の和）が min_grip_gap 以上。
「持ち上がっている」= 目標の中心が机の上で静止した高さより grasped_lift_min 以上高い。
「目標の真上」= 指先の中心（真の状態）の水平位置が、目標の中心から above_tol 以内。

狙い方: 指先の中心を把持点に、運ぶときは立方体の中心を置き場所に合わせ、x_des の目標をその差だけずらす。
x_des で狙うと、手先と x_des の定常的なずれ（最大約 5.6 mm）が残り、置き場所からのずれは中央値 3.5 mm・最大 11.2 mm。
放す位置だけ合わせると、指の中心から外れて握った立方体が転がって最大 16.5 mm。両方を合わせると中央値 1.9 mm・最大 11.7 mm。

置き場所の口: タスクごとの値（机の上の静止の高さ・目標の場所にあるか・x_des が置き場所の上か・置き場所の水平の位置・
掴む高さ・放す高さ）は place から引く（既定 BOX ＝ 3 色の場面の箱。カップと皿は expert/cup_plate.CupPlatePlace）。
段階の並び・指令の式・速さ・時計はタスクによらない。

settle の「開いていて目標が宙にある」: 放した直後は指が開き切るまで立方体が指の間に残り、descend の条件に当たって
手が立方体の上に下りてしまうため。lift を reopen より先に判定し、閉じた直後（指がまだ動いている間）を lift に含める。
"""
import dataclasses
import enum

import numpy as np

from recovla.common import config
from recovla.sim import frames

_CFG = config.load()


class Phase(enum.IntEnum):
    """段階の符号（記録の phase の列と共通）。"""
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


@dataclasses.dataclass(frozen=True)
class Truth:
    """1 時点の真の状態（データ生成と評価でだけ読む。実行時の方策には渡さない）。色の並びは COLORS。"""
    t: float
    target: int                  # 目標の色の添字
    cube_pos: np.ndarray         # (3, 3)
    cube_quat: np.ndarray        # (3, 4)
    cube_linvel: np.ndarray      # (3, 3) 世界座標
    fingers: np.ndarray          # (2,) finger_joint1・2 [m]
    finger_vel: np.ndarray       # (2,)
    gripper_closed: bool
    hand_pos: np.ndarray         # (3,) hand の原点
    hand_vel: np.ndarray         # (3,)
    fingertip: np.ndarray        # (3,) 指先の中心
    x_cmd: np.ndarray            # (3,) 手先参照位置（積分器の値。制御器の desired_pos と同じ）
    box: np.ndarray              # (3,) 箱の中心

    @property
    def target_pos(self) -> np.ndarray:
        return self.cube_pos[self.target]

    @property
    def target_speed(self) -> float:
        return float(np.linalg.norm(self.cube_linvel[self.target]))


@dataclasses.dataclass(frozen=True)
class PhaseParams:
    rest_speed: float
    rest_hold_s: float
    grasped_lift_min: float
    drop_dist: float
    min_grip_gap: float
    above_tol: float
    grasp_z: float
    grasp_z_tol: float
    release_z: float
    release_z_tol: float
    retreat_pose: np.ndarray
    retreat_tol: float

    @classmethod
    def from_config(cls, cfg: dict = None) -> "PhaseParams":
        e = (cfg or _CFG)["expert"]
        p = e["phase"]
        return cls(float(p["rest_speed"]), float(p["rest_hold_s"]), float(p["grasped_lift_min_m"]),
                   float(p["drop_dist_m"]), float(p["min_grip_gap_m"]), float(p["above_tol_m"]),
                   float(e["grasp_z"]), float(p["grasp_z_tol_m"]), float(e["release_z"]),
                   float(p["release_z_tol_m"]), np.array(e["retreat_pose"], dtype=float),
                   float(p["retreat_tol_m"]))


class BoxPlace:
    """3 色の場面の置き場所（箱）の決まり。frames の関数を呼ぶ。"""
    name = "box"

    def phase_params(self, pp: "PhaseParams") -> "PhaseParams":
        return pp

    def rest_z(self, truth: Truth) -> float:
        return frames.CUBE_REST_Z

    def in_place(self, truth: Truth) -> bool:
        return frames.in_box(truth.target_pos, truth.box)

    def over_place(self, x_cmd, truth: Truth) -> bool:
        return frames.over_box_interior(x_cmd, truth.box)

    def slot_xy(self, expert: "Expert", truth: Truth) -> np.ndarray:
        return frames.slot_xy(truth.box, expert.slot(truth))

    def success(self, truth: Truth) -> bool:
        return frames.in_box(truth.target_pos, truth.box)


BOX = BoxPlace()


def holding(truth: Truth, pp: PhaseParams) -> bool:
    if not truth.gripper_closed:
        return False
    near = np.linalg.norm(truth.target_pos - truth.fingertip) <= pp.drop_dist
    return bool(near and float(np.sum(truth.fingers)) >= pp.min_grip_gap)


def lifted(truth: Truth, pp: PhaseParams, place=BOX) -> bool:
    return bool(truth.target_pos[2] - place.rest_z(truth) >= pp.grasped_lift_min)


def phase_of(truth: Truth, target_rest_s: float, pp: PhaseParams, place=BOX) -> Phase:
    """純関数。target_rest_s は目標の速さが rest_speed 未満の状態が続いている時間 [s]（呼ぶ側が数える）。
    place は置き場所の決まり（既定 BOX。pp は place.phase_params を通したものを渡す）。"""
    closed = bool(truth.gripper_closed)
    in_box = place.in_place(truth)
    rested = target_rest_s >= pp.rest_hold_s - 1e-9
    hold = holding(truth, pp)
    up = lifted(truth, pp, place)
    if not closed and in_box and rested:
        at_retreat = np.linalg.norm(truth.x_cmd - pp.retreat_pose) <= pp.retreat_tol
        return Phase.done if at_retreat else Phase.retreat
    if not hold:
        moving = truth.target_speed >= pp.rest_speed
        airborne = (not closed) and up and not in_box
        if moving or (in_box and not rested) or airborne:
            return Phase.settle
    if hold and up:
        at_release = (place.over_place(truth.x_cmd, truth)
                      and truth.x_cmd[2] <= pp.release_z + pp.release_z_tol)
        return Phase.release if at_release else Phase.carry
    if hold:
        return Phase.lift
    if closed:
        return Phase.reopen
    above = np.hypot(*(truth.fingertip[:2] - truth.target_pos[:2])) <= pp.above_tol
    if above and truth.x_cmd[2] <= pp.grasp_z + pp.grasp_z_tol:
        return Phase.close
    if above:
        return Phase.descend
    return Phase.approach


# ------------------------------------------------------------------------------------ 生成プログラム

@dataclasses.dataclass(frozen=True)
class ScriptParams:
    grasp_offset: tuple          # (dx, dy) [m] 把持点のずれ（±grasp_jitter_m の一様）
    speed_scale: float           # speed_ref に掛ける
    slot_order: tuple            # 箱の置き場所の好みの順（空いている最初のものを使う）

    def to_json(self) -> dict:
        return {"grasp_offset": [float(v) for v in self.grasp_offset], "speed_scale": float(self.speed_scale),
                "slot_order": [int(s) for s in self.slot_order]}


def sample_params(rng: np.random.Generator, cfg: dict = None) -> ScriptParams:
    e = (cfg or _CFG)["expert"]
    j = float(e["grasp_jitter_m"])
    return ScriptParams(grasp_offset=tuple(float(v) for v in rng.uniform(-j, j, 2)),
                        speed_scale=float(rng.uniform(*e["speed_scale"])),
                        slot_order=tuple(int(s) for s in rng.permutation(len(frames.BOX_SLOTS))))


@dataclasses.dataclass
class Command:
    vel: np.ndarray              # (3,) [m/s] 手先参照速度（積分器に渡す）
    press: bool                  # グリッパのボタンを押す（制御器は押し始めで開閉を切り替える）
    phase: Phase
    finished: bool = False


class Clocks:
    """真の状態の列から数え直せる時間。dt ごとに update する。"""

    def __init__(self):
        self.target_rest_s = 0.0
        self.finger_rest_s = 0.0
        self.done_s = 0.0

    def update(self, truth: Truth, dt: float, rest_speed: float, finger_rest_speed: float) -> None:
        self.target_rest_s = self.target_rest_s + dt if truth.target_speed < rest_speed else 0.0
        fv = float(np.sum(np.abs(truth.finger_vel)))
        self.finger_rest_s = self.finger_rest_s + dt if fv < finger_rest_speed else 0.0


class Expert:
    """act(truth) を入力の読み取りごと（dt 秒ごと）に呼ぶ。引き継ぎは新しい Expert を作るだけ。"""

    def __init__(self, params: ScriptParams, dt: float, cfg: dict = None, place=None):
        cfg = cfg or _CFG
        e = cfg["expert"]
        self.params = params
        self.dt = float(dt)
        self.place = BOX if place is None else place
        self.pp = self.place.phase_params(PhaseParams.from_config(cfg))
        self.gain = float(e["gain_per_s"])
        self.xy_max = float(e["speed_ref"]["xy"]) * params.speed_scale
        self.z_max = float(e["speed_ref"]["z"]) * params.speed_scale
        self.approach_z = float(e["approach_height"])
        self.lift_m = float(e["lift_m"])
        self.carry_z = float(e["carry_z"])
        self.move_tol = float(e["move_tol_m"])
        self.still = float(e["hand_still_speed"])
        self.close_settle_s = float(e["close_settle_s"])
        self.finger_rest_speed = float(e["finger_rest_speed"])
        self.idle_s = float(e["idle_after_place_s"])
        self.slot_radius = float(cfg["scene"]["slot_occupied_radius"])
        fa = e.get("final_approach")              # 把持点の近くでの減速（configs/expert_v3.yaml）。なければ減速しない
        self.final = None if not fa else {
            "radius": float(fa["radius_m"]), "blend": float(fa["blend_m"]), "xy": float(fa["xy"]) * params.speed_scale,
            "z_band": float(fa["z_band_m"]), "z_blend": float(fa["z_blend_m"]), "z": float(fa["z"]) * params.speed_scale}
        self.clocks = Clocks()

    # -- 補助 --
    @staticmethod
    def _ramp(dist: float, near: float, blend: float, low: float, high: float) -> float:
        """dist ≤ near で low、near + blend 以上で high、その間は直線（上限を距離について連続にする）。"""
        if dist <= near:
            return low
        if dist >= near + blend:
            return high
        return low + (high - low) * (dist - near) / blend

    def _vel_to(self, x_cmd, goal, final_aim=None) -> np.ndarray:
        """goal へ向かう速度（比例＋上限）。final_aim（把持点に向かう段の水平の狙い）を渡したときだけ、把持点の近くの減速を掛ける。"""
        x_cmd = np.asarray(x_cmd, float)
        v = self.gain * (np.asarray(goal, float) - x_cmd)
        xy_max, z_max = self.xy_max, self.z_max
        if self.final is not None and final_aim is not None:
            f = self.final
            xy_max = min(xy_max, self._ramp(float(np.hypot(*(np.asarray(final_aim) - x_cmd[:2]))),
                                            f["radius"], f["blend"], f["xy"], self.xy_max))
            z_max = min(z_max, self._ramp(float(x_cmd[2] - self.pp.grasp_z), f["z_band"], f["z_blend"], f["z"], self.z_max))
        n = np.linalg.norm(v[:2])
        if n > xy_max:
            v[:2] *= xy_max / n
        v[2] = float(np.clip(v[2], -z_max, z_max))
        return v

    def _rise_first(self, x_cmd, goal, safe_z: float) -> np.ndarray:
        """低い位置から水平に動くときは、先に真上へ safe_z まで上がる。"""
        far = np.hypot(*(np.asarray(goal[:2]) - x_cmd[:2])) > self.pp.above_tol
        if far and x_cmd[2] < safe_z - self.pp.grasp_z_tol:
            return np.array([x_cmd[0], x_cmd[1], safe_z])
        return np.asarray(goal, float)

    def grasp_point(self, truth: Truth) -> np.ndarray:
        return truth.target_pos[:2] + np.asarray(self.params.grasp_offset)

    def slot(self, truth: Truth) -> int:
        others = [truth.cube_pos[i] for i in range(len(truth.cube_pos))
                  if i != truth.target and frames.in_box(truth.cube_pos[i], truth.box)]
        for s in self.params.slot_order:
            if not frames.slot_occupied(truth.box, s, others, self.slot_radius):
                return int(s)
        return int(self.params.slot_order[0])

    def _at(self, x_cmd, goal) -> bool:
        return bool(np.linalg.norm(np.asarray(goal, float) - x_cmd) <= self.move_tol)

    def _hand_still(self, truth: Truth) -> bool:
        return bool(np.linalg.norm(truth.hand_vel) < self.still)

    # -- 入力の読み取り 1 回分 --
    def act(self, truth: Truth) -> Command:
        self.clocks.update(truth, self.dt, self.pp.rest_speed, self.finger_rest_speed)
        ph = phase_of(truth, self.clocks.target_rest_s, self.pp, self.place)
        x = np.asarray(truth.x_cmd, float)
        zero = np.zeros(3)
        self.clocks.done_s = self.clocks.done_s + self.dt if ph == Phase.done else 0.0
        if ph == Phase.done:
            return Command(zero, False, ph, finished=self.clocks.done_s >= self.idle_s - 1e-9)
        if ph == Phase.retreat:
            goal = self._rise_first(x, self.pp.retreat_pose, min(self.carry_z, self.pp.retreat_pose[2]))
            return Command(self._vel_to(x, goal), False, ph)
        if ph == Phase.settle:
            return Command(zero, False, ph)
        if ph in (Phase.carry, Phase.release):
            # 立方体の中心を置き場所の真上に合わせる: x_des の目標を、立方体と置き場所の水平の差だけずらす。
            # 手先と x_des の定常的なずれ（最大約 5 mm）と、指の閉じる向きと直交する把持のずれを打ち消す
            s = self.place.slot_xy(self, truth)
            off = s - truth.target_pos[:2]
            aim = x[:2] + off
            placed = np.hypot(*off) <= self.move_tol
            if ph == Phase.release:
                goal = np.array([aim[0], aim[1], self.pp.release_z])
                if placed and abs(x[2] - self.pp.release_z) <= self.move_tol and self._hand_still(truth):
                    return Command(zero, True, ph)
                return Command(self._vel_to(x, goal), False, ph)
            if np.hypot(*off) > self.pp.above_tol:       # まだ置き場所の上にいない: 運ぶ高さで向かう
                goal = self._rise_first(x, np.array([aim[0], aim[1], self.carry_z]), self.carry_z)
            else:                                         # 上にいる: 水平を合わせながら放す高さへ下ろす
                goal = np.array([aim[0], aim[1], self.pp.release_z])
            return Command(self._vel_to(x, goal), False, ph)
        if ph == Phase.lift:
            if self.clocks.finger_rest_s < self.close_settle_s - 1e-9:
                return Command(zero, False, ph)
            goal = np.array([x[0], x[1], self.pp.grasp_z + self.lift_m])
            return Command(self._vel_to(x, goal), False, ph)
        if ph == Phase.reopen:
            return Command(np.array([0.0, 0.0, self.z_max]), True, ph)
        # 指先の中心を把持点に合わせる:x_des の水平の目標を、把持点と指先の中心の差だけずらす
        # （手先と x_des の定常的なずれを打ち消す。ずれが残ると立方体を指の中心から外れて握り、放したときに転がる）
        g = self.grasp_point(truth)
        off = g - truth.fingertip[:2]
        aim = x[:2] + off
        if ph == Phase.close:
            goal = np.array([aim[0], aim[1], self.pp.grasp_z])
            if (np.hypot(*off) <= self.move_tol and abs(x[2] - self.pp.grasp_z) <= self.move_tol
                    and self._hand_still(truth)):
                return Command(zero, True, ph)
            return Command(self._vel_to(x, goal, aim), False, ph)
        if ph == Phase.descend:
            return Command(self._vel_to(x, np.array([aim[0], aim[1], self.pp.grasp_z]), aim), False, ph)
        goal = self._rise_first(x, np.array([aim[0], aim[1], self.approach_z]), self.approach_z)
        return Command(self._vel_to(x, goal, aim), False, ph)
