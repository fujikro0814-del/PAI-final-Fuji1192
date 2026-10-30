"""Franka Hand のモデル（シミュレーションの側）。

実機のハンドと同じ口だけを持つ:
    hand.move(width, speed)                                   # 目標の開き幅へ、開き幅の速さ speed [m/s] 以下で動かす
    hand.grasp(width, speed, force, eps_inner, eps_outer)     # speed で閉じ、物に当たったら指 1 本あたり force [N] で握る
    hand.step                                               # 物理ステップごと（mj_step の前）に呼ぶ
    hand.width, hand.is_grasped                           # ハンドが報告する値（センサのモデルが間引いて渡す）

作動のモデル:
- 指は 2 本で 1 自由度（等式の拘束で連動）。開き幅 w = finger_joint1 + finger_joint2（0〜0.08 m）
- グリッパの作動器を、腱 split（長さ = w/2）の**速度のサーボ**（力 = kv·(指令の速さ − 腱の速さ)、力の上限つき）に替える。
  速度の項は implicitfast の積分で陰に扱われるので、kv を大きくしても安定する。指の速さは指令の速さで、力は上限で抑えられる
  - move: 指令の速さ = clip(k_pos·(目標 − w), ±speed)。力の上限は指 1 本 move_force
  - grasp: 指令の速さ = −min(speed, k_pos·w)（閉じ続ける。開き幅 0 の手前 speed/k_pos＝4 mm で減速し、閉じ切ったら 0）。
    力の上限を move_force から force へ ramp_s で上げる（物に当たれば force で握る。立方体の幅では指令は −speed のまま）。
    閉じ切りで減速しないと、空を掴んだとき左右の指の腹を force で押し合わせて指が震える（実機のハンドは閉じ切って止まる）
    把持の判定（is_grasped）は、開き幅が [width − eps_inner, width + eps_outer] に入り、指がほぼ止まっていること
    （libfranka の grasp の判定と同じ窓）
- 上限: 開き幅の速さ 0.1 m/s（指 1 本 50 mm/s）、指 1 本の力 70 N（Panda の公称）。上限を超える指令は受け付けない
"""
import dataclasses

import numpy as np

MAX_WIDTH = 0.08
MAX_WIDTH_SPEED = 0.10          # 開き幅の速さ [m/s]（指 1 本 0.05 m/s）
MAX_FINGER_FORCE = 70.0         # 指 1 本の連続の把持力 [N]


class HandCommandError(ValueError):
    """上限を超える指令。実機のハンドも拒む。"""


@dataclasses.dataclass
class HandParams:
    kv: float = 10000.0          # 腱の速度のサーボの利得 [N/(m/s)]。止まったときの力 kv·speed/2 が力の上限（腱 140 N）を超える大きさ
    k_pos: float = 20.0          # move の位置の利得 [1/s]（目標の手前で減速する）
    move_force: float = 10.0     # move の指 1 本あたりの力の上限 [N]
    ramp_s: float = 0.2          # 握る力を move_force から force へ上げる時間 [s]
    still_speed: float = 0.005   # 把持の判定で「指が止まっている」とみなす開き幅の速さ [m/s]

    @classmethod
    def from_config(cls, cfg: dict) -> "HandParams":
        return cls(**{k: float(v) for k, v in (cfg or {}).items() if k in cls.__dataclass_fields__})


class FrankaHand:
    def __init__(self, model, data, act_id: int, finger_qadr, finger_vadr, params: HandParams = None):
        self.m, self.d = model, data
        self.act = int(act_id)
        self.qadr = np.asarray(finger_qadr)
        self.vadr = np.asarray(finger_vadr)
        self.p = params or HandParams()
        self.dt = float(model.opt.timestep)
        # panda.xml の位置のサーボを、腱の速度のサーボに替える: force = kv·ctrl − kv·(腱の速さ)
        kv = self.p.kv
        model.actuator_gainprm[self.act, :] = 0.0
        model.actuator_gainprm[self.act, 0] = kv
        model.actuator_biasprm[self.act, :] = 0.0
        model.actuator_biasprm[self.act, 2] = -kv
        vmax = MAX_WIDTH_SPEED / 2.0
        model.actuator_ctrlrange[self.act] = (-vmax, vmax)
        model.actuator_ctrllimited[self.act] = 1
        model.actuator_forcelimited[self.act] = 1
        self.reset(opened=True)

    # ---------------------------------------------------------------- state
    def width(self) -> float:
        return float(self.d.qpos[self.qadr].sum())

    def width_speed(self) -> float:
        return float(self.d.qvel[self.vadr].sum())

    def is_grasped(self) -> bool:
        return self._grasped

    def _set_force_limit(self, finger_force: float) -> None:
        t = 2.0 * float(finger_force)                          # 腱の力 = 指 2 本分
        self.m.actuator_forcerange[self.act] = (-t, t)

    def reset(self, opened: bool = True) -> None:
        self.mode = "move"
        self.w_goal = MAX_WIDTH if opened else self.width()
        self.speed = MAX_WIDTH_SPEED * 0.9
        self.force = self.p.move_force
        self.window = None
        self._t = 0.0
        self._grasped = False
        self._set_force_limit(self.p.move_force)
        self.d.ctrl[self.act] = 0.0
        self.log = []                                          # (時刻, 口, 引数): 指令の監査に渡す記録

    # ------------------------------------------------------------- commands
    def _check(self, width, speed, force=None):
        if not 0.0 <= width <= MAX_WIDTH + 1e-9:
            raise HandCommandError(f"width {width} outside [0, {MAX_WIDTH}]")
        if not 0.0 < speed <= MAX_WIDTH_SPEED + 1e-12:
            raise HandCommandError(f"speed {speed} m/s outside (0, {MAX_WIDTH_SPEED}]（指 1 本 0.05 m/s）")
        if force is not None and not 0.0 < force <= MAX_FINGER_FORCE + 1e-9:
            raise HandCommandError(f"force {force} N outside (0, {MAX_FINGER_FORCE}]")

    def move(self, width: float, speed: float) -> None:
        self._check(width, speed)
        self.log.append((float(self.d.time), "move", (float(width), float(speed))))
        self.mode, self.w_goal, self.speed = "move", float(width), float(speed)
        self._set_force_limit(self.p.move_force)
        self._grasped = False

    def grasp(self, width: float, speed: float, force: float, eps_inner: float = 0.005,
              eps_outer: float = 0.005) -> None:
        self._check(width, speed, force)
        self.log.append((float(self.d.time), "grasp", (float(width), float(speed), float(force),
                                                       float(eps_inner), float(eps_outer))))
        self.mode, self.speed, self.force = "grasp", float(speed), float(force)
        self.window = (float(width) - float(eps_inner), float(width) + float(eps_outer))
        self._t = 0.0
        self._grasped = False

    # ----------------------------------------------------------------- step
    def step(self) -> None:
        """物理ステップの前に 1 回。速度の指令と力の上限を書く。"""
        p, w = self.p, self.width()
        if self.mode == "move":
            v = float(np.clip(p.k_pos * (self.w_goal - w), -self.speed, self.speed))
            self._grasped = False
        else:
            self._t += self.dt
            ramp = min(1.0, self._t / p.ramp_s) if p.ramp_s > 0 else 1.0
            self._set_force_limit(p.move_force + (self.force - p.move_force) * ramp)
            v = -float(np.clip(p.k_pos * w, 0.0, self.speed))     # 閉じ切りで指の腹どうしを押し込まない
            lo, hi = self.window
            self._grasped = lo <= w <= hi and abs(self.width_speed()) < p.still_speed
        self.d.ctrl[self.act] = v / 2.0                        # 腱の長さ = w/2
