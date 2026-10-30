"""Franka Hand のモデルに、実機の Panda で測った指令の遅延と速さの形を足したもの（configs/real_v1.yaml の real_v1.gripper）。

harness/hand.py の FrankaHand を継ぎ、口（move・grasp）と判定は同じ。違いは次のとおり:
- dead_time_s: 指令を受けてから指が動き始めるまでの遅延 [s]。その間は前の指令のまま動く（実機: 約 0.15 s。センサの値で
  見ると 0.23 s で、差は値の届く間隔 約 0.15 s の半分。届く間隔はセンサのモデルの rate_hz で出す）
- dead_time_sd・dead_time_s_range: 指令ごとに遅延を正規分布で引き、範囲で切る。sd が 0 なら毎回 dead_time_s
  （実機のばらつきは値の届く位相で説明できるので、real_v1 は 0）
- speed_scale: 指令の速さに掛ける（実機: 動いている間 0.080 m/s ＝ 指令のまま。real_v1 は 1）
- k_pos: 0 より大きければ HandParams.k_pos を置き換える（目標の手前で減速する強さ。実機: 開く最後の詰めが 9.1 /s、
  シミュレーションは 20 /s）
- speed_cap: 開き幅の速さの上限 [m/s]。0 なら切る（実機の速さは指令のままなので real_v1 では使わない）
どれも既定の値（0・1）なら FrankaHand と同じ動き（tests/test_real_replay.py）。

    hand = RealTimingHand(model, data, act, qadr, vadr, params, RealTiming(dead_time_s=0.15, k_pos=9.1))
    install(rig, RealTiming.from_config(cfg["real_v1"]["gripper"]))   # 世界（WorldRig）のハンドを差し替える
"""
import dataclasses

import numpy as np

from recovla.harness.hand import FrankaHand, HandParams


@dataclasses.dataclass(frozen=True)
class RealTiming:
    dead_time_s: float = 0.0
    speed_cap: float = 0.0
    speed_scale: float = 1.0
    k_pos: float = 0.0
    dead_time_sd: float = 0.0
    dead_time_s_range: tuple = (0.0, float("inf"))
    seed: int = 0

    @classmethod
    def from_config(cls, cfg: dict) -> "RealTiming":
        cfg = cfg or {}
        rng = cfg.get("dead_time_s_range") or (0.0, float("inf"))
        return cls(float(cfg.get("dead_time_s", 0.0)), float(cfg.get("speed_cap", 0.0)),
                   float(cfg.get("speed_scale", 1.0)), float(cfg.get("k_pos", 0.0)),
                   float(cfg.get("dead_time_sd", 0.0)), (float(rng[0]), float(rng[1])), int(cfg.get("seed", 0)))

    @property
    def is_identity(self) -> bool:
        return (self.dead_time_s <= 0 and self.dead_time_sd <= 0 and self.speed_cap <= 0 and self.speed_scale == 1.0
                and self.k_pos <= 0)


class RealTimingHand(FrankaHand):
    def __init__(self, model, data, act_id: int, finger_qadr, finger_vadr, params: HandParams = None,
                 timing: RealTiming = None):
        self.timing = timing or RealTiming()
        self._pending = None
        self._rng = np.random.default_rng(self.timing.seed)
        if self.timing.k_pos > 0:
            params = dataclasses.replace(params or HandParams(), k_pos=self.timing.k_pos)
        super().__init__(model, data, act_id, finger_qadr, finger_vadr, params)

    def reseed(self, seed: int) -> None:
        """指令ごとの遅延を引く乱数を、試行のシードから作り直す（dead_time_sd が 0 なら使わない）。"""
        self._rng = np.random.default_rng(int(seed))

    def reset(self, opened: bool = True) -> None:
        super().reset(opened)
        self._pending = None                                   # 遅延して効く指令 (効く時刻, 口, 引数)

    def _dead_time(self) -> float:
        t = self.timing
        if t.dead_time_sd <= 0:
            return t.dead_time_s
        lo, hi = t.dead_time_s_range
        return float(np.clip(self._rng.normal(t.dead_time_s, t.dead_time_sd), lo, hi))

    def _shape(self) -> None:
        if self.timing.speed_scale != 1.0:
            self.speed = self.speed * self.timing.speed_scale
        if self.timing.speed_cap > 0:
            self.speed = min(self.speed, self.timing.speed_cap)

    def _delayed(self, kind: str, args: tuple) -> None:
        delay = self._dead_time()
        if delay <= 0:
            (FrankaHand.move if kind == "move" else FrankaHand.grasp)(self, *args)
            self._shape()
            return
        self.log.append((float(self.d.time), kind, args))
        self._pending = (float(self.d.time) + delay, kind, args)

    def move(self, width: float, speed: float) -> None:
        self._check(width, speed)
        self._delayed("move", (float(width), float(speed)))

    def grasp(self, width: float, speed: float, force: float, eps_inner: float = 0.005,
              eps_outer: float = 0.005) -> None:
        self._check(width, speed, force)
        self._delayed("grasp", (float(width), float(speed), float(force), float(eps_inner), float(eps_outer)))

    def step(self) -> None:
        if self._pending is not None and float(self.d.time) >= self._pending[0] - 1e-12:
            _, kind, args = self._pending
            self._pending = None
            n = len(self.log)
            (FrankaHand.move if kind == "move" else FrankaHand.grasp)(self, *args)
            del self.log[n:]                                   # 記録は受けた時刻の 1 行だけ（効いた時刻は足さない）
            self._shape()
        super().step()


def install(rig, timing: RealTiming) -> RealTimingHand:
    """WorldRig のハンドを、同じモデル・同じ値の RealTimingHand に差し替える（rig.reset の前に呼ぶ）。"""
    old = rig.hand
    rig.hand = RealTimingHand(old.m, old.d, old.act, old.qadr, old.vadr, old.p, timing)
    return rig.hand
