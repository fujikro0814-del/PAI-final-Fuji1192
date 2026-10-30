"""実行系に渡す RobotIO のシミュレーション版。世界・センサのモデル・計算の時間のモデルをまとめる（実機では libfranka の接続に当たる）。

    io = SimRobotIO(world, suite, seed)
    io.sense(cameras)                 # SensorFrame（センサのモデルを通したもの）
    io.command_joints(q)              # 1 kHz 相当の関節の位置の指令（物理の 1 ステップで 2 つ）
    io.gripper_move / gripper_grasp   # ハンドの口（上限を超える指令はハンドが拒む）
    fut = io.compute(kind, fn)        # 計算（推論・知覚）: 壁の時計で fn を今すぐ実行し、シミュレーションの時刻の上では
                                      # 分布から引いた時間の後に使えるようになる（Pending）。世界はその間も進む
    io.now                          # シミュレーションの時刻（実行系の時計）

計算の時間の分布は configs/latency_v1.json（GPU での実測、前処理を含む）。知覚の遅延は固定の値（configs）。
"""
import json
import time

import numpy as np

from recovla.common import config, seeds

LATENCY = json.loads((config.ROOT / "configs" / "latency_v1.json").read_text(encoding="utf-8"))


def fixed_latency(rtv: dict) -> dict:
    """configs の runtime_v2 → {kind: 秒}（固定の遅延）。基本の知覚は "perception"（perception_latency_s。compute_latency_s に
    あればそちら）。推定を足した知覚（"perception_est"）・手で書いたスクリプト方策の知覚（"perception_cl2"）・
    起動時の 1 回（"perception_startup"）は compute_latency_s（計算の時間の実測から。configs/runtime_v2.yaml）。"""
    out = {"perception": float(rtv.get("perception_latency_s", 0.0))}
    out.update({str(k): float(v) for k, v in (rtv.get("compute_latency_s") or {}).items()})
    return out


class EarlyUse(RuntimeError):
    """使えるようになる前に結果を使った（監査 g2 の違反）。"""


class Pending:
    def __init__(self, value, t_start: float, latency: float, wall_s: float, log: list):
        self._value = value
        self.t_start = t_start
        self.latency = float(latency)
        self.t_ready = t_start + self.latency
        self.wall_s = wall_s
        self._log = log

    def ready(self, t: float) -> bool:
        return t >= self.t_ready - 1e-12

    def result(self, t: float):
        if not self.ready(t):
            self._log.append({"early_use": True, "t": t, "t_ready": self.t_ready})
            raise EarlyUse(f"t={t:.4f} < t_ready={self.t_ready:.4f}")
        self._log.append({"use": True, "t": t, "t_ready": self.t_ready})
        return self._value


class SimRobotIO:
    def __init__(self, world, suite, seed: int, fixed_latency: dict = None):
        self.world, self.suite = world, suite
        self.rng = np.random.default_rng(seeds.seed_sequence(seed, "latency", 1))    # 計算の時間（カメラの遅延は 0）
        self.fixed = dict(fixed_latency or {})             # {kind: 秒}（知覚など、固定の遅延）
        self.cmd_buffer = []
        self.compute_log = []
        self.use_log = []

    def now(self) -> float:
        return float(self.world.data.time)

    def sense(self, cameras: bool = True):
        return self.suite.sense(self.world.data, self.world.hand, self.world._last_cmd, cameras)

    def command_joints(self, q) -> None:
        self.cmd_buffer.append(np.asarray(q, float).copy())

    def take_commands(self) -> list:
        out, self.cmd_buffer = self.cmd_buffer, []
        return out

    def gripper_move(self, width: float, speed: float) -> None:
        self.world.hand.move(width, speed)

    def gripper_grasp(self, width, speed, force, eps_inner, eps_outer) -> None:
        self.world.hand.grasp(width, speed, force, eps_inner, eps_outer)

    def draw_latency(self, kind: str) -> float:
        if kind in self.fixed:
            return float(self.fixed[kind])
        q = LATENCY["kinds"][kind]["quantiles_s"]
        u = float(self.rng.random())
        return float(np.interp(u, np.linspace(0, 1, len(q)), q))

    def compute(self, kind: str, fn) -> Pending:
        t = self.now()
        w0 = time.perf_counter()
        value = fn()
        wall = time.perf_counter() - w0
        lat = self.draw_latency(kind)
        self.compute_log.append({"kind": kind, "t_start": t, "latency_s": lat, "t_ready": t + lat, "wall_s": wall})
        return Pending(value, t, lat, wall, self.use_log)

    def wall_by_kind(self) -> dict:
        """{kind: [各回の壁時計の計算の時間 [s]]}（試行をまたいで p99 を取り、固定の遅延を決めるため）。"""
        out = {}
        for c in self.compute_log:
            out.setdefault(c["kind"], []).append(round(float(c["wall_s"]), 4))
        return out

    def audit(self) -> dict:
        early = [e for e in self.use_log if e.get("early_use")]
        kinds = sorted({c["kind"] for c in self.compute_log})
        wall = {k: np.array([c["wall_s"] for c in self.compute_log if c["kind"] == k]) for k in kinds}
        return {"n_compute": len(self.compute_log), "n_use": sum(1 for e in self.use_log if e.get("use")),
                "early_use": len(early), "fixed_latency": dict(self.fixed),
                "compute_wall_s": {k: {"n": int(len(w)), "median": round(float(np.median(w)), 4),
                                       "p90": round(float(np.percentile(w, 90)), 4),
                                       "p99": round(float(np.percentile(w, 99)), 4), "max": round(float(w.max()), 4)}
                                   for k, w in wall.items()},
                "latency_by_kind": {k: [c["latency_s"] for c in self.compute_log if c["kind"] == k]
                                    for k in {c["kind"] for c in self.compute_log}}}
