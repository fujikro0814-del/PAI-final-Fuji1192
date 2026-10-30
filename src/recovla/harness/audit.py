"""監査。評価の側で、試行ごとに記録して集計する。記録の欄 g1・g2・g3 は次の 3 つ:
  g1  実行系から世界の物（真の状態・シミュレータ）に参照で届かない（reachable_forbidden）
  g2  世界は物理の周期で止まらずに進み、実行系は届く前の観測を使わない（harness/loop.py・robot_io.py）
  g3  指令が Franka Panda の公称の上限の中にある（CommandAudit）

CommandAudit: RobotIO の口を通った指令（関節の位置 1 kHz、ハンドの move・grasp）を、Panda の公称の上限と比べる。
  関節の速度・加速度・躍度は 1 kHz の差分（libfranka と同じ）、直交座標は指令の順運動学の手先。上限は公称の値そのもの
  （制限層の margin は掛けない）。位置のサーボの力とその変化（500 Hz）は参考（reference）として別に書く
"""
import gc
import types

import numpy as np

from recovla.runtime import limiter as L


def _cell_ok(c) -> bool:
    try:
        c.cell_contents
        return True
    except ValueError:
        return False


def reachable_forbidden(root, forbidden_ids: set, forbidden_types: tuple, boundary_types: tuple, limit: int = 2_000_000):
    """g1 の到達検査: root（実行系）から参照を辿って届くものに、世界の物（forbidden_ids の id を持つもの、
    forbidden_types の型）がないかを数える。boundary_types（RobotIO の実装）の中には入らない（実機では libfranka の接続に
    あたる境界。実行系が使える口は静的な検査で 6 つに限っている）。numpy の配列・torch のテンソル・文字列・数は葉として扱う。
    → {"visited": 数, "violations": [型の名前, ...], "truncated": 真偽}"""
    leaf = (np.ndarray, str, bytes, int, float, bool, complex, type(None), types.ModuleType, type)
    try:
        import torch
        leaf = leaf + (torch.Tensor,)
    except ImportError:
        pass
    seen, stack, bad = set(), [root], []
    while stack:
        o = stack.pop()
        i = id(o)
        if i in seen:
            continue
        seen.add(i)
        if i in forbidden_ids or isinstance(o, forbidden_types):
            bad.append(type(o).__name__)
            continue
        if isinstance(o, boundary_types) or isinstance(o, leaf):
            continue
        if len(seen) > limit:
            return {"visited": len(seen), "violations": bad, "truncated": True}
        if isinstance(o, types.FunctionType):                # モジュールの大域（ライブラリ全体）には入らず、閉包と既定値だけ
            stack.extend(c.cell_contents for c in (o.__closure__ or ()) if _cell_ok(c))
            stack.extend(o.__defaults__ or ())
            continue
        if isinstance(o, types.MethodType):
            stack.extend([o.__self__, o.__func__])
            continue
        if isinstance(o, types.BuiltinFunctionType):
            s = getattr(o, "__self__", None)
            if s is not None and not isinstance(s, types.ModuleType):
                stack.append(s)
            continue
        stack.extend(gc.get_referents(o))
    return {"visited": len(seen), "violations": bad, "truncated": False}

TORQUE = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
TORQUE_RATE = 1000.0
CART_VEL, CART_ACC, CART_JERK = 1.7, 13.0, 6500.0
TOL = 1e-6                                  # 数値の丸めの許容（比で）


class CommandAudit:
    def __init__(self, fk=None):
        self.fk = fk                                           # q -> 手先の位置 (3,)（直交座標の監査。None なら省く）
        self.reset()

    def reset(self) -> None:
        self.q = []                                            # 1 kHz の指令
        self.tau = []                                          # 500 Hz の作動器の力（重力の補償を含む）
        self.hand = []                                         # (時刻, 口, 引数)
        self.finger_speed_max = 0.0                            # 参考: 測った指の速さの最大（指令ではない）

    def joint_command(self, q) -> None:
        self.q.append(np.array(q, float))

    def torque(self, tau) -> None:
        self.tau.append(np.array(tau, float))

    def hand_commands(self, log) -> None:
        self.hand = list(log)

    def finger_speed(self, v) -> None:
        self.finger_speed_max = max(self.finger_speed_max, float(v))

    def summary(self) -> dict:
        out = {"n_joint_commands": len(self.q), "violations": {}, "max_ratio": {}}

        def put(name, ratio):
            r = np.asarray(ratio, float)
            out["max_ratio"][name] = float(r.max()) if r.size else 0.0
            out["violations"][name] = int(np.sum(r.reshape(len(r), -1).max(axis=1) > 1.0 + TOL)) if r.size else 0

        if len(self.q) >= 4:
            q = np.array(self.q)
            v = np.diff(q, axis=0) / L.DT
            a = np.diff(v, axis=0) / L.DT
            j = np.diff(a, axis=0) / L.DT
            put("joint_vel", np.abs(v) / L.MAX_VEL)
            put("joint_acc", np.abs(a) / L.MAX_ACC)
            put("joint_jerk", np.abs(j) / L.MAX_JERK)
            if self.fk is not None:
                x = np.array([self.fk(qq) for qq in q])
                cv = np.diff(x, axis=0) / L.DT
                ca = np.diff(cv, axis=0) / L.DT
                cj = np.diff(ca, axis=0) / L.DT
                put("cart_vel", np.linalg.norm(cv, axis=1) / CART_VEL)
                put("cart_acc", np.linalg.norm(ca, axis=1) / CART_ACC)
                put("cart_jerk", np.linalg.norm(cj, axis=1) / CART_JERK)
        if len(self.tau) >= 2:
            tau = np.array(self.tau)
            put("torque", np.abs(tau) / TORQUE)
            put("torque_rate", np.abs(np.diff(tau, axis=0)) / 0.002 / TORQUE_RATE)
        from recovla.harness.hand import MAX_FINGER_FORCE, MAX_WIDTH_SPEED
        speeds = [args[1] for _, kind, args in self.hand]
        forces = [args[2] for _, kind, args in self.hand if kind == "grasp"]
        put("gripper_speed", np.array(speeds) / MAX_WIDTH_SPEED if speeds else np.zeros(0))
        put("gripper_force", np.array(forces) / MAX_FINGER_FORCE if forces else np.zeros(0))
        out["n_hand_commands"] = len(self.hand)
        out["finger_speed_measured_max"] = self.finger_speed_max
        # 口を通る指令の項目だけを数える。位置のサーボの力（torque・torque_rate）は実機の内部の制御器の出力で、
        # 関節の位置の指令の口では指令されないので参考として別に書く（力の大きさは作動器の上限で切られる）
        ref = ("torque", "torque_rate")
        out["reference"] = {k: {"max_ratio": out["max_ratio"].pop(k, 0.0), "violations": out["violations"].pop(k, 0)} for k in ref}
        out["total_violations"] = int(sum(out["violations"].values()))
        return out
