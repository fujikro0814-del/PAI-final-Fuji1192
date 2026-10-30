"""関節の指令の制限層。libfranka の limitRate（rate_limiting.cpp）と同じ式。

1 kHz 相当の刻みごとに、望む関節の速さを受け取り、躍度 → 加速度 → 速度の順に上限へ切り詰めて、
前の指令の位置に積分した関節の位置の指令を返す。上限は Franka の関節の上限（MAX_*）に margin（既定 0.99）を掛けたもの。

    lim = JointLimiter; lim.reset(q)
    q_c = lim.step(v_des)          # v_des: 望む関節の速さ (7,) [rad/s]。返り値: 関節の位置の指令 (7,)
"""
import numpy as np

DT = 1e-3                                                   # libfranka の kDeltaT
# Franka の関節の上限: 速度 [rad/s]・加速度 [rad/s²]・躍度 [rad/s³]
MAX_VEL = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])
MAX_ACC = np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0])
MAX_JERK = np.array([7500.0, 3750.0, 5000.0, 6250.0, 7500.0, 10000.0, 10000.0])


def limit_rate(max_vel, max_acc, max_jerk, v_cmd, v_last, a_last, dt=DT):
    """libfranka の limitRate（関節の速さの版）。返り値は切り詰めた速さ。"""
    jerk = ((v_cmd - v_last) / dt - a_last) / dt
    acc = a_last + np.clip(jerk, -max_jerk, max_jerk) * dt
    safe_max_acc = np.minimum((max_jerk / max_acc) * (max_vel - v_last), max_acc)
    safe_min_acc = np.maximum((max_jerk / max_acc) * (-max_vel - v_last), -max_acc)
    return v_last + np.clip(acc, safe_min_acc, safe_max_acc) * dt


class JointLimiter:
    def __init__(self, margin: float = 0.99, q_min=None, q_max=None, enabled: bool = True):
        self.max_vel, self.max_acc, self.max_jerk = MAX_VEL * margin, MAX_ACC * margin, MAX_JERK * margin
        self.q_min = None if q_min is None else np.asarray(q_min, float)
        self.q_max = None if q_max is None else np.asarray(q_max, float)
        self.enabled = enabled
        self.q = self.v = self.a = None
        self.n_clipped = 0                                   # 切り詰めた刻みの数（記録用）
        self.max_clip = 0.0                                  # 切り詰めた量の最大 [rad/s]
        self.n_pos_clipped = 0

    def reset(self, q) -> None:
        self.q = np.array(q, dtype=float)
        self.v = np.zeros(7)
        self.a = np.zeros(7)

    def propose(self, v_des) -> np.ndarray:
        """切り詰めた速さの候補（まだ確定しない）。"""
        v_des = np.asarray(v_des, float)
        if not self.enabled:
            return v_des
        if self.q_min is not None:
            # 可動域の端の手前で止まれる速さに抑える。端で位置を切り詰めると加速度・躍度が上限を超え、減速の候補がその値を
            # 引き継いで発散するため。止まるまでの距離は、加速度の上限の半分 a で減速し、
            # 減速に入るまでの躍度の立ち上がり（a / 躍度の上限）の分を足す: v²/(2a) + v·a/j ≤ 残り（2 mrad の余白を引く）
            a, j = 0.5 * self.max_acc, self.max_jerk
            b = a * a / j

            def cap(room):
                r = np.maximum(room - 0.002, 0.0)
                return -b + np.sqrt(b * b + 2.0 * a * r)
            v_des = np.clip(v_des, -cap(self.q - self.q_min), cap(self.q_max - self.q))
        v = limit_rate(self.max_vel, self.max_acc, self.max_jerk, v_des, self.v, self.a)
        d = float(np.max(np.abs(v - v_des)))
        if d > 1e-9:
            self.n_clipped += 1
            self.max_clip = max(self.max_clip, d)
        return v

    def shrink(self, v_cand, s: float) -> np.ndarray:
        """躍度の項だけを s 倍に縮めた速さ（加速度は前の値と候補の間、躍度は候補の s 倍なので関節の上限は保たれる）。"""
        v0 = self.v + self.a * DT
        return self._bounded(v0 + s * (np.asarray(v_cand) - v0))

    def brake(self) -> np.ndarray:
        """加速度の大きさを躍度の上限いっぱいで 0 へ近づける速さ（関節の上限を保ったまま、いちばん早く加速度を落とす）。"""
        da = np.clip(-self.a, -self.max_jerk * DT, self.max_jerk * DT)
        return self._bounded(self.v + (self.a + da) * DT)

    def _bounded(self, v) -> np.ndarray:
        """加速度と速度を上限の中に収める（前の加速度が上限を超えていても、それを引き継がない）。"""
        a = np.clip((np.asarray(v, float) - self.v) / DT, -self.max_acc, self.max_acc)
        return np.clip(self.v + a * DT, -self.max_vel, self.max_vel)

    def between(self, v_from, v_to, s: float) -> np.ndarray:
        return np.asarray(v_from) + s * (np.asarray(v_to) - np.asarray(v_from))

    def position_of(self, v) -> np.ndarray:
        q = self.q + np.asarray(v) * DT
        return q if self.q_min is None else np.clip(q, self.q_min, self.q_max)

    def step(self, v_des) -> np.ndarray:
        return self.commit(self.propose(v_des))

    def commit(self, v) -> np.ndarray:
        v = np.asarray(v, float)
        q = self.q + v * DT
        if self.q_min is not None:
            qc = np.clip(q, self.q_min, self.q_max)
            if np.any(qc != q):
                self.n_pos_clipped += 1                     # 記録だけ: 可動域の端で位置を切り詰めた刻み
            q = qc
            v = (q - self.q) / DT
        self.a = np.clip((v - self.v) / DT, -self.max_acc, self.max_acc) if self.enabled else (v - self.v) / DT
        self.v = v
        self.q = q
        return q.copy()
