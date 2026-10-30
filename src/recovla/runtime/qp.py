"""安全フィルタの二次計画（変数 3 つ・不等式は高々数本）。シミュレーションの側（recovla.sim.safety）と実行系（recovla.runtime.safety）が共に使う。
外部の解法器は使わない。有効な組を数え上げ、KKT を満たす解を取る。numpy だけに依る。"""
import itertools

import numpy as np

_EPS = 1e-12

def project(dx, A, b):
    """min ‖x − dx‖² s.t. A x ≥ b（A は (m, 3)）。満たす解がなければ、残る違反の小さい近似（Dykstra）を返す。"""
    dx = np.asarray(dx, float)
    if not len(A) or np.all(A @ dx >= b - 1e-15):
        return dx, True
    m = len(A)
    viol = [i for i in range(m) if A[i] @ dx < b[i]]
    order = viol + [i for i in range(m) if i not in viol]
    for size in range(1, min(3, m) + 1):
        for S in itertools.combinations(order, size):
            As, bs = A[list(S)], b[list(S)]
            G = As @ As.T
            if abs(np.linalg.det(G)) < 1e-12:
                continue
            lam = np.linalg.solve(G, bs - As @ dx)
            if np.any(lam < -1e-12):
                continue
            x = dx + As.T @ lam
            if np.all(A @ x >= b - 1e-12):
                return x, True
    x = dx.copy()                                       # Dykstra（交わりが空に近い、向きが揃った組）
    p = np.zeros((m, 3))
    for _ in range(200):
        for i in range(m):
            y = x + p[i]
            s = A[i] @ y - b[i]
            nn = A[i] @ A[i]
            x_new = y - (min(s, 0.0) / nn) * A[i] if nn > _EPS else y
            p[i] = y - x_new
            x = x_new
    return x, False
