"""推定の関数の確かめ用の合成の点群（机の座標、机 = z 0）。面に一様に点を置き、カメラから見える面だけ残し（裏の面を落とす。
隠れは見ない）、視線の向きに深度の雑音を足し、欠け（ばらばらの抜けと、アクションチャンクの抜け）を作る。"""
import math

import numpy as np

CAMS = (np.array([0.45, -0.55, 0.75]),)                         # 俯瞰（手前の斜め上）
CAMS2 = CAMS + (np.array([0.75, 0.30, 0.45]),)                   # 俯瞰＋手首に当たる 2 つ目


def rot(yaw=0.0, roll=0.0, pitch=0.0) -> np.ndarray:
    cz, sz, cx, sx, cy, sy = math.cos(yaw), math.sin(yaw), math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return Rz @ Rx @ Ry


def _rect(rng, c, a, b, ha, hb, nrm, density):
    n = max(1, int(density * 4 * ha * hb))
    u, v = rng.uniform(-ha, ha, n), rng.uniform(-hb, hb, n)
    return c + np.outer(u, a) + np.outer(v, b), np.repeat(nrm[None], n, axis=0)


def finish(rng, P, N, cams, noise, drop=0.0, hole=None, hole_r=0.0):
    keep = np.zeros(len(P), bool)
    for cam in cams:
        keep |= np.einsum("ij,ij->i", N, cam - P) > 0
    keep &= P[:, 2] >= -1e-9
    P = P[keep]
    cam = cams[0]
    ray = (P - cam) / np.linalg.norm(P - cam, axis=1, keepdims=True)
    P = P + ray * rng.normal(0, noise, (len(P), 1))
    if drop:
        P = P[rng.random(len(P)) >= drop]
    if hole is not None:
        P = P[np.linalg.norm(P - np.asarray(hole), axis=1) > hole_r]
    return P


def cuboid(rng, center, R, size, cams=CAMS, noise=0.0015, density=6e5, drop=0.0, hole=None, hole_r=0.0):
    """向き R・辺 size の直方体の表面の点。"""
    size = np.asarray(size, float)
    P, N = [], []
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        for s in (-1, 1):
            nrm = s * R[:, i]
            p, nn = _rect(rng, center + nrm * size[i] / 2, R[:, j], R[:, k], size[j] / 2, size[k] / 2, nrm, density)
            P.append(p)
            N.append(nn)
    return finish(rng, np.vstack(P), np.vstack(N), cams, noise, drop, hole, hole_r)


def resting_center(R, size) -> np.ndarray:
    """向き R の直方体を机に置いたときの中心の高さ（いちばん低い角が机）。"""
    c = np.array([[a, b, d] for a in (-1, 1) for b in (-1, 1) for d in (-1, 1)]) * np.asarray(size) / 2 @ R.T
    return -c[:, 2].min()


def container(rng, xy, yaw, outer, wall, height, floor_t=0.01, roll=0.0, pitch=0.0, cams=CAMS, noise=0.0015,
              density=6e5, drop=0.0, hole=None, hole_r=0.0):
    """箱（外寸の半分 outer=(hx, hy)、壁の厚さ、縁の高さ、底の厚さ）を傾けて机に置いた点。返り値: 点と真の値の辞書。"""
    hx, hy = outer
    ix, iy = hx - wall, hy - wall
    ex, ey, ez = np.eye(3)
    P, N = [], []

    def add(p_n):
        P.append(p_n[0])
        N.append(p_n[1])
    for s in (-1, 1):
        add(_rect(rng, np.array([s * hx, 0, height / 2]), ey, ez, hy, height / 2, s * ex, density))          # 外の面
        add(_rect(rng, np.array([0, s * hy, height / 2]), ex, ez, hx, height / 2, s * ey, density))
        add(_rect(rng, np.array([s * ix, 0, (height + floor_t) / 2]), ey, ez, iy, (height - floor_t) / 2, -s * ex, density))
        add(_rect(rng, np.array([0, s * iy, (height + floor_t) / 2]), ex, ez, ix, (height - floor_t) / 2, -s * ey, density))
        add(_rect(rng, np.array([s * (hx - wall / 2), 0, height]), ey, ex, hy, wall / 2, ez, density))       # 縁
        add(_rect(rng, np.array([0, s * (hy - wall / 2), height]), ex, ey, ix, wall / 2, ez, density))
    add(_rect(rng, np.array([0, 0, floor_t]), ex, ey, ix, iy, ez, density))                                 # 底
    Pl, Nl = np.vstack(P), np.vstack(N)
    R = rot(yaw, roll, pitch)
    base = np.array([[a * hx, b * hy, 0] for a in (-1, 1) for b in (-1, 1)]) @ R.T
    t = np.array([xy[0], xy[1], -base[:, 2].min()])
    Pw, Nw = Pl @ R.T + t, Nl @ R.T
    truth = {"rim_center": R @ np.array([0, 0, height]) + t, "R": R, "outer": np.array([hx, hy]),
             "inner": np.array([ix, iy]), "wall": wall, "floor_local": floor_t - height, "t": t}
    return finish(rng, Pw, Nw, cams, noise, drop, hole, hole_r), truth


def _disk(rng, c, a, b, r_in, r_out, nrm, density):
    """中心 c・面の軸 a, b の輪（r_in〜r_out）の点（面積に比例する数）。"""
    n = max(1, int(density * math.pi * (r_out ** 2 - r_in ** 2)))
    rr = np.sqrt(rng.uniform(r_in ** 2, r_out ** 2, n))
    th = rng.uniform(0, 2 * math.pi, n)
    return c + np.outer(rr * np.cos(th), a) + np.outer(rr * np.sin(th), b), np.repeat(np.asarray(nrm, float)[None], n, axis=0)


def _tube(rng, c, ax, a, b, r, h, outward, density):
    """軸 ax の円筒の側面（底の中心 c、半径 r、高さ h）。outward なら外向きの法線。"""
    n = max(1, int(density * 2 * math.pi * r * h))
    th = rng.uniform(0, 2 * math.pi, n)
    z = rng.uniform(0, h, n)
    radial = np.outer(np.cos(th), a) + np.outer(np.sin(th), b)
    return c + r * radial + np.outer(z, ax), radial * (1.0 if outward else -1.0)


def cup(rng, bottom, R=np.eye(3), radius=0.03, height=0.08, wall=0.0045, base=0.005, cams=CAMS, noise=0.0015,
        density=6e5, drop=0.0, hole=None, hole_r=0.0):
    """上が開いた円筒（底の中心 bottom、向き R の 3 列目が軸）の表面の点: 外の側面・縁・内の側面・内の底・外の底。"""
    a, b, ax = R[:, 0], R[:, 1], R[:, 2]
    c = np.asarray(bottom, float)
    parts = [_tube(rng, c, ax, a, b, radius, height, True, density),
             _disk(rng, c + height * ax, a, b, radius - wall, radius, ax, density),
             _tube(rng, c + base * ax, ax, a, b, radius - wall, height - base, False, density),
             _disk(rng, c + base * ax, a, b, 0.0, radius - wall, ax, density),
             _disk(rng, c, a, b, 0.0, radius, -ax, density)]
    P = np.vstack([p for p, _ in parts])
    N = np.vstack([n for _, n in parts])
    # 内の面は、縁の上から見える範囲だけ（壁に隠れる所を雑に落とす: カメラから縁の円の内側を通る視線だけ）
    keep = np.ones(len(P), bool)
    loc = (P - c) @ R
    inner = np.hypot(loc[:, 0], loc[:, 1]) < radius - wall + 1e-9
    for i in np.flatnonzero(inner & (loc[:, 2] < height - 1e-9)):
        ok = False
        for cam in cams:
            lc = (cam - c) @ R
            if lc[2] <= height:
                continue
            s = (height - loc[i, 2]) / (lc[2] - loc[i, 2])
            q = loc[i] + s * (lc - loc[i])
            ok |= math.hypot(q[0], q[1]) < radius - wall
        keep[i] = ok
    return finish(rng, P[keep], N[keep], cams, noise, drop, hole, hole_r)


def plate(rng, center_xy, radius=0.09, thickness=0.012, roll=0.0, pitch=0.0, cams=CAMS, noise=0.0015, density=6e5,
          drop=0.0, hole=None, hole_r=0.0):
    """平らな円板を傾けて机に置いた点（上面・側面）。返り値: 点と真の値（上面の中心・法線）。"""
    R = rot(0.0, roll, pitch)
    a, b, ax = R[:, 0], R[:, 1], R[:, 2]
    base = np.array([radius * np.array([math.cos(t), math.sin(t), 0.0]) for t in np.linspace(0, 2 * math.pi, 72)]) @ R.T
    c = np.array([center_xy[0], center_xy[1], -base[:, 2].min()])
    top, nt = _disk(rng, c + thickness * ax, a, b, 0.0, radius, ax, density)
    side, ns = _tube(rng, c, ax, a, b, radius, thickness, True, density)
    P = finish(rng, np.vstack([top, side]), np.vstack([nt, ns]), cams, noise, drop, hole, hole_r)
    return P, {"center": c + thickness * ax, "normal": ax, "radius": radius}
