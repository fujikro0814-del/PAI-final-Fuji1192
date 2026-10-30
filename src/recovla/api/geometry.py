"""点群からの形の推定（api/estimated.EstimatedGeometryFinder の中身）。純関数（numpy・scipy・cv2 だけ）。

既知の大きさ・寸法（cube_size・box_outer・box_wall・box_nominal_xy）は読まない。使う既知の値は机の平面（較正の誤差を含む値）と、
深度の雑音の目安 noise_m（センサの性質。物の大きさではない）だけ。点は机の座標（机 = z 0、上向き +z）で受ける。

    g = fit_object(pts)        # 物: 向きつきの外接箱（中心・向き R・3 辺・上面の高さ）と不確かさ。ObjGeom
    b = fit_container(pts)     # 箱らしい物: 縁の平面・中心・向き・外寸・内寸・壁・縁の高さ・底・傾き（roll・pitch）と不確かさ
    b.depth(p)                 # 推定した 3 次元の内側の領域からの深さ（正なら中、負なら外。距離 [m]）

物の決め方:
  1. その場の法線のまとまりから面を探す（find_faces）。いちばん大きい面の法線と、ほぼ直交する 2 つ目の法線から向き R を作る
     （面が 1 つしか無ければ、その面へ投影した点の最小外接長方形で残りの向きを決める）。R の 3 列目は机の上向きにいちばん近い軸、
     机に沿う 2 軸のうち長い方を x
  2. R で点を射影し、面が当てはまった端はその面の位置、ほかの端は 0.5・99.5 百分位を辺の端にする（外れ値に強い）
  3. 置かれている（外接箱のいちばん低い角と、支える面（机、箱の中なら推定の底）の隙間が、推定した最小の辺の半分以下）なら、
     見えない下の側を支える面まで延ばす（上からのカメラでは側面の下が欠け、切り出しの高さより下の点も無いため）。
     延ばした量は不確かさに足す
箱らしい物の決め方:
  1. 高い方の 10% の点から縁の平面を RANSAC で探し、全点から平面の近く（3σ）を縁として取り直して最小二乗で当てはめ直す
  2. 全点を縁の平面へ投影して最小外接長方形の向きを決め、百分位で外寸（正方形・90° の対称は仮定しない。長い辺を x にする）
  3. 縁の点を 4 つの辺に分け、辺ごとに外の縁と壁の厚さを、点が壁の厚さの中に一様に並ぶことから求める（平均 ± √3 × 標準偏差、
     雑音の分を引く）。外寸・中心はこの外の縁から取り直し、壁の厚さは辺の中央値。内寸 = 外寸 − 壁
  4. 内側の点の高さの中央値を底にする（見えなければ机）。傾きは縁の法線の机に対する roll（x 軸まわり）・pitch（y 軸まわり）
不確かさは目安（雑音 σ と点の数・延ばした量から）。失敗の検知・スクリプト方策の余白はこの値の倍数で書く（api/relations）。
"""
import dataclasses
import math
from typing import Optional

import cv2
import numpy as np
from scipy.spatial import cKDTree

NOISE_M = 0.002                     # 深度の雑音の目安（センサの性質。configs/sensor_v1.yaml の深度の雑音の大きさ）
WORKERS = 4                         # k 近傍の問い合わせの並列の数（結果は同じ。速さだけ）


@dataclasses.dataclass(frozen=True)
class ObjGeom:
    center: np.ndarray              # (3,) 机の座標
    R: np.ndarray                   # (3, 3) 列が物の軸。3 列目は机の上向きにいちばん近い軸（上向き）
    extent: np.ndarray              # (3,) 各軸の辺の長さ（全長）
    top_h: float                    # いちばん高い角の机からの高さ
    cov: np.ndarray                 # (3, 3) 中心の共分散（目安）
    extent_sigma: np.ndarray        # (3,) 辺の長さの標準偏差（目安）
    n: int                          # 使った点の数
    faces: int                      # 当てはまった面の数
    supported: bool                 # 机に置かれていると判定した
    extended: float                 # 机まで延ばした量 [m]

    @property
    def yaw(self) -> float:
        """x 軸の、机の上での向き（(-π/2, π/2]。長方形の 180° の対称だけ使う）。"""
        a = math.atan2(self.R[1, 0], self.R[0, 0])
        return float((a + math.pi / 2) % math.pi - math.pi / 2)

    @property
    def tilt(self) -> float:
        """上向きの軸と机の法線のなす角 [rad]。"""
        return float(math.acos(min(1.0, abs(self.R[2, 2]))))

    @property
    def height(self) -> float:
        """机の法線の向きの見かけの高さ（斜めなら辺より大きい）。"""
        return float(np.sum(np.abs(self.R[2, :]) * self.extent))

    @property
    def sigma(self) -> float:
        return float(math.sqrt(max(np.linalg.eigvalsh(self.cov).max(), 0.0)))


@dataclasses.dataclass(frozen=True)
class BoxGeom:
    center: np.ndarray              # (3,) 縁の平面の上の、外寸の長方形の中心（机の座標）
    R: np.ndarray                   # (3, 3) 列: x（長い辺）・y（縁の平面の中）・z（縁の法線、上向き）
    outer: np.ndarray               # (2,) 外寸の半分（x, y）
    inner: np.ndarray               # (2,) 内寸の半分
    wall: float                     # 壁の厚さ
    rim_h: float                    # 縁の中心の机からの高さ
    floor: float                    # 底の、箱の座標の z（負。縁から下へ）
    roll: float                     # 縁の法線の、机に対する x 軸まわりの傾き [rad]
    pitch: float                    # 同じく y 軸まわり [rad]
    sigma: dict                     # {"center", "yaw", "inner", "rim_h", "tilt"} の標準偏差（目安）
    n: int
    n_rim: int
    n_edges: int = 4                # 縁の点から外の縁を求められた辺の数（0〜4。腕に隠れると減る）

    @property
    def yaw(self) -> float:
        a = math.atan2(self.R[1, 0], self.R[0, 0])
        return float((a + math.pi / 2) % math.pi - math.pi / 2)

    @property
    def tilt(self) -> float:
        return float(math.acos(min(1.0, abs(self.R[2, 2]))))

    def local(self, p) -> np.ndarray:
        """机の座標の点 → 箱の座標（原点は縁の中心、z は縁の法線）。"""
        return (np.asarray(p, float) - self.center) @ self.R

    def depth(self, p) -> float:
        """推定した内側の領域（|x| ≤ 内寸、|y| ≤ 内寸、底 ≤ z ≤ 縁）からの深さ。正なら中（いちばん近い面までの距離）、
        負なら外（はみ出しの大きい方の量）。"""
        q = self.local(p)
        d = np.array([self.inner[0] - abs(q[0]), self.inner[1] - abs(q[1]), -q[2], q[2] - self.floor])
        return float(d.min()) if d.min() >= 0 else float(-np.linalg.norm(np.minimum(d, 0.0)))

    def over_xy(self, p, margin: float = 0.0) -> bool:
        """箱の柱（内寸の長方形を縁の法線の向きに伸ばした所）の中か。高さは見ない。"""
        q = self.local(p)
        return bool(abs(q[0]) <= self.inner[0] - margin and abs(q[1]) <= self.inner[1] - margin)

    def slot(self, i: int, n: int = 2) -> np.ndarray:
        """内寸を n×n に分けた置き場所の中心（机の座標、縁の平面の上）。並びは (+x+y, +x−y, −x+y, −x−y)。"""
        sx, sy = (1, 1, -1, -1)[i % 4], (1, -1, 1, -1)[i % 4]
        q = np.array([sx * self.inner[0] / n, sy * self.inner[1] / n, 0.0])
        return self.center + self.R @ q


# ------------------------------------------------------------------------------ 道具
def spacing(pts: np.ndarray) -> float:
    """点の間隔の目安（最近傍の距離の中央値）。"""
    if len(pts) < 2:
        return 0.0
    d, _ = cKDTree(pts).query(pts, k=2, workers=WORKERS)
    return float(np.median(d[:, 1]))


def voxel_down(pts: np.ndarray, v: float) -> np.ndarray:
    """一辺 v の格子ごとに点を平均する（密なカメラとまばらなカメラの点の数をそろえ、後の計算を軽くする）。"""
    if len(pts) == 0 or v <= 0:
        return pts
    key = np.floor(pts / v).astype(np.int64)
    key -= key.min(axis=0)
    span = key.max(axis=0) + 1
    lin = (key[:, 0] * span[1] + key[:, 1]) * span[2] + key[:, 2]     # 行の辞書順と同じ順の 1 つの整数
    _, inv, cnt = np.unique(lin, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    out = np.zeros((len(cnt), 3))
    np.add.at(out, inv, pts)
    return out / cnt[:, None]


def clean_points(pts: np.ndarray, k: int = 8, k_r: float = 1.5, min_nb: int = 4) -> np.ndarray:
    """まばらな外れ点（深度の境目の飛んだ画素・自己除去の残り）を外す印。k 番目に近い点までの距離の中央値の k_r 倍の半径に、
    自分のほかに min_nb 点以上ある点だけ残す（格子に並ぶ描画の点でも、ばらばらに並ぶ点でも、面の上の点はほぼ残る）。
    1 台のカメラの点に使う（カメラごとに点の間隔が違うため）。"""
    n = len(pts)
    if n < k + 2:
        return np.ones(n, bool)
    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=k + 1, workers=WORKERS)
    r = k_r * max(float(np.median(d[:, -1])), 1e-6)
    if min_nb > k:
        cnt = tree.query_ball_point(pts, r, return_length=True)
        return np.asarray(cnt) - 1 >= min_nb
    # 半径 r に自分のほかに min_nb 点以上 ⇔ (min_nb + 1) 番目に近い点（自分を含む）までの距離 ≤ r。境目の丸めの近くだけ数え直す
    dm = d[:, min_nb]
    out = dm <= r
    near = np.flatnonzero(np.abs(dm - r) <= 1e-9 * r)
    if len(near):
        out[near] = np.asarray(tree.query_ball_point(pts[near], r, return_length=True)) - 1 >= min_nb
    return out


def largest_cluster(pts: np.ndarray, eps: float = None) -> np.ndarray:
    """距離 eps でつながる点の、いちばん大きいアクションチャンクの印。eps を省くと、最近傍の距離の中央値の 3 倍。"""
    n = len(pts)
    if n < 3:
        return np.ones(n, bool)
    tree = cKDTree(pts)
    if eps is None:
        d, _ = tree.query(pts, k=2)
        eps = 3.0 * float(np.median(d[:, 1]))
    pairs = tree.query_pairs(eps, output_type="ndarray")
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) if len(pairs) else coo_matrix((n, n))
    _, lab = connected_components(g, directed=False)
    return lab == np.bincount(lab).argmax()


def fit_plane(pts: np.ndarray, tol: float, rng, iters: int = 200):
    """RANSAC の平面 → (単位法線, 平面の点, 内点の印)。最後に内点で最小二乗に当てはめ直す。点が足りなければ None。"""
    n = len(pts)
    if n < 3:
        return None
    best, best_k = None, -1
    for _ in range(iters):
        s = pts[rng.choice(n, 3, replace=False)]
        v = np.cross(s[1] - s[0], s[2] - s[0])
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            continue
        v /= nv
        k = int(np.sum(np.abs((pts - s[0]) @ v) <= tol))
        if k > best_k:
            best, best_k = (v, s[0]), k
    if best is None:
        return None
    v, p = best
    inl = np.abs((pts - p) @ v) <= tol
    for _ in range(2):
        c = pts[inl].mean(axis=0)
        _, _, vt = np.linalg.svd(pts[inl] - c, full_matrices=False)
        v = vt[-1]
        inl = np.abs((pts - c) @ v) <= tol
    return v, c, inl


def local_normals(pts: np.ndarray, noise: float = NOISE_M, k: int = None) -> np.ndarray:
    """各点の近くの k 点の主成分の、いちばん小さい向き（その場の面の法線。向きの符号は決めない）。k を省くと、近くの範囲の
    半径が雑音の 4 倍ほどになる点の数（点の間隔から決める。12〜60）。"""
    tree = cKDTree(pts)
    if k is None:
        d, _ = tree.query(pts, k=min(2, len(pts)), workers=WORKERS)
        spacing = float(np.median(d[:, -1])) if len(pts) > 1 else noise
        k = int(np.clip(np.pi * (4.0 * noise / max(spacing, 1e-6)) ** 2, 12, 60))
    k = min(k, len(pts))
    _, idx = tree.query(pts, k=k, workers=WORKERS)
    cov = np.empty((len(pts), 3, 3))
    for a in range(0, len(pts), _CHUNK):                            # アクションチャンクごとに（キャッシュに載る大きさ。各点の計算は同じ）
        nb = pts[idx[a:a + _CHUNK]]
        nb = nb - nb.mean(axis=1, keepdims=True)
        cov[a:a + _CHUNK] = np.einsum("nki,nkj->nij", nb, nb)
    return np.linalg.eigh(cov)[1][:, :, 0]


_CHUNK = 2048


def find_faces(pts: np.ndarray, ln: np.ndarray, tol: float, rng, min_count: int, max_faces: int = 6,
               cos_member: float = math.cos(math.radians(25.0))) -> list:
    """その場の法線（外向きに揃える）のまとまりから面を探す → [(外向きの法線, 面の点, 点の数)]（多い順）。各面は、まとまりの点の法線の向きの位置の中央値から tol の中で
    最小二乗に当てはめ直す。小さな物に RANSAC の平面を当てると、雑音の許しが面の幅に比べて広く、2 つの面にまたがる。"""
    c0 = pts.mean(axis=0)
    nn = ln * np.sign(np.einsum("ij,ij->i", pts - c0, ln))[:, None]
    nn[np.all(nn == 0, axis=1)] = ln[np.all(nn == 0, axis=1)]
    free = np.ones(len(pts), bool)
    out = []
    for _ in range(2 * max_faces):
        idx = np.flatnonzero(free)
        if len(idx) < min_count or len(out) >= max_faces:
            break
        cand = idx[rng.choice(len(idx), min(200, len(idx)), replace=False)]
        best = cand[int(np.argmax((nn[cand] @ nn[idx].T > cos_member).sum(axis=1)))]
        v = nn[best]
        for _ in range(3):
            mem = idx[nn[idx] @ v > cos_member]
            v = nn[mem].mean(axis=0)
            v /= np.linalg.norm(v)
        off = float(np.median(pts[mem] @ v))
        on = mem[np.abs(pts[mem] @ v - off) <= tol]
        free[mem] = False
        if len(on) < min_count:
            continue
        c = pts[on].mean(axis=0)
        w = np.linalg.svd(pts[on] - c, full_matrices=False)[2][-1]
        w = w if w @ v >= 0 else -w
        out.append((w, c, int(len(on))))
    return sorted(out, key=lambda x: -x[2])


def _basis_from(a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
    a1 = a1 / np.linalg.norm(a1)
    a2 = a2 - (a2 @ a1) * a1
    a2 = a2 / np.linalg.norm(a2)
    return np.column_stack([a1, a2, np.cross(a1, a2)])


def _rect_dir(uv: np.ndarray) -> float:
    """2 次元の点の最小外接長方形の 1 辺の向き [rad]。"""
    (_, _), (_, _), ang = cv2.minAreaRect(uv.astype(np.float32))
    return math.radians(ang)


def _up_last(R: np.ndarray) -> np.ndarray:
    """3 列目を机の上向き（+z）にいちばん近い軸にし、上向きに揃え、右手系にする。"""
    k = int(np.argmax(np.abs(R[2, :])))
    z = R[:, k] * (1.0 if R[2, k] >= 0 else -1.0)
    x = R[:, (k + 1) % 3]
    x = x - (x @ z) * z
    x = x / np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


# ------------------------------------------------------------------------------ 物
def _corners(lo, hi) -> np.ndarray:
    return np.array([[a, b, c] for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for c in (lo[2], hi[2])])


def fit_object(pts: np.ndarray, noise: float = NOISE_M, rng=None, min_points: int = 20,
               support_h: float = 0.0, assume_supported: bool = False) -> Optional[ObjGeom]:
    """物の点（机の座標）→ ObjGeom。点が足りなければ None。support_h は下で支える面の高さ（机なら 0、箱の中なら推定の底）。
    assume_supported: 手に持たれていない物は宙に浮かない（重力）とみなし、見えない下の側を必ず支える面まで延ばす。上の面しか
    写らない（側面が陰で色の切り出しから外れる）と、隙間からは置かれているか決められず、薄い板が宙にあると推定してしまう。"""
    pts = np.asarray(pts, float)
    if len(pts) < min_points:
        return None
    rng = np.random.default_rng(0) if rng is None else rng
    tol = 3.0 * noise
    faces = find_faces(pts, local_normals(pts, noise), tol, rng, min_count=max(8, int(0.06 * len(pts))))
    normals = []
    for v, _, k in faces:                                           # 軸にするのは前の面とほぼ直交する面（向かい合う面は増やさない）
        if all(abs(v @ u) < 0.35 for u, _ in normals):
            normals.append((v, k))
    normals.sort(key=lambda x: -x[1])
    if len(normals) >= 2:
        R = _basis_from(normals[0][0], normals[1][0])
    else:
        a1 = normals[0][0] if normals else np.linalg.svd(pts - pts.mean(0), full_matrices=False)[2][-1]
        R0 = _basis_from(a1, np.array([1.0, 0.0, 0.0]) if abs(a1[0]) < 0.9 else np.array([0.0, 1.0, 0.0]))
        uv = (pts - pts.mean(0)) @ R0[:, 1:]
        th = _rect_dir(uv)
        a2 = math.cos(th) * R0[:, 1] + math.sin(th) * R0[:, 2]
        R = _basis_from(a1, a2)
    R = _up_last(R)
    m = pts.mean(axis=0)
    q = (pts - m) @ R
    lo, hi = np.percentile(q, 0.5, axis=0), np.percentile(q, 99.5, axis=0)
    for v, c, _ in faces:                                           # 当てはめた面がある端は、その面の位置にする（雑音の幅を足さない）
        i = int(np.argmax(np.abs(R.T @ v)))
        if abs(R[:, i] @ v) > 0.9:
            o = float((c - m) @ R[:, i])
            if o > 0:
                hi[i] = o
            else:
                lo[i] = o
    ext = hi - lo
    corners = _corners(lo, hi) @ R.T + m
    h_min = float(corners[:, 2].min()) - float(support_h)
    supported = bool(assume_supported) or h_min <= max(3.0 * noise, 0.5 * float(ext.min()))
    extended = 0.0
    if supported and h_min > 0.5 * noise:
        k = int(np.argmax(np.abs(R[2, :])))
        d = h_min / abs(R[2, k])
        if R[2, k] > 0:
            lo[k] -= d
        else:
            hi[k] += d
        extended = float(d)
        ext = hi - lo
    ax = np.zeros(3)
    ax[int(np.argmax(np.abs(R[2, :])))] = extended
    es = 2.0 * noise + ax / math.sqrt(3.0)
    cs = es / 2.0 + noise / math.sqrt(len(pts))
    center = m + R @ ((lo + hi) / 2.0)
    top = float((_corners(lo, hi) @ R.T + m)[:, 2].max())
    if ext[1] > ext[0]:                                             # 机に沿う 2 軸のうち長い方を x にする（右手系のまま）
        R = np.column_stack([R[:, 1], -R[:, 0], R[:, 2]])
        ext, es, cs = ext[[1, 0, 2]], es[[1, 0, 2]], cs[[1, 0, 2]]
    return ObjGeom(center, R, ext, top, R @ np.diag(cs ** 2) @ R.T, es, len(pts), len(normals), bool(supported), extended)


# ------------------------------------------------------------------------------ 箱らしい物
def fit_container(pts: np.ndarray, noise: float = NOISE_M, rng=None, min_points: int = 60,
                  max_rim_tilt_deg: float = 30.0) -> Optional[BoxGeom]:
    """箱らしい物の点（机の座標）→ BoxGeom。縁が見つからなければ None。"""
    pts = np.asarray(pts, float)
    if len(pts) < min_points:
        return None
    rng = np.random.default_rng(0) if rng is None else rng
    tol = 3.0 * noise
    ln = local_normals(pts, noise)
    f = _rim_plane(pts, ln, tol, rng, max_rim_tilt_deg)
    if f is None:
        return None
    nrm = f[0]
    rim = (np.abs((pts - f[1]) @ nrm) <= tol) & (np.abs(ln @ nrm) > 0.8)
    for _ in range(2):                                              # 全点から縁（平面の近くで、その場の面が縁と同じ向き）を取り直す
        if rim.sum() < 3:
            return None
        c = pts[rim].mean(axis=0)
        _, _, vt = np.linalg.svd(pts[rim] - c, full_matrices=False)
        nrm = vt[-1] if vt[-1][2] >= 0 else -vt[-1]
        rim = (np.abs((pts - c) @ nrm) <= tol) & (np.abs(ln @ nrm) > 0.8)
    if rim.sum() < 20:
        return None
    B = _basis_from(nrm, np.array([1.0, 0.0, 0.0]))                 # 列: 法線・e1・e2
    e1, e2 = B[:, 1], B[:, 2]
    uv = np.column_stack([(pts - c) @ e1, (pts - c) @ e2])
    th = _rect_dir(uv)
    ux, uy = np.array([math.cos(th), math.sin(th)]), np.array([-math.sin(th), math.cos(th)])
    x, y = uv @ ux, uv @ uy
    lx, hx = np.percentile(x, [0.5, 99.5])
    ly, hy = np.percentile(y, [0.5, 99.5])
    half = np.array([(hx - lx) / 2.0, (hy - ly) / 2.0])
    mid = np.array([(hx + lx) / 2.0, (hy + ly) / 2.0])
    if half[1] > half[0]:                                           # 長い辺を x にする
        ux, uy = uy, -ux
        half, mid = half[::-1].copy(), np.array([mid[1], -mid[0]])
        x, y = uv @ ux, uv @ uy
    xr, yr = x[rim] - mid[0], y[rim] - mid[1]
    d = np.clip(np.minimum(half[0] - np.abs(xr), half[1] - np.abs(yr)), 0.0, None)
    wall0 = float(min(2.0 * np.median(d), 0.9 * half.min()))
    # 辺ごとに縁の点を分け、壁の厚さの中に一様に並ぶことから外の縁・内の縁を求める（平均 ± √3 × 標準偏差。雑音の分を引く）。
    # 百分位だけでは、外の面が見えない奥の辺が縮む
    edges = _rim_edges(xr, yr, half, wall0, noise)
    lo_x, hi_x = edges["-x"], edges["+x"]
    lo_y, hi_y = edges["-y"], edges["+y"]
    outer_lo = np.array([lo_x[0] if lo_x else -half[0], lo_y[0] if lo_y else -half[1]])
    outer_hi = np.array([hi_x[0] if hi_x else half[0], hi_y[0] if hi_y else half[1]])
    walls = [e[1] for e in (lo_x, hi_x, lo_y, hi_y) if e]
    wall = float(np.median(walls)) if walls else wall0
    shift = (outer_lo + outer_hi) / 2.0
    mid = mid + shift
    half = (outer_hi - outer_lo) / 2.0
    d = np.clip(np.minimum(half[0] - np.abs(xr - shift[0]), half[1] - np.abs(yr - shift[1])), 0.0, None)
    inner = half - wall
    if np.any(inner <= 0.0) or wall > 0.5 * float(half.min()):     # 中が無い・壁が厚すぎる: 箱らしくない当てはめ
        return None
    ex = ux[0] * e1 + ux[1] * e2
    ey = uy[0] * e1 + uy[1] * e2
    R = np.column_stack([ex, ey, nrm])
    center = c + mid[0] * ex + mid[1] * ey
    q = (pts - center) @ R
    ins = (np.abs(q[:, 0]) <= inner[0] - 0.5 * wall) & (np.abs(q[:, 1]) <= inner[1] - 0.5 * wall) & (q[:, 2] < -tol)
    table_local = -float(center[2]) / max(nrm[2], 1e-6)            # 机までの、縁の法線の向きの距離（負）
    floor = float(np.median(q[ins, 2])) if ins.sum() >= 10 else table_local
    floor = max(floor, table_local)
    u = np.array([ex[0], ex[1], 0.0])
    u /= max(np.linalg.norm(u), 1e-12)
    w = np.cross([0.0, 0.0, 1.0], u)
    roll = math.atan2(-(nrm @ w), nrm[2])
    pitch = math.atan2(nrm @ u, nrm[2])
    n_rim = int(rim.sum())
    sig = {"center": float(noise), "yaw": float(noise / max(half.min(), 1e-6)),
           "inner": float(noise + 2.0 * np.std(d) / math.sqrt(n_rim)), "rim_h": float(noise),
           "tilt": float(noise / max(half.min(), 1e-6))}
    return BoxGeom(center, R, half, inner, wall, float(center[2]), floor, float(roll), float(pitch), sig, len(pts), n_rim,
                   len(walls))


def _rim_plane(pts, ln, tol, rng, max_tilt_deg: float, max_planes: int = 6):
    """縁の平面: その場の面が上を向く点（机の法線から max_tilt_deg 以内）から平面を順に取り出し、点の数が足りるもののうち
    いちばん高いもの（底・机に置いた物の上面より上）。高い方の何割かの点から探すと、飛んだ画素や自己除去の残りの高い点に
    引かれて壁の面を拾う。→ (上向きの法線, 平面の点) か None。"""
    cmax = math.cos(math.radians(max_tilt_deg))
    span = np.ptp(pts[:, :2], axis=0)

    def on_top(v, c) -> bool:
        """縁はいちばん上の面: 平面より 2 tol 以上高い点がほぼ無い（max(10 点, 0.5%) 以下）。手首のカメラが箱の中を
        近くから見ると点の大半が底で、縁の点は 3% ほどしかなく、高い方の 10% から底の平面を拾うことがある（傾いた箱）。"""
        return int(np.sum((pts - c) @ v > 2.0 * tol)) <= max(10, int(0.005 * len(pts)))

    top = pts[pts[:, 2] >= np.percentile(pts[:, 2], 90)]             # まず高い方の 10% から（壁が薄く雑音が大きいときに強い）
    f = fit_plane(top, tol, rng)
    if f is not None:
        v = f[0] if f[0][2] >= 0 else -f[0]
        on = np.abs((pts - f[1]) @ v) <= tol
        if v[2] >= cmax and np.all(np.ptp(pts[on, :2], axis=0) >= 0.5 * span) and on_top(v, f[1]):
            return v, f[1]
    up = np.flatnonzero(np.abs(ln[:, 2]) >= cmax)                   # 囲まない・傾きすぎなら、上を向く面から順に
    min_k = max(20, int(0.02 * len(up)))
    free = np.ones(len(up), bool)
    planes = []
    for _ in range(max_planes):
        idx = up[free]
        if len(idx) < min_k:
            break
        f = fit_plane(pts[idx], tol, rng, iters=300)
        if f is None:
            break
        v, c, inl = f
        v = v if v[2] >= 0 else -v
        free[np.flatnonzero(free)[inl]] = False
        if inl.sum() < min_k or v[2] < cmax:
            continue
        if np.any(np.ptp(pts[idx[inl], :2], axis=0) < 0.5 * span):   # 縁は全体を囲む（外れの小さなアクションチャンクを縁にしない）
            continue
        if not on_top(v, c):
            continue
        planes.append((v, c, int(inl.sum())))
    if not planes:
        return None
    v, c, _ = max(planes, key=lambda p: p[1][2])
    return v, c


def _rim_edges(xr, yr, half, wall0, noise) -> dict:
    """縁の点（外寸の長方形の座標）を 4 つの辺に分け、辺ごとに (外の縁, 壁の厚さ) を返す（点が足りない辺は None）。
    辺の向きの座標は [内の縁, 外の縁] に一様（雑音）なので、外の縁 = 平均 ＋ √3 × s、厚さ = 2√3 × s（s² = 分散 − 雑音²）。"""
    out = {}
    band = 2.0 * wall0 + 3.0 * noise
    for key, c, other, h, ho in (("+x", xr, yr, half[0], half[1]), ("-x", -xr, yr, half[0], half[1]),
                                 ("+y", yr, xr, half[1], half[0]), ("-y", -yr, xr, half[1], half[0])):
        sel = (c > h - band) & (np.abs(other) < ho - band)              # その辺の縁（角は 2 つの辺にまたがるので外す）
        v = c[sel]
        if len(v) < 15:
            out[key] = None
            continue
        med = np.median(v)
        mad = 1.4826 * np.median(np.abs(v - med)) + 1e-9
        v = v[np.abs(v - med) <= 3.0 * mad + noise]
        sd = math.sqrt(max(float(np.var(v)) - (0.5 * noise) ** 2, 0.0))
        edge = float(np.mean(v)) + math.sqrt(3.0) * sd
        out[key] = (edge if key[0] == "+" else -edge, 2.0 * math.sqrt(3.0) * sd)
    return out


# ------------------------------------------------------------------------------ 机の座標
@dataclasses.dataclass(frozen=True)
class TableFrame:
    """机の座標（原点は机の平面の上、z は机の法線）。世界 ← 机: p = origin + R @ q。"""
    origin: np.ndarray
    R: np.ndarray

    @classmethod
    def from_plane(cls, normal, table_z: float) -> "TableFrame":
        n = np.asarray(normal, float)
        n = n / np.linalg.norm(n)
        R = _basis_from(n, np.array([1.0, 0.0, 0.0]))
        return cls(np.array([0.0, 0.0, float(table_z)]), np.column_stack([R[:, 1], R[:, 2], n]))

    def to_table(self, p) -> np.ndarray:
        return (np.asarray(p, float) - self.origin) @ self.R

    def to_world(self, q) -> np.ndarray:
        return np.asarray(q, float) @ self.R.T + self.origin
