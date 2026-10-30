"""点群からのカップと皿の形の推定（タスク「カップを皿に」）。純関数（numpy・scipy）。

使ってよい物の種類の知識は「カップ＝上が開いた、立った円筒」「皿＝机の上の平らな円板」だけ。大きさ・位置・向き・高さは点群から
推定する（既知の半径・高さ・厚さを読まない）。点は机の座標（机 = z 0、上向き +z。api/geometry と同じ）で受ける。
使う既知の値は机の平面（較正の値）と深度の雑音の目安 noise（api/geometry.NOISE_M）だけ。

    c = fit_cylinder(pts)      # 立った円筒: 中心・外の半径・高さ・上端の高さ・上が開いているか（CylGeom）
    p = fit_plate(pts)         # 平らな円板: 上面の中心・法線（傾き）・半径（PlateGeom）
    p.depth(q)                 # 上面の円の柱（上面より上）からの深さ（正なら中、負なら外）。BoxGeom.depth と同じ口
    p.on_top(bottom, ...)      # 物の底が上面に載って円の内側か

立った円筒の決め方:
  1. 上端: 高さの 99.5 百分位の近く（雑音の 3 倍）の点を縁とし、その高さの中央値を上端にする
  2. 中心: 縁の点の水平の位置に円を当てはめる（代数の当てはめ。縁は上から全周が見える）。手前の壁だけ・奥の壁だけが見えても
     ずれないように、縁の点が少なければ全点に当てはめる
  3. 外の半径: 中心からの水平の距離の、いちばん外のアクションチャンク（95 百分位 − 雑音の 4 倍より外）の中央値（外の壁の面）
  4. 下端: 置かれている（下で支える面との隙間が高さの半分以下、または assume_supported）なら支える面（机、皿の上なら推定の
     上面）まで延ばす（api/geometry.fit_object と同じ考え。延ばした量は不確かさに足す）。宙の物は見えたいちばん下
  5. 上が開いているか: 中心のそば（外の半径の 0.6 倍の内側）に、上端の近く（上端から高さの 1/4 以内）の点がほとんど無い
     （面の密度が、縁の輪（外の半径の 0.8〜1 倍の上端の点）の 1/4 未満）。開いたカップを上（斜め上）から見ると、中心の上には
     面が無く、見えるのは低い内の底か奥の内壁。倒れたカップ（ふたのある物）は中心の上が上の面で覆われる
平らな円板の決め方:
  1. 上面: 高い方の半分の点から平面を RANSAC で探し、平面の近くの点で当てはめ直す（法線 = 傾き）
  2. 縁: 上面の座標で、仮の中心のまわりを 10° ごとの扇に分け、扇ごとのいちばん外を縁の標本にする（面に一様な点の半径の
     98 百分位は 半径 × √0.98 なので、それで割る）。円を当てはめ、内に大きく外れた扇（腕・上の物に隠れた所）を外して当てはめ
     直す。中心と半径
"""
import dataclasses
import math
from typing import Optional

import numpy as np

from recovla.api import geometry as G

OPEN_FRAC = 0.25                    # 上が開いている: 上端から 高さ × これ 以内の点を「上の面」とみなす
OPEN_R_FRAC = 0.6                   # 同じく、中心から 外の半径 × これ の内側を「中心のそば」とみなす
OPEN_DENSITY = 0.25                 # 同じく、中心のそばの上の面の点の面の密度が、縁の輪の密度 × これ 未満なら開いている
OPEN_MIN = 8                        # 縁の輪の点がこれ未満なら決めない（開いているとみなす）
EDGE_Q = 98.0                       # 皿の縁の標本: 扇の中の半径の百分位（面に一様なら 半径 × √(q/100)）


@dataclasses.dataclass(frozen=True)
class CylGeom:
    center: np.ndarray              # (3,) 机の座標（軸の上の、上端と下端の中点）
    radius: float                   # 外の半径
    height: float                   # 下端から上端まで
    top_h: float                    # 上端（縁）の机からの高さ
    bottom_h: float                 # 下端の机からの高さ
    open_top: bool                  # 上が開いている（立っている）と見えた
    cov: np.ndarray                 # (3, 3) 中心の共分散（目安）
    radius_sigma: float
    height_sigma: float
    n: int
    supported: bool
    extended: float                 # 支える面まで延ばした量 [m]

    @property
    def extent(self) -> np.ndarray:
        """向きつきの外接箱の 3 辺（ObjState.extent と同じ口。立った円筒なので R は机の座標の単位行列）。"""
        return np.array([2.0 * self.radius, 2.0 * self.radius, self.height])

    @property
    def R(self) -> np.ndarray:
        return np.eye(3)

    @property
    def sigma(self) -> float:
        return float(math.sqrt(max(np.linalg.eigvalsh(self.cov).max(), 0.0)))


@dataclasses.dataclass(frozen=True)
class PlateGeom:
    center: np.ndarray              # (3,) 上面の円の中心
    R: np.ndarray                   # (3, 3) 列: x・y（上面の中）・z（上面の法線、上向き）
    radius: float
    top_h: float                    # 上面の中心の机からの高さ
    roll: float                     # 法線の、机に対する x 軸まわりの傾き [rad]
    pitch: float                    # 同じく y 軸まわり [rad]
    sigma: dict                     # {"center", "radius", "top", "tilt"} の標準偏差（目安）
    n: int
    n_edges: int                    # 縁の標本に使えた扇の数（36 のうち。隠れると減る）
    kind: str = "surface"

    @property
    def yaw(self) -> float:
        return 0.0                  # 円は向きを持たない

    @property
    def tilt(self) -> float:
        return float(math.acos(min(1.0, abs(self.R[2, 2]))))

    def local(self, p) -> np.ndarray:
        """点 → 上面の座標（原点は上面の中心、z は上面の法線）。"""
        return (np.asarray(p, float) - self.center) @ self.R

    def depth(self, p) -> float:
        """上面の円の柱（半径の内側で、上面より上）からの深さ。正なら中（近い方の面までの距離）、負なら外。"""
        q = self.local(p)
        d = np.array([self.radius - math.hypot(q[0], q[1]), q[2]])
        return float(d.min()) if d.min() >= 0 else float(-np.linalg.norm(np.minimum(d, 0.0)))

    def over_xy(self, p, margin: float = 0.0) -> bool:
        q = self.local(p)
        return bool(math.hypot(q[0], q[1]) <= self.radius - margin)

    def slot(self, i: int = 0, n: int = 2) -> np.ndarray:
        """置き場所の最初の候補（上面の中心）。スクリプト方策は地図で空いている所を選び直す（skills_script_self._surface_spot）。"""
        return self.center.copy()

    def on_top(self, bottom, margin: float, tol: float) -> dict:
        """物の底の中心 bottom が、上面の円の内側（半径 − margin）で、上面から tol 以内の高さか。"""
        q = self.local(bottom)
        rho = math.hypot(q[0], q[1])
        return {"on": bool(rho <= self.radius - margin and abs(q[2]) <= tol), "rho": rho, "height": float(q[2])}


# ------------------------------------------------------------------------------ 円の当てはめ
def fit_circle(xy: np.ndarray) -> Optional[tuple]:
    """代数の円の当てはめ（Kasa）→ (中心 (2,), 半径)。点が 3 つ未満・一直線なら None。"""
    xy = np.asarray(xy, float)
    if len(xy) < 3:
        return None
    A = np.column_stack([2.0 * xy, np.ones(len(xy))])
    b = (xy ** 2).sum(axis=1)
    sol, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    if rank < 3:
        return None
    c = sol[:2]
    r2 = float(sol[2] + c @ c)
    return (c, math.sqrt(r2)) if r2 > 0 else None


# ------------------------------------------------------------------------------ 立った円筒（カップ）
def fit_cylinder(pts: np.ndarray, noise: float = G.NOISE_M, rng=None, min_points: int = 20, support_h: float = 0.0,
                 assume_supported: bool = False, held: bool = False) -> Optional[CylGeom]:
    """物の点（机の座標）→ CylGeom。点が足りない・円が当てはまらなければ None。support_h は下で支える面の高さ
    （机なら 0、皿の上なら推定の上面）。assume_supported は api/geometry.fit_object と同じ（手に持たれていない物は宙に浮かない）。
    held（手が把持を報告していて指先のそば）なら延ばさない（高い物は持ち上げた直後も隙間が高さの半分より小さく、置かれていると
    見誤るため）。どちらでもなければ、隙間が 外径・見えた高さの短い方の半分以下なら置かれているとみなす（fit_object と同じ考え）。"""
    pts = np.asarray(pts, float)
    if len(pts) < min_points:
        return None
    z = pts[:, 2]
    band = 3.0 * noise
    top = float(np.percentile(z, 99.5))
    rim = pts[z >= top - band]
    top_h = float(np.median(rim[:, 2]))
    src = rim if len(rim) >= 12 else pts
    f = fit_circle(src[:, :2])
    if f is None:
        return None
    c = f[0]
    for _ in range(2):                                              # 縁から大きく外れた点（中の底の高い点など）を外して当てはめ直す
        rho = np.linalg.norm(src[:, :2] - c, axis=1)
        keep = np.abs(rho - np.median(rho)) <= max(3.0 * noise, 0.25 * float(np.median(rho)))
        if keep.sum() < 3:
            break
        f2 = fit_circle(src[keep, :2])
        if f2 is None:
            break
        c = f2[0]
    rho = np.linalg.norm(pts[:, :2] - c, axis=1)
    outer = rho[rho >= np.percentile(rho, 95) - 4.0 * noise]
    radius = float(np.median(outer))
    low = float(np.percentile(z, 0.5))
    h_seen = top_h - low
    gap = low - float(support_h)
    supported = (not held) and (bool(assume_supported) or gap <= max(3.0 * noise, 0.5 * min(2.0 * radius, h_seen)))
    extended = 0.0
    bottom = low
    if supported and gap > 0.5 * noise:
        extended, bottom = float(gap), float(support_h)
    height = top_h - bottom
    high = z >= top_h - OPEN_FRAC * height
    ring = (z >= top_h - band) & (rho >= 0.8 * radius) & (rho <= radius + band)
    n_mid = int(np.sum(high & (rho <= OPEN_R_FRAC * radius)))
    if ring.sum() < OPEN_MIN:
        open_top = True
    else:
        d_ring = ring.sum() / (np.pi * ((radius + band) ** 2 - (0.8 * radius) ** 2))
        open_top = bool(n_mid / (np.pi * (OPEN_R_FRAC * radius) ** 2) < OPEN_DENSITY * d_ring)
    n_rim = max(len(rim), 3)
    cs = np.array([noise / math.sqrt(n_rim)] * 2 + [0.5 * (noise + extended / math.sqrt(3.0))]) + noise / math.sqrt(len(pts))
    center = np.array([c[0], c[1], 0.5 * (top_h + bottom)])
    return CylGeom(center, radius, float(height), top_h, float(bottom), open_top, np.diag(cs ** 2),
                   float(noise / math.sqrt(max(len(outer), 1)) + 0.5 * noise), float(noise + extended / math.sqrt(3.0)),
                   len(pts), supported, extended)


# ------------------------------------------------------------------------------ 平らな円板（皿）
def fit_plate(pts: np.ndarray, noise: float = G.NOISE_M, rng=None, min_points: int = 60, max_tilt_deg: float = 30.0,
              n_bins: int = 36) -> Optional[PlateGeom]:
    """皿らしい物の点（机の座標）→ PlateGeom。上面か縁が見つからなければ None。"""
    pts = np.asarray(pts, float)
    if len(pts) < min_points:
        return None
    rng = np.random.default_rng(0) if rng is None else rng
    tol = 3.0 * noise
    hi = pts[pts[:, 2] >= np.percentile(pts[:, 2], 50)]
    pl = G.fit_plane(hi if len(hi) >= 30 else pts, tol, rng)
    if pl is None:
        return None
    nrm, c0, _ = pl
    nrm = nrm if nrm[2] >= 0 else -nrm
    if math.degrees(math.acos(min(1.0, abs(float(nrm[2]))))) > max_tilt_deg:
        return None
    on = np.abs((pts - c0) @ nrm) <= tol                            # 上面の点（平面の近く）で当てはめ直す
    if on.sum() < 30:
        return None
    c0 = pts[on].mean(axis=0)
    nrm = np.linalg.svd(pts[on] - c0, full_matrices=False)[2][-1]
    nrm = nrm if nrm[2] >= 0 else -nrm
    R = G._basis_from(nrm, np.array([1.0, 0.0, 0.0]))
    R = np.column_stack([R[:, 1], R[:, 2], nrm])                    # x・y は上面の中、z は法線
    uv = (pts[on] - c0) @ R[:, :2]
    cen = np.median(uv, axis=0)
    edge = []
    for _ in range(2):                                              # 扇ごとのいちばん外 → 円。中心が動いたら扇を作り直す
        d = uv - cen
        ang = np.arctan2(d[:, 1], d[:, 0])
        rr = np.hypot(d[:, 0], d[:, 1])
        k = np.floor((ang + math.pi) / (2 * math.pi) * n_bins).astype(int) % n_bins
        edge = []
        for b in range(n_bins):
            m = k == b
            if m.sum() >= 3:
                re = float(np.percentile(rr[m], EDGE_Q)) / math.sqrt(EDGE_Q / 100.0)
                a = (b + 0.5) / n_bins * 2 * math.pi - math.pi
                edge.append(cen + re * np.array([math.cos(a), math.sin(a)]))
        edge = np.array(edge)
        if len(edge) < 6:
            return None
        f = fit_circle(edge)
        if f is None:
            return None
        for _ in range(3):                                          # 内に外れた扇（隠れた縁）を外す
            res = np.linalg.norm(edge - f[0], axis=1) - f[1]
            keep = res >= -max(4.0 * noise, 0.05 * f[1])
            if keep.all() or keep.sum() < 6:
                break
            edge = edge[keep]
            f = fit_circle(edge) or f
        cen = f[0]
    radius = float(f[1])
    center = c0 + R[:, :2] @ cen
    roll = float(math.atan2(-nrm[1], nrm[2]))
    pitch = float(math.atan2(nrm[0], nrm[2]))
    n_e = len(edge)
    res = np.linalg.norm(edge - cen, axis=1) - radius
    sr = float(np.std(res)) if n_e > 1 else noise
    sig = {"center": float(math.hypot(noise, sr) / math.sqrt(max(n_e, 1)) + noise / math.sqrt(int(on.sum()))),
           "radius": float(sr + noise), "top": float(noise / math.sqrt(int(on.sum())) + 0.5 * noise),
           "tilt": float(noise / max(radius, 1e-3))}
    return PlateGeom(center, R, radius, float(center[2]), roll, pitch, sig, len(pts), int(n_e))


def to_world(g, frame: G.TableFrame):
    """机の座標の CylGeom・PlateGeom → 世界の座標（中心と向きだけ変わる）。"""
    if isinstance(g, PlateGeom):
        return dataclasses.replace(g, center=frame.to_world(g.center), R=frame.R @ g.R)
    return dataclasses.replace(g, center=frame.to_world(g.center), cov=frame.R @ g.cov @ frame.R.T)
