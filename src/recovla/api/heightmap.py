"""手で書いたスクリプト方策（条件 CL）が使う高さの地図と経路の決まり。

    hm = HeightMap(frame)                         # frame: api.geometry.TableFrame（机の座標）
    hm.update(clouds, t, shift, exclude)          # 点群（Perception.points の出力、カメラごと）から、升ごとのいちばん高い点
    hm.band_max(a, b, r)                          # 線分 a→b（机の座標の水平）から半径 r のシード範囲の中の最大の高さ
    prof = gripper_profile(model, tip_offset)     # ロボットのモデルから、指先の中心に対する手と指の形（半径・下端）
    h = pass_tip_height(hm, a, b, prof, held)     # そのシード範囲を通る指先の高さ（机からの高さ）
    goal, stage = path_goal(x, goal, pass_z, tol) # ①真上へ通る高さまで ②水平に ③下ろす

地図: 机の座標の 2 次元の格子（升の幅はまばらな所の点の間隔 × map_cell_k、最初の更新で決める）。更新ごとに、点が 2 つ以上
あった升は「その升の 2 番目に高い点」（飛んだ画素の 1 点で上げない）に置き換え、点の無い升（腕に隠れた・写らない）は前の値を保つ。机の点も入れる（物が無くなった升を下げるため）。
ロボットの点は知覚の自己除去で除かれている。持っている物の点（指先のそば）は exclude で外す。
通る高さ: 手と指の形の各部（半径 r、指先の中心に対する下端の高さ l）について「シード範囲の最大 − l」の最大に、持っている物（半径・
下端 = 見かけの高さの半分の下）を足して、余白（地図の σ × map_unc_k）を加える。箱の壁・ほかの物を越える高さはここで決まる。
固定の高さ（approach_height・carry_z・release_z）・待機位置は使わない。
"""
import dataclasses
import math
from typing import Optional

import numpy as np

from recovla.api import geometry as G

GRIPPER_BODIES = ("hand", "left_finger", "right_finger")


@dataclasses.dataclass(frozen=True)
class Part:
    radius: float                   # 手先の軸からの水平の半径（手の向きによらない円で見る）
    low: float                      # 指先の中心に対する下端の高さ（負なら指先より下）


def gripper_profile(model, tip_offset: float, width: float = 0.08, bodies=GRIPPER_BODIES) -> tuple:
    """ロボットのモデル（robot_only.xml、runtime/robot_model.load）の手と指の形 → (Part, ...)。指は開き幅 width のとき。
    手の座標の z は指先の向き（指先の中心 = 手の原点 ＋ tip_offset × z）。各部は geom の外接箱の 8 隅から。
    bodies を ("hand",) にすると手のひら（指を除く手）だけ（スクリプト方策の掴む高さの「手のひらが物の上端より上」に使う）。"""
    import mujoco
    d = mujoco.MjData(model)
    for n in ("finger_joint1", "finger_joint2"):
        d.qpos[model.joint(n).qposadr[0]] = 0.5 * float(width)
    mujoco.mj_kinematics(model, d)
    hid = model.body("hand").id
    hp, hR = d.xpos[hid].copy(), d.xmat[hid].reshape(3, 3).copy()
    bodies = {model.body(n).id for n in bodies}
    parts = []
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) not in bodies:
            continue
        c, h = model.geom_aabb[g][:3], model.geom_aabb[g][3:]
        corners = np.array([[a, b, e] for a in (-1, 1) for b in (-1, 1) for e in (-1, 1)]) * h + c
        w = corners @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        q = (w - hp) @ hR                                           # 手の座標
        parts.append(Part(float(np.max(np.hypot(q[:, 0], q[:, 1]))), float(tip_offset - np.max(q[:, 2]))))
    return simplify(parts)


def robot_boxes(model, data, skip=("world", "probe_mount", "target")) -> list:
    """今の姿勢（data は順運動学の後）のロボットの当たり判定の geom（見た目の細かい geom を包む）の外接箱 →
    [(中心, 回転, 半分の大きさ)]（世界の座標）。地図から自分の点を外すのに使う（画像の自己除去は較正の誤差のぶん縁が漏れる）。"""
    skip_ids = {model.body(n).id for n in skip if _has_body(model, n)}
    out = []
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) in skip_ids or int(model.geom_contype[g]) == 0:
            continue
        c, h = model.geom_aabb[g][:3], model.geom_aabb[g][3:]
        R = data.geom_xmat[g].reshape(3, 3).copy()
        out.append((R @ c + data.geom_xpos[g], R, h.copy()))
    return out


def _has_body(model, name) -> bool:
    try:
        model.body(name)
        return True
    except KeyError:
        return False


def depth_edges(depth, jump: float) -> np.ndarray:
    """深度の画像 (H, W) [m] の段差の縁（3×3 の中の最大 − 最小 ≥ jump）の印。縁の画素は近い側と遠い側の間の深さになりやすい
    （空飛ぶ画素）ので地図に入れない。欠け（0）のそばも縁とみなす。"""
    import cv2
    z = np.asarray(depth, np.float32)
    k = np.ones((3, 3), np.uint8)
    return (cv2.dilate(z, k) - cv2.erode(z, k)) >= float(jump)


def near_boxes(pw: np.ndarray, boxes, margin: float) -> np.ndarray:
    """点 (N, 3) のうち、どれかの箱を margin だけ太らせた内側にある点の印。"""
    pw = np.asarray(pw, float)
    hit = np.zeros(len(pw), bool)
    if not boxes or len(pw) == 0:
        return hit
    exts = [np.abs(R) @ h + margin for _, R, h in boxes]
    rects = [(c[:2] - e[:2] - _SLACK, c[:2] + e[:2] + _SLACK, c[2] - e[2] - _SLACK, c[2] + e[2] + _SLACK)
             for (c, _, _), e in zip(boxes, exts)]
    cand = coarse_candidates(pw, rects)                 # 粗い升で、どれかの箱の外接箱に入りうる点だけに絞る（昇順）
    if len(cand) == 0:
        return hit
    q = pw[cand]
    inside = np.zeros(len(cand), bool)
    for (c, R, h), ext in zip(boxes, exts):
        sub = np.flatnonzero(np.all(np.abs(q - c) <= ext, axis=1) & ~inside)   # その箱の世界の軸にそろえた外接箱で先に絞る
        if len(sub) == 0:
            continue
        loc = np.abs((q[sub] - c) @ R)
        inside[sub[np.all(loc <= h + margin, axis=1)]] = True
    hit[cand] = inside
    return hit


_SLACK = 1e-9                                           # 粗い升の範囲を少し広げる幅 [m]（丸めの差で点を落とさない）


def coarse_candidates(p: np.ndarray, rects, w: float = 0.01) -> np.ndarray:
    """点 (N, 3) のうち、どれかの範囲 (水平の下限 (2,), 上限 (2,), 高さの下限, 上限) に入りうる点の添字（昇順。入る点は
    必ず含む）。幅 w の水平の粗い升ごとに、そこに掛かる範囲の高さの下限の最小・上限の最大を持って比べる（細かい判定は呼び手）。"""
    if not rects or len(p) == 0:
        return np.zeros(0, np.int64)
    lo = np.min([r[0] for r in rects], axis=0)
    hi = np.max([r[1] for r in rects], axis=0)
    n = (np.floor((hi - lo) / w).astype(np.int64) + 1)
    zlo = np.full(n, np.inf)
    zhi = np.full(n, -np.inf)
    for a, b, z0, z1 in rects:
        ia, ja = np.floor((np.asarray(a) - lo) / w).astype(np.int64)
        ib, jb = np.floor((np.asarray(b) - lo) / w).astype(np.int64)
        np.minimum(zlo[ia:ib + 1, ja:jb + 1], z0, out=zlo[ia:ib + 1, ja:jb + 1])
        np.maximum(zhi[ia:ib + 1, ja:jb + 1], z1, out=zhi[ia:ib + 1, ja:jb + 1])
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    cand = np.flatnonzero((x >= lo[0]) & (x <= hi[0]) & (y >= lo[1]) & (y <= hi[1]) & (z >= zlo.min()) & (z <= zhi.max()))
    if len(cand) == 0:
        return cand
    i = np.minimum(np.floor((x[cand] - lo[0]) / w).astype(np.int64), n[0] - 1)
    j = np.minimum(np.floor((y[cand] - lo[1]) / w).astype(np.int64), n[1] - 1)
    zc = z[cand]
    return cand[(zc >= zlo[i, j]) & (zc <= zhi[i, j])]


def simplify(parts) -> tuple:
    """ほかの部に含まれる部（半径がそれ以下で、下端がそれ以上の高さ）を外す（通る高さは変わらず、シード範囲の計算が減る）。"""
    parts = sorted(parts, key=lambda p: (-p.radius, p.low))
    out = []
    for p in parts:
        if not any(q.radius >= p.radius and q.low <= p.low for q in out):
            out.append(p)
    return tuple(out)


def sparse_spacing(q: np.ndarray) -> float:
    """まばらな所の点の間隔（水平の最近傍の距離の 90 百分位）。遠い所ほど点がまばらなので、中央値では升が細かすぎる。"""
    if len(q) < 2:
        return 0.0
    from scipy.spatial import cKDTree
    d, _ = cKDTree(np.asarray(q, float)[:, :2]).query(np.asarray(q, float)[:, :2], k=2)
    return float(np.percentile(d[:, 1], 90))


def finger_below(profile) -> float:
    """手と指のいちばん低い所が、指先の中心からどれだけ下か（0 以上）。"""
    return max(0.0, -min(p.low for p in profile)) if profile else 0.0


class HeightMap:
    def __init__(self, frame: G.TableFrame, cell: float = None, cell_k: float = 2.0, noise: float = G.NOISE_M,
                 self_k: float = 2.0, below_k: float = 3.0, edge_k: float = 5.0):
        self.frame = frame
        self.cell = None if cell is None else float(cell)
        self.cell_k = float(cell_k)
        self.noise = float(noise)
        self.self_k = float(self_k)                     # 自分の部品の箱を太らせる幅: 升の幅 × これ
        self.below_k = float(below_k)                   # 机の平面より σ × これ 低い点は入れない
        self.edge_k = float(edge_k)                     # 深度の段差の縁: 3×3 の差 ≥ σ × これ の画素は入れない
        self._h = np.zeros((0, 0))                      # 升の高さの格子（空の升は NaN）。行 i − _i0・列 j − _j0
        self._i0, self._j0 = 0, 0
        self.t = None

    @property
    def sigma(self) -> float:
        """地図の高さの σ の目安（深度の雑音）。"""
        return self.noise

    @property
    def cells(self) -> dict:
        """{(i, j): 高さ} の写し（確かめ用。地図の本体は格子）。"""
        ii, jj = np.nonzero(~np.isnan(self._h))
        return {(int(i) + self._i0, int(j) + self._j0): float(self._h[i, j]) for i, j in zip(ii, jj)}

    def clear(self) -> None:
        self._h = np.zeros((0, 0))

    def _keys(self, xy) -> np.ndarray:
        return np.floor(np.asarray(xy, float) / self.cell).astype(np.int64)

    def _grow(self, lo, hi, pad: int = 8) -> None:
        """升の番号 lo〜hi（どちらも含む）が入るように格子を広げる（前の値は保つ）。"""
        ni, nj = self._h.shape
        if ni and lo[0] >= self._i0 and lo[1] >= self._j0 and hi[0] < self._i0 + ni and hi[1] < self._j0 + nj:
            return
        lo, hi = [int(v) for v in lo], [int(v) for v in hi]
        if ni:
            lo = [min(lo[0], self._i0), min(lo[1], self._j0)]
            hi = [max(hi[0], self._i0 + ni - 1), max(hi[1], self._j0 + nj - 1)]
        i0, j0 = lo[0] - pad, lo[1] - pad
        h = np.full((hi[0] - i0 + 1 + pad, hi[1] - j0 + 1 + pad), np.nan)
        if ni:
            h[self._i0 - i0:self._i0 - i0 + ni, self._j0 - j0:self._j0 - j0 + nj] = self._h
        self._h, self._i0, self._j0 = h, i0, j0

    def _span(self, lo: float, hi: float, k0: int, n: int):
        """中心 (k + 0.5) × 升の幅 が [lo, hi] に入りうる升の、格子の添字の範囲 [a, b)（1 升ずつ広げる。判定は呼び手が中心で）。"""
        a = int(math.floor(lo / self.cell)) - 1 - k0
        b = int(math.floor(hi / self.cell)) + 2 - k0
        return max(a, 0), min(b, n)

    def _window(self, lo, hi):
        """水平の範囲 lo〜hi に中心が入りうる、空でない升の (中心 (M, 2), 高さ (M,))（少し広めに取る）。"""
        ni, nj = self._h.shape
        ia, ib = self._span(lo[0], hi[0], self._i0, ni)
        ja, jb = self._span(lo[1], hi[1], self._j0, nj)
        if ia >= ib or ja >= jb:
            return np.zeros((0, 2)), np.zeros(0)
        sub = self._h[ia:ib, ja:jb]
        ii, jj = np.nonzero(~np.isnan(sub))
        xy = (np.column_stack([ii + (ia + self._i0), jj + (ja + self._j0)]).astype(float) + 0.5) * self.cell
        return xy, sub[ii, jj]

    def update_points(self, q: np.ndarray, t: float = None) -> None:
        """机の座標の点 (N, 3)（z = 机からの高さ）で、点が 2 つ以上あった升を「2 番目に高い点」に置き換える（飛んだ画素の
        1 点では升を上げない。1 点だけの升は前の値のまま）。"""
        q = np.asarray(q, float)
        if len(q) == 0:
            return
        if self.cell is None:
            self.cell = max(self.cell_k * sparse_spacing(q), 1e-3)
        k = self._keys(q[:, :2])
        self._grow(k.min(axis=0), k.max(axis=0))
        flat = self._h.reshape(-1)
        lin = (k[:, 0] - self._i0) * self._h.shape[1] + (k[:, 1] - self._j0)
        z = q[:, 2]
        # 升ごとの 2 番目に高い点（並べ替えない）: いちばん高い値が 2 点以上ならその値、1 点ならそれを除いた最大
        m1 = np.full(flat.size, -np.inf)
        np.maximum.at(m1, lin, z)
        top = z == m1[lin]
        m2 = np.full(flat.size, -np.inf)
        np.maximum.at(m2, lin[~top], z[~top])
        second = np.where(np.bincount(lin[top], minlength=flat.size) >= 2, m1, m2)
        two = np.bincount(lin, minlength=flat.size) >= 2
        flat[two] = second[two]
        self.t = t

    def update(self, clouds, t: float = None, shift: dict = None, exclude=()) -> None:
        """clouds: Perception.points の出力（"pw"・"h"・"name"、あれば "robot_boxes"）。shift: {カメラ: 水平のずれ}（api/estimated の
        カメラの合わせと同じ）。exclude: [(世界の点, 半径)]（球）か [(世界の点, 水平の半径, 世界の高さの下限)]（柱）（持っている物）。升の幅は、最初に点のあった更新で、まばらな方のカメラの点の
        間隔から決める。机の平面より σ × below_k 以上低い点（机の外の床）と、そのフレームのロボットの部品の箱を升の幅 × self_k
        太らせた内側の点（画像の自己除去から漏れた自分の縁）は入れない。"""
        kept = []
        for P in clouds:
            if P is None or len(P["pw"]) == 0:
                continue
            pw = np.asarray(P["pw"], float)
            h = np.asarray(P["h"], float)
            keep = h >= -self.below_k * self.sigma
            if P.get("edge") is not None:
                keep &= ~np.asarray(P["edge"], bool)
            for ex in exclude:
                c = np.asarray(ex[0], float)
                if len(ex) >= 3:                        # 柱: 水平の半径 r の内側で、高さ z_min（世界）より上
                    keep &= ~((np.linalg.norm(pw[:, :2] - c[:2], axis=1) <= float(ex[1])) & (pw[:, 2] >= float(ex[2])))
                else:                                   # 球
                    keep &= np.linalg.norm(pw - c, axis=1) > float(ex[1])
            kept.append((P, pw[keep], h[keep]))
        kept = [k for k in kept if len(k[1])]
        if not kept:
            return
        if self.cell is None:
            sp = [sparse_spacing(self.frame.to_table(pw)) for _, pw, _ in kept if len(pw) > 1]
            self.cell = max(self.cell_k * (max(sp) if sp else 0.0), 1e-3)
        qs, shadows = [], []
        margin = self.self_k * self.cell
        for P, pw, h in kept:
            boxes = P.get("robot_boxes")
            if boxes:
                m = ~near_boxes(pw, boxes, margin)
                pw, h = pw[m], h[m]
                shadows += self._shadows(boxes, margin)
            q = self.frame.to_table(pw)
            q[:, 2] = h
            sh = (shift or {}).get(P.get("name"))
            if sh is not None:
                q[:, :2] -= np.asarray(sh, float)[:2]
            if shadows:
                q = q[~self._in_shadow(q, shadows)]
            qs.append(q)
        self.update_points(np.vstack(qs), t)
        if shadows:
            self._clear_shadows(shadows)

    def _shadows(self, boxes, margin: float) -> list:
        """ロボットの部品の箱 → 机の座標の [(水平の下限, 水平の上限, 下の端の高さ)]（margin だけ太らせる）。机の上の場面では、
        部品の下の端より上に物は無い（あればぶつかっている）ので、そこの点・升はロボット自身か、隠れて古くなった値。"""
        out = []
        signs = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], float)
        for c, R, h in boxes:
            corners = self.frame.to_table(np.asarray(c, float) + (signs * h) @ np.asarray(R).T)
            out.append((corners[:, :2].min(axis=0) - margin, corners[:, :2].max(axis=0) + margin,
                        float(corners[:, 2].min()) - margin))
        return out

    @staticmethod
    def _in_shadow(q: np.ndarray, shadows) -> np.ndarray:
        hit = np.zeros(len(q), bool)
        cand = coarse_candidates(q, [(lo, hi, zb, np.inf) for lo, hi, zb in shadows])   # 粗い升で先に絞る
        if len(cand) == 0:
            return hit
        c = q[cand]
        sub = np.zeros(len(cand), bool)
        for lo, hi, zb in shadows:
            sub |= np.all((c[:, :2] >= lo) & (c[:, :2] <= hi), axis=1) & (c[:, 2] >= zb)
        hit[cand] = sub
        return hit

    def _clear_shadows(self, shadows) -> None:
        """ロボットの部品の下の端より上にある升を消す（点が無ければ前の値を保つ決まりで、腕の下に古い自分の点が残らないように）。"""
        ni, nj = self._h.shape
        for lo, hi, zb in shadows:
            ia, ib = self._span(lo[0], hi[0], self._i0, ni)
            ja, jb = self._span(lo[1], hi[1], self._j0, nj)
            if ia >= ib or ja >= jb:
                continue
            x = (np.arange(ia + self._i0, ib + self._i0).astype(float) + 0.5) * self.cell
            y = (np.arange(ja + self._j0, jb + self._j0).astype(float) + 0.5) * self.cell
            sub = self._h[ia:ib, ja:jb]
            sub[((x >= lo[0]) & (x <= hi[0]))[:, None] & ((y >= lo[1]) & (y <= hi[1]))[None, :] & (sub >= zb)] = np.nan

    def snapshot(self) -> "HeightMap":
        out = HeightMap(self.frame, self.cell, self.cell_k, self.noise, self.self_k, self.below_k, self.edge_k)
        out._h, out._i0, out._j0 = self._h.copy(), self._i0, self._j0
        out.t = self.t
        return out

    def band_max(self, a, b, radius: float) -> Optional[float]:
        """机の座標の水平の線分 a→b から radius 以内（升の半対角を足す）の升の最大の高さ。升が無ければ None。"""
        if self.cell is None:
            return None
        a, b = np.asarray(a, float)[:2], np.asarray(b, float)[:2]
        reach = float(radius) + 0.5 * math.sqrt(2.0) * self.cell
        lo, hi = np.minimum(a, b) - reach, np.maximum(a, b) + reach
        xy, h = self._window(lo, hi)
        if len(h) == 0:
            return None
        box = np.all((xy >= lo) & (xy <= hi), axis=1)
        if not box.any():
            return None
        xy, h = xy[box], h[box]
        ab = b - a
        L2 = float(ab @ ab)
        s = np.zeros(len(xy)) if L2 < 1e-18 else np.clip((xy - a) @ ab / L2, 0.0, 1.0)
        d = np.linalg.norm(xy - (a + s[:, None] * ab), axis=1)
        m = d <= reach
        return float(h[m].max()) if m.any() else None

    def band_low(self, a, radius: float, q: float = 20.0) -> Optional[float]:
        """点 a（机の座標の水平）から radius 以内の升の高さの q 百分位（箱の底など、低い面の高さの目安）。升が無ければ None。"""
        if self.cell is None:
            return None
        a = np.asarray(a, float)[:2]
        xy, h = self._window(a - float(radius), a + float(radius))
        if len(h) == 0:
            return None
        m = np.linalg.norm(xy - a, axis=1) <= float(radius)
        return float(np.percentile(h[m], q)) if m.any() else None

    def to_table_xy(self, p) -> np.ndarray:
        return self.frame.to_table(np.asarray(p, float))[:2]


def pass_tip_height(hm: HeightMap, a_xy, b_xy, profile, held_radius: float = None, held_below: float = 0.0,
                    unc_k: float = 3.0) -> Optional[float]:
    """机の座標の水平 a→b を、手・指（profile）と持っている物（半径 held_radius、指先の中心から held_below 下まで）が
    ぶつからずに通る指先の中心の高さ（机から）。シード範囲に升が無ければ None。"""
    parts = [(p.radius, p.low) for p in profile]
    if held_radius is not None:
        parts.append((float(held_radius), -float(held_below)))
    need = None
    for r, low in parts:
        m = hm.band_max(a_xy, b_xy, r)
        if m is None:
            continue
        v = m - low
        need = v if need is None else max(need, v)
    return None if need is None else need + unc_k * hm.sigma


def path_goal(x, goal, pass_z: float, h_tol: float, z_tol: float):
    """手先の参照 x から goal（どちらも 3 次元）への今の目標と段: ①真上へ通る高さ pass_z まで（"rise"）、②その高さで水平に
    （"move"。通る高さが下がれば斜めに下りる）、③真上に着いたら下ろす（"lower"）。行き先が pass_z より高ければその高さを通る。"""
    x, goal = np.asarray(x, float), np.asarray(goal, float)
    travel = max(float(pass_z), float(goal[2]))                     # 行き先が通る高さより高ければ、その高さで動く
    if math.hypot(*(goal[:2] - x[:2])) > h_tol:
        if x[2] < travel - z_tol:
            return np.array([x[0], x[1], travel]), "rise"
        return np.array([goal[0], goal[1], travel]), "move"
    return goal.copy(), "lower"
