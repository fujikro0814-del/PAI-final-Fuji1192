"""点群から形を推定する知覚 EstimatedGeometryFinder。色で切り出すところは既存の知覚と同じで、
物の大きさ・向き・高さと、箱らしい物の中心・向き・内寸・縁の高さ・傾きを点群から推定する（api/geometry）。
既知の box_outer・box_wall・cube_size・box_nominal_xy は読まない。使う既知の値は机の平面（setup.table_normal・table_z。
較正の値）と、深度の雑音の目安・切り出しの高さ（知覚の引数 table_h）・見失うまでの時間（同 lost_s）だけ。

    finder = EstimatedGeometryFinder.from_spec(spec, perception, setup)
    est = finder.estimate_clouds(clouds, t, gripper, tip)      # clouds: Perception.points の出力（カメラごと）
    est = finder.update(senses, t)                             # ObjectFinder の口（perception.points で点群を作る）
    per = GeometryPerception.attach(perception, finder)        # 既存の Perception に推定を足す（WorldModel の計算は同じ）
    wm = per.update(...); wm.geometry                          # 同じ点群からの WorldEstimate（io.compute の中で計算される）

状態: その時刻に点が足りて形が当てはまった物は seen。当てはまらない物は、直前に手のそば（near_hand）でハンドが把持を報告して
いれば occluded かつ near_hand（中心は指先との差を保って動かす）、そうでなければ lost_s までは occluded、その後は lost。
箱らしい物は、見つからない時刻は前の推定を保つ（腕に隠れても箱は動かない）。中か外かは推定した 3 次元の内側の領域で決める。
"""
import math

import numpy as np

from recovla.api import geometry as G
from recovla.api import relations as REL
from recovla.api import task_spec as TS
from recovla.api.perception import OPEN_VOCAB_MSG, ObjState, PlaceState, WorldEstimate
from recovla.runtime.perception import COLORS, Perception

YELLOW = 3                                    # runtime/perception.Perception.points の箱の黄の印
REF_CAMERA = "wrist"                          # 位置を合わせる基準のカメラ（手に付いたカメラ。手との相対の誤差が小さい）


def _names_and_place(spec):
    names, place = {}, None
    for o in spec.objects:
        kind, arg = TS.finder_kind(o.finder)
        if kind == "open_vocab":
            raise NotImplementedError(OPEN_VOCAB_MSG)
        if kind != "color" or arg not in COLORS:
            raise ValueError(f"物 {o.name} の finder {o.finder} は色の知覚では切り出せない")
        names[o.name] = arg
    for p in spec.places:
        kind, arg = TS.finder_kind(p.finder)
        if kind == "open_vocab":
            raise NotImplementedError(OPEN_VOCAB_MSG)
        if kind == "box" and place is None:
            place = p.name
    if place is None:
        raise ValueError("箱らしい場所（finder 'box:…'）が 1 つ要る")
    return names, place


def _world_box(b: G.BoxGeom, frame: G.TableFrame) -> G.BoxGeom:
    """机の座標の BoxGeom → 世界の座標（中心と向きだけ変わる。大きさ・高さ・傾きは同じ）。"""
    import dataclasses
    return dataclasses.replace(b, center=frame.to_world(b.center), R=frame.R @ b.R)


def _yaw(R) -> float:
    a = math.atan2(R[1, 0], R[0, 0])
    return float((a + math.pi / 2) % math.pi - math.pi / 2)


class EstimatedGeometryFinder:
    needs = ("joints", "gripper", "camera/overhead", "camera/wrist")
    kind = "estimated"

    def __init__(self, names: dict, place: str, frame: G.TableFrame, noise: float = G.NOISE_M, seg_h: float = None,
                 lost_s: float = 1.0, perception=None, seed: int = 0):
        self.names, self.place, self.frame = dict(names), str(place), frame
        self.noise = float(noise)
        self.seg_h = 3.0 * self.noise if seg_h is None else float(seg_h)
        self.lost_s = float(lost_s)
        self.perception = perception
        self.seed = int(seed)
        self.reset()

    @classmethod
    def from_spec(cls, spec, perception=None, setup=None, noise: float = G.NOISE_M, **kw) -> "EstimatedGeometryFinder":
        names, place = _names_and_place(spec)
        normal = getattr(setup, "table_normal", None) if setup is not None else None
        tz = getattr(setup, "table_z", None) if setup is not None else None
        frame = G.TableFrame.from_plane([0.0, 0.0, 1.0] if normal is None else normal, 0.0 if tz is None else tz)
        p = getattr(perception, "p", None)
        kw.setdefault("seg_h", None if p is None else float(p.table_h))
        kw.setdefault("lost_s", 1.0 if p is None else float(p.lost_s))
        return cls(names, place, frame, noise, perception=perception, **kw)

    def reset(self) -> None:
        self.objs = {}                        # {名前: {"state": ObjState, "t_seen", "tip_off"}}
        self.box = None                       # 世界の座標の BoxGeom
        self._box_q = (0, 0)                  # 採った箱の当てはめの (見えた辺の数, 縁の点の数)
        self.cam_shift = {}                   # {カメラ: 基準のカメラに対する水平のずれ (2,)}（机の座標）
        self._shift_buf = {}                  # {カメラ: [ずれの標本]}
        self.last = None
        self.n_frames = 0

    # ------------------------------------------------------------------ 推定
    def estimate_clouds(self, clouds, t: float, gripper=None, tip=None) -> WorldEstimate:
        """カメラごとの点群（Perception.points の {"pw", "h", "col"}）から、その時刻の WorldEstimate。"""
        rng = np.random.default_rng(self.seed + self.n_frames)
        self.n_frames += 1
        tip = None if tip is None else np.asarray(tip, float)
        grasped = bool(gripper is not None and getattr(gripper, "is_grasped", False))
        cams = {}                                           # {カメラ: {色の添字: 点}}
        for i, P in enumerate(clouds):
            if P is None or len(P["pw"]) == 0:
                continue
            h = np.asarray(P["h"], float)
            col = np.asarray(P["col"])
            sel = np.flatnonzero((h > self.seg_h) & (col >= 0))   # 切り出しの高さより上の色の点だけを机の座標へ
            q = self.frame.to_table(np.asarray(P["pw"], float)[sel])
            q[:, 2] = h[sel]                                # 高さはそのカメラ自身の机の平面から（Perception と同じ）
            col = col[sel]
            got = cams.setdefault(str(P.get("name", f"cam{i}")), {})
            for lab in np.unique(col):
                p = q[col == lab]
                p = p[G.clean_points(p)]                    # 飛んだ画素などのまばらな点をカメラごとに外す
                if len(p):
                    got[int(lab)] = p
        self._align(cams, grasped, tip, rng)
        per = {}                                            # {色の添字: [(カメラの点, 点の間隔)]}
        for name, got in cams.items():
            sh = self.cam_shift.get(name)
            for lab, p in got.items():
                if sh is not None:
                    p = p - np.array([sh[0], sh[1], 0.0])  # 基準のカメラに合わせる（水平だけ）
                per.setdefault(lab, []).append((p, G.spacing(p)))

        def points_of(lab, dims=3):
            """カメラの点を合わせ、いちばん大きいアクションチャンク。つなぐ距離は、まばらな方のカメラの点の間隔の 3 倍。dims=2 は机の上へ
            投影してつなぐ（箱: 陰の壁の面が色で切り出せないと、縁と底が高さの向きに離れて別のアクションチャンクになるため）。"""
            parts = per.get(lab, [])
            if not parts:
                return np.zeros((0, 3))
            s = max(sp for _, sp in parts)
            p = np.vstack([x for x, _ in parts])
            if len(parts) > 1:
                p = G.voxel_down(p, s)                      # 密な手首のカメラの点を、まばらな方の間隔の格子で平均する
            if len(p) < 3:
                return p
            return p[G.largest_cluster(p[:, :dims], 3.0 * s)]

        pb = points_of(YELLOW, dims=2)
        if len(pb) >= 60:
            b = G.fit_container(pb, self.noise, rng)
            if b is not None and self._box_better(b):
                self.box = _world_box(b, self.frame)
                self._box_q = (b.n_edges, b.n_rim)
        region = self.box
        place = PlaceState(np.zeros(2) if region is None else region.center[:2].copy(), 0.0 if region is None else region.yaw,
                           region is not None, None, region)
        out = {}
        for name, color in self.names.items():
            po = points_of(COLORS.index(color))
            g = None
            if len(po) >= 20:
                g = G.fit_object(po, self.noise, rng, support_h=self._support_h(po.mean(axis=0)),
                                 assume_supported=not (grasped and self._at_tip(po, tip)))
            if g is not None:
                st = self._seen(g, place, t, tip)
                self.objs[name] = {"state": st, "t_seen": float(t), "tip_off": None if tip is None else st.pos - tip}
            elif name in self.objs:
                st = self._unseen(name, place, t, tip, grasped)
            else:
                continue
            out[name] = st
        self.last = WorldEstimate(float(t), out, {self.place: place}, float(self.frame.origin[2]), self.kind)
        return self.last

    def _align(self, cams: dict, grasped: bool, tip, rng, keep: int = 40, every: int = 5) -> None:
        """カメラどうしの水平のずれ（較正の誤差。2〜3 cm ある）を測り、cam_shift に持つ。両方のカメラに写った物
        （手が持っているかもしれない物を除く）をカメラごとに当てはめ、中心の差の最近の keep 個の中央値を使う。
        試行の中ではずれはほぼ一定（散らばり 2〜4 mm）。合わせずに点を混ぜると、立方体が 2 cm 大きく出る。
        基準は手首のカメラ（立方体の中心の誤差: 手首 3〜10 mm、俯瞰 1〜26 mm。掴むのに要るのは手との相対）。基準の
        カメラが無い・標本が無いときは合わせない。標本が集まったら every フレームに 1 回だけ測る（計算を減らす）。"""
        if not cams:
            return
        ref = REF_CAMERA if REF_CAMERA in cams else next(iter(cams))
        for name, got in cams.items():
            if name == ref:
                continue
            buf = self._shift_buf.setdefault(name, [])
            if len(buf) >= keep // 2 and self.n_frames % every:
                continue
            for lab, p in got.items():
                q = cams[ref].get(lab)
                if lab == YELLOW or q is None or len(p) < 20 or len(q) < 20:
                    continue
                if grasped and (self._at_tip(p, tip) or self._at_tip(q, tip)):
                    continue
                gp = G.fit_object(p, self.noise, rng, support_h=self._support_h(p.mean(axis=0)), assume_supported=True)
                gq = G.fit_object(q, self.noise, rng, support_h=self._support_h(q.mean(axis=0)), assume_supported=True)
                if gp is not None and gq is not None:
                    buf.append((gp.center - gq.center)[:2])
            del buf[:-keep]
            if buf:
                self.cam_shift[name] = np.median(np.array(buf), axis=0)

    def _at_tip(self, po, tip) -> bool:
        """物の点（机の座標）が指先のそばか: 重心と指先の距離が、点の広がりの半径の 1.5 倍 ＋ 雑音の 3 倍以内。手が把持を
        報告しているときに、持たれているかもしれない物（宙にあってよい物）を決める。指先が分からなければそばとみなす。"""
        if tip is None:
            return True
        c = po.mean(axis=0)
        r = float(np.max(np.linalg.norm(po - c, axis=1)))
        return bool(np.linalg.norm(c - self.frame.to_table(tip)) <= 1.5 * r + 3.0 * self.noise)

    def _box_better(self, b: G.BoxGeom) -> bool:
        """新しい当てはめで箱の推定を替えるか。腕に縁が隠れた時刻の当てはめ（辺が欠ける・縁の点が少ない）では替えない
        （箱は動かないとみなす。4 辺とも見えて縁の点が前の半分以上あれば、動いた箱にも付いていく）。"""
        if self.box is None:
            return True
        e0, n0 = self._box_q
        if b.n_edges < e0:
            return False
        # 見えた辺が増えたら替える。同じ数なら 4 辺とも見えたときだけ（片側しか見えない当てはめどうしで入れ替えない）
        return b.n_edges > e0 or (b.n_edges == 4 and b.n_rim >= 0.5 * n0)

    def _support_h(self, q) -> float:
        """机の座標の点 q の下で支える面の高さ: 箱の柱の中なら推定の底、そうでなければ机（0）。"""
        b = self.box
        if b is None:
            return 0.0
        p = self.frame.to_world(q)
        if not b.over_xy(p):
            return 0.0
        loc = b.local(p)
        return float(self.frame.to_table(b.center + b.R @ np.array([loc[0], loc[1], b.floor]))[2])

    def _seen(self, g: G.ObjGeom, place, t, tip) -> ObjState:
        R = self.frame.R @ g.R
        cov = self.frame.R @ g.cov @ self.frame.R.T
        pos = self.frame.to_world(g.center)
        o = ObjState(pos, _yaw(R), "seen", {}, None, float(t), g.extent.copy(), R, cov)
        return self._with_relations(o, place, tip)

    def _with_relations(self, o: ObjState, place, tip, near=None) -> ObjState:
        import dataclasses
        inside = REL.inside(o, place)
        nh = REL.near_hand(o, tip) if near is None else near
        return dataclasses.replace(o, in_place={self.place: bool(inside)}, near_hand=nh)

    def _unseen(self, name, place, t, tip, grasped) -> ObjState:
        import dataclasses
        rec = self.objs[name]
        prev = rec["state"]
        if grasped and tip is not None and prev.near_hand and rec["tip_off"] is not None:
            o = dataclasses.replace(prev, pos=tip + rec["tip_off"], status="occluded")       # 手の中で隠れた: 指先と動く
            o = self._with_relations(o, place, tip, near=True)
        else:
            st = "occluded" if float(t) - rec["t_seen"] < self.lost_s - 1e-9 else "lost"
            o = self._with_relations(dataclasses.replace(prev, status=st), place, tip)
        rec["state"] = o
        return o

    def update(self, senses, t: float) -> WorldEstimate:
        if self.perception is None:
            raise RuntimeError("perception が無い（estimate_clouds に点群を渡す）")
        g = senses.get("gripper")
        clouds = [dict(self.perception.points(fr, g.width), name=n) for n, fr in senses.cameras().items()
                  if getattr(fr, "depth", None) is not None]
        return self.estimate_clouds(clouds, t, g, senses.tip())


class GeometryPerception(Perception):
    """既存の Perception に、同じ点群からの推定（EstimatedGeometryFinder）を足したもの。WorldModel の計算は同じで、update の
    返り値の WorldModel に geometry（WorldEstimate）を付ける。点群は update が作ったものを使い回す（自己除去の描画を増やさない）。
    推定の分だけ計算が重いので、io.compute の種類は "perception_est"（遅延は configs/runtime_v2.yaml の compute_latency_s）。"""
    compute_kind = "perception_est"

    @classmethod
    def attach(cls, per: Perception, finder: EstimatedGeometryFinder) -> "GeometryPerception":
        per.__class__ = cls
        per.geom = finder
        per._clouds = {}
        finder.perception = per
        return per

    def points(self, frame, width: float, margin: float = None):
        P = super().points(frame, width, margin)
        if margin is None:
            self._clouds[frame.name] = dict(P, name=frame.name)
        return P

    def update(self, frames: dict, gripper, joints_t: float, hand_tip=None):
        self._clouds = {}
        wm = super().update(frames, gripper, joints_t, hand_tip)
        wm.geometry = self.geom.estimate_clouds(list(self._clouds.values()), float(joints_t), gripper, hand_tip)
        return wm
