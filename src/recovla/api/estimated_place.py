"""点群から推定する知覚を、箱でない置き場所（皿）と円筒の物（カップ）に広げたもの（タスク「カップを皿に」）。
api/estimated.EstimatedGeometryFinder（立方体と箱）を継ぎ、物と置き場所の当てはめだけを、タスクの
書式の物の形（shape）と場所の finder で選ぶ:
  物     shape "cup" → api/geometry_place.fit_cylinder（立った円筒: 中心・外の半径・高さ・上端・上が開いているか）。
         ほか → api/geometry.fit_object
  場所   finder "plate:…" → api/geometry_place.fit_plate（平らな円板: 上面の中心・法線・半径）。"box:…" → fit_container
物の種類の知識として使うのは「カップ＝その色の、上が開いた立った円筒」「皿＝その色の、机の上の平らな円板」だけ（色の切り出しは
既存の Perception.points。皿は箱と同じ黄）。大きさ・位置・向き・高さは点群から推定する（既知の寸法は読まない）。
ObjState の extent は立った円筒の外接箱（外径・外径・高さ。短い幅 = 外径）、R は机の座標の単位行列。PlaceState の region は
PlateGeom（上面の円）。目標の場所にあるか（in_place）は置き場所の種類で分ける（api/relations.in_place: 箱は中、皿は上に載って
円の内側、カップは上が開いて見えた＝立っている）。皿の上の物は、見えない下の側を皿の上面まで延ばす（_support_h）。

    finder = finder_class(spec).from_spec(spec, perception, setup)   # 立方体と箱だけのタスクは EstimatedGeometryFinder のまま
"""
import numpy as np

from recovla.api import geometry as G
from recovla.api import geometry_place as GP
from recovla.api import relations as REL
from recovla.api import task_spec as TS
from recovla.api.estimated import YELLOW, EstimatedGeometryFinder, _world_box
from recovla.api.perception import OPEN_VOCAB_MSG, PlaceState, WorldEstimate
from recovla.runtime.perception import COLORS

PLACE_FITS = ("box", "plate")
PLATE_SHRINK = 0.85                           # 皿の推定を替えるのは、半径が現在の推定のこの倍以上のときだけ（皿は縮まない）
CYLINDER_SHAPES = ("cup",)


def needs_place_finder(spec) -> bool:
    """箱でない置き場所か、円筒の物があるタスクか。"""
    return (any(TS.finder_kind(p.finder)[0] == "plate" for p in spec.places)
            or any(o.shape in CYLINDER_SHAPES for o in spec.objects))


def finder_class(spec):
    return PlaceGeometryFinder if needs_place_finder(spec) else EstimatedGeometryFinder


def _parse(spec):
    names, shapes, place, fit = {}, {}, None, None
    for o in spec.objects:
        kind, arg = TS.finder_kind(o.finder)
        if kind == "open_vocab":
            raise NotImplementedError(OPEN_VOCAB_MSG)
        if kind != "color" or arg not in COLORS:
            raise ValueError(f"物 {o.name} の finder {o.finder} は色の知覚では切り出せない")
        names[o.name], shapes[o.name] = arg, o.shape
    for p in spec.places:
        kind, arg = TS.finder_kind(p.finder)
        if kind == "open_vocab":
            raise NotImplementedError(OPEN_VOCAB_MSG)
        if kind in PLACE_FITS and place is None:
            place, fit = p.name, kind
    if place is None:
        raise ValueError(f"箱らしい場所か皿らしい場所（finder {PLACE_FITS} の 'box:…'・'plate:…'）が 1 つ要る")
    return names, shapes, place, fit


class PlaceGeometryFinder(EstimatedGeometryFinder):
    def __init__(self, names: dict, place: str, frame: G.TableFrame, noise: float = G.NOISE_M, seg_h: float = None,
                 lost_s: float = 1.0, perception=None, seed: int = 0, shapes: dict = None, place_fit: str = "box"):
        self.shapes = dict(shapes or {})
        self.place_fit = str(place_fit)
        super().__init__(names, place, frame, noise, seg_h, lost_s, perception, seed)

    @classmethod
    def from_spec(cls, spec, perception=None, setup=None, noise: float = G.NOISE_M, **kw) -> "PlaceGeometryFinder":
        names, shapes, place, fit = _parse(spec)
        normal = getattr(setup, "table_normal", None) if setup is not None else None
        tz = getattr(setup, "table_z", None) if setup is not None else None
        frame = G.TableFrame.from_plane([0.0, 0.0, 1.0] if normal is None else normal, 0.0 if tz is None else tz)
        p = getattr(perception, "p", None)
        kw.setdefault("seg_h", None if p is None else float(p.table_h))
        kw.setdefault("lost_s", 1.0 if p is None else float(p.lost_s))
        return cls(names, place, frame, noise, perception=perception, shapes=shapes, place_fit=fit, **kw)

    def reset(self) -> None:
        super().reset()
        self.plate = None                     # 世界の座標の PlateGeom
        self._plate_q = 0                     # 採った皿の当てはめの、縁の標本に使えた扇の数
        self.upright = {}                     # {名前: 最後の当てはめで立っていると見えたか}
        self._current = None

    # ------------------------------------------------------------------ 当てはめ
    def _fit_shape(self, name, po, rng, support_h: float, assume_supported: bool):
        if self.shapes.get(name) in CYLINDER_SHAPES:            # 持っているかもしれない物（assume_supported 偽）は延ばさない
            return GP.fit_cylinder(po, self.noise, rng, support_h=support_h, assume_supported=assume_supported,
                                   held=not assume_supported)
        return G.fit_object(po, self.noise, rng, support_h=support_h, assume_supported=assume_supported)

    def _plate_better(self, p: GP.PlateGeom) -> bool:
        """腕・上の物に隠れた時刻の当てはめでは替えない（皿は動かないとみなす）: 使えた扇が前の 3/4 未満、または半径が現在の推定の
        PLATE_SHRINK 倍未満（隠れてアクションチャンクが切れると、いちばん大きいアクションチャンクが縁の小さな切れ端になる。小さなアクションチャンクでも扇は全部埋まりうるので、
        扇の数だけでは見分けられない）。"""
        if self.plate is None:
            return True
        return p.n_edges >= 0.75 * self._plate_q and p.radius >= PLATE_SHRINK * self.plate.radius

    def _place_state(self, pb, rng) -> PlaceState:
        if self.place_fit == "plate":
            if len(pb) >= 60:
                p = GP.fit_plate(pb, self.noise, rng)
                if p is not None and self._plate_better(p):
                    self.plate = GP.to_world(p, self.frame)
                    self._plate_q = max(self._plate_q, p.n_edges)
            region = self.plate
        else:                                                 # 箱（api/estimated と同じ決まり）
            if len(pb) >= 60:
                b = G.fit_container(pb, self.noise, rng)
                if b is not None and self._box_better(b):
                    self.box = _world_box(b, self.frame)
                    self._box_q = (b.n_edges, b.n_rim)
            region = self.box
        return PlaceState(np.zeros(2) if region is None else region.center[:2].copy(), 0.0 if region is None else region.yaw,
                          region is not None, None, region)

    def _support_h(self, q) -> float:
        """机の座標の点 q の下で支える面の高さ: 皿の上面の円の中なら、その点の真下の上面の高さ。ほかは api/estimated のまま。"""
        p = self.plate
        if p is not None:
            w = self.frame.to_world(q)
            if p.over_xy(w):
                loc = p.local(w)
                return float(self.frame.to_table(p.center + p.R @ np.array([loc[0], loc[1], 0.0]))[2])
        return super()._support_h(q)

    def _with_relations(self, o, place, tip, near=None):
        import dataclasses
        up = self.upright.get(self._current, True)          # いま当てはめている物（estimate_clouds の繰り返しの中）
        nh = REL.near_hand(o, tip) if near is None else near
        return dataclasses.replace(o, in_place={self.place: bool(REL.in_place(o, place, up))}, near_hand=nh)

    # ------------------------------------------------------------------ 推定
    def estimate_clouds(self, clouds, t: float, gripper=None, tip=None) -> WorldEstimate:
        """親クラスと同じ流れ（カメラごとの切り出し・まばらな点の除去・カメラどうしの合わせ・格子の平均・いちばん大きいアクションチャンク）で、
        物と置き場所の当てはめだけを形で選ぶ。"""
        rng = np.random.default_rng(self.seed + self.n_frames)
        self.n_frames += 1
        tip = None if tip is None else np.asarray(tip, float)
        grasped = bool(gripper is not None and getattr(gripper, "is_grasped", False))
        cams = {}
        for i, P in enumerate(clouds):
            if P is None or len(P["pw"]) == 0:
                continue
            q = self.frame.to_table(P["pw"])
            q[:, 2] = np.asarray(P["h"], float)
            col = np.asarray(P["col"])
            above = q[:, 2] > self.seg_h
            got = cams.setdefault(str(P.get("name", f"cam{i}")), {})
            for lab in np.unique(col[above]):
                if lab < 0:
                    continue
                p = q[above & (col == lab)]
                p = p[G.clean_points(p)]
                if len(p):
                    got[int(lab)] = p
        self._align(cams, grasped, tip, rng)
        per = {}
        for name, got in cams.items():
            sh = self.cam_shift.get(name)
            for lab, p in got.items():
                if sh is not None:
                    p = p - np.array([sh[0], sh[1], 0.0])
                per.setdefault(lab, []).append((p, G.spacing(p)))

        def points_of(lab, dims=3):
            parts = per.get(lab, [])
            if not parts:
                return np.zeros((0, 3))
            s = max(sp for _, sp in parts)
            p = np.vstack([x for x, _ in parts])
            if len(parts) > 1:
                p = G.voxel_down(p, s)
            if len(p) < 3:
                return p
            return p[G.largest_cluster(p[:, :dims], 3.0 * s)]

        place = self._place_state(points_of(YELLOW, dims=2), rng)
        out = {}
        for name, color in self.names.items():
            self._current = name
            po = points_of(COLORS.index(color))
            g = None
            if len(po) >= 20:
                g = self._fit_shape(name, po, rng, self._support_h(po.mean(axis=0)),
                                    not (grasped and self._at_tip(po, tip)))
            if g is not None:
                self.upright[name] = bool(getattr(g, "open_top", True))
                st = self._seen(g, place, t, tip)
                self.objs[name] = {"state": st, "t_seen": float(t), "tip_off": None if tip is None else st.pos - tip}
            elif name in self.objs:
                st = self._unseen(name, place, t, tip, grasped)
            else:
                continue
            out[name] = st
        self._current = None
        self.last = WorldEstimate(float(t), out, {self.place: place}, float(self.frame.origin[2]), self.kind)
        return self.last

    def _align(self, cams: dict, grasped: bool, tip, rng, keep: int = 40, every: int = 5) -> None:
        """カメラどうしの水平のずれを測る（親クラスと同じ）。物の当てはめを形で選ぶところだけ違う。"""
        if not cams:
            return
        from recovla.api.estimated import REF_CAMERA
        ref = REF_CAMERA if REF_CAMERA in cams else next(iter(cams))
        names = {COLORS.index(c): n for n, c in self.names.items()}
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
                n = names.get(lab)
                gp = self._fit_shape(n, p, rng, self._support_h(p.mean(axis=0)), True)
                gq = self._fit_shape(n, q, rng, self._support_h(q.mean(axis=0)), True)
                if gp is not None and gq is not None:
                    buf.append((gp.center - gq.center)[:2])
            del buf[:-keep]
            if buf:
                self.cam_shift[name] = np.median(np.array(buf), axis=0)
