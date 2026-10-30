"""知覚の口。物の推定を、タスクの書式の名前で引ける WorldEstimate にする。

    finder = make_finder(spec, perception, setup)       # 書式の finder から選ぶ
    est = finder.update(senses, t)                      # ObjectFinder の口（知覚の更新も行う）
    est = finder.estimate(wm, tip)                      # 計算済みの WorldModel（PolicyRuntime.wm）を写すだけ

ObjState.status: seen（写っている）・occluded（隠れて保持していると推定）・lost。
ColorWorldModelFinder（既知の値を使う実装）は runtime/perception の WorldModel を包む。WorldModel の計算は変えない。
  写し方: seen → seen、held → occluded、in_hand → occluded かつ near_hand、lost → lost、in_box → in_place["box"]。
  near_hand は in_hand のときだけ真、ほかは None（分からない）。大きさ・箱の領域は既知の値から作る。
EstimatedGeometryFinder（点群から推定する実装。api/estimated）は同じ点群から大きさ・向き・箱の内側を推定する。
言葉で物を指す切り出し（finder "open_vocab:<言葉>"）は口だけ（make_finder が NotImplementedError）。
"""
import dataclasses
from typing import Optional, Protocol

import numpy as np

from recovla.api import task_spec as TS

STATUSES = ("seen", "occluded", "lost")
WM_STATUS = {"seen": "seen", "held": "occluded", "in_hand": "occluded", "lost": "lost"}
OPEN_VOCAB_MSG = ("言葉で物を指す切り出し（finder 'open_vocab:<言葉>'、開語彙の検出＋セグメンテーション）は口だけで、"
                  "中身はまだ無い（設計書 task_api.md 3-3・6。今後の課題）")


@dataclasses.dataclass(frozen=True)
class ObjState:
    pos: np.ndarray                       # (3,) 中心
    yaw: float
    status: str                           # STATUSES のどれか
    in_place: dict                        # {場所の名前: 中か}
    near_hand: Optional[bool]             # 手の中・手のそばと推定（None は分からない）
    t_seen: Optional[float] = None        # 最後に写った時刻
    extent: Optional[np.ndarray] = None   # (3,) 推定の辺の長さ（R の各軸。全長）
    R: Optional[np.ndarray] = None        # (3, 3) 推定の向き（列が軸、3 列目が上向き）
    cov: Optional[np.ndarray] = None      # (3, 3) 中心の共分散（目安）


@dataclasses.dataclass(frozen=True)
class PlaceState:
    pos: np.ndarray                       # (2,) 水平の中心
    yaw: float
    ok: bool                              # 当てはめに成功したか
    inner_half_m: Optional[float] = None  # 内寸の半分（正方形の既知の寸法の版だけ。推定の版は region）
    region: object = None                 # api.geometry.BoxGeom（世界の座標）。3 次元の内側の領域・置き場所

    def contains_xy(self, xy, shrink: float = 0.0) -> bool:
        """水平位置が内寸の中か（region が無ければ runtime/perception.Perception.in_box_xy と同じ式）。"""
        if not self.ok:
            return False
        if self.region is not None:
            return self.region.over_xy(np.r_[np.asarray(xy, float)[:2], self.region.center[2]], shrink)
        if self.inner_half_m is None:
            return False
        d = np.asarray(xy, float)[:2] - np.asarray(self.pos, float)[:2]
        ct, st = np.cos(self.yaw), np.sin(self.yaw)
        qx, qy = d[0] * ct + d[1] * st, -d[0] * st + d[1] * ct
        return bool(max(abs(qx), abs(qy)) <= self.inner_half_m - shrink)

    def depth(self, p) -> Optional[float]:
        """内側の領域からの深さ（正なら中、負なら外 [m]）。領域が無ければ None。"""
        return None if (self.region is None or not self.ok) else self.region.depth(p)

    @property
    def sigma(self) -> float:
        return 0.0 if self.region is None else float(self.region.sigma.get("center", 0.0))


@dataclasses.dataclass(frozen=True)
class WorldEstimate:
    t: float
    objects: dict                         # {名前: ObjState}
    places: dict                          # {名前: PlaceState}
    table_z: Optional[float] = None       # 机の高さ（較正の値）
    finder: str = ""                      # 推定を作った知覚の口（"color"・"estimated"）


class ObjectFinder(Protocol):
    needs: tuple

    def update(self, senses, t: float) -> WorldEstimate: ...


def inner_half_of(setup) -> Optional[float]:
    """既知の情報の箱の寸法から内寸の半分（runtime/perception と同じ: 外寸の半分 − 壁）。"""
    outer, wall = getattr(setup, "box_outer", None), getattr(setup, "box_wall", None)
    if outer is None or wall is None:
        return None
    return 0.5 * float(outer) - float(wall)


def _rz(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class ColorWorldModelFinder:
    """runtime/perception.Perception（色の知覚）の WorldModel を包む。names は {書式の物の名前: 色}。
    既知の値を使う: 物の大きさは既知の情報の cube_size、箱の内側の領域は box_outer・box_wall・box_height・
    box_floor から作る（点群から推定する api/estimated.EstimatedGeometryFinder と同じ口で比べるため）。"""
    needs = ("joints", "gripper", "camera/overhead", "camera/wrist")
    kind = "color"

    def __init__(self, names: dict, place: str = "box", perception=None, inner_half_m: float = None, known: dict = None):
        self.names, self.place = dict(names), str(place)
        self.perception = perception
        self.inner_half_m = inner_half_m
        self.known = dict(known or {})    # 既知の値（cube・outer_half・wall・height・floor・table_z）。無ければ大きさ・領域は None
        self.last = None

    def _region(self, b: dict):
        k = self.known
        if not {"outer_half", "wall", "height", "floor", "table_z"} <= set(k):
            return None
        from recovla.api.geometry import BoxGeom
        oh, ih = float(k["outer_half"]), float(k["outer_half"]) - float(k["wall"])
        c = np.array([float(b["xy"][0]), float(b["xy"][1]), float(k["table_z"]) + float(k["height"])])
        return BoxGeom(c, _rz(float(b["yaw"])), np.array([oh, oh]), np.array([ih, ih]), float(k["wall"]), float(k["height"]),
                       -(float(k["height"]) - float(k["floor"])), 0.0, 0.0, {"center": 0.0}, 0, 0)

    @classmethod
    def from_spec(cls, spec, perception=None, setup=None) -> "ColorWorldModelFinder":
        names, place = {}, None
        for o in spec.objects:
            kind, arg = TS.finder_kind(o.finder)
            if kind == "open_vocab":
                raise NotImplementedError(OPEN_VOCAB_MSG)
            if kind != "color":
                raise ValueError(f"物 {o.name} の finder {o.finder} は色の知覚では見つけられない")
            names[o.name] = arg
        for p in spec.places:
            kind, arg = TS.finder_kind(p.finder)
            if kind == "open_vocab":
                raise NotImplementedError(OPEN_VOCAB_MSG)
            if kind == "box" and arg == "v1":
                place = p.name
        if place is None:
            raise ValueError("色の知覚は箱（finder 'box:v1'）の場所が 1 つ要る")
        known = {}
        for key, attr in (("cube", "cube_size"), ("wall", "box_wall"), ("height", "box_height"), ("floor", "box_floor"),
                          ("table_z", "table_z")):
            v = getattr(setup, attr, None)
            if v is not None:
                known[key] = float(v)
        if getattr(setup, "box_outer", None) is not None:
            known["outer_half"] = 0.5 * float(setup.box_outer)
        return cls(names, place, perception, inner_half_of(setup), known)

    def estimate(self, wm, tip=None) -> Optional[WorldEstimate]:
        """WorldModel（None なら None）→ WorldEstimate。tip は使わない（口だけ。near_hand は WorldModel の in_hand だけ）。"""
        if wm is None:
            return None
        objs = {}
        for name, color in self.names.items():
            e = wm.cubes.get(color)
            if e is None:
                continue
            cube = self.known.get("cube")
            var = getattr(e, "var", None)
            objs[name] = ObjState(np.asarray(e.pos, float).copy(), float(e.yaw), WM_STATUS[e.status],
                                  {self.place: bool(e.in_box)}, True if e.status == "in_hand" else None,
                                  float(e.t_seen) if getattr(e, "t_seen", None) is not None else None,
                                  None if cube is None else np.full(3, cube), _rz(float(e.yaw)),
                                  None if var is None else float(var) * np.eye(3))
        b = wm.box
        places = {self.place: PlaceState(np.asarray(b["xy"], float).copy(), float(b["yaw"]), bool(b["ok"]), self.inner_half_m,
                                         self._region(b))}
        self.last = WorldEstimate(float(wm.t), objs, places, getattr(wm, "table_z", None), self.kind)
        return self.last

    def update(self, senses, t: float) -> WorldEstimate:
        """ObjectFinder の口: 知覚の更新（Perception.update）をしてから写す。PolicyRuntime の中では知覚の更新は io.compute の
        中で行われるので、実行系では estimate(prt.wm) を使う（同じ知覚を 2 回回さない）。"""
        if self.perception is None:
            raise RuntimeError("perception が無い（estimate(wm) を使う）")
        wm = self.perception.update(senses.cameras(), senses.get("gripper"), float(t), senses.tip())
        return self.estimate(wm, senses.tip())


FINDER_KINDS = ("estimated", "color")


def make_finder(spec, perception=None, setup=None, kind: str = "color", **kw) -> ObjectFinder:
    """知覚の口を選ぶ。kind "color" = 既知の値の版（ColorWorldModelFinder）、"estimated" = 点群から大きさ・向き・箱の内側を
    推定する版（api/estimated.EstimatedGeometryFinder。既知の寸法を読まない）。書式の finder は両方とも色で切り出す物だけ。"""
    if kind == "color":
        return ColorWorldModelFinder.from_spec(spec, perception, setup)
    if kind == "estimated":                     # カップ・皿のあるタスクは api/estimated_place.PlaceGeometryFinder
        from recovla.api.estimated_place import finder_class
        return finder_class(spec).from_spec(spec, perception, setup, **kw)
    raise ValueError(f"知覚の口 {kind!r} は {FINDER_KINDS} のどれか")
