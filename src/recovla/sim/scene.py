"""3 色の場面の配置。

    layout = sample_layout(seed)                  # 乱数のシードから配置を決める（配置の乱数列だけを使う）
    apply_layout(model, data, layout)             # 立方体 3 個を置く（静止させるのは SimRig.reset）

配置の決め方（すべて配置の乱数列 streams(seed)["layout"] から、この順に引く）:
  1. 配置の種類（kind を渡さないとき scene.layout_kinds の割合で。渡しても 1 つ引いて捨て、以後の列を揃える）
  2. 開始姿勢（scene.start_at_retreat_prob で retreat、ほかは home。start を渡したときも同じく 1 つ引いて捨てる）
  3. 初めから箱に入っている立方体の色と置き場所（1 個・2 個のとき。色の並びと置き場所の並びを無作為に並べ替えて先頭から使う）
  4. 机上の立方体（残りの色、色の並びの順に）: 範囲 scene.region の一様、向き ±scene.yaw_range_deg の一様。
     すでに置いた机上の立方体との中心間距離が scene.min_center_dist 未満なら、その 1 個を引き直す

別のタスク「カップを皿に置く」の場面（variant "cup_plate"、configs/scenes/cup_plate.yaml）:

    sc = cup_plate_config
    model = build_model("cup_plate")                     # assets/mjcf/scene_cup_plate.xml（寸法・色を設定と照合する）
    lay = sample_cup_plate_layout(seed, sc)              # 皿 → カップの順に引く（配置の乱数列 stream(seed, "layout")）
    apply_cup_plate_layout(model, data, lay, sc)         # カップの自由関節と、皿（固定の体）の model.body_pos を書く
  皿の中心は layout.plate_region の一様、カップの中心は layout.cup_region の一様（向きはどちらも ±yaw_range_deg）。
  カップは皿と重ならない（中心間の距離 ≥ 皿の半径 ＋ カップの半径 ＋ min_gap_m。だめなら引き直す）。
"""
import dataclasses
import functools

import mujoco
import numpy as np
import yaml

from recovla.common import config, seeds
from recovla.common.seeds import COLORS
from recovla.sim import frames

_CFG = config.load()
_SCENE = _CFG["scene"]
KINDS = ("empty", "prefilled_1", "prefilled_2")
N_PREFILLED = {"empty": 0, "prefilled_1": 1, "prefilled_2": 2}
STARTS = ("home", "retreat")


@dataclasses.dataclass(frozen=True)
class Layout:
    seed: int
    kind: str                         # empty | prefilled_1 | prefilled_2
    start: str                        # home | retreat
    cubes: dict                       # 机上の色 -> (x, y, yaw)
    prefilled: dict                   # 箱の中の色 -> 置き場所の番号（scene.box_slots の添字）
    tries: int = 1                    # 机上の立方体を引いた回数（棄却を含む）

    @property
    def table_colors(self) -> tuple:
        return tuple(c for c in COLORS if c in self.cubes)

    def to_json(self) -> dict:
        return {"seed": self.seed, "kind": self.kind, "start": self.start,
                "cubes": {c: [float(v) for v in self.cubes[c]] for c in self.table_colors},
                "prefilled": {c: int(s) for c, s in self.prefilled.items()}, "tries": self.tries}

    @classmethod
    def from_json(cls, d: dict) -> "Layout":
        return cls(int(d["seed"]), d["kind"], d["start"], {c: tuple(v) for c, v in d["cubes"].items()},
                   {c: int(s) for c, s in d["prefilled"].items()}, int(d.get("tries", 1)))


def sample_layout(seed: int, kind: str = None, cfg: dict = None, start: str = None) -> Layout:
    sc = (cfg or _CFG)["scene"]
    rng = seeds.stream(seed, "layout")
    probs = np.array([float(sc["layout_kinds"][k]) for k in KINDS])
    u = rng.random()
    drawn = KINDS[int(np.searchsorted(np.cumsum(probs / probs.sum()), u, side="right"))]
    kind = kind or drawn
    if kind not in KINDS:
        raise ValueError(f"kind {kind!r} not in {KINDS}")
    drawn_start = "retreat" if rng.random() < float(sc["start_at_retreat_prob"]) else "home"
    start = start or drawn_start                  # 渡したときも 1 つ引いて捨て、以後の列を揃える
    if start not in STARTS:
        raise ValueError(f"start {start!r} not in {STARTS}")
    color_order = [COLORS[i] for i in rng.permutation(len(COLORS))]
    slot_order = [int(i) for i in rng.permutation(len(sc["box_slots"]))]
    n = N_PREFILLED[kind]
    prefilled = {color_order[i]: slot_order[i] for i in range(n)}
    rx, ry = sc["region"]["x"], sc["region"]["y"]
    half_yaw = np.radians(float(sc["yaw_range_deg"]))
    min_d = float(sc["min_center_dist"])
    cubes, tries = {}, 0
    for color in COLORS:
        if color in prefilled:
            continue
        while True:
            tries += 1
            if tries > int(sc["layout_max_tries"]):
                raise RuntimeError(f"layout seed {seed}: no placement after {tries - 1} draws")
            x, y = float(rng.uniform(*rx)), float(rng.uniform(*ry))
            yaw = float(rng.uniform(-half_yaw, half_yaw))
            if all(np.hypot(x - p[0], y - p[1]) >= min_d for p in cubes.values()):
                cubes[color] = (x, y, yaw)
                break
    return Layout(int(seed), kind, start, cubes, prefilled, tries)


def cube_qpos_adr(model, color: str) -> tuple:
    jnt = model.body(frames.cube_body(color)).jntadr[0]
    return int(model.jnt_qposadr[jnt]), int(model.jnt_dofadr[jnt])


def apply_layout(model, data, layout: Layout) -> None:
    """立方体 3 個の位置・姿勢を書き、速さを 0 にする。箱の中の立方体は置き場所の中心、箱の底からわずかに浮かせる
    （着地させて静止させるのは SimRig.reset の役目）。mj_forward は呼ぶ側で行う。"""
    box = frames.box_pos(model)
    pre_yaw = np.radians(float(_SCENE["prefilled_yaw_deg"]))
    lift = 1e-4                                            # 底の上面との初期の重なりを避ける
    for color in COLORS:
        qa, va = cube_qpos_adr(model, color)
        if color in layout.cubes:
            x, y, yaw = layout.cubes[color]
            pos = (x, y, frames.CUBE_REST_Z)
        elif color in layout.prefilled:
            sx, sy = frames.slot_xy(box, layout.prefilled[color])
            pos, yaw = (sx, sy, frames.BOX_FLOOR_Z + frames.CUBE_HALF + lift), pre_yaw
        else:
            raise ValueError(f"layout {layout.seed}: color {color} neither on the table nor in the box")
        data.qpos[qa:qa + 3] = pos
        data.qpos[qa + 3:qa + 7] = frames.yaw_quat(yaw)
        data.qvel[va:va + 6] = 0.0


def check_scene_colors(model) -> None:
    """場面の XML の立方体の色が設定（scene.colors）と同じであること。"""
    for color in COLORS:
        got = [float(v) for v in model.geom_rgba[model.geom(frames.cube_geom(color)).id]]
        want = [float(v) for v in _SCENE["colors"][color]]
        if not np.allclose(got, want):
            raise RuntimeError(f"scene color {color}: xml {got} != configs {want}")


def check_box_geometry(model) -> None:
    """場面の XML の箱が設定（scene.box）と同じ寸法であること: 壁の内面が中心から ±inner_half、壁の上端が
    wall_top_z、底の上面が floor_z、箱の中心が pos。"""
    box = _SCENE["box"]
    b = model.body(frames.BOX_BODY)
    got = {"pos": [float(v) for v in b.pos]}
    for w, axis, sign in (("box_wall_xp", 0, 1), ("box_wall_xn", 0, -1), ("box_wall_yp", 1, 1), ("box_wall_yn", 1, -1)):
        g = model.geom(w)
        got[f"{w}_inner"] = float(sign * (g.pos[axis] - sign * g.size[axis]))
        got[f"{w}_top"] = float(g.pos[2] + g.size[2])
    fb = model.geom("box_bottom")
    got["floor"] = float(fb.pos[2] + fb.size[2])
    ok = np.allclose(got["pos"], box["pos"]) and abs(got["floor"] - float(box["floor_z"])) < 1e-9
    for w in ("box_wall_xp", "box_wall_xn", "box_wall_yp", "box_wall_yn"):
        ok &= abs(got[f"{w}_inner"] - float(box["inner_half"])) < 1e-9 and abs(got[f"{w}_top"] - float(box["wall_top_z"])) < 1e-9
    if not ok:
        raise RuntimeError(f"scene box {got} != configs scene.box {box}")


def build_model(variant: str = "3cube") -> mujoco.MjModel:
    if variant == "cup_plate":                       # 別のタスクの場面（XML の場所は configs/scenes/cup_plate.yaml）
        sc = cup_plate_config()
        model = mujoco.MjModel.from_xml_path(str(config.path(sc["xml"])))
        apply_cup_plate_material(model, sc)
        check_cup_plate_geometry(model, sc)
        return model
    key = {"3cube": "scene", "g0": "scene_g0"}[variant]
    model = mujoco.MjModel.from_xml_path(str(config.path(_CFG["paths"][key])))
    if variant == "3cube":
        check_scene_colors(model)
        check_box_geometry(model)
    return model


# ------------------------------------------------------------------------------ カップと皿（variant "cup_plate"）
CUP_PLATE = "configs/scenes/cup_plate.yaml"


@functools.lru_cache(maxsize=None)
def _cup_plate_file(path: str) -> dict:
    return yaml.safe_load(config.path(path).read_text(encoding="utf-8"))["cup_plate"]


def cup_plate_config(path: str = CUP_PLATE) -> dict:
    """configs/scenes/cup_plate.yaml の cup_plate（場面と評価だけが読む）。写しを返す。"""
    import copy
    return copy.deepcopy(_cup_plate_file(str(path)))


@dataclasses.dataclass(frozen=True)
class CupPlateLayout:
    seed: int
    start: str                        # home | retreat
    cup: tuple                        # カップの (x, y, yaw)（底の中心）
    plate: tuple                      # 皿の (x, y, yaw)（底の中心）
    tries: int = 1                    # カップを引いた回数（棄却を含む）
    kind: str = "cup_plate"

    @property
    def table_colors(self) -> tuple:  # 3 色の場面の Layout と同じ口（立方体は無い）
        return ()

    def to_json(self) -> dict:
        return {"seed": self.seed, "kind": self.kind, "start": self.start, "cup": [float(v) for v in self.cup],
                "plate": [float(v) for v in self.plate], "tries": self.tries}

    @classmethod
    def from_json(cls, d: dict) -> "CupPlateLayout":
        return cls(int(d["seed"]), d["start"], tuple(d["cup"]), tuple(d["plate"]), int(d.get("tries", 1)))


def sample_cup_plate_layout(seed: int, sc: dict = None, start: str = "home") -> CupPlateLayout:
    """乱数のシードから皿とカップの位置・向きを決める（皿 → カップの順。カップは皿と重なれば引き直す）。"""
    sc = sc or cup_plate_config()
    lay = sc["layout"]
    if start not in STARTS:
        raise ValueError(f"start {start!r} not in {STARTS}")
    rng = seeds.stream(seed, "layout")
    half_yaw = np.radians(float(lay["yaw_range_deg"]))
    pr = lay["plate_region"]
    plate = (float(rng.uniform(*pr["x"])), float(rng.uniform(*pr["y"])), float(rng.uniform(-half_yaw, half_yaw)))
    need = float(sc["plate"]["radius"]) + float(sc["cup"]["radius"]) + float(lay["min_gap_m"])
    cr = lay["cup_region"]
    for tries in range(1, int(lay["max_tries"]) + 1):
        x, y = float(rng.uniform(*cr["x"])), float(rng.uniform(*cr["y"]))
        yaw = float(rng.uniform(-half_yaw, half_yaw))
        if np.hypot(x - plate[0], y - plate[1]) >= need:
            return CupPlateLayout(int(seed), start, (x, y, yaw), plate, tries)
    raise RuntimeError(f"cup_plate layout seed {seed}: no placement after {lay['max_tries']} draws")


def apply_cup_plate_layout(model, data, layout: CupPlateLayout, sc: dict = None) -> None:
    """カップ（自由関節）を机の上に立てて置き、速さを 0 にする。皿は固定の体なので model.body_pos・body_quat を書く
    （同じモデルを使うセンサのモデルの描画にも写る）。机からわずかに浮かせる（静止させるのは SimRig.reset）。mj_forward は呼ぶ側。"""
    sc = sc or cup_plate_config()
    lift = 1e-4
    top = frames.TABLE_TOP_Z
    jnt = model.body(sc["cup"]["body"]).jntadr[0]
    qa, va = int(model.jnt_qposadr[jnt]), int(model.jnt_dofadr[jnt])
    x, y, yaw = layout.cup
    data.qpos[qa:qa + 3] = (x, y, top + lift)
    data.qpos[qa + 3:qa + 7] = frames.yaw_quat(yaw)
    data.qvel[va:va + 6] = 0.0
    b = model.body(sc["plate"]["body"]).id
    px, py, pyaw = layout.plate
    model.body_pos[b] = (px, py, top)
    model.body_quat[b] = frames.yaw_quat(pyaw)


def _partner_geoms(model, sc: dict) -> list:
    """カップが触れる相手の geom: 机と、material.partner_bodies（ハンド・指）の体の衝突する geom。"""
    mat = sc.get("material") or {}
    bodies = {model.body(b).id for b in mat.get("partner_bodies", ())}
    return [g for g in range(model.ngeom) if (model.geom(g).name == "table_geom" or int(model.geom_bodyid[g]) in bodies)
            and (model.geom_contype[g] or model.geom_conaffinity[g])]


def apply_cup_plate_material(model, sc: dict) -> None:
    """陶器の滑りの係数（configs/scenes/cup_plate.yaml の material）: 相手（ハンド・指）の滑りの係数をこの値以下へ下げる
    （接触の係数は大きい方なので、カップとの接触がこの値になる。panda.xml は他の場面と共用なので読み込みの後に書く）。"""
    mat = sc.get("material")
    if mat is None:
        return
    mu = float(mat["friction"][0])
    for g in _partner_geoms(model, sc):
        model.geom_friction[g, 0] = min(float(model.geom_friction[g, 0]), mu)


def check_cup_plate_geometry(model, sc: dict) -> None:
    """場面の XML のカップ・皿が設定（configs/scenes/cup_plate.yaml）と同じ寸法・色であること。"""
    cup, plate = sc["cup"], sc["plate"]
    got, want = {}, {}
    base = model.geom("cup_base")
    walls = [model.geom(g).id for g in range(model.ngeom) if model.geom(g).name.startswith("cup_wall_")]
    got["cup_radius"] = float(base.size[0])
    got["cup_base"] = float(2.0 * base.size[1])
    got["cup_height"] = float(max(model.geom_pos[g][2] + model.geom_size[g][2] for g in walls))
    got["cup_outer"] = float(max(np.hypot(*model.geom_pos[g][:2]) + model.geom_size[g][0] for g in walls))
    got["cup_wall"] = float(2.0 * model.geom_size[walls[0]][0])
    got["cup_segments"] = len(walls)
    want.update(cup_radius=float(cup["radius"]), cup_base=float(cup["base"]), cup_height=float(cup["height"]),
                cup_outer=float(cup["radius"]), cup_wall=float(cup["wall"]), cup_segments=int(cup["segments"]))
    pg = model.geom("plate_geom")
    got["plate_radius"], got["plate_thickness"] = float(pg.size[0]), float(2.0 * pg.size[1])
    got["plate_top"] = float(pg.pos[2] + pg.size[1])
    want.update(plate_radius=float(plate["radius"]), plate_thickness=float(plate["thickness"]),
                plate_top=float(plate["thickness"]))
    bad = {k: (got[k], want[k]) for k in want if abs(float(got[k]) - float(want[k])) > 1e-6}
    mat = sc.get("material")
    for name, geoms, rgba in (("cup", [base.id] + walls, cup["rgba"]), ("plate", [pg.id], plate["rgba"])):
        for g in geoms:
            if not np.allclose(model.geom_rgba[g], rgba):
                bad[f"{name}_rgba"] = ([float(v) for v in model.geom_rgba[g]], list(rgba))
            if mat is not None and not np.allclose(model.geom_friction[g], mat["friction"]):
                bad[f"{name}_friction"] = ([float(v) for v in model.geom_friction[g]], list(mat["friction"]))
            if int(model.geom_priority[g]) != 0:
                bad[f"{name}_priority"] = int(model.geom_priority[g])
    if mat is not None:                              # 相手（机・ハンド・指）の滑りの係数が陶器の値以下（接触がこの値になる）
        hi = [model.geom(g).name or g for g in _partner_geoms(model, sc)
              if float(model.geom_friction[g, 0]) > float(mat["friction"][0]) + 1e-12]
        if hi or not _partner_geoms(model, sc):
            bad["partner_friction"] = hi or "no partner geoms"
    if bad:
        raise RuntimeError(f"scene cup_plate (xml, configs): {bad}")
