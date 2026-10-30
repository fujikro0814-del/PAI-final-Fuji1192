"""点群からの形の推定（api/geometry）と推定の版の知覚（api/estimated）の確かめ。
合成の点群（tests/api/synth_points.py）: 斜めの箱（向き 0〜180°・傾き数度・正方形でない箱）、斜めの立方体（机の上で向き、
手の中で傾き）、直方体、深度の雑音、欠け（ばらばらの抜けとアクションチャンクの抜け）。既知の寸法は推定に渡さない。CPU だけ。"""
import math
import types

import numpy as np
import pytest

from recovla.api import estimated as EST
from recovla.api import geometry as G
from recovla.api import task_spec as TS

from . import synth_points as S

CUBE = np.array([0.04, 0.04, 0.04])


def yaw_err_deg(est_yaw, true_yaw, period):
    d = math.degrees(est_yaw - true_yaw)
    return abs((d + period / 2) % period - period / 2)


def cube_case(rng, yaw, tilt=0.0, lift=0.0, size=CUBE, cams=S.CAMS, **kw):
    R = S.rot(yaw, tilt)
    c = np.array([0.45, -0.10, S.resting_center(R, size) + lift])
    return S.cuboid(rng, c, R, size, cams=cams, **kw), c, R


# ---------------------------------------------------------------- 物
@pytest.mark.parametrize("yaw", [0.0, 0.3, 0.8, 1.2, 1.5])
def test_cube_on_the_table_at_any_yaw(yaw):
    rng = np.random.default_rng(10)
    P, c, R = cube_case(rng, yaw)
    g = G.fit_object(P, noise=0.0015)
    assert g is not None and g.supported and g.faces >= 2
    assert np.linalg.norm(g.center - c) < 0.0015
    assert np.abs(np.sort(g.extent) - 0.04).max() < 0.002
    assert yaw_err_deg(g.yaw, yaw, 90.0) < 1.0                                   # 立方体の向きは 90° ごとに同じ
    assert math.degrees(g.tilt) < 1.0 and g.top_h == pytest.approx(0.04, abs=0.002)
    assert g.height == pytest.approx(0.04, abs=0.002) and g.sigma < 0.01


@pytest.mark.parametrize("tilt", [0.2, 0.35, 0.6])
def test_tilted_cube_in_the_hand(tilt):
    """斜めに掴んで持ち上げた立方体（宙、2 つのカメラ）: 向き・傾き・3 辺。見かけの高さは辺より大きい。"""
    rng = np.random.default_rng(11)
    P, c, R = cube_case(rng, 0.5, tilt, lift=0.08, cams=S.CAMS2)
    g = G.fit_object(P, noise=0.0015)
    assert not g.supported and g.extended == 0.0
    assert np.linalg.norm(g.center - c) < 0.002
    assert np.abs(np.sort(g.extent) - 0.04).max() < 0.002
    assert abs(math.degrees(g.tilt) - math.degrees(math.acos(R[2, 2]))) < 1.5
    assert g.height > 0.04 + 0.01 * math.sin(tilt)


def test_block_is_not_assumed_to_be_a_cube():
    rng = np.random.default_rng(12)
    size = np.array([0.06, 0.03, 0.02])
    P, c, R = cube_case(rng, 0.6, size=size)
    g = G.fit_object(P, noise=0.0015)
    assert np.abs(g.extent - size).max() < 0.002                                  # 長い辺が x、上向きが z
    assert yaw_err_deg(g.yaw, 0.6, 180.0) < 1.0 and np.linalg.norm(g.center - c) < 0.0015


def test_cube_with_noise_and_missing_points():
    """雑音 2.5 mm、40% の抜け、側面のアクションチャンクの抜け、机の上（向き）と手の中（傾き）を 40 通り。"""
    rng = np.random.default_rng(5)
    worst = {"c": 0.0, "ext": 0.0, "yaw": 0.0, "tilt": 0.0}
    for k in range(40):
        yaw, tilt = rng.uniform(0, math.pi / 2), (rng.uniform(0.05, 0.5) if k % 2 else 0.0)
        R = S.rot(yaw, tilt)
        c = np.array([0.45, -0.10, S.resting_center(R, CUBE) + (0.06 if tilt else 0.0)])
        P = S.cuboid(rng, c, R, CUBE, cams=S.CAMS2 if tilt else S.CAMS, noise=0.0025, drop=0.4,
                     hole=c + R @ [0.02, 0.0, 0.02], hole_r=0.012)
        g = G.fit_object(P, noise=0.0025)
        worst["c"] = max(worst["c"], np.linalg.norm(g.center - c))
        worst["ext"] = max(worst["ext"], np.abs(np.sort(g.extent) - 0.04).max())
        if tilt:
            worst["tilt"] = max(worst["tilt"], abs(math.degrees(g.tilt) - math.degrees(math.acos(R[2, 2]))))
        else:
            worst["yaw"] = max(worst["yaw"], yaw_err_deg(g.yaw, yaw, 90.0))
    assert worst["c"] < 0.003 and worst["ext"] < 0.004 and worst["yaw"] < 2.0 and worst["tilt"] < 2.0, worst


def test_hidden_bottom_is_extended_to_the_support():
    """切り出しの高さ（机から 6 mm）より下の点が無い: 置かれていると判定して、支える面まで延ばす。宙の物は延ばさない。"""
    rng = np.random.default_rng(13)
    P, c, R = cube_case(rng, 0.4)
    g = G.fit_object(P[P[:, 2] > 0.006], noise=0.0015)
    assert g.supported and g.extended > 0.003
    assert g.extent[2] == pytest.approx(0.04, abs=0.002) and g.center[2] == pytest.approx(0.02, abs=0.0015)
    assert g.extent_sigma[2] > g.extent_sigma[0]                                  # 延ばした向きは不確か
    P2, c2, _ = cube_case(rng, 0.4, lift=0.05)
    g2 = G.fit_object(P2, noise=0.0015)
    assert not g2.supported and g2.extended == 0.0
    # 箱の底（机から 1 cm）の上の物: 支える面を渡すとそフレームで延ばす
    P3 = S.cuboid(rng, np.array([0.45, 0.25, 0.03]), S.rot(0.2), CUBE)
    g3 = G.fit_object(P3[P3[:, 2] > 0.016], noise=0.0015, support_h=0.01)
    assert g3.center[2] == pytest.approx(0.03, abs=0.002) and g3.extent[2] == pytest.approx(0.04, abs=0.002)


def test_too_few_points_is_none():
    assert G.fit_object(np.zeros((5, 3))) is None and G.fit_container(np.zeros((10, 3))) is None


# ---------------------------------------------------------------- 箱らしい物
@pytest.mark.parametrize("yaw, roll, pitch", [(0.0, 0.0, 0.0), (0.4, 0.0, 0.0), (0.9, 0.05, 0.0), (1.3, 0.03, -0.04),
                                              (2.6, -0.06, 0.02)])
def test_square_box_any_yaw_and_tilt(yaw, roll, pitch):
    rng = np.random.default_rng(20)
    P, T = S.container(rng, (0.45, 0.25), yaw, (0.085, 0.085), 0.005, 0.06, roll=roll, pitch=pitch)
    b = G.fit_container(P, noise=0.0015)
    assert b is not None
    assert np.linalg.norm(b.center - T["rim_center"]) < 0.003
    assert yaw_err_deg(b.yaw, yaw, 90.0) < 1.0                                    # 正方形の箱は 90° ごとに同じ
    assert np.abs(b.outer - 0.085).max() < 0.0015 and np.abs(b.inner - 0.080).max() < 0.0015
    assert b.wall == pytest.approx(0.005, abs=0.0015)
    assert b.rim_h == pytest.approx(T["rim_center"][2], abs=0.0015)
    assert b.floor == pytest.approx(T["floor_local"], abs=0.002)
    assert math.degrees(math.acos(min(1.0, b.R[:, 2] @ T["R"][:, 2]))) < 0.5         # 縁の法線（傾き）
    assert abs(math.degrees(b.tilt) - math.degrees(math.acos(T["R"][2, 2]))) < 0.5


def test_box_roll_and_pitch_are_reported_about_the_box_axes():
    rng = np.random.default_rng(21)
    for roll, pitch in ((0.05, 0.0), (0.0, 0.05), (-0.04, 0.03)):
        P, T = S.container(rng, (0.45, 0.25), 0.0, (0.10, 0.07), 0.005, 0.06, roll=roll, pitch=pitch)
        b = G.fit_container(P, noise=0.0015)
        s = 1.0 if b.R[0, 0] > 0 else -1.0                                        # 長い辺の向きの符号は決めない
        assert b.roll * s == pytest.approx(roll, abs=0.006) and b.pitch * s == pytest.approx(pitch, abs=0.006)


def test_box_is_not_assumed_square():
    rng = np.random.default_rng(22)
    P, T = S.container(rng, (0.45, 0.25), 1.1, (0.10, 0.065), 0.006, 0.05, roll=0.03)
    b = G.fit_container(P, noise=0.0015)
    assert b.outer[0] == pytest.approx(0.10, abs=0.0015) and b.outer[1] == pytest.approx(0.065, abs=0.0015)
    assert b.inner[0] == pytest.approx(0.094, abs=0.0015) and b.inner[1] == pytest.approx(0.059, abs=0.0015)
    assert yaw_err_deg(b.yaw, 1.1, 180.0) < 1.0                                   # 長い辺の向き（180° の対称だけ）
    assert b.rim_h == pytest.approx(T["rim_center"][2], abs=0.0015)


def test_box_with_noise_missing_corner_and_dropout():
    rng = np.random.default_rng(23)
    worst = {"c": 0.0, "inner": 0.0, "normal": 0.0, "yaw": 0.0, "rim": 0.0}
    for k in range(15):
        yaw, (roll, pitch) = rng.uniform(0, math.pi), rng.uniform(-0.06, 0.06, 2)
        outer = (rng.uniform(0.08, 0.11), rng.uniform(0.06, 0.075))
        P, T = S.container(rng, (0.45, 0.25), yaw, outer, 0.005, 0.06, roll=roll, pitch=pitch, noise=0.0025, drop=0.4)
        corner = T["rim_center"] + T["R"] @ [outer[0], outer[1], 0.0]
        P = P[np.linalg.norm(P - corner, axis=1) > 0.03]                          # 腕に隠れた角
        b = G.fit_container(P, noise=0.0025)
        worst["c"] = max(worst["c"], np.linalg.norm(b.center - T["rim_center"]))
        err_inner = np.abs(b.inner - (np.array(outer) - 0.005)).max()
        worst["inner"] = max(worst["inner"], err_inner)
        assert err_inner <= 2.5 * b.sigma["inner"], (k, err_inner, b.sigma)          # 誤差は不確かさの目安の中
        worst["normal"] = max(worst["normal"], math.degrees(math.acos(min(1.0, b.R[:, 2] @ T["R"][:, 2]))))
        worst["yaw"] = max(worst["yaw"], math.degrees(math.acos(min(1.0, abs(b.R[:, 0] @ T["R"][:, 0])))))
        worst["rim"] = max(worst["rim"], abs(b.rim_h - T["rim_center"][2]))
    # 雑音 2.5 mm は壁の厚さ 5 mm と同じ程度なので、内寸は 5 mm まで（雑音 1.5 mm の場合は上のテストで 1.5 mm）
    assert worst["c"] < 0.004 and worst["inner"] < 0.005 and worst["normal"] < 0.6 and worst["yaw"] < 1.5, worst
    assert worst["rim"] < 0.002, worst


def test_box_region_depth_over_xy_and_slots():
    rng = np.random.default_rng(24)
    P, T = S.container(rng, (0.45, 0.25), 0.5, (0.085, 0.085), 0.005, 0.06, roll=0.05)
    b = G.fit_container(P, noise=0.0015)
    inside = T["rim_center"] + T["R"] @ [0.02, -0.03, -0.03]
    above = T["rim_center"] + T["R"] @ [0.0, 0.0, 0.02]
    out = T["rim_center"] + T["R"] @ [0.12, 0.0, -0.03]
    assert b.depth(inside) > 0.015 and b.depth(above) < 0 and b.depth(out) < -0.03
    assert b.over_xy(above) and not b.over_xy(out)
    # 傾きを含む: 縁の法線の向きに沿った柱の中か。机の鉛直で見ると外でも、箱の座標では中の点
    tilted = T["rim_center"] + T["R"] @ [0.078, 0.0, -0.045]
    assert b.depth(tilted) > 0
    s = np.array([b.slot(i) for i in range(4)])
    assert all(b.over_xy(p) for p in s) and len({tuple(np.round(p, 3)) for p in s}) == 4
    assert np.linalg.norm(s[0] - s[3]) == pytest.approx(math.hypot(*b.inner), abs=0.002)    # 内寸の半分ずつ


def test_table_frame_round_trip():
    f = G.TableFrame.from_plane([0.02, -0.01, 1.0], 0.003)
    p = np.array([[0.4, 0.1, 0.05], [0.5, -0.2, 0.0]])
    assert np.allclose(f.to_world(f.to_table(p)), p)
    assert f.R[:, 2] @ np.array([0.0, 0.0, 1.0]) > 0.99


# ---------------------------------------------------------------- 推定の版の知覚
class NoKnownSizes:
    """既知の情報の代わり: 机の平面だけ読める。既知の大きさ・寸法を読んだら失敗する。"""
    table_normal = np.array([0.0, 0.0, 1.0])
    table_z = 0.0

    def __getattr__(self, k):
        raise AssertionError(f"推定の版が既知の値を読んだ: {k}")


def scene(rng, cube_states, box_yaw=0.4, box_roll=0.04, noise=0.0015):
    """箱と色の立方体の点群（Perception.points と同じ形の辞書）。cube_states: {色: (中心, R)}。"""
    Pb, T = S.container(rng, (0.45, 0.25), box_yaw, (0.085, 0.085), 0.005, 0.06, roll=box_roll, noise=noise)
    pts, col = [Pb], [np.full(len(Pb), 3)]
    for ci, color in enumerate(("red", "green", "blue")):
        if color in cube_states:
            c, R = cube_states[color]
            P = S.cuboid(rng, c, R, CUBE, noise=noise)
            pts.append(P)
            col.append(np.full(len(P), ci))
    pw = np.vstack(pts)
    return {"pw": pw, "h": pw[:, 2].copy(), "col": np.concatenate(col)}, T


def test_estimated_finder_without_known_sizes():
    rng = np.random.default_rng(30)
    spec = TS.load_task()
    f = EST.EstimatedGeometryFinder.from_spec(spec, setup=NoKnownSizes())
    probe = S.container(rng, (0.45, 0.25), 0.4, (0.085, 0.085), 0.005, 0.06, roll=0.04)[1]
    inbox = probe["rim_center"] + probe["R"] @ [0.03, 0.02, -0.03]
    states = {"red": (np.array([0.40, -0.05, 0.02]), S.rot(0.3)), "green": (inbox, probe["R"] @ S.rot(0.7))}
    P, T = scene(rng, states)
    est = f.estimate_clouds([P], 1.0)
    assert est.finder == "estimated" and est.table_z == 0.0
    r, g = est.objects["red"], est.objects["green"]
    assert r.status == g.status == "seen" and "blue" not in est.objects
    assert np.linalg.norm(r.pos - states["red"][0]) < 0.002 and np.linalg.norm(g.pos - inbox) < 0.002
    assert np.abs(np.sort(r.extent) - 0.04).max() < 0.002 and np.abs(np.sort(g.extent) - 0.04).max() < 0.002
    assert r.in_place == {"box": False} and g.in_place == {"box": True}            # 傾いた箱の 3 次元の内側で判定
    pl = est.places["box"]
    assert pl.ok and np.linalg.norm(pl.region.center - T["rim_center"]) < 0.003
    assert np.abs(pl.region.inner - 0.08).max() < 0.0015 and pl.inner_half_m is None
    assert abs(math.degrees(pl.region.tilt) - math.degrees(math.acos(T["R"][2, 2]))) < 0.5
    assert r.cov.shape == (3, 3) and r.R.shape == (3, 3)


def test_estimated_finder_tracks_hidden_and_in_hand():
    rng = np.random.default_rng(31)
    spec = TS.load_task()
    f = EST.EstimatedGeometryFinder.from_spec(spec, setup=NoKnownSizes(), lost_s=1.0)
    tip = np.array([0.40, -0.05, 0.02])
    P, _ = scene(rng, {"red": (tip.copy(), S.rot(0.3)), "blue": (np.array([0.5, -0.1, 0.02]), S.rot(1.0))})
    est = f.estimate_clouds([P], 1.0, types.SimpleNamespace(is_grasped=True), tip)
    assert est.objects["red"].near_hand is True and est.objects["blue"].near_hand is False
    Pbox = {k: v[P["col"] == 3] for k, v in P.items()}                             # 次の時刻は箱だけ写る（立方体は隠れた）
    tip2 = tip + [0.0, 0.05, 0.10]
    est = f.estimate_clouds([Pbox], 1.2, types.SimpleNamespace(is_grasped=True), tip2)
    red, blue = est.objects["red"], est.objects["blue"]
    assert red.status == "occluded" and red.near_hand is True and np.allclose(red.pos - tip2, est.objects["red"].pos - tip2)
    assert np.linalg.norm(red.pos - tip2) < 0.002                                  # 指先と一緒に動く
    assert blue.status == "occluded" and np.linalg.norm(blue.pos - [0.5, -0.1, 0.02]) < 0.002
    est = f.estimate_clouds([Pbox], 2.5, types.SimpleNamespace(is_grasped=False), tip2)
    assert est.objects["blue"].status == "lost" and est.places["box"].ok           # 箱は前の推定を保つ
    est = f.estimate_clouds([], 3.0)
    assert est.places["box"].ok and est.objects["red"].status == "lost"


def test_geometry_perception_keeps_the_world_model(monkeypatch):
    """GeometryPerception は今の Perception と同じ WorldModel を返し、同じ点群から geometry を足す（点群を作り直さない）。"""
    from recovla.common import config
    from recovla.harness.setup import nominal_setup                             # テストの道具（既知の情報の公称値）
    from recovla.runtime import cue as C
    from recovla.runtime.perception import Params, Perception
    cfg = config.load(*config.V2_LAYERS)
    su = nominal_setup(cfg)
    thr = C.Thresholds.from_dict(config.color_detect(cfg))
    rng = np.random.default_rng(32)
    P, _ = scene(rng, {"red": (np.array([0.40, -0.05, 0.02]), S.rot(0.3))}, box_yaw=0.0, box_roll=0.0)
    P["plane"] = (np.array([0.0, 0.0, 1.0]), np.zeros(3))
    calls = []

    def fake_points(self, frame, width, margin=None):
        calls.append(frame.name)
        return P if frame.name == "overhead" else {"pw": np.zeros((0, 3)), "h": np.zeros(0), "col": np.zeros(0, int),
                                                   "plane": P["plane"]}
    monkeypatch.setattr(Perception, "points", fake_points)
    frames = {n: types.SimpleNamespace(name=n, depth=np.zeros((2, 2)), t_capture=1.0) for n in ("overhead", "wrist")}
    grip = types.SimpleNamespace(width=0.08, is_grasped=False)
    plain = Perception(su, Params.from_config(cfg["runtime_v2"]["perception"]), thr)
    geo = Perception(su, Params.from_config(cfg["runtime_v2"]["perception"]), thr)
    EST.GeometryPerception.attach(geo, EST.EstimatedGeometryFinder.from_spec(TS.load_task(), geo, su))
    a, b = plain.update(frames, grip, 1.0), geo.update(frames, grip, 1.0)
    assert sorted(a.cubes) == sorted(b.cubes) == ["red"]
    for c in a.cubes:
        ea, eb = a.cubes[c], b.cubes[c]
        assert np.array_equal(ea.pos, eb.pos) and (ea.status, ea.in_box, ea.yaw) == (eb.status, eb.in_box, eb.yaw)
    assert not hasattr(a, "geometry") and b.geometry.finder == "estimated"
    assert calls == ["overhead", "wrist"] * 2                                      # どちらも 1 フレームに 1 回ずつ
    assert np.linalg.norm(b.geometry.objects["red"].pos - [0.40, -0.05, 0.02]) < 0.002
    assert b.geometry.places["box"].ok


def test_fit_container_returns_none_on_degenerate_clouds():
    """縁の点が取れない点群（ばらばら・線・立てた板）でも例外を出さず None か箱を返す。"""
    rng = np.random.default_rng(40)
    clouds = [rng.uniform(-0.05, 0.05, (300, 3)) + [0.45, 0.25, 0.06],
              np.column_stack([np.linspace(0.4, 0.5, 200), np.full(200, 0.25), np.full(200, 0.06)]),
              np.column_stack([np.full(400, 0.45), rng.uniform(0.2, 0.3, 400), rng.uniform(0.0, 0.06, 400)])]
    for P in clouds:
        b = G.fit_container(P, noise=0.002, rng=np.random.default_rng(0))
        assert b is None or isinstance(b, G.BoxGeom)
