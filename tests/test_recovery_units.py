"""立て直しデータの生成の部品の単体検査（描画なし・outputs なしで通る）: 足跡の距離、失敗の意図的な失敗のパラメータ、
意図的な失敗から保存・再生まで。"""
import json

import numpy as np
import pytest

from recovla.common import config, seeds
from recovla.expert import generate as G
from recovla.expert import inject
from recovla.sim import frames, scene

CFG = config.load()
IC = CFG["inject"]


def test_polygon_distance_known_values():
    a = frames.rect_corners((0.0, 0.0), (0.02, 0.02))
    assert frames.polygon_distance(a, frames.rect_corners((0.07, 0.0), (0.02, 0.02))) == pytest.approx(0.03)
    assert frames.polygon_distance(a, frames.rect_corners((0.07, 0.07), (0.02, 0.02))) == pytest.approx(0.03 * np.sqrt(2))
    assert frames.polygon_distance(a, frames.rect_corners((0.03, 0.0), (0.02, 0.02))) == 0.0        # 重なる
    # 45° 回した立方体: 角が 0.02·√2 まで出る
    b = frames.rect_corners((0.1, 0.0), (0.02, 0.02), np.pi / 4)
    assert frames.polygon_distance(a, b) == pytest.approx(0.1 - 0.02 - 0.02 * np.sqrt(2))


def test_box_outer_half_and_walls():
    m = scene.build_model("3cube")
    outer = frames.box_outer_half(m)
    assert outer > frames.BOX_INNER_HALF
    box = frames.box_pos(m)
    inside = frames.rect_corners(box[:2], (frames.BOX_INNER_HALF - 1e-4, frames.BOX_INNER_HALF - 1e-4))
    for w, fp in frames.wall_footprints(m).items():
        assert frames.polygon_distance(inside, fp) == pytest.approx(1e-4, abs=1e-9), w


@pytest.mark.parametrize("kind", inject.KINDS)
def test_inject_params_deterministic_and_in_range(kind):
    modes = set()
    for seed in range(50000, 50040):
        lay = scene.sample_layout(seed)
        color = lay.table_colors[0]
        p = inject.sample_params(kind, seeds.inject_rng(seed, color, 0), lay, color)
        assert p == inject.sample_params(kind, seeds.inject_rng(seed, color, 0), lay, color)
        assert p.kind == kind
        if kind == "A":
            modes.add(p.a_mode)
            rng_ = IC["A"]["lateral_offset_m"] if p.a_mode == "lateral" else IC["A"]["raise_close_m"]
            assert rng_[0] <= abs(p.a_offset_m) <= rng_[1]
            assert IC["A"]["lift_m"][0] <= p.a_lift_m <= IC["A"]["lift_m"][1]
        elif kind == "B":
            assert 0.0 <= p.b_u < 1.0
        else:
            x, y = p.c_drop_xy
            assert CFG["scene"]["region"]["x"][0] <= x <= CFG["scene"]["region"]["x"][1]
            assert CFG["scene"]["region"]["y"][0] <= y <= CFG["scene"]["region"]["y"][1]
            for c, (ox, oy, _) in lay.cubes.items():
                if c != color:
                    assert np.hypot(x - ox, y - oy) >= 2 * frames.CUBE_HALF * np.sqrt(2) + IC["C"]["min_clearance_m"]
        json.dumps(p.to_json())
    if kind == "A":
        assert modes == {"lateral", "raise"}


# 描画なしで最後まで通るシード（検査用の 50000〜）。A は横ずらし
CASES = [("A", 50001, 0), ("B", 50001, 0), ("C", 50000, 0)]


@pytest.fixture(scope="module")
def rig():
    from recovla.sim.rig import SimRig
    r = SimRig(render=False)
    yield r
    r.close()


def _pick(kind, seed):
    lay = scene.sample_layout(seed)
    return lay.table_colors[seed % len(lay.table_colors)]


@pytest.mark.windows          # シードごとの結果（どのシードで確定して成功するか）はこの PC で選んだ（浮動小数の差で OS により変わりうる）
@pytest.mark.parametrize("kind,seed,retry", CASES)
def test_inject_save_and_replay(rig, tmp_path, kind, seed, retry):
    """意図的な失敗が確定し、10 Hz の境目から保存が始まり、最初のフレームが保存を始める時点の条件を満たし、立て直して成功し、
    保存した途中の状態から step 再生でビット一致する（test_recovery_data の 2・3 の小さな版）。"""
    from recovla.record import replay as legacy, replay_scene as RS
    color = _pick(kind, seed)
    a = G.run_attempt(rig, G.EpisodeSpec(seed, color, kind=kind), retry, tmp_path, render=False)
    assert a["inject"]["status"] == "confirmed", a["inject"]
    assert a["success"], a["failure"]
    assert a["record_start_step"] % (CFG["sim"]["record_every"] * CFG["sim"]["stride"]) == 0
    if kind == "A":
        assert a["inject"]["params"]["a_mode"] == "lateral"
    ep = legacy.load_episode(tmp_path / a["name"])
    cond = inject.start_condition(ep.data, ep.meta)
    assert cond["ok"], cond
    assert ep.data["start_step"] == a["record_start_step"]
    rep = RS.replay(ep, "step", rig, render=False)
    r = RS.compare(ep, rep, images=False)
    assert r["bit_exact_state"] and r["success_replayed"], r


def test_normal_attempt_unchanged_by_recovery_path(rig):
    """通常（n）の試みは、同じ指定なら同じ列になる（決定性）。"""
    spec = G.EpisodeSpec(50000, _pick("n", 50000))
    a = G.run_attempt(rig, spec, 0, None, False)
    b = G.run_attempt(rig, spec, 0, None, False)
    assert a["success"] and a["duration_s"] == b["duration_s"] and a["n_frames"] == b["n_frames"]
    fa, fb = a["_frames"], b["_frames"]
    assert all(np.array_equal(x["joints"], y["joints"]) for x, y in zip(fa, fb))
    assert "inject" not in a
