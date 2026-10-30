"""自分の失敗から学び直すデータの道具（expert/handover_real.py・expert/displace.py・generate_real の D の包み・
scripts/failure_relearn/failure_data.py）の確かめ。既存の種類（n・A・B・C）の生成が変わらないこと、外乱 D の行き先の決まり、
VLA から生成プログラムへの交代の判定、シードの範囲と学習の手数の決まり。シード 74900〜 は動作確認用（学習に使わない）。"""
import importlib.util
import pathlib
import re

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _r1():
    spec = importlib.util.spec_from_file_location("r1_data_t", ROOT / "scripts" / "failure_relearn" / "failure_data.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _truth(t=0.0, cube=None, tip=(0.45, -0.1, 0.2), closed=False, fingers=(0.04, 0.04), target=0, vel=None, quat=None):
    from recovla.expert import script as S
    cube = np.array(cube if cube is not None else [[0.45, -0.1, 0.02], [0.35, -0.15, 0.02], [0.45, 0.25, 0.03]], float)
    return S.Truth(t=t, target=target, cube_pos=cube,
                   cube_quat=np.array(quat if quat is not None else [[1, 0, 0, 0]] * 3, float),
                   cube_linvel=np.zeros((3, 3)) if vel is None else np.asarray(vel, float),
                   fingers=np.array(fingers, float), finger_vel=np.zeros(2), gripper_closed=closed,
                   hand_pos=np.array(tip, float) + [0, 0, 0.1], hand_vel=np.zeros(3), fingertip=np.array(tip, float),
                   x_cmd=np.array(tip, float), box=np.array([0.45, 0.25, 0.0]))


# ------------------------------------------------------------------ 既存の生成は変わらない
def test_inject_kinds_unchanged_and_d_is_separate():
    from recovla.expert import displace, inject
    assert inject.KINDS == ("A", "B", "C") and displace.KIND not in inject.KINDS


def test_inject_params_unchanged_for_abc():
    """D を足しても A・B・C の意図的な失敗の乱数の引き方は同じ（inject.sample_params は変えていない）。"""
    from recovla.common import seeds
    from recovla.expert import inject
    from recovla.sim import scene
    lay = scene.sample_layout(74950)
    c = lay.table_colors[0]
    a = inject.sample_params("A", seeds.inject_rng(74950, c, 0), lay, c)
    assert a == inject.sample_params("A", seeds.inject_rng(74950, c, 0), lay, c)
    b = inject.sample_params("B", seeds.inject_rng(74950, c, 0), lay, c)
    rng = seeds.inject_rng(74950, c, 0)
    for _ in range(5):                                  # A の 5 つ（向き・横のずれ・符号・上げ幅・持ち上げ）の後が B の u
        rng.random()
    assert b.b_u == pytest.approx(float(rng.random()))


def test_generate_real_passes_through_non_d(monkeypatch):
    from recovla.expert import generate as G
    from recovla.expert import generate_real as GR
    calls = []

    def fake(rig, spec, retry, run_dir=None, render=True, handover_at_s=None):
        calls.append((spec.kind, retry, run_dir, render, handover_at_s))
        return {"success": True, "kind": spec.kind}
    monkeypatch.setitem(GR._ORIG, "run_attempt", fake)
    for k in ("n", "A", "B", "C", "h"):
        out = GR.run_attempt_any(object(), G.EpisodeSpec(74951, "red", kind=k), 2, "x", False, 1.5)
        assert out == {"success": True, "kind": k}
    assert calls == [(k, 2, "x", False, 1.5) for k in ("n", "A", "B", "C", "h")]


def test_generate_real_patch_restores_run_attempt():
    from recovla.expert import generate as G
    from recovla.expert import generate_real as GR
    orig = G.run_attempt
    with GR.patched():
        assert G.run_attempt is GR.run_attempt_any
    assert G.run_attempt is orig


# ------------------------------------------------------------------ 外乱 D（狙いの物を動かす）の行き先
def test_displace_goal_respects_rules():
    from recovla.common import config
    from recovla.expert import displace as D
    cfg = config.load()
    sc = cfg["scene"]
    rng = np.random.default_rng(0)
    n_ok = 0
    for i in range(200):
        p = D.sample_params(D.rng_for(74960 + i % 20, "red", i // 20))
        start = np.array([rng.uniform(*sc["region"]["x"]), rng.uniform(*sc["region"]["y"])])
        yaw0 = float(np.radians(rng.uniform(-30, 30)))
        others = [start + np.array([0.0, 0.1]), start + np.array([0.12, 0.0])]
        g = D.choose_goal(p, start, yaw0, others, cfg)
        if not g["ok"]:
            continue
        n_ok += 1
        xy = np.array(g["xy"])
        assert D.DIST_M[0] - 1e-9 <= np.hypot(*(xy - start)) <= D.DIST_M[1] + 1e-9
        assert sc["region"]["x"][0] <= xy[0] <= sc["region"]["x"][1] and sc["region"]["y"][0] <= xy[1] <= sc["region"]["y"][1]
        assert abs(np.degrees(g["yaw"])) <= sc["yaw_range_deg"] + 1e-9 and abs(g["dyaw_deg"]) <= D.DYAW_DEG + 1e-9
        assert all(np.hypot(*(xy - o)) >= sc["min_center_dist"] for o in others)
        assert all(D.seg_point_dist(o, start, xy) >= D.PATH_CLEAR_M for o in others)
    assert n_ok > 100


def test_displace_rng_is_its_own_stream():
    from recovla.common import seeds
    from recovla.expert import displace as D
    a = D.sample_params(D.rng_for(74961, "green", 0))
    assert a == D.sample_params(D.rng_for(74961, "green", 0))
    assert a.u[0] != float(seeds.inject_rng(74961, "green", 0).random())
    assert D.TRIGGER_R_M[0] <= a.trigger_r_m <= D.TRIGGER_R_M[1] and D.SLIDE_S[0] <= a.slide_s <= D.SLIDE_S[1]


def test_displace_attempt_in_v3_real():
    """D の 1 本（描画なし）: 発動して動かし終わり、生成プログラムが動いた目標を掴んで箱に入れる。包みは外れる。"""
    from recovla.expert import generate as G
    from recovla.expert import generate_real as GR
    from recovla.harness import real_matched as V
    from recovla.sim import scene
    rig = V.V3RealGenRig(V.load_config("v3_real", generation=True))
    keys = ("pad_read", "physics_step", "episode_meta")
    before = {k: rig.__dict__.get(k) for k in keys}            # 腕のサーボは physics_step を自分で包んでいる
    try:
        lay = scene.sample_layout(74900)
        color = lay.table_colors[74900 % len(lay.table_colors)]
        out = GR.run_attempt(rig, G.EpisodeSpec(74900, color, kind="D"), 0, None, False)
        assert out["disturb"]["status"] == "done" and out["success"]
        moved = np.hypot(*(np.array(out["disturb"]["end_xy"]) - np.array(out["disturb"]["start_xy"])))
        assert 0.03 - 1e-6 <= moved <= 0.08 + 1e-6
        assert {k: rig.__dict__.get(k) for k in keys} == before
    finally:
        rig.close()


# ------------------------------------------------------------------ VLA から生成プログラムへの交代
def test_rollout_layout_strata():
    from recovla.common import config
    from recovla.expert import handover_real as H
    cfg = config.load()
    seen = set()
    for s in range(74900, 74940):
        lay, color, info = H.rollout_layout(s, cfg)
        assert (lay, color, info) == H.rollout_layout(s, cfg)
        assert color in lay.table_colors and info["stratum"] == H.stratum(s)
        x, y, yaw = lay.cubes[color]
        seen.add(info["stratum"])
        if info["stratum"] == "pf2":
            assert lay.kind == "prefilled_2" and len(lay.prefilled) == 2
        else:
            assert lay.kind in ("empty", "prefilled_1")
        if info["stratum"] == "yaw":
            assert 20.0 - 1e-6 <= abs(np.degrees(yaw)) <= 30.0 + 1e-6
        if info["stratum"] == "near":
            assert x < H.NEAR_X
            assert all(np.hypot(x - v[0], y - v[1]) >= cfg["scene"]["min_center_dist"] for c, v in lay.cubes.items() if c != color)
    assert seen == {"pf2", "yaw", "near", "free"}


def test_watch_miss_closed_empty():
    from recovla.expert import handover_real as H
    from recovla.expert import script as S
    w = H.Watch(S.PhaseParams.from_config(), 0.1)
    w.update(_truth(0.0, tip=(0.46, -0.1, 0.03)), False)
    t = 0.1
    while w.trigger is None and t < 2.0:
        w.update(_truth(t, tip=(0.46, -0.1, 0.03), closed=True, fingers=(0.001, 0.001)), True)
        t += 0.1
    assert (w.trigger, w.detail) == ("miss", "closed_empty") and t < 0.6


def test_watch_miss_reopened_and_far_close_ignored():
    from recovla.expert import handover_real as H
    from recovla.expert import script as S
    w = H.Watch(S.PhaseParams.from_config(), 0.1)
    w.update(_truth(0.0, tip=(0.45, 0.1, 0.2)), True)          # 目標から遠い空中で閉じた: 数えない
    w.update(_truth(0.1, tip=(0.45, 0.1, 0.2), closed=True), False)
    assert w.trigger is None and w.closes[0]["near"] is False
    w.update(_truth(0.2, tip=(0.45, -0.1, 0.03)), True)
    w.update(_truth(0.3, tip=(0.45, -0.1, 0.03), closed=True, fingers=(0.015, 0.015)), False)
    assert (w.trigger, w.detail) == ("miss", "reopened")


def test_watch_drop_and_misplace():
    from recovla.expert import handover_real as H
    from recovla.expert import script as S
    pp = S.PhaseParams.from_config()
    w = H.Watch(pp, 0.1)
    w.update(_truth(0.0, tip=(0.45, -0.1, 0.03)), True)
    up = [[0.45, -0.1, 0.10], [0.35, -0.15, 0.02], [0.45, 0.25, 0.03]]
    w.update(_truth(0.1, cube=up, tip=(0.45, -0.1, 0.10), closed=True), True)
    assert w.trigger is None
    down = [[0.45, -0.05, 0.02], [0.35, -0.15, 0.02], [0.45, 0.25, 0.03]]
    w.update(_truth(0.2, cube=down, tip=(0.45, -0.1, 0.12), closed=True), True)
    assert (w.trigger, w.detail) == ("drop", "closed_apart")
    w = H.Watch(pp, 0.1)
    w.update(_truth(0.0, tip=(0.45, -0.1, 0.03)), True)
    w.update(_truth(0.1, cube=up, tip=(0.45, -0.1, 0.10), closed=True), True)
    w.update(_truth(0.2, cube=down, tip=(0.45, -0.05, 0.12)), False)    # 放した（箱の外）
    for t in (0.5, 1.0, 1.5, 2.3):
        w.update(_truth(t, cube=down, tip=(0.45, -0.05, 0.2)), False)
    assert (w.trigger, w.detail) == ("misplace", "not_in_box_after_release")


def test_watch_wrong_cube_discards():
    from recovla.expert import handover_real as H
    from recovla.expert import script as S
    w = H.Watch(S.PhaseParams.from_config(), 0.1)
    w.update(_truth(0.0, tip=(0.45, -0.1, 0.03)), True)
    other_up = [[0.45, -0.1, 0.02], [0.35, -0.15, 0.06], [0.45, 0.25, 0.03]]
    w.update(_truth(0.1, cube=other_up, closed=True), True)
    assert w.discard == "wrong_cube" and w.trigger is None


def test_release_check_cube_and_wall():
    from recovla.expert import handover_real as H
    from recovla.sim import scene
    model = scene.build_model("3cube")
    box = np.array([0.45, 0.25])
    # 箱の中の立方体（置き場所 +4,+4 cm）の真上近くで放す → 悪い
    cube = [[0.47, 0.27, 0.10], [0.49, 0.29, 0.03], [0.30, -0.2, 0.02]]
    assert H.release_check(_truth(cube=cube, tip=(0.47, 0.27, 0.12)), model)["bad_cube"]
    # 空いた置き場所（-4,-4 cm）の上で放す → 良い
    cube = [[0.41, 0.21, 0.10], [0.49, 0.29, 0.03], [0.30, -0.2, 0.02]]
    r = H.release_check(_truth(cube=cube, tip=(0.41, 0.21, 0.12)), model)
    assert not r["bad"], r
    # 放す位置（中心）が壁の内面から 5 mm → 悪い
    cube = [[box[0], box[1] + 0.08 - 0.005, 0.10], [0.49, 0.29, 0.03], [0.30, -0.2, 0.02]]
    assert H.release_check(_truth(cube=cube), model)["bad_wall"]
    # 足跡は壁から 5 mm だが中心は 2.5 cm → 良い（箱に入る）
    cube = [[box[0], box[1] + 0.08 - 0.02 - 0.005, 0.10], [0.49, 0.29, 0.03], [0.30, -0.2, 0.02]]
    r = H.release_check(_truth(cube=cube), model)
    assert not r["bad_wall"] and r["wall_gap_m"] < 0.01


def test_state_check():
    from recovla.common import config
    from recovla.expert import handover_real as H
    from recovla.sim import scene
    cfg, model = config.load(), scene.build_model("3cube")
    ok = _truth(cube=[[0.36, -0.1, 0.02], [0.45, 0.25, 0.03], [0.30, -0.2, 0.02]])
    assert H.state_check(ok, cfg, model) is None
    wall = _truth(cube=[[0.45, 0.08, 0.02], [0.35, -0.15, 0.02], [0.30, -0.2, 0.02]])
    assert H.state_check(wall, cfg, model) == "near_wall"
    rim = _truth(cube=[[0.45, 0.16, 0.08], [0.35, -0.15, 0.02], [0.30, -0.2, 0.02]])
    assert H.state_check(rim, cfg, model) == "rim"


# ------------------------------------------------------------------ シードの範囲・チェックポイント・学習の手数
def test_bands_are_training_and_disjoint():
    m = _r1()
    rs = sorted(m.BANDS.values())
    for lo, hi in rs:
        assert 0 <= lo <= hi < 100000
    for (a0, a1), (b0, b1) in zip(rs, rs[1:]):
        assert a1 < b0
    with pytest.raises(SystemExit):
        m.check_band(171400, 171699)


def test_rreal_checkpoint_matches_ladder():
    m = _r1()
    text = (ROOT / "scripts" / "api" / "ladder.py").read_text(encoding="utf-8")
    assert re.search(r'^RREAL = "([^"]+)"', text, re.M).group(1) == m.RREAL


def test_training_steps_rule():
    m = _r1()
    assert m.training_steps(50659)["steps"] == 20000
    r = m.training_steps(120000)
    assert r["steps"] % 1000 == 0 and abs(r["epochs"] - r["epochs_rreal"]) < 0.3
    assert m.training_steps(120000)["steps"] == 48000


def test_recovery_share():
    m = _r1()
    fk = {"n": {"frames": 60}, "A": {"frames": 20}, "H": {"frames": 30}, "D": {"frames": 10}}
    assert m.recovery_share(fk) == pytest.approx(50 / 120, abs=1e-4)
