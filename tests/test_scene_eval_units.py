"""場面と評価の部品の単体検査（描画なし・outputs なしで通る）: 配置の抽選、開始状態の保存と復元、距離のモデル、評価器。"""
import collections

import mujoco
import numpy as np
import pytest

from recovla.common import config
from recovla.common.seeds import COLORS
from recovla.sim import contact, frames, scene

CFG = config.load()
SC = CFG["scene"]


def test_layout_constraints_and_determinism():
    kinds = collections.Counter()
    for seed in range(50000, 50600):
        lay = scene.sample_layout(seed)
        assert lay == scene.sample_layout(seed)                               # 同じシードなら同じ配置
        kinds[lay.kind] += 1
        assert len(lay.cubes) + len(lay.prefilled) == 3 and set(lay.cubes) | set(lay.prefilled) == set(COLORS)
        assert len(lay.prefilled) == scene.N_PREFILLED[lay.kind]
        assert len(set(lay.prefilled.values())) == len(lay.prefilled)        # 先客は別々の置き場所
        pts = list(lay.cubes.values())
        for x, y, yaw in pts:
            assert SC["region"]["x"][0] <= x <= SC["region"]["x"][1] and SC["region"]["y"][0] <= y <= SC["region"]["y"][1]
            assert abs(np.degrees(yaw)) <= SC["yaw_range_deg"]
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                assert np.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]) >= SC["min_center_dist"]
        assert scene.Layout.from_json(lay.to_json()) == lay
    ratio = {k: kinds[k] / 600 for k in scene.KINDS}
    want = {k: SC["layout_kinds"][k] / sum(SC["layout_kinds"].values()) for k in scene.KINDS}
    assert all(abs(ratio[k] - want[k]) < 0.06 for k in scene.KINDS), (ratio, want)
    forced = scene.sample_layout(50000, "prefilled_2")
    assert forced.kind == "prefilled_2" and len(forced.cubes) == 1


def test_scene_colors_and_slots():
    m = scene.build_model()                                                     # 色を設定と照合する
    assert frames.BOX_SLOTS.shape == (4, 2)
    box = frames.box_pos(m)
    for s in range(4):
        xy = frames.slot_xy(box, s)
        assert frames.in_box([xy[0], xy[1], frames.BOX_FLOOR_Z + frames.CUBE_HALF], box)


@pytest.fixture(scope="module")
def rig():
    from recovla.sim.rig import SimRig
    r = SimRig(render=False)
    yield r
    r.close()


def test_prefilled_cubes_rest_in_their_slots(rig):
    lay = scene.sample_layout(50200, "prefilled_2")
    rig.reset(lay)
    for c, s in lay.prefilled.items():
        p = rig.data.xpos[rig.cube_ids[COLORS.index(c)]]
        assert np.hypot(*(p[:2] - frames.slot_xy(rig.box, s))) < 1e-3
        assert abs(p[2] - (frames.BOX_FLOOR_Z + frames.CUBE_HALF)) < 1e-3
    assert np.all(np.abs(rig.data.qvel[rig.cube_vadr[0]:rig.cube_vadr[0] + 3]) < 1e-3)
    assert rig.data.time == 0.0 and rig.step == 0


def test_snapshot_restore_continues_bit_exactly(rig):
    from recovla.record import snapshot
    from recovla.sim.rig import quiet
    lay = scene.sample_layout(50003, "empty")
    rig.reset(lay)
    snap = snapshot.capture(rig.model, rig.data, rig.controller, rig.integrator, rig.step)

    def run():
        with quiet():
            for i in range(30):
                rig.pad_read(np.array([0.1, -0.05, -0.05]), press=(i == 10))
        return rig.data.qpos.copy(), rig.controller.q_des.copy(), rig.integrator.x_cmd.copy()

    a = run()
    snapshot.restore(rig.model, rig.data, rig.controller, rig.integrator, snap)
    b = run()
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    arrays = snap.to_arrays()
    snap2 = snapshot.SimSnapshot.from_arrays(arrays)
    assert np.array_equal(snap2.state, snap.state) and snap2.flags == snap.flags


def test_distance_model_matches_the_scene_and_uses_native_ccd(rig):
    meter = rig.meter
    dm, dd = meter.dmodel, meter.ddata
    assert dm.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_NATIVECCD)
    assert not rig.model.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_NATIVECCD)     # 物理のモデルは既定のまま
    rig.reset(scene.sample_layout(50000, "empty"))
    np.copyto(dd.qpos, rig.data.qpos)
    mujoco.mj_kinematics(dm, dd)
    m, d = rig.model, rig.data
    for g in meter.robot_geoms:
        if m.geom_type[g] != int(mujoco.mjtGeom.mjGEOM_BOX):
            continue
        assert dm.geom_type[g] == int(mujoco.mjtGeom.mjGEOM_MESH)
        s = m.geom_size[g]
        corners = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]) * s
        want = corners @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        mid = dm.geom_dataid[g]
        a, n = dm.mesh_vertadr[mid], dm.mesh_vertnum[mid]
        got = dm.mesh_vert[a:a + n] @ dd.geom_xmat[g].reshape(3, 3).T + dd.geom_xpos[g]
        assert max(np.linalg.norm(got - w, axis=1).min() for w in want) < 1e-6


def test_distances_known_case(rig):
    """立方体を、手・指のいちばん低い点（当て板の角とメッシュの頂点のうち最も低いもの）の真下に置く: 最短距離は、
    その点と立方体の上面の高さの差になる（指は垂直に下を向いており、ほかの点はそれより高い）。"""
    rig.reset(scene.sample_layout(50000, "empty"))
    m = rig.model
    d = mujoco.MjData(m)
    d.qpos[:] = rig.data.qpos
    mujoco.mj_forward(m, d)
    unit = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)])
    pts = []
    for g in rig.meter.robot_geoms:
        R, p = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
        if m.geom_type[g] == int(mujoco.mjtGeom.mjGEOM_BOX):
            pts.append(unit * m.geom_size[g] @ R.T + p)
        else:
            mid = m.geom_dataid[g]
            pts.append(m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]] @ R.T + p)
    pts = np.concatenate(pts)
    low = pts[np.argmin(pts[:, 2])]
    top = low[2] - 0.03
    qa, _ = scene.cube_qpos_adr(rig.model, "green")
    d.qpos[qa:qa + 7] = [low[0], low[1], top - frames.CUBE_HALF, 1, 0, 0, 0]
    mujoco.mj_forward(rig.model, d)
    dist = rig.meter.distances(d)[contact.COLUMNS.index("cube_green")]
    assert abs(dist - 0.03) < 2e-4, dist


def test_minimal_evaluator_success_error_and_metrics(rig, tmp_path):
    """記録した 1 本の行動を流す: 目標を正しく言えば成功、別の色を言えば誤り（持ち上げた色が目標でない）。"""
    from recovla.eval import metrics, scene_trial as T
    from recovla.expert import generate as G
    a = G.run_attempt(rig, G.EpisodeSpec(50000, "red", "empty"), 0, None, False)
    x = np.array([f["x_des"] for f in a["_frames"]])
    g = np.array([f["gripper_closed"] for f in a["_frames"]])
    acts = np.array([np.r_[x[2 * k + 2] - x[2 * k], 0, 0, 0, 1.0 if g[2 * k + 2] else -1.0]
                     for k in range((len(x) - 1) // 2)])

    def act(k, frame, raw, task):
        return acts[k] if k < len(acts) else np.r_[0, 0, 0, 0, 0, 0, acts[-1, 6]]

    lay = scene.Layout.from_json(a["layout"])
    rows = {}
    for i, tgt in enumerate(("red", "green")):
        meta, arr, _ = T.run_trial(rig, lay, tgt, act, {"trial": i, "seed": 50000, "experiment": "test",
                                                        "condition": "replay"}, time_limit_s=20.0, render=False)
        p = T.write_trial(tmp_path, i, meta, arr)
        rows[tgt] = metrics.trial_metrics(metrics.load_trial(p), CFG["eval"])
    assert rows["red"]["success"] and not rows["red"]["error"] and rows["red"]["contacts_n"] == 0
    assert not rows["green"]["success"] and rows["green"]["error"]


def test_choose_targets_is_balanced():
    from recovla.eval import scene_trial as T
    seeds_ = list(range(191000, 191030))
    lays = [scene.sample_layout(s, "empty") for s in seeds_]
    targets = T.choose_targets(seeds_, lays)
    assert collections.Counter(targets) == {"red": 10, "green": 10, "blue": 10}
    lays2 = [scene.sample_layout(s, "prefilled_1") for s in seeds_]
    for t, lay in zip(T.choose_targets(seeds_, lays2), lays2):
        assert t in lay.table_colors
