"""安全フィルタの検査: 障害物へまっすぐ向かう参照を与えても、最短距離が d_min を割らない
（許容 1 mm）。フィルタなしでは同じ動きで接触する（試験が意味を持つことの確かめ）。離れる動き・沿う動きは変えない。"""
import numpy as np
import pytest

from recovla.common import config
from recovla.common.seeds import COLORS
from recovla.sim import contact, frames, scene
from recovla.sim.safety import project

CFG = config.load()
TOL = 0.001


def test_project_is_the_nearest_feasible_point():
    A = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    b = np.array([0.0, -0.5])
    x, ok = project(np.array([-2.0, -3.0, 1.0]), A, b)
    assert ok and np.allclose(x, [0.0, -0.5, 1.0])
    x, ok = project(np.array([1.0, 1.0, 1.0]), A, b)        # 満たしていれば変えない
    assert ok and np.array_equal(x, [1.0, 1.0, 1.0])
    x, ok = project(np.array([-1.0, 0.0, 0.0]), np.array([[1.0, 1.0, 0.0]]) / np.sqrt(2), np.array([0.0]))
    assert ok and np.allclose(x, [-0.5, 0.5, 0.0])        # 斜めの面へ直交に


@pytest.fixture(scope="module")
def rig():
    from recovla.sim.rig import SimRig
    r = SimRig(render=False)
    yield r
    r.close()


def _setup(rig, enabled, seed=20000):
    lay = scene.sample_layout(seed, "empty", start="home")
    target = lay.table_colors[0]
    rig.reset(lay)
    rig.safety.enabled = enabled
    rig.safety.d_min, rig.safety.d_detect = float(CFG["safety_filter"]["d_min_m"]), float(CFG["safety_filter"]["d_detect_m"])
    rig.safety.start_trial(target)
    rig.safety.gate = True
    return lay, target


def _drive(rig, hand_goal, target, speed=0.15, max_reads=400):
    """参照（hand の原点）を hand_goal へ一定の速さでまっすぐ動かす。障害物の最短距離の最小を返す。"""
    mask = contact.ContactMeter.obstacle_mask(target, rig.cubes_in_box())
    dmin = np.inf
    for _ in range(max_reads):
        x = rig.integrator.x_cmd
        v = np.asarray(hand_goal) - x
        n = np.linalg.norm(v)
        vel = v / n * min(speed, n / (rig.steps_per_read * rig.timestep)) if n > 1e-6 else np.zeros(3)
        rig.pad_read(vel)
        mujoco_fwd(rig)
        dmin = min(dmin, float(rig.meter.distances(rig.data)[mask].min()))
    return dmin


def mujoco_fwd(rig):
    import mujoco
    mujoco.mj_forward(rig.model, rig.data)


def _cube_approach(rig, target):
    """目標外の机上の立方体へ、指先の高さを立方体の中心に合わせて横から向かう（行き先は立方体の中心の向こう 2 cm）。"""
    other = [c for c in COLORS if c != target and c in rig.layout.table_colors][0]
    cube = rig.data.xpos[rig.cube_ids[COLORS.index(other)]].copy()
    off = rig.data.xpos[rig.hand_id] - frames.fingertip_center(rig.data, rig.hand_id)
    away = np.r_[cube[:2] - rig.box[:2], 0.0]           # 箱と反対の側から近づく（途中で壁に触れないように）
    away /= np.linalg.norm(away)
    start = cube + 0.16 * away
    goal = cube - 0.02 * away
    return start + off, goal + off


def _wall_approach(rig):
    """箱の外から、壁の上端より低い高さで壁へ横から向かう。(始める点, 行き先, 外向きの単位の向き) か None"""
    box = rig.box
    half = frames.BOX_INNER_HALF
    off = rig.data.xpos[rig.hand_id] - frames.fingertip_center(rig.data, rig.hand_id)
    z = frames.TABLE_TOP_Z + 0.04
    cubes = [rig.data.xpos[rig.cube_ids[COLORS.index(c)]][:2] for c in rig.layout.table_colors]
    ws = CFG["sim"]["workspace"]
    for u in ([0, -1], [0, 1], [-1, 0], [1, 0]):        # 始める点が机上の立方体から離れていて、作業空間の中の面
        u = np.array(u, float)
        start = np.r_[box[:2] + (half + 0.11) * u, z]
        s_hand = start + off
        inside = all(ws[a][0] <= s_hand[i] <= ws[a][1] for i, a in enumerate("xy"))
        if inside and all(np.linalg.norm(start[:2] - c) > 0.10 for c in cubes):
            return s_hand, np.r_[box[:2], z] + off, np.r_[u, 0.0]
    return None


def _wall_seed(rig):
    """壁の検査に使う配置: シード 20000 から順に、条件に合う面を持つ最初の配置（決まった手順なので毎回同じ）。"""
    for seed in range(20000, 20060):
        _setup(rig, False, seed)
        if _wall_approach(rig) is not None:
            return seed
    raise AssertionError("no layout with a clear side of the box in 20000-20059")


@pytest.mark.parametrize("which", ["cube", "wall"])
def test_filter_keeps_distance_and_without_it_contact_happens(rig, which):
    res = {}
    seed = 20000 if which == "cube" else _wall_seed(rig)
    for enabled in (False, True):
        lay, target = _setup(rig, enabled, seed)
        start, goal = _cube_approach(rig, target) if which == "cube" else _wall_approach(rig)[:2]
        rig.safety.gate = False                         # 近づくまではフィルタなしで同じ動き
        _drive(rig, start + np.array([0, 0, 0.08]), target)
        _drive(rig, start, target)
        mask = contact.ContactMeter.obstacle_mask(target, rig.cubes_in_box())
        clear = float(rig.meter.distances(rig.data)[mask].min())
        assert clear > float(CFG["safety_filter"]["d_min_m"]) + 0.01, f"{which}: start already near an obstacle ({clear:.4f})"
        rig.safety.gate = True
        res[enabled] = _drive(rig, goal, target)
    assert res[False] < 0.0005, f"without the filter the {which} should be reached (min {res[False]:.4f})"
    assert res[True] >= float(CFG["safety_filter"]["d_min_m"]) - TOL, f"{which}: min distance {res[True]:.4f}"
    assert rig.safety.n_active_steps > 0


def test_moving_away_and_gate_off_unchanged(rig):
    lay, target = _setup(rig, True, _wall_seed(rig))
    start, goal, out = _wall_approach(rig)
    along = np.array([-out[1], out[0], 0.0])
    rig.safety.gate = False
    _drive(rig, start + np.array([0, 0, 0.08]), target)
    _drive(rig, start, target)
    rig.safety.gate = True
    _drive(rig, goal, target)                           # 壁の手前で止まる
    n0 = rig.safety.n_active_steps
    x0 = rig.integrator.x_cmd.copy()
    rig.pad_read(0.1 * out)                             # 壁から離れる
    assert np.allclose(rig.integrator.x_cmd - x0, 0.1 * out * rig.steps_per_read * rig.timestep, atol=1e-12)
    x1 = rig.integrator.x_cmd.copy()
    rig.pad_read(0.1 * along)                           # 壁に沿う
    assert np.allclose(rig.integrator.x_cmd - x1, 0.1 * along * rig.steps_per_read * rig.timestep, atol=1e-9)
    assert rig.safety.n_active_steps == n0
    rig.safety.gate = False                             # 当てる範囲の外では働かない
    assert not rig.safety.on()
    rig.pad_read(-0.2 * out)
    assert rig.integrator.command_filter is None
