"""生成プログラムの終盤の速さの上限（configs/expert_v3.yaml の final_approach）: 距離について連続で、設定がなければ効かない。"""
import numpy as np

from recovla.common import config
from recovla.expert import generate as G
from recovla.expert import script as S


def _expert(cfg):
    params = S.sample_params(np.random.default_rng(0), cfg)
    return S.Expert(params, 0.05, cfg), params


def test_v2_expert_unchanged_without_final_approach():
    cfg = config.load_v2()
    assert "final_approach" not in cfg["expert"]
    ex, _ = _expert(cfg)
    x, goal = np.array([0.4, 0.0, 0.25]), np.array([0.5, 0.05, 0.115])
    assert np.array_equal(ex._vel_to(x, goal, goal[:2]), ex._vel_to(x, goal))


def test_final_approach_caps_are_continuous_and_bounded():
    cfg = G.rig_config("v3")
    fa = cfg["expert"]["final_approach"]
    ex, params = _expert(cfg)
    aim = np.array([0.5, 0.0])
    caps = []
    for d in np.linspace(0.0, 0.3, 601):
        x = np.array([0.5 - d, 0.0, 0.25])
        v = ex._vel_to(x, np.array([0.5, 0.0, 0.25]), aim)
        caps.append(np.hypot(*v[:2]))
        if d <= fa["radius_m"]:
            assert np.hypot(*v[:2]) <= fa["xy"] * params.speed_scale + 1e-12
    steps = np.abs(np.diff(caps))
    assert steps.max() < 0.005                                   # 0.5 mm 刻みで速さが跳ばない
    far = ex._vel_to(np.array([0.1, 0.0, 0.25]), np.array([0.5, 0.0, 0.25]), aim)
    assert np.isclose(np.hypot(*far[:2]), ex.xy_max)             # 遠くでは通常の上限
    low = ex._vel_to(np.array([0.5, 0.0, ex.pp.grasp_z + 0.02]), np.array([0.5, 0.0, ex.pp.grasp_z]), aim)
    assert abs(low[2]) <= fa["z"] * params.speed_scale + 1e-12   # 把持の高さの近くでは z も遅く
