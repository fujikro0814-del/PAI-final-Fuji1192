"""失敗の意図的な失敗 P1〜P3（評価で失敗をわざと起こす 3 種類）と、run_trial への組み込みの確かめ。方策の代わりに生成プログラム
（真値を読む Expert を 10 fps の行動に直したもの）を使い、描画なし・GPU なしで回す。シードごとの結果はこの PC で選んだので
windows の目印。"""
import numpy as np
import pytest

from recovla.common import config, seeds
from recovla.eval import induce as I
from recovla.eval import scene_trial as T
from recovla.expert import script as S
from recovla.sim import scene

CFG = config.load()


class ScriptPolicy:
    """生成プログラムを 10 fps の行動にする（評価器の指令の入口の確認用）。"""

    def __init__(self, rig, target, seed):
        self.rig, self.target = rig, target
        self.ex = S.Expert(S.sample_params(seeds.script_rng(seed, target, 0)), T.ACTION_DT)

    def __call__(self, k, frame, raw, task):
        tr = self.rig.truth(self.target)
        cmd = self.ex.act(tr)
        closed = bool(self.rig.controller.gripper_closed) != bool(cmd.press)
        a = np.zeros(7)
        a[:3] = cmd.vel * T.ACTION_DT
        a[6] = 1.0 if closed else -1.0
        return a


@pytest.fixture(scope="module")
def rig():
    from recovla.sim.rig import SimRig
    r = SimRig(render=False)
    yield r
    r.close()


def run(rig, kind, seed):
    lay = scene.sample_layout(seed, "empty", start="home")
    target = lay.table_colors[seed % len(lay.table_colors)]
    ind = I.Inducer(kind, seed, lay, target, rig) if kind else None
    meta, arr, _ = T.run_trial(rig, lay, target, ScriptPolicy(rig, target, seed),
                               {"trial": 0, "seed": seed, "experiment": "test", "condition": kind or "none"},
                               time_limit_s=30.0, render=False, inducer=ind)
    return meta, arr


def test_params_drawn_up_front_and_deterministic():
    lay = scene.sample_layout(53000, "empty", start="home")
    t = lay.table_colors[0]
    a = I.Inducer("P3", 53000, lay, t, rig=None)
    b = I.Inducer("P3", 53000, lay, t, rig=None)
    assert a.params == b.params
    p1 = I.Inducer("P1", 53000, lay, t, rig=None).params
    lo, hi = CFG["eval"]["P1"]["lateral_offset_m"]
    assert lo <= abs(p1["offset_m"]) <= hi
    assert 0.0 <= I.Inducer("P2", 53000, lay, t, rig=None).params["u"] < 1.0


def test_same_seed_same_layout_and_induction_across_modes():
    """配置と意図的な失敗の条件はシードだけで決まり、実行の方式（sync・naive・rtc）に依らない。
    実行器は意図的な失敗の乱数列に触れず（雑音は noise の列）、意図的な失敗は試行の開始時に全部引く。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("r41", config.ROOT / "scripts" / "eval_subtask_direct.py")
    r41 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r41)
    a = r41.trial_list("induced:198000:6")
    b = r41.trial_list("induced:198000:6")
    assert [(s, l, t) for s, l, t in a] == [(s, l, t) for s, l, t in b]
    for seed, lay, tgt in a:
        for kind in I.KINDS:
            assert I.Inducer(kind, seed, lay, tgt, rig=None).params == I.Inducer(kind, seed, lay, tgt, rig=None).params
    # 雑音は推論の番号だけで決まる（方式に依らない）: 同じ (シード, i) なら同じ生成器の列
    from recovla.policy.schedule import noise_generator
    import torch
    assert torch.equal(torch.randn(3, generator=noise_generator(198000, 4)),
                       torch.randn(3, generator=noise_generator(198000, 4)))


def test_no_inducer_keeps_old_behaviour(rig):
    meta, arr = run(rig, None, 53001)
    assert meta["success"] and meta["induce"]["kind"] is None and not arr["induce_active"].any()


@pytest.mark.windows
@pytest.mark.parametrize("kind", I.KINDS)
def test_inducer_fires_and_is_recorded(rig, kind):
    """発動し、実際の失敗の判定まで進み、記録（meta.induce と induce_active）に残る。生成プログラムは立て直すので成功もする。"""
    got = []
    for seed in range(53010, 53014):
        meta, arr = run(rig, kind, seed)
        ind = meta["induce"]
        assert ind["kind"] == kind and ind["fired"], ind
        assert arr["induce_active"].any()
        got.append(ind["established"])
        if ind["established"]:                 # 実際の失敗の後は方策に操作を戻す（意図的な失敗はもう行動を上書きしない）
            after = arr["sim_time"] > float(ind["t_established"]) + 0.15
            assert not arr["induce_active"][after].any(), (kind, seed, ind)
    assert sum(got) >= 3, got                  # 4 回中 3 回以上実際の失敗（生成プログラムが相手なので、ほぼ実際の失敗する）
