"""知覚の計算時間の種類（io.compute の kind）と固定の遅延: 色の知覚は "perception"、推定を足した知覚（--finder estimated）は
"perception_est"、条件 CL2 の知覚は "perception_cl2"、起動時の 1 回は "perception_startup"。どれも同じ決まり
（max(0.16 s, p99)、10 ms 単位）。偽の io・偽の知覚だけ（CPU）。"""
import types

import numpy as np

from recovla.api import estimated as EST
from recovla.api import skills_script_self as CL2
from recovla.common import config
from recovla.harness.robot_io import SimRobotIO, fixed_latency
from recovla.runtime.perception import Perception
from recovla.runtime.runner import PolicyRuntime


def test_all_perception_kinds_follow_the_same_rule():
    rtv = config.load(*config.V2_LAYERS)["runtime_v2"]
    fx = fixed_latency(rtv)
    assert float(rtv["perception_latency_s"]) == 0.16                          # 生成の側が読む値は変えない
    for k in ("perception", "perception_est", "perception_cl2", "perception_startup"):
        assert fx[k] == float(rtv["compute_latency_s"][k]), k                   # compute_latency_s が優先する
        assert fx[k] >= 0.16 and abs(fx[k] * 100 - round(fx[k] * 100)) < 1e-9, k   # 下限 0.16 s・10 ms 単位


def test_fixed_kinds_do_not_change_the_drawn_latencies_of_other_kinds():
    rtv = config.load(*config.V2_LAYERS)["runtime_v2"]
    a = SimRobotIO(None, None, 5, fixed_latency={"perception": 0.16})          # perception だけ固定
    b = SimRobotIO(None, None, 5, fixed_latency=fixed_latency(rtv))
    got_a = [a.draw_latency(k) for k in ("policy", "perception", "llm", "policy")]
    got_b = []
    for k in ("policy", "perception", "llm", "policy"):            # "perception" の固定の値は違うが、どちらも乱数は引かない
        got_b.append(b.draw_latency(k))
        b.draw_latency("perception_est")                                       # 固定の種類は乱数を引かない
        b.draw_latency("perception_startup")
    assert [got_a[i] for i in (0, 2, 3)] == [got_b[i] for i in (0, 2, 3)]   # 方策・LLM の遅延の列は a と同じ
    assert got_b[1] == fixed_latency(rtv)["perception"]


def test_compute_kind_comes_from_the_perception():
    assert not hasattr(Perception, "compute_kind")                             # 色の知覚は既定の "perception"
    assert EST.GeometryPerception.compute_kind == "perception_est"
    assert CL2.CL2Perception.compute_kind == "perception_cl2"


class _IO:
    def __init__(self):
        self.kinds = []

    def compute(self, kind, fn):
        self.kinds.append(kind)
        return types.SimpleNamespace(ready=lambda t: False)

    def sense(self, cameras=True):
        return types.SimpleNamespace(cameras={"overhead": 1, "wrist": 1}, joints=types.SimpleNamespace(q=np.zeros(7)),
                                     gripper=types.SimpleNamespace(width=0.08), t=0.0)


def _fake_runtime(per):
    return types.SimpleNamespace(perception=per, pend_per=None, stop_reason=None, io=_IO(), ready=True,
                                 _tip=lambda q: np.zeros(3))


def test_runtimes_use_the_kind_of_their_perception():
    plain = types.SimpleNamespace(update=lambda *a: None)
    est = types.SimpleNamespace(update=lambda *a: None, compute_kind="perception_est")
    for per, want in ((plain, "perception"), (est, "perception_est")):
        rt = _fake_runtime(per)
        PolicyRuntime._perception(rt, 1.0)
        assert rt.io.kinds == [want]
    rt = _fake_runtime(types.SimpleNamespace(update=lambda *a: None, compute_kind="perception_cl2"))
    CL2.CL2Skill._perception(rt, 1.0)
    assert rt.io.kinds == ["perception_cl2"]


def test_startup_uses_its_own_kind():
    import recovla.runtime.runner as RN
    for path in (RN.__file__, CL2.__file__):
        text = open(path, encoding="utf-8").read()
        assert 'io.compute("perception_startup", startup)' in text and 'io.compute("perception", startup)' not in text, path


def test_compute_log_keeps_the_wall_time_per_kind_for_the_record():
    w = types.SimpleNamespace(data=types.SimpleNamespace(time=0.0))
    io = SimRobotIO(w, None, 1, fixed_latency={"perception": 0.16, "perception_cl2": 0.52})
    for k in ("perception_cl2", "perception", "perception_cl2"):
        assert io.compute(k, lambda: 1).t_ready == (0.52 if k == "perception_cl2" else 0.16)
    walls = io.wall_by_kind()
    assert sorted(walls) == ["perception", "perception_cl2"] and len(walls["perception_cl2"]) == 2
    a = io.audit()
    assert a["fixed_latency"] == {"perception": 0.16, "perception_cl2": 0.52} and a["compute_wall_s"]["perception_cl2"]["n"] == 2
