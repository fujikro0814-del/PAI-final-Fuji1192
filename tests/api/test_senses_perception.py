"""task API の感覚の口（api/senses）と知覚の口（api/perception）の確かめ。CPU だけ。"""
import math
import types

import numpy as np
import pytest

from recovla.api import perception as AP
from recovla.api import senses as AS
from recovla.api import task_spec as TS
from recovla.runtime.perception import CubeEstimate, Perception, WorldModel
from recovla.runtime.runner import PolicyRuntime
from recovla.runtime.types import CameraFrame, GripperState, JointState, SensorFrame


def frame(t=1.0, cams=("overhead", "wrist")):
    q = np.arange(7) * 0.1
    cameras = {n: CameraFrame(n, t - 0.05, t - 0.01, np.zeros((4, 4, 3), np.uint8), None, 3) for n in cams}
    return SensorFrame(t, JointState(t, q, np.zeros(7), q.copy()), GripperState(t - 0.03, 0.039, True), cameras)


def test_senses_wraps_the_sensor_frame_without_copying():
    f = frame()
    calls = []

    def kin(q):
        calls.append(q)
        return np.array([0.4, 0.0, 0.2]), np.array([0.4, 0.0, 0.0966])
    s = AS.Senses.from_sensor_frame(f, kin)
    assert s.get("joints") is f.joints and s.get("gripper") is f.gripper and s.get("camera/wrist") is f.cameras["wrist"]
    assert s.time("gripper") == f.gripper.t and s.time("camera/overhead") == f.cameras["overhead"].t_capture
    assert s.time("joints") == f.joints.t and s.t == f.t
    assert set(s.names) == {"joints", "gripper", "camera/overhead", "camera/wrist"}
    assert s.cameras() == {"overhead": f.cameras["overhead"], "wrist": f.cameras["wrist"]}
    assert calls == []                                                     # 順運動学は読むまで計算しない
    assert s.hand().tolist() == [0.4, 0.0, 0.2] and s.tip()[2] == pytest.approx(0.0966)
    assert len(calls) == 1 and calls[0] is f.joints.q                       # 1 回だけ、測った関節角で
    s2 = AS.Senses.from_sensor_frame(frame(cams=()))
    assert not s2.has("camera/overhead") and s2.hand() is None and s2.tip() is None


def test_registry_available_and_table():
    sim, real = AS.standard_registry("sim"), AS.standard_registry("real")
    cam = ("joints", "gripper", "camera/overhead")
    assert sim.available(cam) and not real.available(cam) and real.missing(cam) == ["camera/overhead"]
    assert real.available(("joints", "gripper")) and real.available(("ext_wrench",)) and not sim.available(("ext_wrench",))
    assert not sim.available(("tactile",)) and not sim.available(("sonar",))
    rows = {r["name"]: r for r in sim.table()}
    assert list(rows)[:4] == ["joints", "gripper", "camera/overhead", "camera/wrist"]
    assert rows["ext_wrench"]["real"] is True and rows["ext_wrench"]["sim_model"] is False      # モデルは無い（印だけ）
    assert rows["gripper"]["rate_hz"] == 10.0 and rows["joints"]["rate_hz"] == 1000.0
    assert "ext_wrench" in real.format_table() and "○" in real.format_table()
    assert AS.standard_registry("sim", extra=False).names == ("joints", "gripper", "camera/overhead", "camera/wrist")
    with pytest.raises(ValueError):
        sim.register(AS.SenseInfo("joints", True, True, 1.0, 0.0))
    with pytest.raises(ValueError):
        AS.SenseRegistry("lab")


def test_fingertip_is_the_policy_runtime_formula():
    rng = np.random.default_rng(3)
    for _ in range(20):
        pos = rng.normal(size=3)
        quat = rng.normal(size=4)
        quat /= np.linalg.norm(quat)
        fake = types.SimpleNamespace(motion=types.SimpleNamespace(hand_pose=lambda q, p=pos, qq=quat: (p.copy(), qq.copy())),
                                     tip_offset=0.1034)
        assert np.allclose(AS.fingertip(pos, quat, 0.1034), PolicyRuntime._tip(fake, np.zeros(7)))


# ---------------------------------------------------------------- 知覚の口
def wm(statuses: dict, in_box=(), t=2.0, box_ok=True):
    cubes = {c: CubeEstimate(c, np.array([0.4 + 0.01 * i, -0.1, 0.02]), 0.1 * i, 1e-4, t - 0.1, c in in_box, st, "overhead")
             for i, (c, st) in enumerate(statuses.items())}
    return WorldModel(t, 0.0, {"xy": np.array([0.45, 0.25]), "yaw": 0.2, "ok": box_ok}, cubes, {})


def test_color_world_model_mapping():
    spec = TS.load_task()
    su = types.SimpleNamespace(box_outer=0.17, box_wall=0.005)
    f = AP.make_finder(spec, setup=su)
    assert isinstance(f, AP.ColorWorldModelFinder) and f.names == {"red": "red", "green": "green", "blue": "blue"}
    assert f.place == "box" and f.inner_half_m == pytest.approx(0.08)
    w = wm({"red": "seen", "green": "held", "blue": "in_hand"}, in_box=("green",))
    est = f.estimate(w)
    r, g, b = est.objects["red"], est.objects["green"], est.objects["blue"]
    assert (r.status, r.near_hand, r.in_place) == ("seen", None, {"box": False})
    assert (g.status, g.near_hand, g.in_place) == ("occluded", None, {"box": True})
    assert (b.status, b.near_hand) == ("occluded", True)
    assert est.t == 2.0 and np.allclose(r.pos, w.cubes["red"].pos) and r.pos is not w.cubes["red"].pos
    assert r.t_seen == pytest.approx(1.9) and b.yaw == pytest.approx(0.2)
    assert f.estimate(wm({"red": "lost"})).objects["red"].status == "lost"
    assert "green" not in f.estimate(wm({"red": "seen"})).objects                 # 推定の無い物は入れない
    assert f.estimate(None) is None
    pl = est.places["box"]
    assert pl.ok and pl.yaw == pytest.approx(0.2) and np.allclose(pl.pos, [0.45, 0.25])


def test_place_contains_matches_perception_in_box_xy():
    """PlaceState.contains_xy は runtime/perception.Perception.in_box_xy と同じ式。"""
    su = types.SimpleNamespace(box_outer=0.17, box_wall=0.005)
    rng = np.random.default_rng(0)
    for _ in range(200):
        xy, yaw = np.array([0.45, 0.25]) + rng.uniform(-0.12, 0.12, 2), rng.uniform(0, math.pi / 2)
        box = {"xy": np.array([0.45, 0.25]), "yaw": yaw, "ok": True}
        fake = types.SimpleNamespace(setup=su, box=box)
        pl = AP.PlaceState(box["xy"], yaw, True, AP.inner_half_of(su))
        for shrink in (0.0, 0.01):
            assert pl.contains_xy(xy, shrink) == Perception.in_box_xy(fake, xy, shrink)
    assert not AP.PlaceState(np.zeros(2), 0.0, False, 0.08).contains_xy(np.zeros(2))
    assert not AP.PlaceState(np.zeros(2), 0.0, True, None).contains_xy(np.zeros(2))


def test_finder_update_runs_perception_once():
    spec = TS.load_task()
    seen = []

    class FakePer:
        def update(self, frames, gripper, t, tip):
            seen.append((sorted(frames), gripper.width, t, None if tip is None else tip.tolist()))
            return wm({"red": "seen"}, t=t)
    f = AP.ColorWorldModelFinder.from_spec(spec, FakePer())
    s = AS.Senses.from_sensor_frame(frame(), lambda q: (np.zeros(3), np.ones(3)))
    est = f.update(s, 1.0)
    assert seen == [(["overhead", "wrist"], 0.039, 1.0, [1.0, 1.0, 1.0])] and est.objects["red"].status == "seen"
    with pytest.raises(RuntimeError):
        AP.ColorWorldModelFinder.from_spec(spec).update(s, 1.0)
