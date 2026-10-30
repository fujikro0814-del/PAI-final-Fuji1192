"""置く位置の入力: 実行時の空いたマスの選び（api/place_cue）と、状態への足し方（data/vla_state.with_place）。"""
import numpy as np
import pytest

from recovla.api import place_cue as PC
from recovla.api.geometry import BoxGeom
from recovla.api.perception import ObjState, PlaceState, WorldEstimate
from recovla.data import convert, vla_state


def box(center=(0.45, 0.25), inner=0.08, yaw=0.0):
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return BoxGeom(np.array([center[0], center[1], 0.06]), R, np.array([inner + 0.01] * 2), np.array([inner, inner]), 0.01,
                   0.06, -0.05, 0.0, 0.0, {"center": 0.002}, 500, 200)


def est(objs: dict, b=None, ok=True):
    places = {"box": PlaceState(np.array([0.45, 0.25]), 0.0, ok, None, b if b is not None else box())}
    return WorldEstimate(0.0, {n: ObjState(np.array([x, y, 0.03]), 0.0, st, {"box": inb}, None)
                               for n, (x, y, inb, st) in objs.items()}, places)


def test_empty_box_takes_the_first_cell():
    xy, info = PC.free_cell(est({"red": (0.40, -0.10, False, "seen")}), "box", "red")
    assert np.allclose(xy, [0.49, 0.29]) and info["cell"] == 0 and info["in_place"] == []


def test_takes_the_cell_farthest_from_what_is_in_the_box():
    # 緑が +x+y のマスにある → いちばん遠いのは −x−y のマス
    xy, info = PC.free_cell(est({"red": (0.40, -0.10, False, "seen"), "green": (0.49, 0.29, True, "seen")}), "box", "red")
    assert np.allclose(xy, [0.41, 0.21]) and info["in_place"] == ["green"]
    # 2 個入り（+x+y と −x−y）→ 残りの 2 マスは同じ遠さなので並びの先（+x−y）
    xy, _ = PC.free_cell(est({"red": (0.40, -0.10, False, "seen"), "green": (0.49, 0.29, True, "seen"),
                              "blue": (0.41, 0.21, True, "seen")}), "box", "red")
    assert np.allclose(xy, [0.49, 0.21])


def test_ignores_the_target_lost_objects_and_objects_outside():
    e = est({"red": (0.49, 0.29, True, "seen"), "green": (0.41, 0.21, True, "lost"), "blue": (0.49, 0.21, False, "seen")})
    xy, info = PC.free_cell(e, "box", "red")
    assert info["in_place"] == [] and np.allclose(xy, [0.49, 0.29])


def test_uses_the_estimated_box_not_known_sizes():
    xy, _ = PC.free_cell(est({}, b=box(center=(0.50, 0.20), inner=0.06)), "box", "red")
    assert np.allclose(xy, [0.53, 0.23])
    xy, _ = PC.free_cell(est({}, b=box(yaw=np.pi / 2)), "box", "red")     # 回った箱でもマスは箱の座標で
    assert np.allclose(sorted(np.round(xy - [0.45, 0.25], 4)), [-0.04, 0.04])


def test_no_box_gives_none():
    assert PC.free_cell(None, "box", "red")[0] is None
    assert PC.free_cell(est({}, ok=False), "box", "red") == (None, {"source": "no_box"})


def test_with_place_and_state_names():
    s = np.zeros((4, 17), np.float32)
    out = vla_state.with_place(s, [0.4, 0.2])
    assert out.shape == (4, 19) and out.dtype == np.float32 and np.allclose(out[:, 17:], [0.4, 0.2])
    assert vla_state.with_place(np.zeros(17), np.array([0.1, 0.2])).shape == (19,)
    with pytest.raises(ValueError):
        vla_state.with_place(s, [0.1, 0.2, 0.3])
    names = convert.state_names("xy", place_cue=True)
    assert len(names) == 19 and names[-2:] == ["place_x", "place_y"]
    assert convert.state_names("xy") == names[:17]


def test_place_label_is_the_target_rest_position():
    pos = np.zeros((5, 3, 3))
    pos[-1, 1] = [0.41, 0.21, 0.03]
    inbox = np.zeros((5, 3), bool)
    inbox[-1, 1] = True
    assert np.allclose(convert.place_label({"target": "green", "name": "x"}, {"cube_pos": pos, "cube_in_box": inbox}), [0.41, 0.21])
    inbox[-1, 1] = False
    with pytest.raises(ValueError):
        convert.place_label({"target": "green", "name": "x"}, {"cube_pos": pos, "cube_in_box": inbox})
