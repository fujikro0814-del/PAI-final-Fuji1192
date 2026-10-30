"""check_object_decompose の数え方（API は呼ばない）。"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts" / "api"))
import check_object_decompose as C  # noqa: E402


def test_meets():
    assert C.meets([2, 1, 3], {"set": [1, 2, 3]})
    assert not C.meets([1, 2], {"set": [1, 2, 3]})
    assert C.meets([3, 1, 2], {"set": [1, 2, 3], "first": 3}) and not C.meets([1, 3, 2], {"set": [1, 2, 3], "first": 3})
    assert C.meets([2, 1, 3], {"seq": [2, 1, 3]}) and not C.meets([1, 2, 3], {"seq": [2, 1, 3]})
    assert C.meets([2], {"one_of": [1, 2]}) and not C.meets([1, 2], {"one_of": [1, 2]})
    assert C.meets([], {"ask": True}) and not C.meets([1], {"ask": True})
    assert not C.meets([1, 1], {"set": [1]})


def test_color_arm_maps_colors_to_cubes(monkeypatch):
    seen = {}

    def fake(text, table, box):
        seen["in"] = (table, box)
        return {"steps": ["red"], "reply": ""}

    monkeypatch.setattr(C.D4, "decompose", fake)
    objs = [C.O(1, "カップ", "赤", 45, 15), C.O(2, "立方体", "赤", 50, -5), C.O(3, "立方体", "青", 40, -15, "box")]
    out = C.color_arm("カップを片付けて", objs, {"set": [1]})
    assert seen["in"] == (["red"], ["blue"])            # 色の集合だけ。種類は見えない
    assert not out["ok"]                                # 赤は立方体（2）に当たり、カップ（1）には当たらない
    assert C.color_arm("赤を片付けて", objs, {"set": [2]})["ok"]
