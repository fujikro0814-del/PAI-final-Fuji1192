"""立て直しデータの生成の点検。重い測定は scripts/recovery_data.py が行い、outputs/f/*.json に結果を書く。
ここはその結果を読んで判定する（無ければ、何を実行すればよいかを示して落ちる）。"""
import json

import pytest

from recovla.common import config

pytestmark = pytest.mark.needs_outputs
CFG = config.load()
OUT = config.path(CFG["paths"]["outputs"]) / "f"
KINDS = ("A", "B", "C")


def _need(name: str, how: str) -> dict:
    p = OUT / f"{name}.json"
    if not p.is_file():
        pytest.fail(f"{p} がない。先に scripts/recovery_data.py {how} を実行する")
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.mark.parametrize("kind", KINDS)
def test_cond1_recovery_rate_and_landing(kind):
    """1: 失敗の意図的な失敗の種類ごとに、成功 ÷（意図的な失敗が効き、着地が正常だった数）≥ 90%、着地が不適切な割合 ≤ 2 割。
    数えるのは各指定の最初の試み（作り直しの前）。作り直しを含む数と捨てた指定の割合は別に出す。"""
    r = _need("check", "check-gen と check-eval")
    t = r["table"][kind]
    assert "all_attempts" in t and "dropped_ratio" in t, "check.json が古い（recovery_data.py check-eval --table-only で作り直す）"
    assert t["attempts"] == t["specs"]                                    # 最初の試みだけを数えている
    k, n, _ = t["recovery_success"]
    assert n > 0 and k / n >= 0.9, t
    k, n, _ = t["landing_invalid_ratio"]
    assert n > 0 and k / n <= float(CFG["inject"]["landing"]["max_invalid_ratio"]), t


def test_cond2_first_frame_all_episodes():
    """2: 保存した全エピソードの最初のフレームが、保存を始める時点の条件を満たす（全件を機械的に）。"""
    r = _need("check", "check-gen と check-eval")
    assert r["cond2_episodes"] > 0 and r["cond2_pass"], r["cond2_failures"][:5]


def test_cond3_replay_from_saved_state():
    """3: 保存した状態からの再生が、種類ごとに無作為の 3 本で通る（step はビット一致、window は手先 5 mm 以内）。"""
    r = _need("check", "check-gen と check-eval")
    for kind in KINDS:
        assert sum(1 for row in r["cond3_rows"] if row["episode"].startswith(kind + "_")) >= 3, kind
    assert r["cond3_pass"], [(row["episode"], row["step"]["pass"], row["window"]["pass"]) for row in r["cond3_rows"]]


def test_cond5_r1_n1_data():
    """5: 立て直しデータ入りの R1 と成功データだけの N1 で配置と目標の集合が一致し（通常の部分は同じエピソード）、
    構成が計画どおりで、変換の検証が通る。フレーム数の差は 1 割以内。"""
    r = _need("data", "gen-data")
    assert r["code_version"]["git_commit"] and r["config_used"]["inject"]["B"]["min_dist_from_box_m"] == 0.19   # 意図的な失敗 B の箱からの距離の下限
    assert r["same_layouts_and_targets"] and r["normal_part_identical"]
    for name in ("R1", "N1"):
        d = r["datasets"][name]
        assert d["convert_exit"] == 0 and d["verify_exit"] == 0 and d["verify_pass"], d
    assert all(c["short"] == 0 for c in r["recovery"]["cells"]), r["recovery"]["cells"]
    assert r["recovery"]["composition"]["n"] == sum(c["need"] for c in r["recovery"]["cells"])
    assert r["datasets"]["R1"]["episodes"] == r["datasets"]["N1"]["episodes"]
    assert r["frame_diff_ratio_R1_N1"] is not None and r["frame_diff_ratio_R1_N1"] <= 0.10, r["frame_diff_ratio_R1_N1"]
    assert not r["stop"], r["recovery"]["dropped_by_kind"]


def test_cond4_videos():
    """4: 種類ごとに 5 本の映像。"""
    r = _need("check", "check-gen と check-eval")
    assert r["cond4_pass"], r["cond4_videos"]
    for v in r["cond4_videos"]:
        assert (config.ROOT / v).is_file(), v
