"""事前に決めた比較（手順書 r1_protocol.md）の判定の確かめ。独立に書いた 2 つの実装（scripts/api/judge_failure_relearn.py の A と
judge_failure_relearn_b.py の B）について、合成の数と、合成した試行の記録（json・npz）で、数え・並べ替え検定・ブートストラップ・McNemar・
Newcombe・Holm と A・B の一致を固定する。CPU だけ。"""
import importlib.util
import itertools
import json
import math
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


A = _load(ROOT / "scripts" / "api" / "judge_failure_relearn.py", "r1_judge_for_test")
B = _load(ROOT / "scripts" / "api" / "judge_failure_relearn_b.py", "r1_judge_b_for_test")


# ---------------------------------------------------------------- 統計の純関数
def test_perm_test_matches_exact_enumeration():
    kb, nb, ka, na = [1, 0, 2], [3, 1, 4], [2, 1, 3], [3, 2, 4]
    obs = A.pooled(ka, na) - A.pooled(kb, nb)
    hits = 0
    for sw in itertools.product((0, 1), repeat=3):
        k_a = sum(b if s else a for s, a, b in zip(sw, ka, kb))
        n_a = sum(b if s else a for s, a, b in zip(sw, na, nb))
        k_b, n_b = sum(ka) + sum(kb) - k_a, sum(na) + sum(nb) - n_a
        hits += abs(k_a / n_a - k_b / n_b) >= abs(obs) - 1e-12
    r = A.perm_test(kb, nb, ka, na, n_perm=40_000, seed=1)
    assert r["stat"] == pytest.approx(obs)
    assert r["p"] == pytest.approx(hits / 8, abs=0.01)


def test_perm_test_identical_and_large_difference():
    k, n = [1, 2, 0, 3] * 10, [3, 3, 2, 4] * 10
    assert A.perm_test(k, n, k, n, n_perm=1000)["p"] == 1.0
    kb, nb = [0] * 40, [3] * 40
    ka, na = [3] * 40, [3] * 40
    r = A.perm_test(kb, nb, ka, na, n_perm=2000)
    assert r["stat"] == 1.0 and r["p"] == 1 / 2001
    r2 = A.perm_test(ka, na, kb, nb, n_perm=2000)                 # 向きを入れ替えても p は同じ（両側）
    assert r2["stat"] == -1.0 and r2["p"] == r["p"]


def test_boot_ci_contains_the_difference_and_is_seeded():
    rng = np.random.default_rng(3)
    nb, na = rng.integers(0, 4, 100), rng.integers(0, 4, 100)
    kb, ka = rng.binomial(nb, 0.2), rng.binomial(na, 0.6)
    c1, c2 = A.boot_ci(kb, nb, ka, na), A.boot_ci(kb, nb, ka, na)
    d = A.pooled(ka, na) - A.pooled(kb, nb)
    assert c1 == c2 and c1["lo"] < d < c1["hi"] and c1["lo"] > 0
    assert B.boot_b(kb, nb, ka, na) == pytest.approx([c1["lo"], c1["hi"]], rel=1e-9)
    assert B.perm_b(kb, nb, ka, na) == pytest.approx((d, A.perm_test(kb, nb, ka, na)["p"]), rel=1e-9)


def test_no_grasp_failures_gives_p_one_and_nan_interval():
    k0, n0 = [0, 0, 0], [0, 0, 0]
    k1, n1 = [1, 0, 1], [2, 1, 1]
    r = A.perm_test(k0, n0, k1, n1, n_perm=100)
    assert r["p"] == 1.0 and math.isnan(r["stat"])
    c = A.boot_ci(k0, n0, k1, n1, n_boot=50)
    assert math.isnan(c["lo"]) and c["n_used"] == 0
    assert B.perm_b(k0, n0, k1, n1)[1] == 1.0 and all(math.isnan(x) for x in B.boot_b(k0, n0, k1, n1))


def test_success_row_mcnemar_newcombe_and_holm():
    seeds = list(range(50))
    base = {s: s < 20 for s in seeds}                             # 基準 20 本
    arm = {s: (s < 15) or (20 <= s < 35) for s in seeds}          # 比べる側 30 本: b 5・c 15
    r = A.success_row(base, arm, seeds)
    assert (r["n11"], r["b"], r["c"], r["n00"]) == (15, 5, 15, 15) and r["diff"] == pytest.approx(0.2)
    assert r["p"] == pytest.approx(B.binom_two_sided(5, 20), rel=1e-9)
    assert r["p"] == pytest.approx(2 * sum(math.comb(20, i) for i in range(6)) / 2 ** 20, rel=1e-9)
    lo, hi = r["diff_ci_newcombe"]
    d, ci = B.newcombe_b(15, 15, 5, 15)
    assert lo < 0.2 < hi and [lo, hi] == pytest.approx(ci, rel=1e-9) and d == pytest.approx(0.2)
    h = A.holm({"x": 0.01, "y": 0.04, "z": 0.03})
    assert [h["by"][k]["p_holm"] for k in ("x", "z", "y")] == pytest.approx([0.03, 0.06, 0.06])
    assert h["by"]["x"]["rejected"] and not h["by"]["z"]["rejected"]


def _irls(X, y, iters=50):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        W = p * (1 - p)
        b = b + np.linalg.solve(X.T @ (W[:, None] * X), X.T @ (y - p))
    return b


def test_grid_2x2_matches_a_logistic_fit_and_bootstrap():
    rng = np.random.default_rng(5)
    seeds = list(range(200))
    p = {"N0": 0.35, "U0": 0.5, "Ln": 0.55, "Lf": 0.85}
    data = {c: {"success": {s: bool(rng.random() < q) for s in seeds}} for c, q in p.items()}
    g = A.grid_2x2(data, seeds, n_boot=2000)
    X, y = [], []
    for (r, pl), c in A.GRID.items():
        for s in seeds:
            X.append([1, r, pl, r * pl])
            y.append(float(data[c]["success"][s]))
    b = _irls(np.array(X, float), np.array(y))
    e = g["estimate"]
    assert [e["b0"], e["bR"], e["bP"], e["bRP"]] == pytest.approx(list(b), abs=1e-8)
    lo, hi = g["boot"]["bRP"]["ci"]
    assert lo < e["bRP"] < hi and g["bRP_wald_ci"][0] < e["bRP"] < g["bRP_wald_ci"][1]
    k = g["cells"]
    assert e["interaction_pp"] == pytest.approx((k["Lf"] - k["Ln"] - k["U0"] + k["N0"]) / 200)
    assert A.grid_2x2({c: data[c] for c in ("N0", "U0")}, seeds)["status"] == "missing"


# ---------------------------------------------------------------- 合成の記録で A と B
T = np.round(np.arange(0.0, 60.0, 0.05), 4)
RED = np.array([0.45, -0.10, 0.02])
FAR = np.array([0.30, 0.20, 0.30])


def write_trial(d: pathlib.Path, i: int, seed: int, misses, success: bool, arm="U0", model="Rreal", bad=False,
                disturb=False):
    """misses: 試み 0・1 の掴み損ねの (起きるか, 自分で立て直すか)。試み 0 は 0〜30 s、試み 1 は 30〜60 s。"""
    n = len(T)
    closed = np.zeros(n, bool)
    tip = np.tile(FAR, (n, 1))
    pos = np.tile(np.stack([RED, [0.40, 0.0, 0.02], [0.50, 0.0, 0.02]]), (n, 1, 1))
    inbox = np.zeros((n, 3), bool)
    t_in = None
    for att, (happens, selfrec) in enumerate(misses):
        if not happens:
            continue
        a, b = np.searchsorted(T, 2.0 + 30 * att), np.searchsorted(T, 2.5 + 30 * att)
        closed[a:b] = True
        tip[a:b] = RED
        if selfrec and t_in is None:
            t_in = 10.0 + 30 * att
    if t_in is not None:
        inbox[T >= t_in, 0] = True
    np.savez(d / f"run_{i:04d}.npz", sim_time=T, gripper_closed=closed, cube_pos=pos, cube_in_box=inbox,
             plan_step=np.zeros(n, np.int16), fingertip=tip)
    audit = {"g1": {"violations": 1 if bad else 0}, "g2": {"world_stops": 0, "early_use": 0}, "g3": {"total_violations": 0}}
    m = {"seed": seed, "steps": [{"step": 0, "color": "red", "attempts": [{"attempt": 0, "t_begin": 0.0},
                                                                        {"attempt": 1, "t_begin": 30.0}], "t_end": 60.0}],
         "t_end": 60.0, "final_in_box": {"red": success, "green": success, "blue": success}, "all_three_in_box": success,
         "audit": audit, "model": model, "env": {"git_head": "x", "driver": "1", "torch": "2", "torch_cuda": "3", "os_build": "4"},
         "u4": {"arm": arm, "step_reset": "home", "finder": None if arm == "U0" else "estimated"},
         "world": {"world": "v3_real", "grip_debounce": {"open_steps": 5}}}
    if model != "Rreal":
        m["u4"]["model_ckpt"] = A.CKPT_OF[model]
    if disturb:
        m["u4"]["disturb"] = "push"
        m["disturb"] = {"pushed": True, "dropped": bool(seed % 2)}
    (d / f"run_{i:04d}.json").write_text(json.dumps(m), encoding="utf-8")


def test_a_and_b_agree_on_synthetic_records(tmp_path):
    rng = np.random.default_rng(11)
    seeds = list(range(171400, 171430))
    conds = {"N0": 0.1, "U0": 0.6}
    want = {}
    for c, p_self in conds.items():
        d = tmp_path / "EXP" / c
        d.mkdir(parents=True)
        k = n = 0
        for i, s in enumerate(seeds):
            ms = [(bool(rng.random() < 0.7), bool(rng.random() < p_self)) for _ in range(2)]
            # 試み 0 で自分で入れたら、試み 1 の失敗は箱の中なので数えない
            n_s = ms[0][0] + (ms[1][0] and not (ms[0][0] and ms[0][1]))
            k_s = (ms[0][0] and ms[0][1]) + (ms[1][0] and ms[1][1] and not (ms[0][0] and ms[0][1]))
            n, k = n + n_s, k + k_s
            write_trial(d, i, s, ms, bool(rng.random() < 0.5), model="Nreal" if c == "N0" else "Rreal")
        want[c] = (k, n)
    data = {c: A.read_condition([("EXP", tmp_path / "EXP" / c)]) for c in conds}
    for c in conds:
        assert (sum(v[0] for v in data[c]["self"].values()), sum(v[1] for v in data[c]["self"].values())) == want[c]
    fam = {"H2": A.FAMILY["H2"]}
    ja = A.judge(data, seeds, fam)
    jb = B.judge_b(tmp_path, ["EXP"], tuple(conds), fam, seeds)
    assert A.compare({"counts": A.summary_counts(data, seeds), "rows": A.key_numbers(ja)}, jb) == []
    r = ja["primary"]["H2"]
    assert r["diff"] > 0 and r["result"] in ("上がった", "確かめられなかった") and 0 < r["p"] <= 1


def test_a_and_b_agree_on_the_2x2(tmp_path):
    rng = np.random.default_rng(12)
    seeds = list(range(171400, 171440))
    p = {"N0": 0.3, "U0": 0.5, "Ln": 0.5, "Lf": 0.8}
    for c, q in p.items():
        d = tmp_path / "EXP" / c
        d.mkdir(parents=True)
        for i, s in enumerate(seeds):
            write_trial(d, i, s, [(False, False), (False, False)], bool(rng.random() < q))
    data = {c: A.read_condition([("EXP", tmp_path / "EXP" / c)]) for c in p}
    ga = A.grid_2x2(data, seeds, n_boot=500)
    jb = B.judge_b(tmp_path, ["EXP"], tuple(p), {}, seeds)
    assert A.compare({"cells": ga["cells"], "estimate": ga["estimate"]}, jb["grid"]) == []


def test_entry_check_counts_invalid_trials_per_machine(tmp_path):
    seeds = list(range(171400, 171412))
    d = tmp_path / "EXP" / "U0"
    d.mkdir(parents=True)
    for i, s in enumerate(seeds):
        write_trial(d, i, s, [(True, False), (False, False)], True, bad=i < 11)
    data = {"U0": A.read_condition([("EXP", d)])}
    probs = A.entry_check(data, seeds, {"EXP": (171400, 12)}, practice=True)
    assert any("無効な試行が 11 本" in p for p in probs)
    assert B.read_b(tmp_path, ["EXP"], "U0")[171400]["bad"] is True


def _write_cond(root, exp, cond, seeds, rng, disturb=False, bad=()):
    arm, model = A.CONDS[cond]
    d = root / exp / cond
    d.mkdir(parents=True)
    for i, s in enumerate(seeds):
        ms = [(bool(rng.random() < 0.6), bool(rng.random() < 0.5)) for _ in range(2)]
        write_trial(d, i, s, ms, bool(rng.random() < 0.5), arm=arm, model=model, bad=s in bad, disturb=disturb)
    (d / "G_AUDIT.json").write_text(json.dumps({"met": True}), encoding="utf-8")


def test_entry_check_full_and_invalid_sum_per_machine(tmp_path):
    rng = np.random.default_rng(21)
    split = {"E": (171400, 6), "E_B": (171406, 6)}
    seeds = list(range(171400, 171412))
    bad = set(range(171400, 171404))                              # 実験 E（1 台目の PC）で 4 本 × 3 条件 = 12 本の無効
    for c in ("N0", "Ud", "Ld"):
        for e, (s0, n) in split.items():
            _write_cond(tmp_path, e, c, list(range(s0, s0 + n)), rng, bad=bad)
    data = {c: A.read_condition([(e, tmp_path / e / c) for e in split]) for c in ("N0", "Ud", "Ld")}
    probs = A.entry_check(data, seeds, split, practice=False)
    assert probs == ["実験 E（1 台）で無効な試行が 12 本（全条件の和、10 を超える）"]
    assert A.invalid_by_exp(data, split) == {"E": 12, "E_B": 0}
    data["Ud"]["ckpts"] = ["outputs/other"]                        # 別のチェックポイント・外乱つきの記録は通さない
    data["Ld"]["disturbs"] = ["push"]
    probs = A.entry_check(data, seeds, split, practice=False)
    assert any("Ud: 保存点の道" in p for p in probs) and any("Ld: 外乱" in p for p in probs)


def test_practice_check_and_disturb_summary_agree(tmp_path, monkeypatch):
    rng = np.random.default_rng(22)
    for e, s in (("SM", 193998), ("SM_B", 193999)):                 # PC ごとに別のシード（同じシードは 2 本と数える）
        for c in A.CONDS:
            _write_cond(tmp_path, e, c, [s], rng)
    for e, s in (("SMD", 193998), ("SMD_B", 193999)):
        for c in A.DCONDS:
            _write_cond(tmp_path, e, c, [s], rng, disturb=True)
    monkeypatch.setattr(A, "OUTD", tmp_path / "api")
    assert A.main(["check", "--root", str(tmp_path), "--practice", "SM,SM_B", "--practice-disturb", "SMD,SMD_B"]) == 0
    res = json.loads((tmp_path / "api" / "r1_judge_practice_SM.json").read_text(encoding="utf-8"))
    assert res["agree"] and res["status"] == "complete" and set(res["a"]["primary"]) == {"H2", "H-R1a", "H-R1b"}
    assert res["disturb"]["status"] == "complete" and res["grid_2x2"]["status"] == "complete"
    assert res["disturb"]["events"]["Lf"]["pushed"] == 2 and res["disturb"]["events"]["Lf"]["dropped"] == 1
    assert A.main(["entry", "--root", str(tmp_path), "--exp", "SMD", "--disturb"]) == 0
    assert A.main(["entry", "--root", str(tmp_path), "--exp", "SM"]) == 0
    assert A.main(["entry", "--root", str(tmp_path), "--exp", "SMD"]) == 3      # 外乱つきの記録を外乱なしとして通さない
