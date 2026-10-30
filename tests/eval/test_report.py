"""結果の表と図（report.py、results.md）を合成の試行記録で確かめる。"""
import csv

import numpy as np

from recovla.eval import report

from .synth import SynthTrial


def _make(tmp_path, cond, seeds, success):
    d = tmp_path / cond
    for i, (s, ok) in enumerate(zip(seeds, success)):
        tr = SynthTrial(n_frames=120, trial=i, seed=s)
        tr.meta["condition"] = cond
        tr.meta["experiment"] = "E3"
        if not ok:                                   # 目標を机の上に置いたまま（成功しない）
            tr.meta["success"] = False
            for st in tr.meta.get("steps", []):
                st["success"], st["t_success"] = False, None
        tr.write(d)
    return d


def test_write_all_tables_and_figures(tmp_path):
    seeds = list(range(110000, 110010))
    a = _make(tmp_path, "R1", seeds, [True] * 8 + [False] * 2)
    b = _make(tmp_path, "N1", seeds, [True] * 5 + [False] * 5)
    rows = report.collect([a, b])
    assert len(rows) == 20
    out = tmp_path / "results"
    info = report.write_all(out, rows, "E3", pairs=[("R1", "N1")])
    for name in ("trials.csv", "summary.csv", "paired.csv", "continuous.csv", "tables.md",
                 "fig_success.png", "fig_recovery.png", "fig_seam.png", "fig_reaction.png", "fig_stage.png"):
        assert (out / name).is_file(), name
    summ = {r["condition"]: r for r in csv.DictReader(open(out / "summary.csv", encoding="utf-8"))}
    assert set(summ) == {"R1", "N1"} and int(summ["R1"]["n"]) == 10
    p = [r for r in csv.DictReader(open(out / "paired.csv", encoding="utf-8")) if r["metric"] == "success"][0]
    assert int(p["n_pairs"]) == 10
    assert int(p["n11"]) + int(p["n10"]) + int(p["n01"]) + int(p["n00"]) == 10
    assert info["trials"] == 20 and info["conditions"] == ["R1", "N1"]
    # 真偽は true・false
    t = list(csv.DictReader(open(out / "trials.csv", encoding="utf-8")))
    assert {r["success"] for r in t} <= {"true", "false"}
