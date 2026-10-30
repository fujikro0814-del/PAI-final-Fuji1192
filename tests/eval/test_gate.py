"""試行の記録の監査（真値の読み・世界の止め・動きの制限の違反）が 0 でなければ止める仕組み（recovla.eval.gate）の確かめ。"""
import json

import pytest

from recovla.eval import gate


def _trial(d, i, g1=0, stops=0, early=0, g3=0, audit=True):
    m = {"success": True}
    if audit:
        m["audit"] = {"g1": {"violations": g1}, "g2": {"world_stops": stops, "early_use": early},
                      "g3": {"total_violations": g3, "violations": {"cart_jerk": g3}, "max_ratio": {"cart_jerk": 1.0 + g3}}}
    (d / f"trial_{i:04d}.json").write_text(json.dumps(m), encoding="utf-8")


def test_clean_run_passes(tmp_path):
    for i in range(3):
        _trial(tmp_path, i)
    s = gate.require([tmp_path], "test")[str(tmp_path)]
    assert s["met"] and s["trials"] == 3
    assert json.loads((tmp_path / gate.FILE).read_text(encoding="utf-8"))["met"]


@pytest.mark.parametrize("kw", [{"g1": 1}, {"stops": 1}, {"early": 2}, {"g3": 5}, {"audit": False}])
def test_any_violation_stops(tmp_path, kw):
    _trial(tmp_path, 0)
    _trial(tmp_path, 1, **kw)
    with pytest.raises(gate.GateFailure):
        gate.require([tmp_path], "test")
    assert not json.loads((tmp_path / gate.FILE).read_text(encoding="utf-8"))["met"]


def test_task_runs_counted_but_not_runtime_logs(tmp_path):
    _trial(tmp_path, 0)
    (tmp_path / "run_0000.json").write_text(json.dumps({"audit": {"g1": {"violations": 0}, "g2": {"world_stops": 0, "early_use": 0},
                                                                   "g3": {"total_violations": 0}}}), encoding="utf-8")
    (tmp_path / "run_0000_runtime.json").write_text("{}", encoding="utf-8")
    assert gate.audit_dir(tmp_path)["trials"] == 2
