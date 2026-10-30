"""試行の記録の監査を集計する監査の関門。報告・判定に使う数字は、全試行が監査を満たした走行からだけ出す。

監査の 3 項目（記録の欄 g1〜g3）:
  G1  実行系がシミュレーションの真値に触れない
  G2  計算（推論・知覚）の間も世界を止めず、結果を使えるようになる前に使わない
  G3  関節の指令が制限（速度・加速度・躍度などの上限）を超えない

    summ = audit_dir(d)              # 1 つの走行（trial_*.json・run_*.json）の集計
    mark(d)                          # 集計を d/G_AUDIT.json に書く（met が偽なら満たさない）
    require(dirs, purpose)           # どれかの走行に違反のある試行が 1 本でもあれば GateFailure（判定・評価の前に呼ぶ）
"""
import json
import pathlib

FILE = "G_AUDIT.json"


class GateFailure(RuntimeError):
    pass


def _trial_files(d: pathlib.Path) -> list:
    return sorted(d.glob("trial_*.json")) + sorted(p for p in d.glob("run_*.json") if not p.name.endswith("_runtime.json"))


def audit_dir(d) -> dict:
    d = pathlib.Path(d)
    out = {"dir": str(d), "trials": 0, "trials_without_audit": 0,
           "g1_violations": 0, "g1_trials": 0, "g2_world_stops": 0, "g2_early_use": 0, "g2_trials": 0,
           "g3_violations": 0, "g3_trials": 0, "g3_by_item": {}, "g3_max_ratio": {}}
    for p in _trial_files(d):
        m = json.loads(p.read_text(encoding="utf-8"))
        au = m.get("audit")
        out["trials"] += 1
        if not au or not all(k in au for k in ("g1", "g2", "g3")):        # 監査の欠けた記録は監査なしに数える
            out["trials_without_audit"] += 1
            continue
        g1 = int(au["g1"]["violations"])
        s, e = int(au["g2"]["world_stops"]), int(au["g2"]["early_use"])
        g3 = int(au["g3"]["total_violations"])
        out["g1_violations"] += g1
        out["g1_trials"] += int(g1 > 0)
        out["g2_world_stops"] += s
        out["g2_early_use"] += e
        out["g2_trials"] += int(s + e > 0)
        out["g3_violations"] += g3
        out["g3_trials"] += int(g3 > 0)
        for k, v in au["g3"].get("violations", {}).items():
            out["g3_by_item"][k] = out["g3_by_item"].get(k, 0) + int(v)
        for k, v in au["g3"].get("max_ratio", {}).items():
            out["g3_max_ratio"][k] = max(out["g3_max_ratio"].get(k, 0.0), float(v))
    out["met"] = (out["trials"] > 0 and out["trials_without_audit"] == 0 and out["g1_trials"] == 0
                  and out["g2_trials"] == 0 and out["g3_trials"] == 0)
    return out


def mark(d) -> dict:
    s = audit_dir(d)
    (pathlib.Path(d) / FILE).write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
    return s


def require(dirs, purpose: str) -> dict:
    summ = {str(d): mark(d) for d in dirs}
    bad = {k: v for k, v in summ.items() if not v["met"]}
    if bad:
        lines = [f"{k}: G1 {v['g1_trials']} 本・G2 {v['g2_trials']} 本・G3 {v['g3_trials']} 本・監査なし {v['trials_without_audit']} 本"
                 for k, v in bad.items()]
        raise GateFailure(f"{purpose}: G を満たさない走行があるので止める（0124 の ■ 3）\n" + "\n".join(lines))
    return summ
