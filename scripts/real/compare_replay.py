"""生成プログラムの指令の再生（実機の試験 A8）の比較: 実機の記録と、書き出した再生の列（シミュレーションの応答つき）を、
流し始めの時刻で揃えて比べる。

    python scripts/real/compare_replay.py --replay outputs/real/replay_v1/n_20000_red_r0.csv \
        --real logs/a8_x/arm.csv [--real-gripper logs/a8_x/gripper.csv] [--real-meta logs/a8_x/meta.json] [--out result.json]

基準（データを見る前に決めた。実機の試験の許容範囲。列の決まりは docs/real/replay_format.md）:
  errors          実機のエラー 0（meta.json の errors が空）
  success_rate    control_command_success_rate の最小 ≥ 0.9（流し始めから 0.1 s 以降。libfranka の値は直前 100 周期の平均）
  lag             手先の、指令に対する時間の遅延（実機・シミュレーションそれぞれ、0〜300 ms を 0.5 ms 刻みで二乗平均が最小の
                  ずれ）の差 ≤ 10 ms
  position        同じ時刻の手先の位置の差（実機 対 シミュレーション）: 中央値 ≤ 2 mm、最大 ≤ 5 mm、掴む・放す瞬間 ≤ 3 mm
  orientation     手先の向きの差の最大 ≤ 1°
  gripper         閉じ始め・開き終わりの時刻の差 ≤ 0.1 s（開く指令の時点の幅が実機とシミュレーションで 2 mm より違うとき
                  ＝立方体なしの実機は閉じ切っている、は開き終わりを比べられないので開き始めで代え、どちらを使ったかを書く）
参考（判定に使わない）: 実機 対 指令、シミュレーション 対 指令の遅延・遅延を引いた後の最大のずれ、関節の最大のずれ。
どれかが fail なら fail、それ以外で評価できない基準（記録がない）があれば incomplete、全部 pass なら pass。
10 Hz の表（参考、合否なし）: <replay>_ds.csv があれば、実機の記録を real_to_ds.py で学習データと同じ列にし、列ごとの差の
中央値・最大を dataset_view_10hz に出す。python3 と numpy だけで動く（リポジトリを読まない）。
"""
import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from real_to_ds import ACTION_NAMES, STATE_NAMES, read_csv, real_to_ds  # noqa: E402  (python3 と numpy だけ)

TOL = {"success_rate_min": 0.9, "lag_diff_s": 0.010, "pos_median_m": 0.002, "pos_max_m": 0.005,
       "pos_event_m": 0.003, "ori_max_deg": 1.0, "gripper_dt_s": 0.1}
LAG_GRID = np.arange(0.0, 0.3 + 1e-12, 0.0005)        # シミュレーションの追従の遅延は約 0.1 s（書き出しの 3 本で 98 ms）
W_EPS = 0.001                                         # 幅が動き始めた・着いたとみなす変化 [m]


def interp_cols(t_new, t, arr) -> np.ndarray:
    arr = np.asarray(arr, float)
    if arr.ndim == 1:
        return np.interp(t_new, t, arr)
    return np.stack([np.interp(t_new, t, arr[:, j]) for j in range(arr.shape[1])], axis=1)


def estimate_lag(t, meas, cmd_t, cmd) -> tuple:
    """meas(t) ≈ cmd(t − τ) となる τ（0〜100 ms）と、そのときの二乗平均の誤差・最大の誤差。"""
    best = None
    lo = cmd_t[0]
    for tau in LAG_GRID:
        m = t - tau >= lo
        if m.sum() < 10:
            continue
        e = np.linalg.norm(meas[m] - interp_cols(t[m] - tau, cmd_t, cmd), axis=1)
        rms = float(np.sqrt(np.mean(e ** 2)))
        if best is None or rms < best[1]:
            best = (float(tau), rms, float(e.max()))
    return best


def quat_to_R(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def otee_R(row16) -> np.ndarray:
    """libfranka の O_T_EE（列ごと 16 個）の回転。"""
    return np.asarray(row16, float).reshape(4, 4).T[:3, :3]


def angle_deg(Ra, Rb) -> float:
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def first_cross(t, w, t0, test) -> float:
    m = (t >= t0) & test(w)
    return float(t[m][0]) if m.any() else None


def gripper_times(t, w, events) -> list:
    """各指令の、閉じ始め（grasp）・開き始めと開き終わり（開く move）の時刻。w は幅の列。"""
    out = []
    for ev in events:
        t_ev = float(ev["t"])
        w0 = float(np.interp(t_ev, t, w))
        if ev["kind"] == "grasp":
            out.append({"t_cmd": t_ev, "kind": "grasp", "w_at_cmd": w0,
                        "close_start": first_cross(t, w, t_ev, lambda x: x < w0 - W_EPS)})
        elif float(ev["width"]) >= 0.07:
            target = float(ev["width"])
            out.append({"t_cmd": t_ev, "kind": "open", "w_at_cmd": w0,
                        "open_start": first_cross(t, w, t_ev, lambda x: x > w0 + W_EPS),
                        "open_end": first_cross(t, w, t_ev, lambda x: x >= target - W_EPS)})
    return out


def compare(replay: dict, real: dict, gripper_events: list, real_grip: dict = None, meta: dict = None) -> dict:
    crit, ref = {}, {}
    tr = np.asarray(replay["t"], float)
    t = np.asarray(real["t"], float)
    keep = (t >= tr[0]) & (t <= tr[-1])
    t = t[keep]
    ee_cmd_r = np.stack([replay[f"ee_cmd_{a}"] for a in "xyz"], axis=1)
    ee_sim_r = np.stack([replay[f"ee_sim_{a}"] for a in "xyz"], axis=1)
    otee = np.stack([np.asarray(real[f"O_T_EE{i}"], float)[keep] for i in range(16)], axis=1)
    ee_real = otee[:, 12:15]
    ee_sim = interp_cols(t, tr, ee_sim_r)

    # エラー・成功率
    if meta is not None and "errors" in meta:
        crit["errors"] = {"value": list(meta["errors"]), "pass": len(meta["errors"]) == 0}
    else:
        crit["errors"] = {"value": None, "pass": None, "note": "meta.json の errors がない"}
    if "success_rate" in real:
        sr = np.asarray(real["success_rate"], float)[keep]
        sr = sr[t >= 0.1] if (t >= 0.1).any() else sr
        crit["success_rate"] = {"value_min": round(float(sr.min()), 4), "pass": bool(sr.min() >= TOL["success_rate_min"])}
    else:
        crit["success_rate"] = {"value_min": None, "pass": None, "note": "success_rate の列がない"}

    # 遅延
    lag_real = estimate_lag(t, ee_real, tr, ee_cmd_r)
    lag_sim = estimate_lag(tr, ee_sim_r, tr, ee_cmd_r)
    d_lag = abs(lag_real[0] - lag_sim[0])
    crit["lag"] = {"real_s": lag_real[0], "sim_s": lag_sim[0], "diff_s": round(d_lag, 4), "pass": bool(d_lag <= TOL["lag_diff_s"] + 1e-12)}
    ref["real_vs_cmd"] = {"lag_s": lag_real[0], "rms_m_after_lag": round(lag_real[1], 6), "max_m_after_lag": round(lag_real[2], 6)}
    ref["sim_vs_cmd"] = {"lag_s": lag_sim[0], "rms_m_after_lag": round(lag_sim[1], 6), "max_m_after_lag": round(lag_sim[2], 6)}

    # 位置
    d = np.linalg.norm(ee_real - ee_sim, axis=1)
    ev_t = [float(e["t"]) for e in gripper_events if e["kind"] == "grasp" or float(e["width"]) >= 0.07]
    d_ev = [round(float(np.interp(x, t, d)), 6) for x in ev_t if t[0] <= x <= t[-1]]
    pos_pass = bool(np.median(d) <= TOL["pos_median_m"] and d.max() <= TOL["pos_max_m"]
                    and all(v <= TOL["pos_event_m"] for v in d_ev))
    crit["position"] = {"median_m": round(float(np.median(d)), 6), "max_m": round(float(d.max()), 6),
                        "at_events_m": d_ev, "pass": pos_pass}

    # 向き
    qs = np.stack([np.interp(t, tr, replay[k]) for k in ("ee_sim_qw", "ee_sim_qx", "ee_sim_qy", "ee_sim_qz")], axis=1)
    ang = [angle_deg(quat_to_R(q / np.linalg.norm(q)), otee_R(o)) for q, o in zip(qs, otee)]
    crit["orientation"] = {"max_deg": round(float(max(ang)), 4), "pass": bool(max(ang) <= TOL["ori_max_deg"])}

    # 関節（参考）
    if all(f"q{i}" in real for i in range(7)):
        q_real = np.stack([np.asarray(real[f"q{i}"], float)[keep] for i in range(7)], axis=1)
        q_cmd = interp_cols(t, tr, np.stack([replay[f"q_cmd{i}"] for i in range(7)], axis=1))
        q_sim = interp_cols(t, tr, np.stack([replay[f"q_sim{i}"] for i in range(7)], axis=1))
        ref["joint_max_abs_rad"] = {"real_vs_cmd": round(float(np.abs(q_real - q_cmd).max()), 6),
                                    "sim_vs_cmd": round(float(np.abs(q_sim - q_cmd).max()), 6),
                                    "real_vs_sim": round(float(np.abs(q_real - q_sim).max()), 6)}

    # グリッパー
    if real_grip is not None and len(real_grip.get("t", [])):
        gt, gw = np.asarray(real_grip["t"], float), np.asarray(real_grip["width"], float)
        sim_g = gripper_times(tr, np.asarray(replay["width_sim"], float), gripper_events)
        real_g = gripper_times(gt, gw, gripper_events)
        diffs, rows = [], []
        for s, r in zip(sim_g, real_g):
            if s["kind"] == "grasp":
                key = "close_start"
            else:
                key = "open_end" if abs(s["w_at_cmd"] - r["w_at_cmd"]) <= 0.002 else "open_start"
            dv = None if s.get(key) is None or r.get(key) is None else round(abs(s[key] - r[key]), 4)
            rows.append({"t_cmd": s["t_cmd"], "used": key, "sim": s.get(key), "real": r.get(key), "diff_s": dv,
                         "w_at_cmd_sim": round(s["w_at_cmd"], 5), "w_at_cmd_real": round(r["w_at_cmd"], 5)})
            diffs.append(dv)
        ok = None if any(x is None for x in diffs) else bool(all(x <= TOL["gripper_dt_s"] + 1e-12 for x in diffs))
        crit["gripper"] = {"events": rows, "pass": ok}
    else:
        crit["gripper"] = {"events": [], "pass": None, "note": "実機のグリッパーの記録がない"}

    vals = [c["pass"] for c in crit.values()]
    verdict = "fail" if any(v is False for v in vals) else ("incomplete" if any(v is None for v in vals) else "pass")
    return {"verdict": verdict, "criteria": crit, "reference": ref, "tolerances": TOL,
            "n_real_rows": int(keep.sum()), "t_span_s": [float(t[0]), float(t[-1])]}


def compare_ds(replay_ds: dict, real: dict, real_grip: dict = None) -> dict:
    """10 Hz の学習データの列での比べ（参考。合否は付けない）: 実機の記録を real_to_ds で同じ列にし、列ごとの差の
    中央値・最大（絶対値）。行動は実機も再生の値を写すので差 0（並べるだけ）。"""
    rows, cols = real_to_ds(real, real_grip, replay_ds, "base")
    r = {c: np.array([float(x[i]) for x in rows]) for i, c in enumerate(cols)}
    out = {}
    for c in STATE_NAMES + ACTION_NAMES[:3]:
        d = np.abs(r[c] - np.asarray(replay_ds[c], float))
        d = d[~np.isnan(d)]
        out[c] = {"median_abs": round(float(np.median(d)), 6) if d.size else None,
                  "max_abs": round(float(d.max()), 6) if d.size else None, "n": int(d.size)}
    for a, m in (("dx", "meas_dx"), ("dy", "meas_dy"), ("dz", "meas_dz")):
        d = np.abs(r[m] - np.asarray(replay_ds[a], float))
        d = d[~np.isnan(d)]
        out[f"{m}_vs_{a}"] = {"median_abs": round(float(np.median(d)), 6) if d.size else None,
                              "max_abs": round(float(d.max()), 6) if d.size else None, "n": int(d.size)}
    return {"frames": len(rows), "fields": out,
            "note": "シミュレーションの側は記録器の値（センサの模型の雑音・遅れを含む）、実機の側は t ≤ こまの時刻の最後の測定"}


def load_events(path) -> list:
    g = read_csv(path)
    out = []
    for i in range(len(g["t"])):
        out.append({"t": float(g["t"][i]), "kind": str(g["kind"][i]), "width": float(g["width"][i])})
    return out


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--replay", required=True)
    ap.add_argument("--replay-gripper", default=None, help="既定は <replay>_gripper.csv")
    ap.add_argument("--replay-ds", default=None, help="既定は <replay>_ds.csv（あれば 10 Hz の表も出す）")
    ap.add_argument("--real", required=True)
    ap.add_argument("--real-gripper", default=None)
    ap.add_argument("--real-meta", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    rp = pathlib.Path(a.replay)
    rg = pathlib.Path(a.replay_gripper) if a.replay_gripper else rp.with_name(rp.stem + "_gripper.csv")
    meta = json.loads(pathlib.Path(a.real_meta).read_text(encoding="utf-8")) if a.real_meta else None
    real = read_csv(a.real)
    real_grip = read_csv(a.real_gripper) if a.real_gripper else None
    res = compare(read_csv(rp), real, load_events(rg), real_grip, meta)
    rd = pathlib.Path(a.replay_ds) if a.replay_ds else rp.with_name(rp.stem + "_ds.csv")
    if rd.is_file():
        res["dataset_view_10hz"] = compare_ds(read_csv(rd), real, real_grip)
    res["replay"] = {"path": str(rp), "sha256": hashlib.sha256(rp.read_bytes()).hexdigest()}
    # 遅い版（replay_v1_slow）: 同じ場所の manifest.json の variant を結果に写す。基準は同じ
    mp = rp.with_name("manifest.json")
    man = json.loads(mp.read_text(encoding="utf-8")) if mp.is_file() else {}
    variant = man.get("variant")
    res["variant"] = variant or "normal"
    res["slow_variant"] = bool(variant and variant.startswith("slow"))
    # シミュレーションの応答を付け直した列（resim_replay.py）: 流した指令は元の列と同じなので、元の列の sha256 でもよい
    src_sha = {t.get("source_csv_sha256") for t in man.get("trials", []) if t.get("csv") == rp.name} - {None}
    if src_sha:
        res["replay"]["source_sha256"] = sorted(src_sha)[0]
    if meta and meta.get("replay_sha256") and meta["replay_sha256"] not in {res["replay"]["sha256"]} | src_sha:
        res["verdict"] = "fail"
        res["mismatch"] = "meta.json の replay_sha256 が再生の列と違う（別の列を流した記録）"
    txt = json.dumps(res, ensure_ascii=False, indent=2)
    if a.out:
        pathlib.Path(a.out).write_text(txt, encoding="utf-8")
    print(txt)
    return 0 if res["verdict"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
