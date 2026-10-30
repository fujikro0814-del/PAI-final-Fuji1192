"""立方体の本番（CLC・CLC_B）で失敗した試行の、終わり方と元の原因を分ける（事後の集計。判定には使わない、記述）。

    .venv\\Scripts\\python.exe scripts\\api\\failure_causes.py            # N0・U0・L2v2・CL2

目的: 再学習の案（R1: 方策がひとりでに失敗した所からスクリプトに代わって立て直しを記録する）の前に、今の失敗の中身を知る。
読むもの: run_NNNN.json（実行器の記録）・run_NNNN.npz（真値の時系列）・run_NNNN_runtime.json（判定の行・方策の目標位置の入力）。
書くもの: outputs/api/failure_causes_CLC.json だけ（outputs の既存のファイルは変えない）。

把持の 1 回 = グリッパが閉じていた間（natural_recovery.py と同じ区切り）。種類:
- placed: 目標が LIFT_M 以上上がり、放してから SETTLE_S の間に箱に入った
- slip_drop: 目標が上がった後、閉じたまま指先から DROP_M より離れた（運ぶ途中で落とした）
- misplaced: 目標が上がり、閉じたまま運んだが、放した後に箱に入らなかった
- wrong_cube: 目標でない立方体が上がった
- miss: 指先が目標から NEAR_M 以内で閉じ、どれも上がらなかった（指の和が EMPTY_W 未満なら空、以上なら挟んだが上がらず）
- air: 目標から NEAR_M より遠くで閉じ、どれも上がらなかった
手順の失敗の種類（判定で終えられなかった手順）は、真値で決める（下の step_class）。
試行の分類（category）: 元の手順＝失敗した手順のうち、最後に箱にない色の最初のもの（なければ止まる直前の失敗した手順）。
- M_*: 目標が一度も上がらなかった（M_table 机の上に残る・M_pushed_robot_side ロボットの側へ押し出した・M_box_wall_rim 箱の壁ぎわ）
- D_*: 上がったが箱に入らなかった（D_stacked_in_box 箱の中の先客の上・D_box_wall_rim 箱の縁・外壁ぎわ・D_table_then_miss 机へ落として
  拾い直せなかった）
- W_wrong_cube 別の立方体を上げた・J_fn 箱に入ったのに判定が完了を出さなかった・J_fp 完了と判定したのに箱にない・P_not_planned 計画に無い
立て直しの上の問題（overlays、L2v2）: 候補が残るのに LLM が止めた・知覚の箱の中の取り違えで先へ進んだ・見失い・判定の取りこぼしで回数を使った。
"""
import argparse
import collections
import json
import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
COLORS = ("red", "green", "blue")
CONDS = ("N0", "U0", "L2v2", "CL2")
LIFT_M = 0.015          # 持ち上がったとみなす高さ
NEAR_M = 0.06           # 把持の試みとみなす、閉じた時点の指先と目標の距離
SETTLE_S = 2.0          # 放した後、箱へ落ちきるまで待つ時間
DROP_M = 0.04           # 閉じたまま指先と目標がこれより離れたら、運ぶ途中で落とした
EMPTY_W = 0.01          # 閉じた指の開き（2 本の和）がこれ未満なら空を掴んだ
TILT_DEG = 15.0         # これより傾いたら「傾いた・倒れた」
BOX_XY = (0.45, 0.25)   # 箱の中心（シミュレーションの箱の体の位置）
BOX_OUT = 0.09          # 箱の外寸の半分
WALL_NEAR_M = 0.03      # 箱の外壁から立方体の中心までがこれ未満なら「壁ぎわ」（接すると 0.02）
REGION = ((0.33, 0.57), (-0.21, 0.05))   # 立方体を置く範囲（configs/default.yaml の scene.region）
REGION_TOL = 0.02       # 範囲からこれより外なら「範囲の外」
TOUCH_M = 0.05          # 他の立方体の中心までの水平の距離がこれ未満なら「くっついている」
MOVED_M = 0.02          # 初めの位置からこれより動いたら「動かされた」
REST_Z = 0.0199         # 机の上の立方体の中心の高さ
BOX_MIN_PIXELS = 56     # 判定の俯瞰の画素の下限（configs/runtime_v2.yaml）
WRIST_MIN_PIXELS = 33


def quat_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def tilt_yaw(q):
    """→ (傾き [deg], 向き [deg]、90 度の対称で -45〜45 に畳む)。向きは手の閉じる向き（世界の y）との角度と同じ（手の向きは固定）。"""
    R = quat_mat(q)
    k = int(np.argmax(np.abs(R[2, :])))
    tilt = math.degrees(math.acos(min(1.0, abs(R[2, k]))))
    j = (k + 1) % 3
    yaw = math.degrees(math.atan2(R[1, j], R[0, j]))
    yaw = (yaw + 45.0) % 90.0 - 45.0
    return tilt, yaw


def box_cheb(p):
    return max(abs(p[0] - BOX_XY[0]), abs(p[1] - BOX_XY[1]))


def out_of_region(p):
    (x0, x1), (y0, y1) = REGION
    return max(x0 - p[0], p[0] - x1, y0 - p[1], p[1] - y1) > REGION_TOL


def cube_state(i, c, pos, quat, inbox):
    """時刻の添字 i の立方体 c の様子（1 つの名前）。"""
    p = pos[i, c]
    if inbox[i, c]:
        tilt, _ = tilt_yaw(quat[i, c])
        return "in_box" if tilt < TILT_DEG else "in_box_tilted"
    if p[2] < -0.03:
        return "off_table"
    others = [o for o in range(3) if o != c]
    if p[2] > REST_Z + 0.015:
        below = [o for o in others if np.linalg.norm(pos[i, o, :2] - p[:2]) < 0.035 and pos[i, o, 2] < p[2] - 0.02]
        if below:
            return "stacked_in_box" if any(inbox[i, o] for o in below) else "stacked"   # 箱の中の立方体の上（真値では箱の外）
        if box_cheb(p) < BOX_OUT + 0.02:
            return "on_box_rim"
        return "raised"
    tilt, _ = tilt_yaw(quat[i, c])
    near_wall = box_cheb(p) - BOX_OUT < WALL_NEAR_M
    if tilt > TILT_DEG:
        return "tilted_at_wall" if near_wall else "tilted"
    if near_wall:
        return "at_box_wall"
    if out_of_region(p):
        return "pushed_robot_side" if p[0] < REGION[0][0] - REGION_TOL else "out_of_region"   # ロボットの側（x が小さい）へ押した
    if any(np.linalg.norm(pos[i, o, :2] - p[:2]) < TOUCH_M and not inbox[i, o] for o in others):
        return "touching_cube"
    return "table"


def closed_intervals(closed):
    d = np.diff(closed.astype(np.int8), prepend=0, append=0)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def idx(t, x):
    return int(min(len(t) - 1, max(0, np.searchsorted(t, x - 1e-9))))


def episode(a, b, c, z, inf_t, inf_cue):
    """閉じていた部分 [a, b) の把持 1 回の種類と、閉じた時点の様子。"""
    t, pos, quat, inbox, tip, fing = z["t"], z["pos"], z["quat"], z["inbox"], z["tip"], z["wsum"]
    n = len(t)
    rise = pos[a:b, :, 2].max(axis=0) - pos[a, :, 2]
    off = tip[a] - pos[a, c]
    w_min = float(fing[a:b].min())
    w_end = float(fing[b - 1])
    if rise[c] >= LIFT_M:
        i_lift = a + int(np.flatnonzero(pos[a:b, c, 2] - pos[a, c, 2] >= LIFT_M)[0])
        dist = np.linalg.norm(tip[i_lift:b] - pos[i_lift:b, c], axis=1)
        w = np.flatnonzero((t >= t[b - 1]) & (t <= t[b - 1] + SETTLE_S))
        if dist.size and dist.max() > DROP_M:
            kind = "slip_drop"
            i_land = min(n - 1, i_lift + int(np.argmax(dist > DROP_M)) + int(SETTLE_S / 0.05))
        elif inbox[b - 1, c] or inbox[w, c].any():
            kind, i_land = "placed", min(n - 1, b - 1 + int(SETTLE_S / 0.05))
        else:
            kind, i_land = "misplaced", min(n - 1, b - 1 + int(SETTLE_S / 0.05))
        land = cube_state(i_land, c, pos, quat, inbox)
        rel_box = float(box_cheb(tip[b - 1]))
        # 放した所と、箱の中の他の立方体の水平の距離（先客の真上で放したか）
        occ = [float(np.linalg.norm(tip[b - 1, :2] - pos[b - 1, o, :2])) for o in range(3) if o != c and inbox[b - 1, o]]
        rel_occ = min(occ) if occ else None
    elif (rise >= LIFT_M).any():
        kind, land, rel_box, rel_occ = "wrong_cube", None, None, None
    elif np.linalg.norm(off) < NEAR_M:
        kind, land, rel_box, rel_occ = ("miss_empty" if w_min < EMPTY_W else "miss_pinched"), None, None, None
    else:
        kind, land, rel_box, rel_occ = "air", None, None, None
    others = [o for o in range(3) if o != c and not inbox[a, o]]
    nd = min([float(np.linalg.norm(pos[a, o, :2] - pos[a, c, :2])) for o in others], default=None)
    near_other = min([(float(np.linalg.norm(tip[a, :2] - pos[a, o, :2])), o) for o in range(3) if o != c], default=(9, -1))
    tilt, yaw = tilt_yaw(quat[a, c])
    k = int(np.searchsorted(inf_t, t[a] + 1e-9)) - 1
    cue_err = cue_vis = None
    if k >= 0 and inf_cue[k] is not None:
        j = idx(t, inf_t[k])
        cue_err = float(np.linalg.norm(np.asarray(inf_cue[k][:2]) - pos[j, c, :2]))
        cue_vis = float(inf_cue[k][2])
    return {"kind": kind, "t": round(float(t[a]), 2), "dur_s": round(float(t[b - 1] - t[a]), 2),
            "dx": round(float(off[0]), 4), "dy": round(float(off[1]), 4), "dz": round(float(off[2]), 4),
            "w_min": round(w_min, 4), "w_end": round(w_end, 4), "tilt": round(tilt, 1), "yaw": round(yaw, 1),
            "x": round(float(pos[a, c, 0]), 3), "y": round(float(pos[a, c, 1]), 3),
            "box_d": round(float(box_cheb(pos[a, c]) - BOX_OUT), 3), "nb_d": None if nd is None else round(nd, 3),
            "tip_other_d": round(near_other[0], 3), "tip_other": COLORS[near_other[1]] if near_other[1] >= 0 else None,
            "state": cube_state(a, c, pos, quat, inbox), "cue_err": None if cue_err is None else round(cue_err, 4),
            "cue_vis": cue_vis, "land": land, "rel_box_release": None if rel_box is None else round(rel_box, 3),
            "rel_occupied_release": None if rel_occ is None else round(rel_occ, 3),
            "n_in_box": int(inbox[a].sum())}


FAIL_KINDS = ("slip_drop", "misplaced", "wrong_cube", "miss_empty", "miss_pinched", "air")


def step_class(st, eps, z, rows, m):
    """判定で終えられなかった手順（judged_complete 偽）か、完了と判定したのに箱にない手順の、原因の種類。"""
    t, inbox, pos = z["t"], z["inbox"], z["pos"]
    c = COLORS.index(st["color"])
    t_end = st.get("t_end", m["t_end"])                   # 時間切れの試行の最後の手順は t_end がない
    i0, i1 = idx(t, st["attempts"][0]["t_begin"]), idx(t, t_end)
    kinds = [e["kind"] for e in eps]
    info = {"kinds": dict(collections.Counter(kinds))}
    if st.get("judged_complete"):
        ij = idx(t, st["t_judge"])
        return ("ok" if inbox[ij, c] else "judge_fp"), info
    if inbox[i0, c]:
        return "already_in_box", info
    if inbox[i1, c]:
        k0 = i0 + int(np.argmax(inbox[i0:i1 + 1, c]))
        rs = [r for r in rows if r["step"] == st["step"] and r["t"] >= t[k0] - 1e-9 and r["t"] <= t_end + 1e-9]
        info["judge"] = {"t_in": round(float(t[k0]), 1), "rows": len(rs),
                         "box_px_max": max([r["box_pixels"] for r in rs], default=None),
                         "wrist_in_max": max([r["wrist_in"] for r in rs], default=None),
                         "depth_ok_any": any(r["depth_ok"] for r in rs),
                         "retreat_min": round(min([r["retreat_dist_m"] for r in rs], default=9), 3),
                         "ok_any": any(r["ok"] for r in rs), "state_end": cube_state(i1, c, pos, z["quat"], inbox)}
        return "judge_fn", info
    if inbox[i0:i1 + 1, c].any():
        return "in_then_out", info
    if st.get("early_event"):
        info["early"] = True
    if any(k in ("slip_drop", "misplaced") for k in kinds):
        return "lift_then_lost", info
    if "wrong_cube" in kinds:
        return "wrong_cube", info
    if any(k.startswith("miss") for k in kinds):
        return "miss", info
    # 目標の近くで閉じなかった手順: 指先がどフレームで近づいたか・安全の制限が効いていたか
    d = np.linalg.norm(z["tip"][i0:i1 + 1] - pos[i0:i1 + 1, c], axis=1)
    info["approach"] = {"tip_min_d": round(float(d.min()), 3), "tip_min_z": round(float(z["tip"][i0:i1 + 1, 2].min()), 3),
                        "safety_frac": round(float(z["safety"][i0:i1 + 1].mean()), 2)}
    if "air" in kinds:
        return "air_only", info
    return "no_close", info


def trial(js: pathlib.Path, cond: str) -> dict:
    m = json.loads(js.read_text(encoding="utf-8"))
    zz = np.load(js.with_suffix(".npz"))
    rt_p = js.with_name(js.stem + "_runtime.json")
    rt = json.loads(rt_p.read_text(encoding="utf-8")) if rt_p.is_file() else {}
    rows = ((rt.get("executor") or {}).get("judge_rows")) or []
    inf = ((rt.get("runtime") or {}).get("inference")) or []
    inf_t = np.array([r["t_obs"] for r in inf]) if inf else np.zeros(0)
    inf_cue = [r.get("cue") for r in inf]
    z = {"t": zz["sim_time"], "closed": zz["gripper_closed"], "pos": zz["cube_pos"], "quat": zz["cube_quat"],
         "inbox": zz["cube_in_box"], "ps": zz["plan_step"], "tip": zz["fingertip"], "wsum": zz["fingers"].sum(axis=1),
         "safety": zz["safety_active"]}
    t, pos, inbox = z["t"], z["pos"], z["inbox"]
    steps = {s["step"]: s for s in m["steps"]}
    # 把持の 1 回ずつ
    eps_by_step = collections.defaultdict(list)
    for a, b in closed_intervals(z["closed"]):
        k = int(z["ps"][a])
        if k < 0 or k not in steps or b >= len(t):
            continue
        st = steps[k]
        c = COLORS.index(st["color"])
        if inbox[a, c]:
            continue
        e = episode(a, b, c, z, inf_t, inf_cue)
        att = max([x for x in st["attempts"] if x["t_begin"] <= t[a] + 1e-9], key=lambda x: x["t_begin"], default=None)
        e.update(step=k, color=st["color"], attempt=None if att is None else att["attempt"],
                 t_in_att=None if att is None else round(float(t[a] - att["t_begin"]), 1),
                 prior_fail=sum(1 for p in eps_by_step[k] if p["kind"] in FAIL_KINDS))
        eps_by_step[k].append(e)
    # 手順ごと
    srows = []
    for s in m["steps"]:
        cls, info = step_class(s, eps_by_step[s["step"]], z, rows, m)
        c = COLORS.index(s["color"])
        i0, i1 = idx(t, s["attempts"][0]["t_begin"]), idx(t, s.get("t_end", m["t_end"]))
        moved_other = [COLORS[o] for o in range(3) if o != c and np.linalg.norm(pos[i1, o, :2] - pos[i0, o, :2]) > MOVED_M]
        srows.append({"step": s["step"], "color": s["color"], "cls": cls, "n_att": len(s["attempts"]),
                      "t0": round(float(t[i0]), 1), "t1": round(float(t[i1]), 1),
                      "state0": cube_state(i0, c, pos, z["quat"], inbox), "state1": cube_state(i1, c, pos, z["quat"], inbox),
                      "moved_other": moved_other, **info})
    # 試行の終わり方
    final = m.get("final_in_box") or {}
    missing = [c for c in COLORS if not final.get(c)]
    st_ = m.get("stopped")
    ended = (m.get("u4") or {}).get("ended")
    reps = m.get("replans") or []
    if m.get("timed_out"):
        term = "time_limit"
    elif st_:
        by = st_.get("by")
        if by == "replan":
            term = "llm_stop"
        elif by == "replan_fallback":
            term = "llm_fallback_stop"
        elif cond == "L2v2" and len(reps) >= 2:
            term = "replan_limit_stop"
        else:
            term = "executor_stop"
    elif ended and ended.get("kind") == "finish":
        term = "llm_finish"
    else:
        term = "plan_completed"
    failed = [r for r in srows if r["cls"] not in ("ok",)]
    root = None
    for r in failed:
        if r["color"] in missing:
            root = r
            break
    if root is None and failed and term not in ("plan_completed",):
        root = failed[-1]
    planned = set(m["plan"].get("steps_initial") or m["plan"].get("steps") or []) | {s["color"] for s in m["steps"]}
    table0 = list((m.get("layout") or {}).get("table_colors") or [])
    not_planned = [c for c in missing if c not in planned]
    root_cls = root["cls"] if root else ("not_planned" if not_planned else "unknown")
    # 立て直しの様子（L2v2）
    rep_rows = []
    for r in reps:
        i = idx(t, r["t_request"])
        truth = {"in_box": [c for c in COLORS if inbox[i, COLORS.index(c)]]}
        per = r.get("perception") or {}
        wrong = sorted((set(per.get("in_box", [])) ^ set(truth["in_box"])))
        cand = ((r.get("out") or {}).get("candidates"))
        fc = COLORS.index(r["failed_color"])
        rep_rows.append({"n": r["n"], "t": round(r["t_request"], 1), "event": r.get("event"), "failed": r["failed_color"],
                         "action": r.get("action"), "applied": (r.get("applied") or {}).get("kind"),
                         "order": (r.get("applied") or {}).get("order"), "candidates": cand,
                         "percept_in_box_wrong": wrong, "lost": per.get("lost", []),
                         "failed_state": cube_state(i, fc, pos, z["quat"], inbox),
                         "rejected": (r.get("out") or {}).get("rejected"), "fallback": (r.get("out") or {}).get("fallback"),
                         "reason": ((r.get("out") or {}).get("reason") or "")[:120]})
    n_box0 = sum(1 for c in COLORS if inbox[0, COLORS.index(c)])
    disp = {c: round(float(np.linalg.norm(pos[-1, COLORS.index(c), :2] - pos[0, COLORS.index(c), :2])), 3) for c in COLORS}
    untouched = (sum(final.get(c, False) for c in COLORS) == n_box0
                 and all(disp[c] < MOVED_M for c in missing))
    eps = [e for k in sorted(eps_by_step) for e in eps_by_step[k]]
    # 立て直しの上の問題（L2v2）: 候補が残るのに止めた・知覚の取り違えで先へ進んだ・見失いで候補から外れた・判定の取りこぼしで回数を使った
    over = []
    if any(p["action"] == "stop" and p["candidates"] for p in rep_rows):
        over.append("llm_stop_with_candidates")
    if any(p["action"] in ("next", "reorder", "skip", "finish") and set(p["percept_in_box_wrong"]) & set(missing) for p in rep_rows):
        over.append("percept_in_box_wrong_moved_on")
    if any(set(p["lost"]) & set(missing) for p in rep_rows):
        over.append("lost_at_replan")
    if any(s["cls"] == "judge_fn" for s in srows) and root_cls != "judge_fn":
        over.append("judge_fn_used_budget")
    if any(p["action"] == "skip" or p["action"] == "finish" for p in rep_rows):
        over.append("llm_finish_or_skip")
    return {"seed": m["seed"], "overlays": over, "cond": cond, "success": bool(m.get("all_three_in_box")), "layout": (m.get("layout") or {}).get("kind"),
            "table0": table0, "plan0": m["plan"].get("steps_initial") or m["plan"].get("steps"), "t_end": round(m["t_end"], 1),
            "term": term, "root": root_cls, "root_step": None if root is None else root["step"],
            "root_color": None if root is None else root["color"], "missing": missing, "not_planned": not_planned,
            "final_state": {c: cube_state(len(t) - 1, COLORS.index(c), pos, z["quat"], inbox) for c in COLORS},
            "disp": disp, "untouched": untouched, "steps": srows, "replans": rep_rows,
            "early_events": [{"t": round(e["t"], 1), "color": e["color"], "event": e["event"]} for e in m.get("early_events") or []],
            "episodes": eps}


WALL_STATES = ("on_box_rim", "at_box_wall", "tilted_at_wall")


def category(r) -> str:
    """失敗した試行の分類（1 つ）。元の手順の種類と、その立方体の行き先から決める。"""
    root = r["root"]
    if root in ("judge_fn", "judge_fp", "not_planned", "wrong_cube", "in_then_out", "already_in_box", "unknown"):
        return {"judge_fn": "J_fn", "judge_fp": "J_fp", "not_planned": "P_not_planned", "wrong_cube": "W_wrong_cube"}.get(root, root)
    st = next(s for s in r["steps"] if s["step"] == r["root_step"])
    fin = r["final_state"].get(st["color"])
    if root == "lift_then_lost":
        land = next((e["land"] for e in r["episodes"] if e["step"] == st["step"] and e["kind"] in ("slip_drop", "misplaced")), None)
        if land == "stacked_in_box":
            return "D_stacked_in_box"
        if land in WALL_STATES or fin in WALL_STATES:
            return "D_box_wall_rim"
        return "D_table_then_miss"
    if fin == "pushed_robot_side":
        return "M_pushed_robot_side"
    if fin in WALL_STATES:
        return "M_box_wall_rim"
    return "M_table"


def timeline(r) -> str:
    parts = []
    for s in r["steps"]:
        ks = s.get("kinds") or {}
        k = ",".join(f"{a}x{b}" for a, b in ks.items())
        parts.append(f"s{s['step']} {s['color']} [{s['t0']}-{s['t1']}s] {s['cls']} ({k}) {s['state0']}->{s['state1']}")
    for p in r["replans"]:
        parts.append(f"replan{p['n']}@{p['t']} {p['event']}:{p['failed']}->{p['action']}{p['order'] or ''}"
                     + (f" percept_wrong{p['percept_in_box_wrong']}" if p["percept_in_box_wrong"] else "")
                     + (f" lost{p['lost']}" if p["lost"] else ""))
    parts.append(f"END {r['term']} t={r['t_end']} missing={r['missing']} final={ {c: r['final_state'][c] for c in r['missing']} }")
    return " | ".join(parts)


def pct(xs, qs=(25, 50, 75)):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return [round(float(v), 3) for v in np.percentile(xs, qs)] + [len(xs)]


def rate_table(eps, key, bins):
    """把持の 1 回ずつの、範囲ごとの「上がらず失敗の率」（miss・air を失敗、placed など目標が上がったものを成功の分母に）。"""
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        es = [e for e in eps if e[key] is not None and lo <= e[key] < hi]
        n = len(es)
        mi = sum(1 for e in es if e["kind"].startswith("miss"))
        out.append({"bin": [lo, hi], "n": n, "miss": mi, "miss_pct": round(100 * mi / n, 1) if n else None})
    return out


def miss_rates(es: list) -> dict:
    """目標が上がった把持（placed・落とした・置き損ね）と上がらなかった把持（miss）の中の、範囲ごとの miss の率。"""
    for e in es:
        e["_ayaw"], e["_adx"], e["_ady"] = abs(e["yaw"]), abs(e["dx"]), abs(e["dy"])
    out = {"n": len(es), "miss": sum(1 for e in es if e["kind"].startswith("miss")), "by": {
        "abs_yaw_deg": rate_table(es, "_ayaw", [0, 5, 10, 15, 20, 25, 30, 46]),
        "abs_dx_m": rate_table(es, "_adx", [0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.06]),
        "abs_dy_m": rate_table(es, "_ady", [0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.06]),
        "dz_m": rate_table(es, "dz", [-0.05, 0.0, 0.01, 0.02, 0.03, 0.05, 0.1]),
        "tilt_deg": rate_table(es, "tilt", [0, 5, 15, 45, 91]),
        "nb_d_m": rate_table([e for e in es if e["nb_d"] is not None], "nb_d", [0, 0.06, 0.08, 0.10, 0.15, 1]),
        "box_d_m": rate_table(es, "box_d", [-0.2, 0.0, 0.03, 0.06, 0.1, 0.15, 1]),
        "cue_err_m": rate_table([e for e in es if e["cue_err"] is not None], "cue_err", [0, 0.005, 0.01, 0.015, 0.02, 0.03, 1]),
        "x_m": rate_table(es, "x", [0.2, 0.36, 0.40, 0.44, 0.48, 0.52, 0.8]),
        "y_m": rate_table(es, "y", [-0.5, -0.17, -0.13, -0.09, -0.05, -0.01, 0.5]),
        "prior_fail": rate_table(es, "prior_fail", [0, 1, 2, 3, 5, 100]),
        "state": {s: {"n": sum(1 for e in es if e["state"] == s), "miss": sum(1 for e in es if e["state"] == s and e["kind"].startswith("miss"))}
                  for s in sorted({e["state"] for e in es})}}}
    for e in es:
        for k in ("_ayaw", "_adx", "_ady"):
            e.pop(k, None)
    return out


def summarize(rows: list) -> dict:
    fails = [r for r in rows if not r["success"]]
    for r in fails:
        r["cat"] = category(r)
    out = {"trials": len(rows), "failed": len(fails)}
    out["cat"] = dict(collections.Counter(r["cat"] for r in fails).most_common())
    out["cat_x_term"] = {}
    for r in fails:
        out["cat_x_term"].setdefault(r["cat"], collections.Counter())[r["term"]] += 1
    out["cat_x_term"] = {k: dict(v) for k, v in sorted(out["cat_x_term"].items(), key=lambda kv: -sum(kv[1].values()))}
    out["overlays"] = dict(collections.Counter(o for r in fails for o in r["overlays"]).most_common())
    out["cat_examples"] = {c: [r["seed"] for r in fails if r["cat"] == c][:10] for c in out["cat"]}
    # 元の立方体の初めの位置（x が小さいほどロボットに近い）
    out["root_cube_x0"] = {c: pct([next((e["x"] for e in r["episodes"] if e["step"] == r["root_step"]), None)
                                   for r in fails if r["cat"] == c]) for c in out["cat"]}
    # 同じ色の 2 回目の手順（立て直しで同じ色をもう一度）の成績（全試行）
    sec = [s for r in rows for j, s in enumerate(r["steps"]) if any(p["color"] == s["color"] for p in r["steps"][:j])]
    out["second_run_same_color"] = {"n": len(sec), "ok": sum(1 for s in sec if s["cls"] == "ok"),
                                    "state0": dict(collections.Counter(s["state0"] for s in sec).most_common()),
                                    "ok_by_state0": dict(collections.Counter(s["state0"] for s in sec if s["cls"] == "ok"))}
    out["no_close_approach"] = [dict(s["approach"], state0=s["state0"], seed=r["seed"]) for r in rows for s in r["steps"]
                                if s["cls"] in ("no_close", "air_only")]
    out["root_x_term"] = {}
    for r in fails:
        out["root_x_term"].setdefault(r["root"], collections.Counter())[r["term"]] += 1
    out["root_x_term"] = {k: dict(v) for k, v in sorted(out["root_x_term"].items(), key=lambda kv: -sum(kv[1].values()))}
    out["root"] = dict(collections.Counter(r["root"] for r in fails).most_common())
    out["term"] = dict(collections.Counter(r["term"] for r in fails).most_common())
    # 元の手順の、目標の様子（初めと終わり）
    rs = [s for r in fails for s in r["steps"] if s["step"] == r["root_step"]]
    out["root_state0"] = dict(collections.Counter(s["state0"] for s in rs).most_common())
    out["root_state1"] = dict(collections.Counter(s["state1"] for s in rs).most_common())
    out["root_color"] = dict(collections.Counter(r["root_color"] for r in fails).most_common())
    out["root_step_index"] = dict(collections.Counter(r["root_step"] for r in fails).most_common())
    out["any_miss"] = sum(1 for r in fails if any(e["kind"].startswith("miss") for e in r["episodes"]))
    out["root_step_had_miss"] = sum(1 for r in fails for s in r["steps"] if s["step"] == r["root_step"]
                                    and any(k.startswith("miss") for k in (s.get("kinds") or {})))
    out["untouched"] = sum(1 for r in fails if r["untouched"])
    out["nothing_placed"] = sum(1 for r in fails if all(not s["cls"] == "ok" for s in r["steps"]))
    out["t_end"] = pct([r["t_end"] for r in fails], (10, 25, 50, 75, 90))
    out["t_end_hist"] = np.histogram([r["t_end"] for r in fails], bins=[0, 40, 80, 120, 160, 201])[0].tolist()
    out["t_first_fail"] = pct([next((s["t1"] for s in r["steps"] if s["cls"] != "ok"), None) for r in fails], (10, 25, 50, 75, 90))
    # 手順の位置・色ごとの失敗の率（全試行の手順）
    by_pos, by_col = collections.defaultdict(lambda: [0, 0]), collections.defaultdict(lambda: [0, 0])
    for r in rows:
        for s in r["steps"]:
            bad = s["cls"] != "ok"
            by_pos[s["step"]][0] += 1
            by_pos[s["step"]][1] += bad
            by_col[s["color"]][0] += 1
            by_col[s["color"]][1] += bad
    out["step_fail_by_index"] = {k: {"n": v[0], "fail": v[1], "pct": round(100 * v[1] / v[0], 1)} for k, v in sorted(by_pos.items())}
    out["step_fail_by_color"] = {k: {"n": v[0], "fail": v[1], "pct": round(100 * v[1] / v[0], 1)} for k, v in sorted(by_col.items())}
    out["step_cls_all"] = dict(collections.Counter(s["cls"] for r in rows for s in r["steps"]).most_common())
    # 把持の 1 回ずつ（全試行）
    eps = [e for r in rows for e in r["episodes"]]
    out["episode_kinds"] = dict(collections.Counter(e["kind"] for e in eps).most_common())
    groups = {"placed": [e for e in eps if e["kind"] == "placed"], "miss": [e for e in eps if e["kind"].startswith("miss")],
              "miss_first": [e for e in eps if e["kind"].startswith("miss") and e["prior_fail"] == 0],
              "lost": [e for e in eps if e["kind"] in ("slip_drop", "misplaced")], "air": [e for e in eps if e["kind"] == "air"]}
    feats = {}
    for g, es in groups.items():
        feats[g] = {"n": len(es)}
        for f, fn in (("abs_dx", lambda e: abs(e["dx"])), ("abs_dy", lambda e: abs(e["dy"])), ("dz", lambda e: e["dz"]),
                      ("abs_yaw", lambda e: abs(e["yaw"])), ("tilt", lambda e: e["tilt"]), ("nb_d", lambda e: e["nb_d"]),
                      ("box_d", lambda e: e["box_d"]), ("cue_err", lambda e: e["cue_err"]), ("x", lambda e: e["x"]),
                      ("y", lambda e: e["y"]), ("w_min", lambda e: e["w_min"]), ("dur_s", lambda e: e["dur_s"]),
                      ("t_in_att", lambda e: e["t_in_att"])):
            feats[g][f] = pct([fn(e) for e in es])
        feats[g]["state"] = dict(collections.Counter(e["state"] for e in es).most_common())
        feats[g]["kind"] = dict(collections.Counter(e["kind"] for e in es).most_common())
    out["episode_features_q25_50_75_n"] = feats
    lifted_or_miss = [e for e in eps if e["kind"] in ("placed", "slip_drop", "misplaced", "miss_empty", "miss_pinched")]
    out["miss_rate_by"] = miss_rates(lifted_or_miss)
    # 手順の最初の把持で、目標が普通に机の上にあるときだけ（押された・落とした後の様子を混ぜない）
    out["miss_rate_by_first_try_table"] = miss_rates([e for e in lifted_or_miss if e["prior_fail"] == 0 and e["state"] == "table"])
    # 落とした・置き損ねの行き先
    out["lost_landing"] = dict(collections.Counter(f"{e['kind']}:{e['land']}" for e in eps
                                                   if e["kind"] in ("slip_drop", "misplaced")).most_common())
    # 放す所: 箱の中心からの距離（チェビシェフ）と、箱の中の先客からの距離。箱の中の数ごとの置き損ねの率
    lifted = [e for e in eps if e["kind"] in ("placed", "misplaced")]
    out["release"] = {g: {"rel_box": pct([e["rel_box_release"] for e in es]), "rel_occupied": pct([e["rel_occupied_release"] for e in es])}
                      for g, es in (("placed", [e for e in lifted if e["kind"] == "placed"]),
                                    ("misplaced_stacked", [e for e in lifted if e["land"] == "stacked_in_box"]),
                                    ("misplaced_rim_wall", [e for e in lifted if e["kind"] == "misplaced" and e["land"] in WALL_STATES]),
                                    ("misplaced_table", [e for e in lifted if e["kind"] == "misplaced" and e["land"] not in WALL_STATES
                                                         and e["land"] != "stacked_in_box"]))}
    out["misplace_by_n_in_box"] = {k: {"lifted": sum(1 for e in lifted if e["n_in_box"] == k),
                                       "misplaced": sum(1 for e in lifted if e["n_in_box"] == k and e["kind"] == "misplaced"),
                                       "stacked": sum(1 for e in lifted if e["n_in_box"] == k and e["land"] == "stacked_in_box")}
                                   for k in range(3)}
    # 判定の取りこぼし
    jf =[s for r in fails for s in r["steps"] if s["cls"] == "judge_fn"]
    out["judge_fn_detail"] = [dict(s["judge"], step=s["step"], color=s["color"]) for s in jf]
    # 立て直し（L2v2）
    reps = [p for r in fails for p in r["replans"]]
    out["replan_actions_in_failed"] = dict(collections.Counter(f"{p['event']}->{p['action']}" for p in reps).most_common())
    out["replan_percept_wrong_in_failed"] = sum(1 for p in reps if p["percept_in_box_wrong"])
    out["replan_lost_in_failed"] = sum(1 for p in reps if p["lost"])
    out["n_replans_in_failed"] = dict(collections.Counter(len(r["replans"]) for r in fails))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(ROOT / "outputs" / "v2eval"))
    ap.add_argument("--exps", default="CLC,CLC_B")
    ap.add_argument("--conds", default=",".join(CONDS))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "api" / "failure_causes_CLC.json"))
    a = ap.parse_args(argv)
    res = {"note": "事後の集計（記述）。判定には使わない",
           "params": {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float, tuple))},
           "exps": a.exps.split(","), "conds": {}}
    pooled = []
    for cond in a.conds.split(","):
        rows = []
        for exp in a.exps.split(","):
            d = pathlib.Path(a.root) / exp / cond
            rows += [trial(js, cond) for js in sorted(d.glob("run_[0-9][0-9][0-9][0-9].json")) if js.with_suffix(".npz").is_file()]
        if not rows:
            sys.exit(f"{cond} の記録が見つからない（{a.root}）")
        s = summarize(rows)
        fails = [r for r in rows if not r["success"]]
        s["failed_trials"] = [{k: v for k, v in r.items() if k != "episodes"} | {"timeline": timeline(r),
                              "episodes": [e for e in r["episodes"] if e["kind"] != "placed"]} for r in fails]
        res["conds"][cond] = s
        if cond in ("U0", "L2v2"):
            pooled += [e for r in rows for e in r["episodes"]]
        print(f"{cond}: 試行 {s['trials']}  失敗 {s['failed']}  分類 {s['cat']}  終わり方 {s['term']}")
    # 同じ方策（Rreal）の U0 と L2v2 の把持をまとめた、最初の把持の miss の率
    lm = [e for e in pooled if e["kind"] in ("placed", "slip_drop", "misplaced", "miss_empty", "miss_pinched")]
    res["Rreal_pooled_first_try_table"] = miss_rates([e for e in lm if e["prior_fail"] == 0 and e["state"] == "table"])
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                   encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
