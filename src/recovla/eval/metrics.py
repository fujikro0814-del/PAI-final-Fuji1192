"""評価の 1 試行の記録（json と npz）から指標を計算する純関数（docs/interfaces/trial_record.md §4〜§6）。

MuJoCo も方策も呼ばない。入力は読み込んだ記録と、configs/default.yaml の eval（eval_cfg）だけ。
返り値の名前は docs/interfaces/results.md §2 の trials.csv の列と同じ。欠けた数値は NaN、ない値は None
（CSV では空欄）。フレームについての計算は numpy の配列演算で行い、フレームを 1 つずつ回すループは書かない。
"""
import json
import math
import pathlib
from dataclasses import dataclass

import numpy as np

COLORS = ("red", "green", "blue")        # 色の並び（scene.colors の順）
FRAMES_PER_ACTION = 2                    # フレーム 20 Hz、行動 10 Hz

# 段階の符号（phase）→ 段階別の到達の区分（-1 は区分に数えない: reopen・settle）
STAGES = ("接近", "把持", "搬送", "設置", "退避", "完了")
_PHASE_STAGE = np.array([0, 0, 1, -1, 1, 2, 3, -1, 4, 5], dtype=np.int64)   # 添字が phase、値が STAGES の添字


@dataclass(frozen=True)
class TrialRecord:
    meta: dict              # trial_NNNN.json
    arrays: dict            # trial_NNNN.npz の {キー: ndarray}


def load_trial(json_path) -> TrialRecord:
    """trial_NNNN.json と、同じ名前の trial_NNNN.npz を読む。"""
    json_path = pathlib.Path(json_path)
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    with np.load(json_path.with_suffix(".npz"), allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    return TrialRecord(meta=meta, arrays=arrays)


# ------------------------------------------------------------------------------ 個別の量

def seam_jumps(action: np.ndarray, chunk_switch: np.ndarray, dt: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """行動列のチャンク境界での速度の不連続。10 Hz の行動の列から (切り替わりの跳び, 切り替わりでない時点の同じ量) [m/s]。

    v_k = action[k, :3] / dt、跳び_k = |v_k − v_{k−1}|（k ≥ 1）。chunk_switch[k] が真なら切り替わり。
    v_k か v_{k−1} が NaN（行動がない）の k は、どちらにも入れない。
    """
    action = np.asarray(action, dtype=float)
    switch = np.asarray(chunk_switch, dtype=bool)
    if len(action) < 2:
        return np.zeros(0), np.zeros(0)
    v = action[:, :3] / dt
    jump = np.linalg.norm(v[1:] - v[:-1], axis=1)
    ok = np.isfinite(jump)
    sw = switch[1:]
    return jump[ok & sw], jump[ok & ~sw]


def reaction_time(t, fingertip, target_pos, t_failure, close_m=0.02) -> float:
    """反応時間 [s]。t_failure の時点（その時刻以後の最初のフレーム）の指先の中心と目標の距離を d0 とし、
    距離が d0 − close_m 以下になった最初のフレームの時刻 − t_failure。見つからなければ NaN。"""
    t = np.asarray(t, dtype=float)
    if t_failure is None or not np.isfinite(t_failure):
        return math.nan
    i0 = _first_at_or_after(t, t_failure)
    if i0 is None:
        return math.nan
    d = np.linalg.norm(np.asarray(fingertip, dtype=float) - np.asarray(target_pos, dtype=float), axis=1)
    if not np.isfinite(d[i0]):
        return math.nan
    hit = np.flatnonzero(d[i0:] <= d[i0] - close_m)
    return float(t[i0 + hit[0]] - t_failure) if hit.size else math.nan


def recovery_time(t, target_z, z0, t_fire, lift_m=0.02) -> float:
    """復帰時間 [s]。t_fire から、目標が「再び」持ち上げ（z − z0 ≥ lift_m）の状態になった最初のフレームの時刻まで。

    「再び」: t_fire 以後に一度、持ち上げでない状態（落ちた・まだ持ち上げていない）になってから後の、
    最初の持ち上げのフレーム。t_fire の時点で持ち上げていなければ、その時点から数える。見つからなければ NaN。
    """
    t = np.asarray(t, dtype=float)
    if t_fire is None or not np.isfinite(t_fire):
        return math.nan
    i0 = _first_at_or_after(t, t_fire)
    if i0 is None:
        return math.nan
    lifted = (np.asarray(target_z, dtype=float)[i0:] - z0) >= lift_m
    down = np.flatnonzero(~lifted)
    if not down.size:
        return math.nan
    up = np.flatnonzero(lifted[down[0]:])
    return float(t[i0 + down[0] + up[0]] - t_fire) if up.size else math.nan


def jerk_rms(x_des_10hz: np.ndarray, dt: float = 0.1) -> float:
    """躍度 [m/s³]: x_des の 10 Hz の 3 階差分 / dt³ の大きさの二乗平均の平方根。NaN を含む差分は除く。"""
    x = np.asarray(x_des_10hz, dtype=float)
    if len(x) < 4:
        return math.nan
    j = np.linalg.norm(np.diff(x, n=3, axis=0), axis=1) / dt ** 3
    j = j[np.isfinite(j)]
    return float(np.sqrt(np.mean(j ** 2))) if j.size else math.nan


# ------------------------------------------------------------------------------ 1 試行の行

def trial_metrics(rec: TrialRecord, eval_cfg: dict) -> dict:
    """1 試行 1 行の指標。キーは results.md §2 の trials.csv の列（並びも同じ）。"""
    m, a = rec.meta, rec.arrays
    runtime = m.get("runtime") or {}
    layout = m.get("layout") or {}
    steps = m.get("steps") or []
    induce = m.get("induce") or {}
    induce_kind = induce.get("kind")

    t = np.asarray(a["sim_time"], dtype=float)
    frame_dt = float(np.median(np.diff(t))) if len(t) > 1 else math.nan
    action_dt = FRAMES_PER_ACTION * frame_dt
    target = np.asarray(a["target"], dtype=np.int64)
    cube_pos = np.asarray(a["cube_pos"], dtype=float)
    z_rise = cube_pos[:, :, 2] - cube_pos[:1, :, 2]                     # (N, 3) 最初のフレームからの高さ
    lifted_err = z_rise >= eval_cfg["error_lift_m"]
    non_target = np.arange(len(COLORS))[None, :] != target[:, None]      # 手順の外（−1）は全色が目標でない

    fired = bool(induce.get("fired")) if induce_kind else False
    established = bool(induce.get("established")) if induce_kind else False
    t_fire = _num(induce.get("t_fire")) if fired else math.nan
    induced_color = _target_at(target, t, t_fire, steps)

    collateral, collateral_induced = _collateral(rec, eval_cfg, target, t, t_fire)

    reaction = recovery = math.nan
    if established and induced_color is not None:
        tgt_pos = cube_pos[:, induced_color]
        reaction = reaction_time(t, a["fingertip"], tgt_pos, _num(induce.get("t_failure")),
                                 eval_cfg["reaction_close_m"])
        recovery = recovery_time(t, tgt_pos[:, 2], cube_pos[0, induced_color, 2], t_fire,
                                 eval_cfg["recovery_lift_m"])

    k_frames = np.flatnonzero((np.asarray(a["step"]) // _frame_steps(a["step"])) % FRAMES_PER_ACTION == 0)
    seam, nonseam = seam_jumps(np.asarray(a["action"])[k_frames], np.asarray(a["chunk_switch"])[k_frames],
                               action_dt)
    ee_jump = _ee_speed_jumps(a, k_frames, frame_dt)

    wall = np.array([float(e["wall_s"]) for e in (m.get("inference") or []) if e.get("wall_s") is not None])
    success = bool(m.get("success"))
    mode = runtime.get("mode")

    return {
        "experiment": m.get("experiment"),
        "condition": m.get("condition"),
        "model": (m.get("model") or {}).get("name"),
        "mode": mode,
        "s": runtime.get("exec_interval"),
        "d": 0 if mode == "sync" else runtime.get("delay_steps"),
        "safety_filter": runtime.get("safety_filter"),
        "trial": m.get("trial"),
        "seed": m.get("seed"),
        "layout_kind": layout.get("kind"),
        "start_pose": layout.get("start"),
        "target": ">".join(s["target"] for s in steps) if steps else None,
        "induce": induce_kind if induce_kind else "none",
        "success": success,
        "t_success_s": _num(steps[-1].get("t_success")) if (success and steps) else math.nan,
        "error": bool(np.any(lifted_err & non_target)),
        "collateral": collateral,
        "collateral_induced": collateral_induced,
        "stage_reached": _stage_reached(a["phase"], target),
        "induce_fired": fired,
        "induce_established": established,
        "recovered": success if established else None,
        "reaction_time_s": reaction,
        "recovery_time_s": recovery,
        "seam_jump_mean": _mean(seam),
        "seam_jump_max": float(seam.max()) if seam.size else math.nan,
        "nonseam_jump_mean": _mean(nonseam),
        "seam_ee_speed_jump_mean": _mean(ee_jump),
        "jerk_rms": jerk_rms(np.asarray(a["x_des"], dtype=float)[k_frames], action_dt),
        "contacts_n": _contacts_n(rec, target),
        "inference_mean_s": _mean(wall),
        "inference_p95_s": float(np.percentile(wall, 95)) if wall.size else math.nan,
    }


# ------------------------------------------------------------------------------ 内部

def _num(v) -> float:
    return math.nan if v is None else float(v)


def _mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if x.size else math.nan


def _first_at_or_after(t: np.ndarray, t0: float):
    """t0 以後の最初のフレームの添字（物理ステップの刻みの丸めを見込んで 1e-9 s 手前から）。なければ None。"""
    i = int(np.searchsorted(t, t0 - 1e-9, side="left"))
    return i if i < len(t) else None


def _frame_steps(step) -> int:
    """1 フレームの物理ステップ数（記録の step の刻み。通常は 25）。"""
    step = np.asarray(step)
    return int(np.median(np.diff(step))) if len(step) > 1 else 1


def _target_at(target: np.ndarray, t: np.ndarray, t0: float, steps: list):
    """時刻 t0 のフレームの目標の色の添字。フレームの target が −1 なら、t0 を含む手順の target。なければ None。"""
    if not np.isfinite(t0):
        return None
    i = _first_at_or_after(t, t0)
    if i is not None and target[i] >= 0:
        return int(target[i])
    for s in steps:
        if s.get("t_start", -math.inf) <= t0 <= (s.get("t_end") if s.get("t_end") is not None else math.inf):
            return COLORS.index(s["target"])
    return None


# harness/loop.py の試行の記録の contact_robot の列（sim.contact.COLUMNS と同じ並び）。obstacles を書いていない記録で、
# audit の欄があり列の数が合うときは、この並びとして読む
V2_OBSTACLES = ("cube_red", "cube_green", "cube_blue", "wall_xp", "wall_xn", "wall_yp", "wall_yn")


def obstacle_names(meta: dict, n_cols: int = None) -> list:
    names = meta.get("obstacles")
    if names:
        return list(names)
    if "audit" in meta and n_cols == len(V2_OBSTACLES):
        return list(V2_OBSTACLES)
    return []


def _obstacle_cols(meta: dict, n_cols: int = None) -> dict:
    return {name: k for k, name in enumerate(obstacle_names(meta, n_cols))}


def _collateral(rec: TrialRecord, eval_cfg: dict, target, t, t_fire) -> tuple[bool, bool]:
    """巻き添えと、意図的な失敗による移動。

    目標でない立方体が初期位置から水平に collateral_move_m 以上になったフレーム（越えた時点）ごとに、その移動の部分
    ＝越えたフレームの前で最後に静止していた（速さ < success.rest_speed）フレームの次から、越えたフレームまで、を見る。
    - 部分に、意図的な失敗の後（t_fire 以後の窓）の立方体どうしの接触（contact_cube_cube。相手の色は問わない）があり、
      それが部分の最初の contact_robot より先か同じフレームなら「意図的な失敗による移動」。玉突き（落ちた目標が A に当たり、
      A がこの立方体に当たった）もこれに入る
    - そうでなく、部分に contact_robot があれば「巻き添え」
    """
    a = rec.arrays
    cube_pos = np.asarray(a["cube_pos"], dtype=float)
    speed = np.linalg.norm(np.asarray(a["cube_linvel"], dtype=float), axis=2)       # (N, 3)
    moved = np.linalg.norm(cube_pos[:, :, :2] - cube_pos[:1, :, :2], axis=2) >= eval_cfg["collateral_move_m"]
    rest = speed < eval_cfg["success"]["rest_speed"]
    contact_robot = np.asarray(a["contact_robot"], dtype=bool)
    ccc = np.asarray(a["contact_cube_cube"], dtype=bool)
    cols = _obstacle_cols(rec.meta, contact_robot.shape[1] if contact_robot.ndim == 2 else None)
    n = len(t)
    idx = np.arange(n)
    after_fire = t >= t_fire - 1e-9 if np.isfinite(t_fire) else np.zeros(n, dtype=bool)

    robot_any = induced_any = False
    for c, color in enumerate(COLORS):                  # 立方体 3 つ（フレームの向きはすべて配列演算）
        cross = moved[:, c] & ~np.concatenate(([False], moved[:-1, c]))            # 越えたフレーム
        cross &= (target != c)
        if not cross.any():
            continue
        k = cols.get(f"cube_{color}")
        robot = contact_robot[:, k] if k is not None else np.zeros(n, dtype=bool)
        robot = robot & (target != c)
        by_drop = np.any(np.delete(ccc[:, c, :], c, axis=1), axis=1) & after_fire   # 自分以外の立方体との接触
        # 各フレーム i について、i 以前で最後に静止していたフレーム（なければ −1）
        last_rest = np.maximum.accumulate(np.where(rest[:, c], idx, -1))
        # 各フレーム i について、i 以前で最後に robot・by_drop があったフレーム（なければ −1）
        last_robot = np.maximum.accumulate(np.where(robot, idx, -1))
        last_drop = np.maximum.accumulate(np.where(by_drop, idx, -1))
        # 部分の始まり: 越えたフレームの 1 つ前までで最後に静止していたフレームの次
        start = np.concatenate(([0], last_rest[:-1] + 1))
        has_robot = last_robot >= start
        has_drop = last_drop >= start
        # 部分の中の最初の接触: 始まり以後の最初の robot・by_drop
        first_robot = _first_from(robot, start)
        first_drop = _first_from(by_drop, start)
        drop_first = has_drop & (~has_robot | (first_drop <= first_robot))
        ev = np.flatnonzero(cross)
        induced_any |= bool(np.any(drop_first[ev]))
        robot_any |= bool(np.any(has_robot[ev] & ~drop_first[ev]))
    return robot_any, induced_any


def _first_from(flag: np.ndarray, start: np.ndarray) -> np.ndarray:
    """各フレーム i について、start[i] 以後で最初に flag が真のフレーム（なければ len）。"""
    n = len(flag)
    nxt = np.where(flag, np.arange(n), n)
    nxt = np.minimum.accumulate(nxt[::-1])[::-1]                 # i 以後で最初に真のフレーム
    nxt = np.append(nxt, n)
    return nxt[np.minimum(start, n)]


def _stage_reached(phase, target):
    """段階別の到達: 手順の中（target ≥ 0）のフレームの phase から、到達した区分の最も先のもの。なければ None。"""
    phase = np.asarray(phase, dtype=np.int64)
    ok = (np.asarray(target) >= 0) & (phase >= 0) & (phase < len(_PHASE_STAGE))
    st = _PHASE_STAGE[phase[ok]]
    st = st[st >= 0]
    return STAGES[int(st.max())] if st.size else None


def _ee_speed_jumps(a: dict, k_frames: np.ndarray, frame_dt: float) -> np.ndarray:
    """チャンク境界（切り替わりの行動が始まるフレーム i）での手先速度の差 |u_i − u_{i−1}| [m/s]。
    u_i = (ee_pos[i+1] − ee_pos[i]) / frame_dt はフレーム i の後の部分の手先速度。"""
    ee = np.asarray(a["ee_pos"], dtype=float)
    if len(ee) < 3:
        return np.zeros(0)
    u = np.diff(ee, axis=0) / frame_dt                            # (N−1, 3)
    sw = np.asarray(a["chunk_switch"], dtype=bool)
    act = np.asarray(a["action"], dtype=float)
    kf = k_frames[1:] if len(k_frames) and k_frames[0] == 0 else k_frames   # 最初の行動は前がない
    # 切り替わりで、前の行動もある（seam_jumps と同じ時点）。u_i と u_{i−1} が両方ある i だけ
    prev = kf - FRAMES_PER_ACTION
    ok = (prev >= 0) & (kf <= len(u) - 1)
    kf, prev = kf[ok], prev[ok]
    ok = sw[kf] & np.all(np.isfinite(act[kf, :3]), axis=1) & np.all(np.isfinite(act[prev, :3]), axis=1)
    i = kf[ok]
    j = np.linalg.norm(u[i] - u[i - 1], axis=1)
    return j[np.isfinite(j)]


def _contacts_n(rec: TrialRecord, target: np.ndarray) -> int:
    """contact_robot の、目標以外の立方体（cube_*）と壁（wall_*）の列で、偽から真に変わった回数（フレーム 0 の前は偽）。"""
    cr = np.asarray(rec.arrays["contact_robot"], dtype=bool)
    names = obstacle_names(rec.meta, cr.shape[1] if cr.ndim == 2 else None)
    if cr.ndim != 2 or cr.shape[1] == 0:
        return 0
    use = np.zeros(cr.shape, dtype=bool)
    for k, name in enumerate(names):
        if name.startswith("wall_"):
            use[:, k] = True
        elif name.startswith("cube_") and name[5:] in COLORS:
            use[:, k] = target != COLORS.index(name[5:])
    x = cr & use
    prev = np.vstack([np.zeros((1, x.shape[1]), dtype=bool), x[:-1]])
    return int(np.sum(x & ~prev))
