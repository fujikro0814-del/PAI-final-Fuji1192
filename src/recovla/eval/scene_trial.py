"""3 色の場面で、1 色の指示（立方体 1 個）を閉じたループで評価する最小限の試行。

    rig = SimRig(render=True)
    targets = choose_targets(seeds_list, layouts)           # 試行ごとの指示（机上の色から均等に）
    meta, arrays, video = run_trial(rig, layout, target, act, trial_info)
    write_trial(out_dir, index, meta, arrays)               # docs/interfaces/trial_record.md の形
    row = metrics.trial_metrics(metrics.load_trial(json_path), cfg["eval"])   # 成功・誤り・巻き添え・段階など

- 指令の入口は closed_loop.execute_action と同じ: 10 fps の行動 1 つを物理 50 ステップに配り、
  速度 = 行動の xyz / 0.1 s を入力の読み取り 5 回に渡して保つ。グリッパは符号が制御器の状態と違えば最初の読み取りで押す
- 成功（評価器が判定する）: 目標が箱の成功の体積の中で速さ 0.01 m/s 未満の状態が 1.0 s 続く。
  物理ステップごとに判定し、成功か制限時間で打ち切る
- 誤り・巻き添え・段階別の到達・接触回数は、記録から recovla.eval.metrics が計算する
- 真値は判定と記録にだけ使い、方策には渡さない
"""
import json
import pathlib

import numpy as np

from recovla.common import code_version, config, seeds
from recovla.common.seeds import COLORS
from recovla.expert.script import PhaseParams
from recovla.record import episode as E
from recovla.sim import contact, frames

_CFG = config.load()
RECORD_VERSION = 1
STEPS_PER_ACTION = int(_CFG["sim"]["stride"]) * int(_CFG["sim"]["record_every"])          # 50
ACTION_DT = STEPS_PER_ACTION * float(_CFG["sim"]["timestep"])                               # 0.1 s
TRIAL_KEYS = ("step", "sim_time", "ee_pos", "ee_quat", "fingertip", "fingers", "x_des", "gripper_closed",
              "cube_pos", "cube_quat", "cube_linvel", "cube_in_box", "target", "phase", "contact_robot",
              "contact_cube_cube", "min_dist", "safety_active")


def choose_targets(seed_list, layouts) -> list:
    """試行ごとの目標の色。机上の色から均等に: 試行の番号 i について COLORS[i % 3] が机上にあればそれ、
    なければ、そのシードの order の乱数列で机上の色から 1 つ選ぶ。"""
    out = []
    for i, (seed, lay) in enumerate(zip(seed_list, layouts)):
        c = COLORS[i % len(COLORS)]
        if c not in lay.table_colors:
            c = lay.table_colors[int(seeds.stream(seed, "order").integers(len(lay.table_colors)))]
        out.append(c)
    return out


def instruction(color: str) -> str:
    return _CFG["convert"]["instruction"].format(color=color)


def execute_action(rig, action, on_step=None) -> None:
    a = np.asarray(action, dtype=np.float64)
    vel = a[:3] / ACTION_DT
    press = bool(a[6] > 0.0) != bool(rig.controller.gripper_closed)
    for r in range(STEPS_PER_ACTION // rig.steps_per_read):
        rig.pad_read(vel, press and r == 0, on_step)


def run_trial(rig, layout, target: str, act, trial: dict, time_limit_s: float = None, render: bool = True,
              inducer=None):
    """act(k, frame, raw_by_view, task) -> 7 次元の行動（10 fps の k ステップ目）。trial: 記録の属性（trial・seed・
    experiment・condition・model・runtime）。返り値 (meta, arrays, raw_video)。

    inducer（recovla.eval.induce.Inducer）を渡すと、行動を 1 つずつ上書きし、実行の後に実際の失敗を判定する。
    act が trace（recovla.policy.runner.SceneRunner）を持てば、行動列の番号・添字・予測経路・推論の記録を埋める。"""
    ev = _CFG["eval"]
    time_limit_s = float(ev["time_limit_s"]) if time_limit_s is None else float(time_limit_s)
    rest_speed, rest_hold = float(ev["success"]["rest_speed"]), float(ev["success"]["rest_hold_s"])
    pp = PhaseParams.from_config()
    rig.reset(layout)
    rig.safety.start_trial(target)                  # 安全フィルタ: 障害物の組はこの目標で作る
    ti = COLORS.index(target)
    dt = rig.timestep
    every = rig.record_every
    state = {"hold": 0.0, "success_t": None, "rest": 0.0}
    frames_log, actions, video, act_k, induced = [], [], [], [], []

    def capture():
        f, imgs = E.capture_frame(rig, target, state["rest"], pp, render)
        frames_log.append(f)
        actions.append(np.full(7, np.nan))
        act_k.append(-1)
        induced.append(False)
        state["raw"] = dict(zip(rig.cameras, imgs))
        if render:
            video.append(np.hstack(imgs))
        return f, state["raw"]

    def on_step(r):
        d = r.data
        pos = d.xpos[r.cube_ids[ti]]
        v = r.cube_vadr[ti]
        speed = float(np.linalg.norm(d.qvel[v:v + 3]))
        state["rest"] = state["rest"] + dt if speed < pp.rest_speed else 0.0
        if state["success_t"] is None:
            if frames.in_box(pos, r.box) and speed < rest_speed:
                state["hold"] += dt
                t = r.step * dt
                if state["hold"] >= rest_hold - 1e-9 and t <= time_limit_s + 1e-9:
                    state["success_t"] = float(t)
            else:
                state["hold"] = 0.0
        if r.step % every == 0:
            capture()

    from recovla.sim.rig import quiet
    k = 0
    frame, raw = capture()
    task = instruction(target)
    with quiet():
        while state["success_t"] is None and rig.step * dt < time_limit_s - 1e-9:
            a = np.asarray(act(k, frame, raw, task), dtype=np.float64)
            if inducer is not None:
                a = np.asarray(inducer.filter(k, a, rig.truth(target)), dtype=np.float64)
            actions[-1] = a
            act_k[-1] = k
            induced[-1] = bool(inducer is not None and inducer.active)
            rig.safety.gate = not induced[-1]           # 方策が指令を出している間だけ（意図的な失敗の上書き中は切る）
            n0 = len(frames_log)
            execute_action(rig, a, on_step)
            for i in range(n0, len(frames_log) - 1):        # 行動 k は、フレーム 2k と 2k+1 の後に効く
                actions[i] = a
                act_k[i] = k
                induced[i] = induced[n0 - 1]
            if inducer is not None:
                inducer.after(k, rig.truth(target))
            frame, raw = frames_log[-1], state["raw"]            # フレーム 2k+2（行動 k+1 の直前の観測）
            k += 1
    arrays = {key: np.array([f[key] for f in frames_log]) for key in TRIAL_KEYS}
    n = len(frames_log)
    arrays.update({
        "target": np.full(n, ti, dtype=np.int8), "phase": arrays["phase"].astype(np.int8),
        "action": np.array(actions, dtype=np.float64),
        "chunk_id": np.full(n, -1, dtype=np.int32), "chunk_index": np.full(n, -1, dtype=np.int32),
        "chunk_switch": np.zeros(n, dtype=bool), "induce_active": np.array(induced, dtype=bool),
        "chunk_k_valid": np.zeros(0, dtype=np.int32), "chunk_xdes_pred": np.zeros((0, 50, 3)),
        "chunk_grip_pred": np.zeros((0, 50)),
    })
    inference = trial.get("inference", [])
    if hasattr(act, "trace"):                                  # 方策の実行器: 行動列の記録
        tr_ = act.trace()
        ex = tr_["exec"]
        cid = np.array([ex[kk][0] if 0 <= kk < len(ex) else -1 for kk in act_k], dtype=np.int32)
        cix = np.array([ex[kk][1] if 0 <= kk < len(ex) else -1 for kk in act_k], dtype=np.int32)
        sw = np.zeros(n, dtype=bool)
        prev = -1
        for f_i in range(n):                                   # 行動列が切り替わった行動の最初のフレーム
            if cid[f_i] >= 0 and cid[f_i] != prev and prev >= 0:
                sw[f_i] = True
            if cid[f_i] >= 0:
                prev = cid[f_i]
        arrays.update({"chunk_id": cid, "chunk_index": cix, "chunk_switch": sw,
                       "chunk_k_valid": tr_["chunk_k_valid"], "chunk_xdes_pred": tr_["chunk_xdes_pred"],
                       "chunk_grip_pred": tr_["chunk_grip_pred"],
                       # 推論ごとの目標位置の入力 [k_obs, x, y, 旗]。真値は同じ行動の番号のフレーム 2·k_obs の cube_pos
                       "cue_at_inference": tr_.get("cue_at_inference", np.zeros((0, 4)))})
        inference = tr_["inference"]
    success = state["success_t"] is not None
    t_end = float(arrays["sim_time"][-1])
    meta = {
        "record_version": RECORD_VERSION, **trial,
        "layout": {"kind": layout.kind, "start": layout.start,
                   "cubes": {c: [float(v) for v in layout.cubes[c]] for c in layout.table_colors},
                   "prefilled": sorted(layout.prefilled, key=COLORS.index), "seed": layout.seed},
        "steps": [{"target": target, "instruction": task, "t_start": 0.0, "t_end": t_end,
                   "success": success, "t_success": state["success_t"]}],
        "success": success, "time_limit_s": time_limit_s,
        "induce": inducer.record() if inducer is not None else
        {"kind": None, "params": {}, "fired": False, "t_fire": None, "established": False,
         "t_established": None, "t_failure": None, "reason": None},
        "obstacles": list(contact.COLUMNS), "inference": inference,
        "code_version": code_version.code_version(),
        "safety": rig.safety.summary(),
    }
    if isinstance(meta.get("runtime"), dict):             # 安全フィルタの実際の入切（configs を上書きした場合も）
        meta["runtime"] = {**meta["runtime"], "safety_filter": rig.safety.enabled}
    rig.safety.gate = False
    return meta, arrays, video


def write_trial(out_dir, index: int, meta: dict, arrays: dict) -> pathlib.Path:
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"trial_{index:04d}.json"
    if path.exists():
        raise FileExistsError(path)
    np.savez(out_dir / f"trial_{index:04d}.npz", **arrays)
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    return path
