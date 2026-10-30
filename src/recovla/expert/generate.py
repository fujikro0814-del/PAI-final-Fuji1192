"""データ生成の枠: 生成プログラムで 1 本ずつ作り、失敗したら生成プログラムのシードだけを変えて作り直す。

    specs = plan_specs(...)                       # [EpisodeSpec]（配置のシード・配置の種類・目標の色）
    results = generate(specs, run_dir, workers=8, render=True)

- 1 つの指定（配置、色）について、作り直しの回数 r = 0, 1, …, expert.slip_retry_max で順に試み、最初に成功した
  ものを保存する。全部失敗したらその配置と色は捨てて件数を残す（どのデータの組でも成功データを同じに保つため）
- 失敗の区分（生成プログラムの外で決める）:
    grasp    carry に入る前に reopen になった（掴み損ね）
    slip     carry に入った後、release の前に settle・reopen になった（運ぶ途中で指から外れた）
    place    release の後に、接近・把持・運搬の段階に戻った（置き損ね）
    timeout  expert.time_limit_s を超えた
- 描画なし（render=False）でも同じ物理・同じ判定で回る。描画は scratch の複写に対して行うので、
  物理の列は描画の有無で変わらない（tests/test_d_scene.py）
- 並列: multiprocessing の spawn。各プロセスがシミュレータを 1 回だけ作る。仕事の単位は 1 つの配置（その全色の指定）。
  各プロセスは自分のエピソードのフォルダにだけ書き、親が generation.jsonl を指定の順に書く
- 引き継ぎ（kind "h"）: 指定の時刻に生成プログラムを新しいシードで作り直して続ける
"""
import dataclasses
import datetime
import json
import multiprocessing
import pathlib
import time

import numpy as np

from recovla.common import code_version, config, seeds
from recovla.common.seeds import COLORS
from recovla.expert import inject
from recovla.expert import script as S
from recovla.record import episode as E
from recovla.record import snapshot
from recovla.sim import contact, frames, scene

_CFG = config.load()
PROGRESS_ORDER = [S.Phase.approach, S.Phase.descend, S.Phase.close, S.Phase.lift, S.Phase.carry,
                  S.Phase.release, S.Phase.settle, S.Phase.retreat, S.Phase.done]
HANDOVER_RETRY_KEY = 1000          # 引き継ぎ後のシード: script_rng(配置のシード, 色, 1000 + 作り直しの回数)


@dataclasses.dataclass(frozen=True)
class EpisodeSpec:
    layout_seed: int
    color: str
    layout_kind: str = None        # None なら配置の乱数列から割合で決める
    kind: str = "n"                # n 成功データ、h 引き継ぎ、A・B・C 立て直しデータ（失敗の種類）
    handover_u: float = None       # 引き継ぎの時刻（通常の所要時間に対する割合）。kind "h" のとき
    start: str = None              # 開始姿勢の上書き（None なら配置の乱数列から。"home" は待機姿勢）

    def name(self, retry: int) -> str:
        return f"{self.kind}_{self.layout_seed}_{self.color}_r{retry}"


def instruction(color: str) -> str:
    return _CFG["convert"]["instruction"].format(color=color)


def plan_specs(seed_ranges: dict, kind: str = "n", start: str = None) -> list:
    """{配置の種類: range(配置のシード)} から、各配置の机上の全色の指定を作る。"""
    specs = []
    for layout_kind, rng_ in seed_ranges.items():
        for seed in rng_:
            lay = scene.sample_layout(seed, layout_kind, start=start)
            specs += [EpisodeSpec(int(seed), c, layout_kind, kind, start=start) for c in lay.table_colors]
    return specs


# ------------------------------------------------------------------------------ 1 回の試み

def run_attempt(rig, spec: EpisodeSpec, retry: int, run_dir=None, render: bool = True,
                handover_at_s: float = None) -> dict:
    """1 回の試み。run_dir があれば記録し、成功なら保存、失敗なら捨てる。

    立て直しデータ（kind A・B・C）: 失敗の意図的な失敗（recovla.expert.inject）が確定した後の、最初の 10 Hz のフレームの境目
    （物理ステップが record_every × stride の倍数）から記録を始め、そこで生成プログラムを作り直して
    （同じパラメータ、時計は 0 から）立て直す。それより前は記録しない。失敗の区分は記録を始めてからの段階で判定し、
    意図的な失敗の結果が confirmed 以外なら "inject_<結果>" で打ち切る。"""
    cfg = rig.cfg
    layout = scene.sample_layout(spec.layout_seed, spec.layout_kind, start=spec.start)
    rig.reset(layout)
    dt = rig.steps_per_read * rig.timestep
    params = S.sample_params(seeds.script_rng(spec.layout_seed, spec.color, retry), cfg)
    expert = S.Expert(params, dt, cfg)
    pp = expert.pp
    recovery = spec.kind in inject.KINDS
    inj = None
    if recovery:
        ip = inject.sample_params(spec.kind, seeds.inject_rng(spec.layout_seed, spec.color, retry), layout,
                                  spec.color, cfg)
        inj = inject.Injector(ip, expert, rig, cfg)
    mask = contact.ContactMeter.obstacle_mask(spec.color, rig.cubes_in_box())
    every = rig.record_every
    split_every = every * int(cfg["sim"]["stride"])
    name = spec.name(retry)
    frames_log = []
    state = {"expert": expert, "writer": None, "start": None, "rec": False,
             "pre_contact": np.zeros(len(contact.COLUMNS), dtype=bool),
             "pre_cc": np.zeros((len(COLORS), len(COLORS)), dtype=bool)}

    def capture():
        first_frame = not frames_log
        f, imgs = E.capture_frame(rig, spec.color, state["expert"].clocks.target_rest_s, pp,
                                  render and not first_frame)
        if hasattr(rig, "postprocess_frame"):          # センサのモデルつきの生成（harness/gen_sensed.py）: 状態をセンサの値に置き換える
            f = rig.postprocess_frame(f)
        if render and first_frame:
            # 最初のフレーム: GPU の描画は同じ状態でもまれに 1 階調ずれるので、状態（qpos）がビット一致する
            # 直前の画像があればそれを使う（同じ配置の各色の最初の観測を画素単位で揃える）
            key = rig.scratch.qpos.tobytes()
            cached = getattr(rig, "_frame0_cache", None)
            if cached is not None and cached[0] == key:
                imgs, state["frame0_shared"] = cached[1], True
            else:
                imgs = rig.render()
                rig._frame0_cache = (key, imgs)
                state["frame0_shared"] = False
        frames_log.append(f)
        if state["writer"] is not None:
            state["writer"].add_frame(f, imgs)

    def begin_recording():
        state["start"] = snapshot.capture(rig.model, rig.data, rig.controller, rig.integrator, rig.step)
        if recovery:
            # 途中から始めるとき: mj_step の直後は派生量（xpos・ヤコビアン）が 1 ステップ前のままで、制御器はそれを読む。
            # 保存した状態からの再生（snapshot.restore）と同じく mj_forward で揃える
            # （成功データは SimRig.reset の最後に揃えてあるので通さない）
            snapshot.restore(rig.model, rig.data, rig.controller, rig.integrator, state["start"])
        state["rec"] = True
        rig.meter.reset_window()
        if run_dir:
            state["writer"] = E.SceneEpisodeWriter(run_dir, name, rig.cameras if render else ())
            state["writer"].add_step(rig.controller.desired_pos, float(rig.data.ctrl[rig.grip_act]))
        capture()

    def on_step(r):
        if not state["rec"]:
            if r.step % every == 0:                      # 記録の前: 接触だけ集める（意図的な失敗の部分の接触を残す）
                robot, cc = r.meter.take_window()
                state["pre_contact"] |= robot
                state["pre_cc"] |= cc
            return
        if state["writer"] is not None:
            state["writer"].add_step(r.controller.desired_pos, float(r.data.ctrl[r.grip_act]))
        if r.step % every == 0:
            capture()

    if not recovery:
        begin_recording()
    first, pre_first = {}, {}
    failure, t_fail, handed, t_rec0 = None, None, None, 0.0
    time_limit = float(cfg["expert"]["time_limit_s"])
    wall0 = time.perf_counter()
    from recovla.sim.rig import quiet
    with quiet():
        while True:
            tr = rig.truth(spec.color)
            if recovery and not state["rec"]:
                if inj.finished and inj.status != "confirmed":
                    failure = f"inject_{inj.status}"
                elif inj.status == "confirmed" and rig.step % split_every == 0:
                    begin_recording()
                    inj.handover()
                    state["expert"] = S.Expert(params, dt, cfg)
                    t_rec0 = tr.t
                    time_limit = float(cfg["inject"]["recovery_time_limit_s"])
                elif tr.t > time_limit:
                    failure = "timeout"
                if failure:
                    t_fail = round(tr.t, 3)
                    break
            if handover_at_s is not None and handed is None and tr.t >= handover_at_s - 1e-9:
                hp = S.sample_params(seeds.script_rng(spec.layout_seed, spec.color, HANDOVER_RETRY_KEY + retry), cfg)
                state["expert"] = S.Expert(hp, dt, cfg)
                handed = {"t": round(tr.t, 3), "phase_before": None, "params": hp.to_json()}
            cmd = state["expert"].act(tr)
            if inj is not None:
                cmd = inj.modify(cmd, tr)
            if handed is not None and handed["phase_before"] is None:
                handed["phase_before"] = S.Phase(frames_log[-1]["phase"]).name
            ph = cmd.phase
            if recovery and not state["rec"]:
                pre_first.setdefault(ph.name, round(tr.t, 3))
                if inj.status == "armed" and (ph == S.Phase.reopen or (
                        "carry" in pre_first and ph in (S.Phase.settle, S.Phase.reopen))):
                    failure = "natural_before_inject"     # 意図的な失敗の前にひとりでに掴み損ねた・滑った
            else:
                first.setdefault(ph.name, round(tr.t, 3))
                grasp_fail = ph == S.Phase.reopen and "carry" not in first and (not recovery or "close" in first)
                if grasp_fail:
                    failure = "grasp"
                elif "carry" in first and "release" not in first and ph in (S.Phase.settle, S.Phase.reopen):
                    failure = "slip"
                elif "release" in first and ph in (S.Phase.approach, S.Phase.descend, S.Phase.close,
                                                   S.Phase.lift, S.Phase.carry, S.Phase.reopen):
                    failure = "place"
                elif tr.t - t_rec0 > time_limit:
                    failure = "timeout"
            if failure:
                t_fail = round(tr.t, 3)
                break
            if cmd.finished:
                break
            rig.pad_read(cmd.vel, cmd.press, on_step)
    wall = time.perf_counter() - wall0
    tr = rig.truth(spec.color)
    writer = state["writer"]
    success = (failure is None and state["rec"] and frames.in_box(tr.target_pos, tr.box)
               and tr.target_speed < pp.rest_speed)
    if failure is None and not success:
        failure = "place"
    arr = {k: np.array([f[k] for f in frames_log]) for k in ("phase", "min_dist", "contact_robot")}
    if not frames_log:                                   # 記録を始める前に打ち切った（失敗の意図的な失敗が不実際の失敗など）
        arr = {"phase": np.zeros(0), "min_dist": np.full((1, len(contact.COLUMNS)), np.nan),
               "contact_robot": np.zeros((0, len(contact.COLUMNS)), dtype=bool)}
    contact_obst = arr["contact_robot"][:, mask]
    summary = {
        "name": name, "layout_seed": spec.layout_seed, "color": spec.color, "retry": retry, "kind": spec.kind,
        "layout": layout.to_json(), "script": params.to_json(), "success": bool(success), "failure": failure,
        "t_failure": t_fail, "duration_s": round(float(tr.t), 3), "n_frames": len(frames_log),
        "phase_first_t": first, "handover": handed, "wall_s": round(wall, 3),
        "obstacle_mask": mask.tolist(),
        "contact_obstacle_frames": int(contact_obst.any(axis=1).sum()),
        "contact_obstacle_onsets": int((np.diff(contact_obst.astype(int), axis=0, prepend=0) == 1).sum()),
        "min_dist_obstacle": float(arr["min_dist"][:, mask].min()),
        "target_yaw_deg": round(float(np.degrees(layout.cubes[spec.color][2])), 2),
        "final_target_pos": [round(float(v), 5) for v in tr.target_pos],
        "final_target_tilt_deg": round(frames.tilt_deg(tr.cube_quat[tr.target]), 3),
    }
    if recovery:
        summary["inject"] = {"params": inj.p.to_json(), "status": inj.status, "reason": inj.reason,
                             "t_fire": inj.t_fire, "t_confirm": inj.t_confirm, "info": inj.info,
                             "phase_first_t_before": pre_first,
                             "contact_obstacle_before": bool(state["pre_contact"][mask].any()),
                             # 記録の前に目標の立方体が触れた立方体（B・C の落下で当たったか）と、初めから箱の中にある色
                             "cube_contact_before": [c for i, c in enumerate(COLORS)
                                                     if state["pre_cc"][COLORS.index(spec.color), i]],
                             "prefilled": sorted(layout.prefilled)}
        summary["t_record_start"] = round(t_rec0, 3) if state["rec"] else None
        summary["record_start_step"] = int(state["start"].step) if state["rec"] else None
        summary["recovery_duration_s"] = round(float(tr.t) - t_rec0, 3) if state["rec"] else None
    start = state["start"]
    if writer is not None:
        if success:
            meta = {
                **summary, "instruction": instruction(spec.color), "target": spec.color,
                "pair_id": spec.layout_seed, "layout_kind": layout.kind, "start_pose": layout.start,
                "record_every": every, "record_dt": every * rig.timestep,
                "record_hz": round(1.0 / (every * rig.timestep)), "timestep": rig.timestep,
                "cameras": {"names": list(rig.cameras) if render else [],
                            "width": rig.renderer.width if rig.renderer else None,
                            "height": rig.renderer.height if rig.renderer else None},
                "rendered": bool(render),
                "frame0_image_shared_with_pair": bool(state.get("frame0_shared", False)),
            }
            if hasattr(rig, "episode_meta"):               # センサのモデルつきの生成: モデルの版と較正の値
                meta.update(rig.episode_meta())
            summary["path"] = str(writer.finalize(meta, start.to_arrays(), every))
        else:
            writer.discard()
    summary["_frames"] = frames_log if not writer else None
    return summary


def run_spec(rig, spec: EpisodeSpec, run_dir=None, render: bool = True, max_retry: int = None) -> dict:
    """作り直しを含めて 1 つの指定を処理する。"""
    max_retry = int(rig.cfg["expert"]["slip_retry_max"]) if max_retry is None else max_retry
    attempts = []
    handover_at = None
    for retry in range(max_retry + 1):
        if spec.kind == "h":
            base = run_attempt(rig, dataclasses.replace(spec, kind="n"), retry, None, False)
            handover_at = float(spec.handover_u) * float(base["duration_s"])
        a = run_attempt(rig, spec, retry, run_dir, render, handover_at)
        a.pop("_frames", None)
        attempts.append(a)
        if a["success"]:
            break
    return {"layout_seed": spec.layout_seed, "color": spec.color, "layout_kind": spec.layout_kind,
            "kind": spec.kind, "success": attempts[-1]["success"], "retries": len(attempts) - 1,
            "first_try_success": attempts[0]["success"], "attempts": attempts}


# ------------------------------------------------------------------------------------ 並列

_RIG = None


def rig_config(rig_kind: str):
    """v3 なら、v2 の設定に configs/expert_v3.yaml を重ねる（生成プログラムの終盤の速さだけが変わる）。"""
    if rig_kind == "v3":
        return config.load(*[n for n in config.V2_LAYERS if (config.CONFIG_DIR / f"{n}.yaml").is_file()], "expert_v3")
    return None


def _init_worker(render: bool, rig_kind: str = "v1") -> None:
    global _RIG
    if rig_kind in ("v2", "v3"):                      # 実機と同じ指令の口・センサのモデルで記録する
        from recovla.harness.gen_sensed import SensedDrivenRig
        _RIG = SensedDrivenRig(rig_config(rig_kind))
        return
    from recovla.sim.rig import SimRig
    _RIG = SimRig(render=render)


def _work(job):
    """仕事の単位は 1 つの配置（その全色）。同じプロセスで描くので、最初のフレームの画像は並列数によらず
    画素単位で一致する（別のプロセスの描画は、まれに 1 画素が 1 階調ずれる）。"""
    specs, run_dir, render, max_retry = job
    outs = []
    for spec in specs:
        t0 = time.perf_counter()
        out = run_spec(_RIG, spec, run_dir, render, max_retry)
        out["worker_wall_s"] = round(time.perf_counter() - t0, 3)
        out["worker_pid"] = __import__("os").getpid()
        outs.append(out)
    return outs


def generate(specs: list, run_dir, workers: int = 1, render: bool = True, max_retry: int = None,
             log_name: str = "generation.jsonl", rig_kind: str = "v1") -> list:
    """指定の列を生成し、generation.jsonl を指定の順に書く。run_dir は新しく作る（あれば拒否）。"""
    run_dir = pathlib.Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    version = code_version.code_version()
    (run_dir / "run.json").write_text(json.dumps({
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "workers": workers, "render": render, "n_specs": len(specs), "code_version": version,
        "config_expert": (rig_config(rig_kind) or _CFG)["expert"], "config_scene": _CFG["scene"], "rig_kind": rig_kind,
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    groups = {}
    for s in specs:                                   # 同じ配置を 1 つの仕事にまとめる。順は最初に現れた順
        groups.setdefault((s.layout_seed, s.layout_kind, s.kind, s.start), []).append(s)
    jobs = [(g, str(run_dir), render, max_retry) for g in groups.values()]
    results = []
    t0 = time.perf_counter()
    log = open(run_dir / log_name, "w", encoding="utf-8")
    try:
        if workers <= 1:
            _init_worker(render, rig_kind)
            it = map(_work, jobs)
            pool = None
        else:
            pool = multiprocessing.get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(render, rig_kind))
            it = pool.imap(_work, jobs)
        for rs in it:
            for r in rs:
                r["code_version"] = version.get("code_sha256")
                results.append(r)
                log.write(json.dumps(r, ensure_ascii=False) + "\n")
                log.flush()
        if pool is not None:
            pool.close()
            pool.join()
    finally:
        log.close()
        if workers <= 1 and _RIG is not None:
            _RIG.close()
    wall = time.perf_counter() - t0
    (run_dir / "timing.json").write_text(json.dumps({
        "workers": workers, "render": render, "specs": len(specs), "wall_s": round(wall, 2),
        "episodes_saved": sum(r["success"] for r in results),
        "attempts": sum(len(r["attempts"]) for r in results),
        "episodes_per_hour": round(sum(r["success"] for r in results) / wall * 3600.0, 1) if wall > 0 else None,
    }, indent=2), encoding="utf-8")
    return results
