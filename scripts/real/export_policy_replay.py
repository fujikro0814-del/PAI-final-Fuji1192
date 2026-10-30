"""方策の再生（実機の試験 H1）の書き出し: v3_real で VLA が出した関節の指令を、実機に流す列（試験 A8 の生成プログラムの
再生と同じ形）にする。

    python scripts/real/export_policy_replay.py capture [--tree C:\\PAI\\wt_demo_clc] [--seed 171318] [--arm L2v2]
    python scripts/real/export_policy_replay.py process            # capture.pkl から作り直す（試行は回さない）
    python scripts/real/export_policy_replay.py finish
    python scripts/real/export_policy_replay.py all ...            # capture（process）の後に finish
    （capture は --tree の作業場所で、PYTHONPATH=<tree>\\src;<tree>、PYTHONUTF8=1 で起動する）

手順: docs/real/h1_policy_replay.md の A。列の決まり: docs/real/replay_format.md（Linux 側の再生と compare_replay.py をそのまま使う）。

capture（GPU。--tree の作業場所の recovla と scripts/api/ladder.py を使う）:
1. ladder.py run を同じプロセスで呼び、1 本だけ回す（動画の場面と同じ引数。世界 v3_real は ladder が入れる）。
   recovla.harness.task_loop.run_task_trial を包み、試行の始めに世界（world）の reset・apply_joint_commands・physics_step に
   記録の口を被せる（v3_real の腕のサーボが差し替えた後の口を包む）。記録するもの:
   制限層の出力の 1 kHz の指令（物理の 1 ステップに 2 つ）、物理の 1 ステップの後の腕の関節角・速さと指の幅、ハンドの指令（world.hand.log）、
   開始の関節角（制限層の最初の reset）。
2. 切り出し: 始めから、最初の手順（赤）と、その後の待機位置への戻し（step_reset home）の終わりまで。戻しの後に指令が 0.2 s
   変わらない所が次の手順の前にあればそこで切る。無ければ（戻しは関節の目標へ指数的に近づくので、指令はわずかに動き続ける）
   次の手順の始めで切り、制限層（runtime/limiter.py、同じ余裕）に速さ 0 を渡して止まりきるまで続け、さらに 0.2 s 止める（tail）。
   切った後のハンドの指令は捨てる。
3. 実機への変換・確かめ・列は export_replay.py の関数をそのまま使う（to_real・check_stream・FK・build_rows・write_csv）。外れたら書かない。
4. 確かめ（a）: 新しく作った v3_real の評価の世界に、同じ指令・ハンドの指令を開ループで流す（立方体あり＝閉ループと同じ）。
   閉ループの関節角を再現すれば、記録は欠けていない。足した止める分（tail）の応答はこの開ループの値を使う。
5. 遅い版: export_replay_slow.py の引き延ばし（stretch・時刻 2 倍）。応答は v3_real の評価の世界（実機に合わせた腕とハンド）に
   立方体なしで流したもの（export_replay_slow.main は合わせる前の生成の世界を使うので、ここでは関数だけ借りる）。
finish（CPU）: resim_replay.py --hand で通常・遅い版の応答を付け直し（_fit）、build_replay_zip.py で zip、確かめ（b）（立方体が
腕の動きをどれだけ変えたか）と、resim の世界と v3_real の評価の世界の応答の差を出す。
書くもの: outputs/real/policy_replay_v1/（<名前>.csv・<名前>_gripper.csv・manifest.json）、policy_replay_v1_slow/、
policy_replay_v1_capture/（生の記録・試行の json・確かめの結果）、finish で policy_replay_v1_fit/・policy_replay_v1_slow_fit/ と zip。
_ds.csv（学習データの列）は作らない（方策の試行は学習データの本ではなく、照らす行がない）。
"""
import argparse
import copy
import datetime
import hashlib
import importlib.util
import json
import os
import pathlib
import pickle
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]                                      # このスクリプトがある作業場所（書き出し先・既定の --tree）
OUT = "outputs/real/policy_replay_v1"
EXP = "H1REPLAY"
REST_S = 0.2                                                # 最後に止める長さ
REST_TOL = 1e-12                                            # 「指令が変わらない」の 1 刻みの差 [rad]
HOLD_MAX_S = 5.0
MISSING = object()


# ------------------------------------------------------------------ 作業場所（--tree）の切り替え
def use_tree(tree: pathlib.Path):
    """--tree の recovla を先に読み込む（export_replay は自分の作業場所の src を sys.path に足すので、その前に読む）。"""
    for p in (str(tree), str(tree / "src")):
        while p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    import recovla
    got = pathlib.Path(recovla.__file__).resolve()
    if tree.resolve() not in got.parents:
        raise SystemExit(f"recovla が --tree の外から読まれた: {got}（PYTHONPATH を確かめる）")
    if str(HERE) not in sys.path:
        sys.path.insert(1, str(HERE))
    import export_replay as X                               # noqa: F401
    import export_replay_slow as S                          # noqa: F401
    return X, S


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def git_info(root: pathlib.Path) -> dict:
    """作業場所のコミットと、書きかけがあるか（同梱の git を使う）。"""
    for exe in ("git", str(ROOT / ".tools" / "git" / "cmd" / "git.exe")):
        try:
            h = subprocess.run([exe, "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30)
            if h.returncode != 0:
                continue
            st = subprocess.run([exe, "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True, timeout=60)
            return {"root": str(root), "commit": h.stdout.strip(), "dirty": bool(st.stdout.strip())}
        except (OSError, subprocess.SubprocessError):
            continue
    return {"root": str(root), "commit": None, "dirty": None}


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def resolve(p) -> pathlib.Path:
    p = pathlib.Path(p)
    return p if p.is_absolute() else ROOT / p


# ------------------------------------------------------------------ 記録の口
def make_wrapper(orig, cap: dict):
    """run_task_trial の包み。試行の始めに世界の口へ記録を被せ、終わったら元に戻す。"""
    from recovla.runtime import limiter as L

    def wrapper(world, suite, make_task_runtime, layout, text, seed, cfg=None, time_limit_s=200.0):
        if cap.get("done"):
            raise RuntimeError("記録は 1 本だけ（--trials <種>:1）")
        names = ("reset", "apply_joint_commands", "physics_step")
        before = {n: world.__dict__.get(n, MISSING) for n in names}
        o_reset, o_apply, o_step = world.reset, world.apply_joint_commands, world.physics_step
        o_lreset = L.JointLimiter.reset
        cap.update(cmds=[], counts=[], meas=[], lim_resets=[], world_cfg=copy.deepcopy(world.cfg),
                   layout=layout.to_json(), seed=int(seed), text=text, dt=float(world.timestep))

        def reset(lay):
            o_reset(lay)                                     # v3_real: 腕を最初の指令へ置き直す（arm_real.install の reset）
            d = world.data
            cap["reset_q"] = d.qpos[world.arm_qadr].copy()
            cap["t0"] = float(d.time)

        def apply(cmds):
            cmds = [np.array(c, float) for c in cmds]
            cap["cmds"].extend(cmds)
            cap["counts"].append(len(cmds))
            o_apply(cmds)

        def step(on_step=None):
            o_step(on_step)
            d = world.data
            cap["meas"].append((float(d.time), d.qpos[world.arm_qadr].copy(), d.qvel[world.arm_vadr].copy(),
                                float(world.hand.width())))

        def lreset(self, q):
            o_lreset(self, q)
            cap["lim_resets"].append((float(world.data.time), int(world.step), np.array(q, float).copy()))

        world.reset, world.apply_joint_commands, world.physics_step = reset, apply, step
        L.JointLimiter.reset = lreset
        try:
            out = orig(world, suite, make_task_runtime, layout, text, seed, cfg, time_limit_s)
        finally:
            L.JointLimiter.reset = o_lreset
            for n in names:
                if before[n] is MISSING:
                    world.__dict__.pop(n, None)
                else:
                    world.__dict__[n] = before[n]
        cap["hand_log"] = list(world.hand.log)
        cap["done"] = True
        return out
    return wrapper


# ------------------------------------------------------------------ 開ループ（v3_real の評価の世界）
def open_loop_world(cfg: dict, layout_json: dict, cmds_sim, events, cubes: bool) -> dict:
    """新しく作った v3_real の評価の世界（make_eval_world）に、指令（1 kHz）を物理の 1 ステップに 2 つずつ流す。
    ハンドの指令は、その時刻の物理の 1 ステップの前に出す（閉ループの tick と同じ順）。cubes が偽なら立方体を無しにする
    （export_replay_slow.remove_cubes）。"""
    import export_replay_slow as S
    from recovla.harness import real_matched as V
    from recovla.sim import scene
    world, suite = V.make_eval_world(cfg)
    try:
        if not cubes:
            S.remove_cubes(world)
        world.reset(scene.Layout.from_json(layout_json))
        d = world.data
        start_q, t0 = d.qpos[world.arm_qadr].copy(), float(d.time)
        ev = sorted(events, key=lambda e: e[0])
        k, meas = 0, []
        for n in range(len(cmds_sim) // 2):
            while k < len(ev) and ev[k][0] <= float(d.time) - t0 + 1e-9:
                _, kind, args = ev[k]
                (world.hand.move if kind == "move" else world.hand.grasp)(*args)
                k += 1
            world.apply_joint_commands([cmds_sim[2 * n], cmds_sim[2 * n + 1]])
            world.physics_step(None)
            meas.append((float(d.time), d.qpos[world.arm_qadr].copy(), d.qvel[world.arm_vadr].copy(),
                         float(world.hand.width())))
        return {"meas": meas, "start_q": start_q, "t0": t0, "events_sent": k, "hand_log": list(world.hand.log)}
    finally:
        suite.close()


def meas_arrays(meas) -> dict:
    return {"t": np.array([m[0] for m in meas]), "q": np.array([m[1] for m in meas]),
            "dq": np.array([m[2] for m in meas]), "w": np.array([m[3] for m in meas])}


# ------------------------------------------------------------------ 切り出し
def find_cut(cmds, rec: dict, dt: float) -> dict:
    """最初の手順の後の待機位置への戻しの終わりを探す。返り値 n_keep（残す指令の数、偶数）、rest（0.2 s 変わらない所があったか）。"""
    s0 = rec["steps"][0]
    sws = [s for s in ((rec.get("step_reset") or {}).get("switches") or []) if s.get("step") == 1]
    if not sws or sws[0].get("t_end") is None:
        raise SystemExit("最初の手順の後の家への戻し（step_reset の switches の step 1）が記録にない")
    sw = sws[0]
    n_begin, n_end = int(round(sw["t_begin"] / dt)), int(round(sw["t_end"] / dt))
    i_begin, i_end = 2 * n_begin, 2 * n_end                 # 物理の 1 ステップ n の指令は行 2n・2n+1
    moving = np.abs(np.diff(cmds, axis=0)).max(axis=1) > REST_TOL     # moving[i]: 行 i → i+1 で変わった
    after = np.where(moving[i_begin:])[0]
    if not len(after):
        raise SystemExit("家への戻しの間に指令が動いていない")
    i_m = i_begin + int(after[0])                           # 戻しが動き始めた行
    w = int(round(REST_S / 1e-3))
    lo = i_m + w + 2
    for n_keep in range(lo + lo % 2, i_end + 1, 2):         # 行 n_keep−w−1〜n_keep−1 が変わらない（n_keep は偶数）
        if not moving[n_keep - w - 1:n_keep - 1].any():
            return {"n_keep": n_keep, "n_main": n_keep - w - 1, "rest_found": True, "step0": s0, "switch": sw,
                    "i_motion_begin": i_m}
    return {"n_keep": i_end, "n_main": i_end, "rest_found": False, "step0": s0, "switch": sw, "i_motion_begin": i_m}


def hold_tail(cmds, margin: float, q_min, q_max) -> tuple:
    """切った所から、制限層に速さ 0 を渡して止まりきるまで続け、さらに 0.2 s 止める指令。
    制限層の状態（位置・速さ・加速度）は、切る前の 3 つの指令から作る（制限層の commit と同じ式）。"""
    from recovla.runtime import limiter as L
    lim = L.JointLimiter(margin=margin, q_min=q_min, q_max=q_max)
    c = np.asarray(cmds, float)
    v1, v0 = (c[-1] - c[-2]) / L.DT, (c[-2] - c[-3]) / L.DT
    lim.q, lim.v, lim.a = c[-1].copy(), v1, np.clip((v1 - v0) / L.DT, -lim.max_acc, lim.max_acc)
    out, conv = [], None
    for k in range(int(HOLD_MAX_S / L.DT)):
        if np.abs(lim.v).max() < 1e-9 and np.abs(lim.a).max() < 1e-9:
            conv = k
            break
        out.append(lim.step(np.zeros(7)))
    if conv is None:
        return np.array(out), {"converged": False}
    n = int(round(REST_S / L.DT))
    n += (len(c) + len(out) + n) % 2                         # 物理の 1 ステップに 2 つずつ流すので、全体を偶数にする
    out += [lim.q.copy() for _ in range(n)]
    return np.array(out), {"converged": True, "stop_steps": conv, "rest_steps": n,
                           "v_at_cut": float(np.abs(v1).max()), "a_at_cut": float(np.abs((v1 - v0) / L.DT).max())}


# ------------------------------------------------------------------ capture
def cmd_capture(a) -> int:
    """試行を回し、生の記録を <out>_capture/capture.pkl に置く（後の処理が止まっても回し直さずに済むように、先に書く）。"""
    tree = pathlib.Path(a.tree).resolve()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    use_tree(tree)
    out, slow_out, cap_dir = resolve(a.out), resolve(a.out + "_slow"), resolve(a.out + "_capture")
    for p in (out, slow_out, cap_dir):
        if p.exists() and any(p.iterdir()):
            print(f"書き出し先 {p} が空でない。別の場所を指定する")
            return 2
    cond = a.condition or f"h1_s{a.seed}_{a.arm}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = tree / "outputs" / "v2eval" / a.experiment / cond
    if run_dir.exists():
        print(f"{run_dir} がもうある（96 は続きから回すので、別の条件名にする）")
        return 2
    from recovla.harness import task_loop as TL

    cap = {}
    TL.run_task_trial = make_wrapper(TL.run_task_trial, cap)
    argv = ["run", "--trials", f"{a.seed}:1", "--min-free-gb", "6", "--grip-debounce", "5", "--step-reset", "home",
            "--arm", a.arm, "--model", "Rreal", "--finder", "estimated", "--experiment", a.experiment, "--condition", cond]
    cwd = os.getcwd()
    os.chdir(tree)
    w0 = time.perf_counter()
    try:
        ladder = _load(tree / "scripts" / "api" / "ladder.py", "ladder_for_policy_replay")
        rc = ladder.main(argv)
    finally:
        os.chdir(cwd)
    if not cap.get("done"):
        print(f"試行が回らなかった（ladder の戻り値 {rc}）")
        return 3
    cap.update(tree=str(tree), experiment=a.experiment, condition=cond, ladder_argv=argv, ladder_rc=rc,
               wall_trial_s=round(time.perf_counter() - w0, 1), rec_path=str(run_dir / "run_0000.json"))
    rec = json.loads((run_dir / "run_0000.json").read_text(encoding="utf-8"))
    course = course_check(rec)
    cap["course"] = course
    # 経過が選ぶ決まりに合わなければ、書き出さずに <out>_rejected/<条件>/ に置く（回し直すかは人が決める）
    dst = cap_dir if course["ok"] else resolve(a.out + "_rejected") / cond
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(run_dir / "run_0000.json", dst / "run_0000.json")
    with open(dst / "capture.pkl", "wb") as f:
        pickle.dump(cap, f)
    print(f"生の記録 → {dst / 'capture.pkl'}（試行 {cap['wall_trial_s']} s）、経過 {course}", flush=True)
    if not course["ok"]:
        print("経過が選ぶ決まり（docs/real/h1_policy_replay.md の A1: 赤を 1 回で掴んで箱に入れ、家へ戻る）に合わない。書き出さない")
        return 4
    return cmd_process(a)


def course_check(rec: dict) -> dict:
    """試行を選ぶ決まり（docs/real/h1_policy_replay.md の A1）: 最初の手順が赤で、やり直しなしに完了と判定され、その間に赤が
    箱に入り（真値）、待機位置へ戻した。"""
    s0 = rec["steps"][0]
    sws = [s for s in ((rec.get("step_reset") or {}).get("switches") or []) if s.get("step") == 1]
    red_t = (rec.get("truth_success_t") or {}).get("red")
    reasons = []
    if s0.get("color") != "red":
        reasons.append(f"最初の手順が赤でない（{s0.get('color')}）")
    if not s0.get("judged_complete"):
        reasons.append("最初の手順が完了と判定されていない")
    if len(s0.get("attempts") or []) != 1:
        reasons.append(f"最初の手順にやり直しがある（attempt {len(s0.get('attempts') or [])} 回）")
    if red_t is None or red_t > float(s0["t_end"]) + 1e-9:
        reasons.append(f"最初の手順の間に赤が箱に入っていない（真値 {red_t}）")
    if not sws or sws[0].get("t_end") is None:
        reasons.append("最初の手順の後の家への戻しがない")
    return {"ok": not reasons, "reasons": reasons, "step0_t_end": s0.get("t_end"), "red_truth_t": red_t,
            "home_switch": sws[0] if sws else None}


def cmd_process(a) -> int:
    """capture.pkl から、切り出し・確かめ・通常の版・遅い版を作る（CPU）。"""
    out, slow_out, cap_dir = resolve(a.out), resolve(a.out + "_slow"), resolve(a.out + "_capture")
    with open(cap_dir / "capture.pkl", "rb") as f:
        cap = pickle.load(f)
    tree = pathlib.Path(cap["tree"])
    X, S = use_tree(tree)
    for p in (out, slow_out):
        if p.exists() and any(p.iterdir()):
            print(f"書き出し先 {p} が空でない。別の場所を指定する")
            return 2
    from recovla.common import code_version
    from recovla.harness.setup import nominal_setup
    from recovla.runtime.motion import Motion
    argv, rc, cond, wall_trial = cap["ladder_argv"], cap["ladder_rc"], cap["condition"], cap["wall_trial_s"]
    a.seed, a.experiment = cap["seed"], cap["experiment"]
    a.arm = argv[argv.index("--arm") + 1]
    rec_path = cap_dir / "run_0000.json"
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    course = course_check(rec)
    if not course["ok"] and not a.any_course:
        print(f"経過が選ぶ決まりに合わない（{course['reasons']}）。書き出さない")
        return 4
    rejected = []                                            # 決まりに合わず書き出さなかった回し直し（記録に残す）
    rej_dir = resolve(a.out + "_rejected")
    for p in sorted(rej_dir.glob("*/run_0000.json")) if rej_dir.is_dir() else []:
        rejected.append({"condition": p.parent.name, "reasons": course_check(json.loads(p.read_text(encoding="utf-8")))["reasons"]})
    cfg = cap["world_cfg"]
    dt = cap["dt"]
    counts = np.array(cap["counts"])
    cmds_all = np.array(cap["cmds"])
    if not (counts == 2).all() or len(cmds_all) != 2 * len(cap["meas"]):
        print(f"物理の 1 手に指令が 2 つでない所がある（{sorted(set(counts.tolist()))}、"
              f"指令 {len(cmds_all)}、手 {len(cap['meas'])}）。止める")
        return 3
    lr = cap["lim_resets"]
    first = [r for r in lr if r[1] == 0]
    if not first:
        print("試行の始めの制限層の reset が記録にない。止める")
        return 3
    start_sim = first[-1][2]                                 # 制限層が始めた姿勢（確かめの「始めは止まっている」の基準）
    start_vs_placed = float(np.abs(start_sim - cap["reset_q"]).max())

    # 2. 切り出し
    cut = find_cut(cmds_all, rec, dt)
    n_keep = cut["n_keep"]
    kept = cmds_all[:n_keep]
    mo = Motion(nominal_setup(cfg))
    fk = X.FK(mo)
    tail_info = {"appended": 0}
    if cut["rest_found"]:
        cmds = kept
    else:
        tail, tail_info = hold_tail(kept, float(cfg["actuation"]["limiter_margin"]), mo.limiter.q_min, mo.limiter.q_max)
        if not tail_info["converged"]:
            raise SystemExit(f"切った後、{HOLD_MAX_S} s 以内に止まりきらなかった")
        tail_info["appended"] = len(tail)
        cmds = np.vstack([kept, tail])
    t_cut = n_keep // 2 * dt
    hand_log = [e for e in cap["hand_log"] if float(e[0]) - cap["t0"] < t_cut - 1e-9]
    dropped = len(cap["hand_log"]) - len(hand_log)
    events = [(float(e[0]) - cap["t0"], e[1], tuple(e[2])) for e in hand_log]

    # 3. 確かめ（a）: 同じ指令を開ループで流す（立方体あり）。足した tail の応答もここから
    w1 = time.perf_counter()
    ol = open_loop_world(cfg, cap["layout"], cmds, events, cubes=True)
    m_cl, m_ol = meas_arrays(cap["meas"][:n_keep // 2]), meas_arrays(ol["meas"])
    nk = n_keep // 2
    e_cl = np.array([fk(q)[0] for q in m_cl["q"]])
    e_ol = np.array([fk(q)[0] for q in m_ol["q"][:nk]])
    repro = {"q_max_abs_diff_rad": float(np.abs(m_ol["q"][:nk] - m_cl["q"]).max()),
             "dq_max_abs_diff_rad_s": float(np.abs(m_ol["dq"][:nk] - m_cl["dq"]).max()),
             "ee_max_diff_m": float(np.linalg.norm(e_ol - e_cl, axis=1).max()),
             "width_max_abs_diff_m": float(np.abs(m_ol["w"][:nk] - m_cl["w"]).max()),
             "start_q_max_abs_diff_rad": float(np.abs(ol["start_q"] - cap["reset_q"]).max()),
             "events_sent": ol["events_sent"], "events": len(events), "steps_compared": nk}
    meas = list(cap["meas"][:nk]) + list(ol["meas"][nk:])
    wall_ol = time.perf_counter() - w1

    # 4. 列を作り、確かめ、書く（通常の速さ）
    name = f"p_{a.seed}_red"
    res = {"cmds": cmds, "meas": meas, "n_main": cut["n_main"], "t0": 0.0,
           "summary": {"kind": "policy", "t_record_start": 0.0}}     # 全部が方策の出力なので recorded = 1
    rows, cmds_r, ee_cmd = X.build_rows(res, fk)
    grows = X.gripper_rows({"hand_log": hand_log, "t0": cap["t0"]})
    start_real = np.round(X.to_real(start_sim), X.DECIMALS)
    chk = X.check_stream(cmds_r, start_real, [float(r[2]) for r in grows], (a.q1_min, a.q1_max))
    from recovla.diag import return_pose as E7
    home_err = float(np.abs(kept[-1] - np.asarray(E7.HOME_Q)).max())
    instruction = cfg["convert"]["instruction"].format(color="red")
    files = ["scripts/real/export_policy_replay.py", "scripts/real/export_replay.py", "scripts/real/export_replay_slow.py",
             "scripts/real/real_to_ds.py"]
    cfg_sha = {n: sha(tree / "configs" / f"{n}.yaml") for n in ("default", "sensor_v1", "runtime_v2", "runtime_v2_color",
                                                                 "runtime_v2_derived", "runtime_v2_judge", "v3_real", "real_v1")
               if (tree / "configs" / f"{n}.yaml").is_file()}
    trial_info = {"tree": str(tree), "experiment": a.experiment, "condition": cond, "record": cap["rec_path"],
                  "record_sha256": sha(rec_path), "start_q_limiter_vs_placed_rad": start_vs_placed, "ladder_argv": argv, "ladder_rc": rc, "text": cap["text"],
                  "steps0": cut["step0"], "switch_home": cut["switch"], "truth_success_t": rec.get("truth_success_t"),
                  "t_end": rec.get("t_end"), "replans": len(rec.get("replans") or []),
                  "early_events": len(rec.get("early_events") or []), "world": rec.get("world"),
                  "limiter_resets": [{"t": r[0], "step": r[1]} for r in lr], "wall_s": round(wall_trial, 1),
                  "selection": {"rule": "docs/real/h1_policy_replay.md A1: first step red, completed without retry, red in "
                                        "the box (truth) before the step ended, then the home move", "course": course,
                                "rejected_runs": rejected}}
    manifest = {"schema": "recovery_vla.real_replay/1", "replay_source": "policy",
                "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                "code_version": code_version.code_version(), "git_head": X.git_head(), "tree_git": git_info(tree),
                "files_sha256": {f: sha(ROOT / f) for f in files}, "config_sha256": cfg_sha,
                "q1_shift_rad": X.Q1_SHIFT, "q1_range_rad": [a.q1_min, a.q1_max], "dt_s": X.DT,
                "ee_convention": "libfranka O_T_EE (Franka Hand: flange + 0.1034 m along hand z, rotation = hand)",
                "limits": {"max_vel": X.MAX_VEL.tolist(), "max_acc": X.MAX_ACC.tolist(), "max_jerk": X.MAX_JERK.tolist(),
                           "q_min": X.Q_MIN.tolist(), "q_max": X.Q_MAX.tolist(), "source": "libfranka 0.9.2 rate_limiting.h"},
                "sim_response": "closed loop: the policy trial in v3_real (real_v1 arm servo and gripper, cubes present); "
                                "rows after the cut (appended stop) from the open-loop replay with cubes in a fresh v3_real world",
                "ds_view": None,
                "ds_note": "no _ds.csv: a policy trial is not an episode of the training data, so there are no dataset rows "
                           "to match (export_replay.py takes _ds.csv from the dataset rows)",
                "trials": []}
    ee_z = ee_cmd[:, 2]
    entry = {"name": name, "kind": "policy", "seed": int(a.seed), "color": "red", "retry": 0, "instruction": instruction,
             "layout_kind": cap["layout"]["kind"], "layout": cap["layout"], "start_pose": cap["layout"]["start"],
             "arm": a.arm, "start_q_real": start_real.tolist(),
             "start_ee_real": [round(float(v), 6) for v in fk(start_real)[0]], "sim_t0": cap["t0"],
             "t_record_start": 0.0, "stream_s": round(len(cmds_r) * X.DT, 3),
             "tail_s": round((len(cmds_r) - cut["n_main"]) * X.DT, 3),
             "cut": {"t_cut_s": round(t_cut, 4), "rest_found": cut["rest_found"], "n_keep": n_keep,
                     "i_home_motion_begin": cut["i_motion_begin"], "hold": tail_info,
                     "home_q_max_abs_diff_rad": home_err, "gripper_events_dropped": dropped},
             "check": chk, "ee_cart_cmd": X.cart_ratios(ee_cmd),
             "ee_z_cmd_range_m": [round(float(ee_z.min()), 4), round(float(ee_z.max()), 4)],
             "gripper_events": len(grows), "gripper": [r[:4] for r in grows],
             "reproduction_open_loop_with_cubes": repro, "trial": trial_info, "exported": bool(chk["ok"])}
    out.mkdir(parents=True, exist_ok=True)
    if chk["ok"]:
        entry["csv"], entry["gripper_csv"] = f"{name}.csv", f"{name}_gripper.csv"
        entry["csv_sha256"] = X.write_csv(out / entry["csv"], rows, X.COLS)
        entry["gripper_csv_sha256"] = X.write_csv(out / entry["gripper_csv"], grows, X.GRIP_COLS)
    else:
        entry["reasons"] = chk["reasons"]
    manifest["trials"].append(entry)
    manifest["n_exported"] = int(bool(chk["ok"]))
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{name}: {'書き出し' if chk['ok'] else '除外'} 流す {entry['stream_s']} s（切る所 {t_cut:.3f} s、"
          f"静止の窓 {'あり' if cut['rest_found'] else 'なし→止める分を足す'}）、比 {chk['ratio_max_all']}、"
          f"q1 {chk['q1_range_used']}、手先の高さ {entry['ee_z_cmd_range_m']}、グリッパー {len(grows)}"
          + ("" if chk["ok"] else f"、理由 {chk['reasons']}"), flush=True)
    print(f"再現（立方体あり・開ループ）: 関節 {repro['q_max_abs_diff_rad']:.3e} rad、手先 {repro['ee_max_diff_m']:.3e} m、"
          f"指 {repro['width_max_abs_diff_m']:.3e} m", flush=True)

    # 確かめに使う配列
    ol_nc = open_loop_world(cfg, cap["layout"], cmds, events, cubes=False)   # 立方体なし 1.0 倍（resim との照らし）
    m_nc = meas_arrays(ol_nc["meas"])
    np.savez_compressed(cap_dir / "capture.npz", cmds_all=cmds_all, cmds=cmds, start_q=start_sim, reset_q=cap["reset_q"],
                        cl_t=meas_arrays(cap["meas"])["t"], cl_q=meas_arrays(cap["meas"])["q"],
                        cl_dq=meas_arrays(cap["meas"])["dq"], cl_w=meas_arrays(cap["meas"])["w"],
                        ol_q=m_ol["q"], ol_w=m_ol["w"], nc_q=m_nc["q"], nc_w=m_nc["w"])
    (cap_dir / "capture.json").write_text(json.dumps(
        {"hand_log_all": [[float(e[0]), e[1], list(e[2])] for e in cap["hand_log"]], "layout": cap["layout"],
         "cut": entry["cut"], "reproduction": repro, "trial": trial_info, "world_cfg": cfg},
        ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    if not chk["ok"]:
        print("確かめが通らないので、遅い版は作らない")
        return 1

    # 5. 遅い版（export_replay_slow の引き延ばし。応答は v3_real の評価の世界・立方体なし）
    from real_to_ds import read_csv
    r_n = read_csv(out / entry["csv"])                      # 書き出した桁の指令から引き延ばす
    q_cmd = np.stack([np.asarray(r_n[f"q_cmd{j}"], float) for j in range(7)], axis=1)
    q_slow = S.stretch(q_cmd, start_real, S.SCALE)
    ev2 = S.events_from_csv(out / entry["gripper_csv"], S.SCALE)
    o = open_loop_world(cfg, cap["layout"], S.to_sim(q_slow), ev2, cubes=False)
    n_main2 = int(round((len(q_cmd) - round(entry["tail_s"] / X.DT)) * S.SCALE))
    res2 = {"cmds": S.to_sim(q_slow), "meas": o["meas"], "n_main": n_main2, "t0": o["t0"],
            "summary": {"kind": "policy", "t_record_start": 0.0}}
    rows2, cmds_r2, ee_cmd2 = X.build_rows(res2, fk)
    grows2 = [[f"{e[0]:.3f}", e[1]] + ([X.fmt(e[2][0]), X.fmt(e[2][1]), "", "", ""] if e[1] == "move"
                                         else [X.fmt(v) for v in e[2]]) for e in ev2]
    chk2 = X.check_stream(cmds_r2, start_real, [float(g[2]) for g in grows2], (a.q1_min, a.q1_max))
    man2 = {k: manifest[k] for k in ("schema", "replay_source", "code_version", "git_head", "tree_git", "files_sha256",
                                      "config_sha256", "q1_shift_rad", "q1_range_rad", "dt_s", "ee_convention", "limits")}
    man2.update({"variant": "slow_0.5x", "time_scale": S.SCALE,
                 "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                 "source": {"dir": str(out), "manifest_git_head": manifest["git_head"]},
                 "sim_response": "open loop: the scaled q_cmd fed 2 per 500 Hz physics step into a fresh v3_real eval world "
                                 "(real_v1 arm servo and gripper, harness/real_matched.make_eval_world); gripper commands at the "
                                 "scaled times; no cubes (contacts off); no re-decision by the policy",
                 "ds_view": None, "trials": []})
    e2 = {k: entry[k] for k in ("name", "kind", "seed", "color", "retry", "instruction", "layout_kind", "layout", "start_pose",
                                "arm", "start_q_real", "start_ee_real")}
    e2.update({"source_csv_sha256": entry["csv_sha256"], "reproduction_1x": repro, "check": chk2,
               "stream_s": round(len(cmds_r2) * X.DT, 3), "tail_s": round((len(cmds_r2) - n_main2) * X.DT, 3),
               "t_record_start": 0.0, "ee_cart_cmd": X.cart_ratios(ee_cmd2),
               "ee_z_cmd_range_m": [round(float(ee_cmd2[:, 2].min()), 4), round(float(ee_cmd2[:, 2].max()), 4)],
               "gripper_events": len(grows2), "gripper_events_sent": o["events_sent"], "exported": bool(chk2["ok"])})
    slow_out.mkdir(parents=True, exist_ok=True)
    if chk2["ok"]:
        e2["csv"], e2["gripper_csv"] = f"{name}.csv", f"{name}_gripper.csv"
        e2["csv_sha256"] = X.write_csv(slow_out / e2["csv"], rows2, X.COLS)
        e2["gripper_csv_sha256"] = X.write_csv(slow_out / e2["gripper_csv"], grows2, X.GRIP_COLS)
    else:
        e2["reasons"] = chk2["reasons"]
    man2["trials"].append(e2)
    man2["n_exported"] = int(bool(chk2["ok"]))
    (slow_out / "manifest.json").write_text(json.dumps(man2, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"遅い版 {name}: {'書き出し' if chk2['ok'] else '除外'} 流す {e2['stream_s']} s、比 {chk2['ratio_max_all']}"
          + ("" if chk2["ok"] else f"、理由 {chk2['reasons']}"), flush=True)
    print(f"試行 {wall_trial:.0f} s、開ループ {wall_ol:.0f} s → {out}、{slow_out}、{cap_dir}")
    return 0 if chk2["ok"] else 1


# ------------------------------------------------------------------ finish
def ee_of(r) -> np.ndarray:
    return np.stack([np.asarray(r[f"ee_sim_{c}"], float) for c in "xyz"], axis=1)


def q_of(r, key="q_sim") -> np.ndarray:
    return np.stack([np.asarray(r[f"{key}{j}"], float) for j in range(7)], axis=1)


def diff_stats(e_a, e_b, q_a, q_b) -> dict:
    d = np.linalg.norm(e_a - e_b, axis=1)
    return {"ee_median_m": float(np.median(d)), "ee_max_m": float(d.max()), "ee_p95_m": float(np.percentile(d, 95)),
            "t_ee_max_s": round(float((int(np.argmax(d)) + 1) * 1e-3), 3), "q_max_abs_rad": float(np.abs(q_a - q_b).max())}


def cmd_finish(a) -> int:
    out, slow_out, cap_dir = resolve(a.out), resolve(a.out + "_slow"), resolve(a.out + "_capture")
    tree = pathlib.Path(json.loads((cap_dir / "capture.json").read_text(encoding="utf-8"))["trial"]["tree"])
    X, S = use_tree(tree)
    import build_replay_zip as B
    import resim_replay as R
    from real_to_ds import read_csv
    report = {"created": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}
    for src in (out, slow_out):
        dst = src.with_name(src.name + "_fit")
        rc = R.main(["--src", str(src), "--out", str(dst), "--hand"])
        if rc != 0:
            print(f"resim_replay が止まった（{src}）")
            return rc
        # resim の manifest は A8 の欄だけを写すので、実機の前に見る欄（手先の高さ・確かめ・切り出し）を元の manifest から足す
        sm = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
        dm = json.loads((dst / "manifest.json").read_text(encoding="utf-8"))
        for te in dm["trials"]:
            st = [x for x in sm["trials"] if x["name"] == te["name"]][0]
            for k in ("ee_z_cmd_range_m", "check", "ee_cart_cmd", "gripper_events", "cut", "arm", "layout"):
                if k in st:
                    te[k] = st[k]
        (dst / "manifest.json").write_text(json.dumps(dm, ensure_ascii=False, indent=2), encoding="utf-8")
        rc = B.main(["--src", str(dst), "--zip", str(dst.with_suffix(".zip"))])
        if rc != 0:
            print(f"build_replay_zip が止まった（{dst}）")
            return rc
        report[f"zip_{src.name}"] = {"zip": str(dst.with_suffix(".zip")), "sha256": sha(dst.with_suffix(".zip"))}
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    t = man["trials"][0]
    r_cl = read_csv(out / t["csv"])                          # 閉ループ（立方体あり）
    r_fit = read_csv(out.with_name(out.name + "_fit") / t["csv"])     # 実機に合わせた腕とハンド・立方体なし（resim）
    r_slow = read_csv(slow_out / t["csv"])                   # 遅い版・v3_real の評価の世界・立方体なし
    r_sfit = read_csv(slow_out.with_name(slow_out.name + "_fit") / t["csv"])
    # 確かめ（b）: 立方体が腕の動きをどれだけ変えたか（同じ指令、閉ループ 対 resim の立方体なし）
    report["b_cubes_effect_closed_loop_vs_fit"] = diff_stats(ee_of(r_cl), ee_of(r_fit), q_of(r_cl), q_of(r_fit))
    # 参考: resim（合わせる前の生成の世界＋real_v1）と v3_real の評価の世界（立方体なし）が同じ応答か
    cap = np.load(cap_dir / "capture.npz")
    idx = np.arange(1, len(r_fit["t"]), 2)
    q_nc = X.to_real(cap["nc_q"])
    report["resim_world_vs_v3_real_world_no_cubes_1x"] = {
        "q_max_abs_rad": float(np.abs(q_of(r_fit)[idx] - q_nc[:len(idx)]).max()),
        "width_max_abs_m": float(np.abs(np.asarray(r_fit["width_sim"], float)[idx] - cap["nc_w"][:len(idx)]).max())}
    report["resim_world_vs_v3_real_world_no_cubes_slow"] = diff_stats(ee_of(r_slow), ee_of(r_sfit), q_of(r_slow),
                                                                     q_of(r_sfit))
    # 指令と応答の差（参考: 手先が指令からどれだけ遅延して離れるか）
    e_cmd = np.stack([np.asarray(r_fit[f"ee_cmd_{c}"], float) for c in "xyz"], axis=1)
    report["fit_vs_cmd_ee"] = {"median_m": float(np.median(np.linalg.norm(ee_of(r_fit) - e_cmd, axis=1))),
                               "max_m": float(np.linalg.norm(ee_of(r_fit) - e_cmd, axis=1).max())}
    report["check_normal"] = {k: t["check"][k] for k in ("ok", "ratio_max_all", "frac_over_90pct", "q1_range_used")}
    report["stream_s"], report["ee_z_cmd_range_m"], report["gripper"] = t["stream_s"], t["ee_z_cmd_range_m"], t["gripper"]
    sm = json.loads((slow_out / "manifest.json").read_text(encoding="utf-8"))["trials"][0]
    report["check_slow"] = {k: sm["check"][k] for k in ("ok", "ratio_max_all", "frac_over_90pct", "q1_range_used")}
    report["slow_stream_s"], report["slow_ee_cart_cmd"] = sm["stream_s"], sm["ee_cart_cmd"]
    report["reproduction_open_loop_with_cubes"] = t["reproduction_open_loop_with_cubes"]
    (cap_dir / "verify.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:                                    # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("capture", "process", "finish", "all"))
    ap.add_argument("--tree", default=str(ROOT), help="試行を回す木（recovla・ladder.py・保存点・LLM のキャッシュ）")
    ap.add_argument("--seed", type=int, default=171318)
    ap.add_argument("--arm", default="L2v2")
    ap.add_argument("--experiment", default=EXP)
    ap.add_argument("--condition", default=None, help="既定は h1_s<種>_<腕>_<時刻>（96 は同じ条件の続きから回すので毎回新しく）")
    ap.add_argument("--out", default=OUT, help="通常の版。遅い版は <out>_slow、生の記録は <out>_capture、応答の付け直しは _fit")
    ap.add_argument("--any-course", action="store_true", help="process だけ: 選ぶ決まりに合わない経過でも作る（道具の確かめ用。実機には使わない）")
    ap.add_argument("--q1-min", type=float, default=-2.80)
    ap.add_argument("--q1-max", type=float, default=-1.66)
    a = ap.parse_args(argv)
    if a.cmd in ("capture", "process", "all"):
        rc = cmd_capture(a) if a.cmd != "process" else cmd_process(a)
        if rc != 0 or a.cmd != "all":
            return rc
    return cmd_finish(a)


if __name__ == "__main__":
    sys.exit(main())
