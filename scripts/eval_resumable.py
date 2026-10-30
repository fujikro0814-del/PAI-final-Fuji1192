"""評価の試行を、止まっても続きから回せる形で回す（1 つのサブタスクの run と、3 個の連続タスクの task）。

scripts/eval_trials.py の run・task と同じ試行を、同じ記録の形で回す（eval_trials を読み込み、試行の本体を写して使う。eval_trials は書き換えず、
SHA-256 を固定して照らす）。scripts/eval_replan_task.py・scripts/real/eval_real_matched.py などが、この包みを読み込み、試行の回し方
（Engine など）を差し替えて使う。

    .venv\\Scripts\\python.exe scripts\\eval_resumable.py run --experiment EXP --condition COND --model R1v3 \\
        --trials natural:44400:1 --mode naive --exec-interval 6 --no-safety
    .venv\\Scripts\\python.exe scripts\\eval_resumable.py task --experiment EXP --condition COND --model R1v3 --trials 44400:2 --exec-interval 6
    .venv\\Scripts\\python.exe scripts\\eval_resumable.py run ... --dry-run          # 何を飛ばし何を回すかを見るだけ（何も書かない）
    .venv\\Scripts\\python.exe scripts\\eval_resumable.py score --experiment EXP --condition COND [--at 30,45,60]   # 読むだけ

eval_trials に足した引数:
  --max-new N（新しい試行を N 本回したら止まる）、--stop-file、--progress-file、--min-free-gb・--min-commit-free-gb・
  --mem-timeout-min（空きメモリが足りなければ待つ）、--quiet-window HH:MM-HH:MM（その時刻の窓では新しい試行を始めない。既定は無し）、
  --accept-spec-change・--accept-env-change・--allow-82-change（食い違いを承知で進める）、--reuse-world（世界を使い回す）、
  task の --planner s4|legacy（手順の分解の LLM。既定 s4 は planner/decompose_haiku.py、claude-haiku-5-5）、
  run の --reflex drop_only|full と --reflex-set KEY=VALUE（握り損ねの反射。recovla.runtime.reflex。既定は切）、
  run の --induce P2S（引き抜きの落下。recovla.eval.induce_slip）。
制限時間: run は --time-limit-s 60、task は 1 手順 --step-timeout-s 30・やり直し 1 回・全体 --task-time-limit-s 200（既定）。
  configs を読んだ辞書の写しに重ねて渡し、使った値は試行の json の "time_limits"、run.json、resume_spec.json に残す。
  既にある完全な記録と制限時間が違えば、混ぜずに止める。
環境: 回の始めにドライバ・torch と CUDA・OS・git の HEAD を読んで記録に残す。完全な記録か前の回と違えば止める
  （--accept-env-change のときだけ進め、報告では環境ごとに分ける）。
score: 同じ記録から 30・45・60 秒の採点を出す（定義は score_condition）。

書くもの: outputs\\v2eval\\<実験>\\<条件>\\ に、eval_trials と同じ形の試行の記録（run は trial_NNNN.json/.npz・runtime_NNNN.json、
  task は run_NNNN.json/.npz・run_NNNN_runtime.json）、全部そろったときだけ run.json と G_AUDIT.json（監査の結果）、
  progress.json（進み具合。ops_tools.py wait が読む）、resume_spec.json（引数の控え。再開のとき食い違えば止める）、
  resume_log.json（回ごとの環境・本数・時間）、_incomplete_<時刻>\\（壊れた・欠けた記録の退避）。

動き:
  1. 3 つのファイルが全部あり、読めて、並びと合う試行は「完全」として飛ばす。欠けた・壊れた試行は退避して同じ番号で回し直す。
  2. 新しい試行の前に、止める合図のファイル（<条件>\\STOP と outputs\\s4\\STOP）、時刻の窓、空きメモリを確かめる。
  3. 試行ごとに世界（WorldRig・SensorSuite）を作り直す（方策は使い回す）。1 つの世界で続けて回すと、2 本目以降の開始の
     指の開きが 1 本目と最大 0.94 mm 違い、止めて再開した試行と一致しなくなるため。
  再開した試行は、GPU の推論の揺らぎで成否・行動が元の実行と一致するとは限らない。
終了コード: 0 全部そろった、1 途中で止まった（合図・--max-new・メモリ待ちの時間切れ・Ctrl+C）、2 エラー、3 引数・前提の食い違い。
"""
import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import sys
import threading
import time
import traceback
import zipfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT82 = ROOT / "scripts" / "eval_trials.py"
SHA82 = "26fd3e7ad5d4dac338409242cea83c88d3c6fcd55e5c793930107fced58fb659"      # タグ v3-s3-freeze の 82_v2_eval.py を、名前の付け直しに合わせて書き換えた eval_trials.py（読み込むスクリプトのパスと docstring だけが違う）
S4 = ROOT / "outputs" / "s4"
EXIT = {"done": 0, "stopped": 1, "interrupted": 1, "memory_timeout": 1, "error": 2}


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_ops():
    return _load(ROOT / "scripts" / "ops_tools.py", "s4_ops")


def load_82(allow_change: bool):
    sha = sha256_file(SCRIPT82)
    if sha != SHA82 and not allow_change:
        raise SystemExit(f"scripts/eval_trials.py のハッシュが固定した値と違う（{sha}）。このスクリプトは 82 の本体を写しているので、"
                         f"82 が変わったなら写しを見直してから SHA82 を直す（見直さずに進めるなら --allow-82-change）")
    return _load(SCRIPT82, "v82_eval")


# ---------------------------------------------------------------- 進捗
def _now_s() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _write_atomic(path: pathlib.Path, text: str, tries: int = 8) -> bool:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    for k in range(tries):
        try:
            os.replace(tmp, path)
            return True
        except PermissionError:                  # 読んでいる最中の相手がいる（Windows）。少し待つ
            time.sleep(0.2 * (k + 1))
    return False


class Progress:
    """progress.json を原子的に書く。心拍の別スレッドが更新時刻・空きメモリ・自分のメモリの最大も書く。"""

    def __init__(self, path: pathlib.Path, ops, base: dict):
        self.path, self.ops = path, ops
        self.lock = threading.RLock()
        self.d = dict(base)
        self.stop_evt = threading.Event()
        self.min_free = None
        self.min_commit_free = None
        self.peak = {"wset_gb": 0.0, "pagefile_gb": 0.0}
        self.t0 = time.time()
        try:
            import psutil
            self.proc = psutil.Process()
            self.d["proc_create_time"] = self.proc.create_time()
        except Exception:                        # noqa: BLE001
            self.proc = None

    def _sample(self) -> None:
        m = self.ops.memory_gb()
        self.d["free_phys_gb"], self.d["free_commit_gb"] = m["phys_free_gb"], m["commit_free_gb"]
        self.min_free = m["phys_free_gb"] if self.min_free is None else min(self.min_free, m["phys_free_gb"])
        self.min_commit_free = m["commit_free_gb"] if self.min_commit_free is None else min(self.min_commit_free, m["commit_free_gb"])
        if self.proc is not None:
            try:
                mi = self.proc.memory_info()
                g = 1024 ** 3
                self.peak["wset_gb"] = max(self.peak["wset_gb"], getattr(mi, "peak_wset", mi.rss) / g)
                self.peak["pagefile_gb"] = max(self.peak["pagefile_gb"], getattr(mi, "peak_pagefile", mi.vms) / g)
            except Exception:                    # noqa: BLE001
                pass

    def update(self, **kw) -> None:
        with self.lock:
            self.d.update(kw)
            self._sample()
            self.d["updated"] = _now_s()
            self.d["elapsed_min"] = round((time.time() - self.t0) / 60, 2)
            _write_atomic(self.path, json.dumps(self.d, ensure_ascii=False, indent=1, default=str))

    def start_heartbeat(self, every: float = 30.0) -> None:
        def loop():
            while not self.stop_evt.wait(every):
                try:
                    self.update()
                except Exception:                # noqa: BLE001
                    pass
        self.hb = threading.Thread(target=loop, daemon=True)
        self.hb.start()

    def stop_heartbeat(self) -> None:
        self.stop_evt.set()


# ---------------------------------------------------------------- 試行の完全さ
def trial_paths(kind: str, out: pathlib.Path, i: int) -> dict:
    if kind == "run":
        return {"json": out / f"trial_{i:04d}.json", "npz": out / f"trial_{i:04d}.npz", "runtime": out / f"runtime_{i:04d}.json"}
    return {"json": out / f"run_{i:04d}.json", "npz": out / f"run_{i:04d}.npz", "runtime": out / f"run_{i:04d}_runtime.json"}


def check_complete(kind: str, out: pathlib.Path, i: int, expect: dict) -> tuple:
    """(完全か, 理由, meta)。expect = run なら {"seed","target"}、task なら {"seed"}。"""
    ps = trial_paths(kind, out, i)
    missing = [k for k, p in ps.items() if not p.is_file()]
    if missing:
        return False, "missing:" + ",".join(missing), None
    empty = [k for k, p in ps.items() if p.stat().st_size == 0]
    if empty:
        return False, "empty:" + ",".join(empty), None
    try:
        meta = json.loads(ps["json"].read_text(encoding="utf-8"))
        rt = json.loads(ps["runtime"].read_text(encoding="utf-8"))
    except Exception as e:                       # noqa: BLE001
        return False, f"json:{type(e).__name__}", None
    if not isinstance(meta, dict) or not isinstance(rt, (dict, list)):
        return False, "json:shape", None
    need = ("success", "seed", "target", "trial", "audit") if kind == "run" else ("all_three_in_box", "audit", "run")
    absent = [k for k in need if k not in meta]
    if absent:
        return False, "meta_keys:" + ",".join(absent), None
    if kind == "run" and (meta["trial"] != i or meta["seed"] != expect["seed"] or meta["target"] != expect["target"]):
        return False, f"mismatch:trial={meta['trial']} seed={meta['seed']} target={meta['target']}", None
    if kind == "task" and (meta["run"] != i or meta.get("seed", expect["seed"]) != expect["seed"]):
        return False, f"mismatch:run={meta['run']}", None
    try:
        if not zipfile.is_zipfile(ps["npz"]):
            return False, "npz:not_zip", None
        with zipfile.ZipFile(ps["npz"]) as z:
            bad = z.testzip()
            if bad is not None or not z.namelist():
                return False, f"npz:crc:{bad}", None
    except Exception as e:                       # noqa: BLE001
        return False, f"npz:{type(e).__name__}", None
    return True, "ok", meta


def quarantine(kind: str, out: pathlib.Path, i: int, stamp: str) -> list:
    moved = []
    for p in trial_paths(kind, out, i).values():
        if p.exists():
            d = out / f"_incomplete_{stamp}"
            d.mkdir(exist_ok=True)
            shutil.move(str(p), str(d / p.name))
            moved.append(p.name)
    return moved


def _json_text(obj, v82) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=1, default=v82._json_default)


def write_trial_files(kind, out, i, meta, arrays, rlog, v82) -> None:
    """eval_trials と同じ順（npz → json → runtime）・同じ書式。途中で止まっても半端なファイルが残らないよう、一時ファイルから置き換える。"""
    ps = trial_paths(kind, out, i)
    tmp = ps["npz"].with_name(ps["npz"].name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, ps["npz"])
    _write_atomic(ps["json"], _json_text(meta, v82))
    _write_atomic(ps["runtime"], json.dumps(rlog, ensure_ascii=False, default=v82._json_default))


# ---------------------------------------------------------------- 始める前の確認
def stop_requested(paths) -> str:
    for p in paths:
        if p.exists():
            return f"stop_file:{p}"
    return ""


def sleep_checking(sec: float, paths) -> str:
    end = time.monotonic() + sec
    while time.monotonic() < end:
        r = stop_requested(paths)
        if r:
            return r
        time.sleep(min(1.0, max(0.0, end - time.monotonic())))
    return ""


def quiet_window_of(a):
    """効いている窓（HH:MM-HH:MM）か None。--quiet-window を指定し、--ignore-quiet を付けていないときだけ効く。"""
    w = (getattr(a, "quiet_window", None) or "").strip()
    return None if (not w or getattr(a, "ignore_quiet", False)) else w


def gate_before_trial(a, ops, prog, stop_paths) -> str:
    """新しい試行を始めてよくなるまで待つ。返り値: "" なら始める、それ以外は止まる理由。"""
    mem_t0 = None
    while True:
        r = stop_requested(stop_paths)
        if r:
            return r
        if quiet_window_of(a):                   # 窓は --quiet-window を指定したときだけ（既定は無し）
            ws = ops.window_state(ops._now(), a.quiet_window)
            if ws["in_window"]:
                prog.update(status="quiet_wait", wait_note=f"窓 {a.quiet_window} の終わりまであと {ws['minutes_to_window_end']} 分")
                r = sleep_checking(min(20.0, ws["minutes_to_window_end"] * 60 + 1), stop_paths)
                if r:
                    return r
                continue
        m = ops.memory_gb()
        if m["phys_free_gb"] < a.min_free_gb or m["commit_free_gb"] < a.min_commit_free_gb:
            mem_t0 = mem_t0 or time.time()
            if a.mem_timeout_min and (time.time() - mem_t0) / 60 > a.mem_timeout_min:
                return "memory_timeout"
            prog.update(status="memory_wait", wait_note=f"空き 物理 {m['phys_free_gb']} GB（{a.min_free_gb} 未満で待つ）・"
                                                        f"コミット {m['commit_free_gb']} GB（{a.min_commit_free_gb} 未満で待つ）")
            r = sleep_checking(30.0, stop_paths)
            if r:
                return r
            continue
        prog.update(status="running", wait_note=None)
        return ""


# ---------------------------------------------------------------- 環境の記録
# 回（session）の始めに読み、resume_log.json の session・progress.json・run.json・各試行の meta["env"] に残す。
# 完全な記録や前の回と ENV_STOP_KEYS が違えば止める（終了コード 3）。--accept-env-change のときだけ進め、session に env_segment を書く。
ENV_STOP_KEYS = ("driver", "torch", "torch_cuda", "os_build")
ENV_BRIEF_KEYS = ENV_STOP_KEYS + ("git_head",)


def parse_nvidia_smi_driver(text: str):
    """`nvidia-smi --query-gpu=driver_version --format=csv,noheader` の出力から版（例 "610.88"）。GPU が複数なら最初の行。
    読めなければ None。"""
    import re
    for ln in (text or "").splitlines():
        s = ln.strip()
        if re.fullmatch(r"\d+(?:\.\d+)+", s):
            return s
    return None


def read_driver():
    import subprocess
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, timeout=60)
        return parse_nvidia_smi_driver(r.stdout.decode("utf-8", "replace"))
    except Exception:                            # noqa: BLE001
        return None


def read_env(ops) -> dict:
    """今の環境（ドライバ・torch と CUDA・OS・git の HEAD）。torch は版を読むだけ（CUDA の初期化はしない）。"""
    import platform
    try:
        import torch
        tv, tc = torch.__version__, torch.version.cuda
    except Exception as e:                       # noqa: BLE001
        tv, tc = f"error: {type(e).__name__}", None
    try:
        build = ops._reg_value(r"SOFTWARE\Microsoft\Windows NT\CurrentVersion", "CurrentBuildNumber")
        ubr = ops._reg_value(r"SOFTWARE\Microsoft\Windows NT\CurrentVersion", "UBR")
        os_build = f"{build}.{ubr}" if build is not None else platform.version()
    except Exception:                            # noqa: BLE001
        os_build = platform.version()
    status = ops._git("status", "--short")
    return {"read": _now_s(), "driver": read_driver(), "torch": tv, "torch_cuda": tc, "os_build": os_build,
            "os": platform.platform(), "python": platform.python_version(), "git_head": ops._git("rev-parse", "HEAD"),
            "git_dirty_n": None if status.startswith("error") else sum(1 for ln in status.splitlines() if ln.strip()),
            "source": "nvidia-smi --query-gpu=driver_version、torch.__version__・torch.version.cuda、"
                      "HKLM CurrentBuildNumber.UBR、git rev-parse HEAD（96_s4_resume.read_env）"}


def env_brief(env: dict) -> dict:
    return {k: (env or {}).get(k) for k in ENV_BRIEF_KEYS}


def _env_diff(old: dict, cur: dict) -> dict:
    return {k: ((old or {}).get(k), cur.get(k)) for k in ENV_STOP_KEYS if (old or {}).get(k) != cur.get(k)}


def env_conflicts(cur: dict, states: list, log: dict) -> dict:
    """今の環境と、完全な記録の meta["env"]・前の回（resume_log の session["env"]）を照らす。
    返り値 {"records": {i: 差}, "unknown_records": [i...], "previous_session": 差 or {}}。空なら食い違いなし。
    - env の無い完全な記録は「環境の分からない記録」として食い違いに数える（記録のない実行は報告に使わない）。
    - 前に --accept-env-change で今の環境へ切り替えた回があれば、それより前の環境の記録は分けてあるので食い違いに数えない。"""
    sessions = [s for s in (log or {}).get("sessions", []) if s.get("env")]
    cur_key = json.dumps({k: cur.get(k) for k in ENV_STOP_KEYS}, sort_keys=True)
    accepted = any(s.get("env_segment") and json.dumps({k: s["env"].get(k) for k in ENV_STOP_KEYS}, sort_keys=True) == cur_key
                   for s in sessions)
    rec, unknown = {}, []
    if not accepted:
        for i, st in enumerate(states):
            if not st[0]:
                continue
            e = (st[2] or {}).get("env")
            if not e:
                unknown.append(i)
                continue
            d = _env_diff(e, cur)
            if d:
                rec[i] = d
    prev = _env_diff(sessions[-1]["env"], cur) if sessions else {}
    out = {}
    if rec:
        out["records"] = rec
    if unknown:
        out["unknown_records"] = unknown
    if prev:
        out["previous_session"] = prev
    return out


def env_segments(metas: list) -> list:
    """完全な記録を環境（ENV_STOP_KEYS）ごとに分けた並び。[{"env": {...}, "trials": [i...]}]（環境が 1 つなら長さ 1）。"""
    segs = []
    for i, m in metas:
        e = (m or {}).get("env")
        key = {k: (e or {}).get(k) for k in ENV_STOP_KEYS} if e else None
        if segs and segs[-1]["env"] == key:
            segs[-1]["trials"].append(i)
            continue
        hit = next((s for s in segs if s["env"] == key), None)
        if hit is not None:
            hit["trials"].append(i)
        else:
            segs.append({"env": key, "trials": [i]})
    return segs


# ---------------------------------------------------------------- 引数の控え
SPEC_KEYS =("cmd", "experiment", "condition", "model", "trials", "mode", "exec_interval", "induce", "no_limiter", "no_safety",
             "diag_ik", "diag_no_gravcomp", "ablate", "grip_gate", "xcmd_leash", "cart_margin", "text",
             "time_limit_s", "step_timeout_s", "task_time_limit_s", "world_per_trial", "planner")


# ---------------------------------------------------------------- 制限時間（run は 60 s、task は 1 手順 30 s・全体 200 s）
TASK_STEP_TIMEOUT_DEFAULT = 30.0         # task の 1 手順の持ち時間（configs の planner.step_timeout_s と同じ値）
TASK_TIME_LIMIT_DEFAULT = 200.0          # task の試行全体の打ち切り（harness/task_loop.py の既定と同じ）


def default_task_time_limit(step_timeout_s: float = TASK_STEP_TIMEOUT_DEFAULT, retry: int = 1) -> float:
    """task の試行全体の打ち切りの既定 200 s（引数は互換のため残す。値には使わない）。"""
    return TASK_TIME_LIMIT_DEFAULT


# ---------------------------------------------------------------- 手順の分解の LLM（task）
PLANNERS = {
    "s4": {"module": "recovla.planner.decompose_haiku", "model": "claude-haiku-5-5", "variant": "s4_haiku55",
           "note": "段階 4 の計画役（温度は送らない、thinking disabled、鍵にモデル名を入れたキャッシュ）"},
    "legacy": {"module": "recovla.planner.decompose", "model": "configs planner.model（claude-haiku-4-5）", "variant": "legacy",
               "note": "凍結した段階 3 の計画役（温度 0）"},
}


def planner_of(name: str):
    """(TaskRuntime に渡す decompose の関数, 記録に残す辞書)。eval_trials は書き換えず、写した make の中でこの関数を渡す。"""
    import importlib
    if name not in PLANNERS:
        raise SystemExit(f"--planner {name}: {sorted(PLANNERS)} のどれか")
    info = dict(PLANNERS[name], name=name)
    return importlib.import_module(info["module"]).decompose, info


def limits_of(a, base_cfg) -> dict:
    """今回の試行に使う制限時間。run は試行の制限時間、task は 1 手順の持ち時間・やり直しの回数・全体の打ち切り。"""
    if a.cmd == "run":
        return {"time_limit_s": float(a.time_limit_s), "source": "96_s4_resume --time-limit-s"}
    retry = int(base_cfg["planner"]["retry"])
    total = float(a.task_time_limit_s) if a.task_time_limit_s else default_task_time_limit(float(a.step_timeout_s), retry)
    return {"step_timeout_s": float(a.step_timeout_s), "retry": retry, "task_time_limit_s": total,
            "source": "96_s4_resume --step-timeout-s/--task-time-limit-s"}


def overlay_limits(cfg: dict, lim: dict) -> dict:
    """configs を読んだ辞書の写しに制限時間を重ねる（ファイルは書き換えない）。"""
    import copy
    c = copy.deepcopy(cfg)
    if "time_limit_s" in lim:
        c["eval"]["time_limit_s"] = lim["time_limit_s"]
    if "step_timeout_s" in lim:
        c["planner"]["step_timeout_s"] = lim["step_timeout_s"]
    return c


def record_limits(kind: str, meta: dict, base_cfg) -> dict:
    """既にある記録が使った制限時間。eval_trials の記録には "time_limits" がないので、run は meta の time_limit_s、
    task は configs の値（eval_trials の task は planner.step_timeout_s と task_loop の既定 200 s）とみなす。"""
    if meta.get("time_limits"):
        return meta["time_limits"]
    if kind == "run":
        return {"time_limit_s": float(meta.get("time_limit_s"))}
    return {"step_timeout_s": float(base_cfg["planner"]["step_timeout_s"]), "retry": int(base_cfg["planner"]["retry"]),
            "task_time_limit_s": 200.0}


def limits_conflict(kind: str, old: dict, new: dict) -> dict:
    keys = ("time_limit_s",) if kind == "run" else ("step_timeout_s", "retry", "task_time_limit_s")
    return {k: (old.get(k), new.get(k)) for k in keys
            if old.get(k) is None or abs(float(old.get(k)) - float(new.get(k))) > 1e-9}


def reflex_config(a):
    """--reflex NAME のときだけ、握り損ねの反射の設定 {"preset": NAME, "params": ReflexParams を辞書にしたもの}。切なら None。
    NAME は recovla.runtime.reflex.PRESETS（drop_only = 落下だけ、full = 落下＋空掴み）。--reflex-set KEY=VALUE で
    さらに上書きする（値は JSON として読み、読めなければ文字列）。"""
    name = getattr(a, "reflex", None)
    if not name:
        if getattr(a, "reflex_set", None):
            raise SystemExit("--reflex-set は --reflex NAME と一緒に使う")
        return None
    from recovla.runtime.reflex import preset_params
    d = {}
    for kv in getattr(a, "reflex_set", None) or []:
        k, sep, v = kv.partition("=")
        if not sep:
            raise SystemExit(f"--reflex-set {kv!r}: KEY=VALUE の形で書く")
        try:
            d[k.strip()] = json.loads(v)
        except ValueError:
            d[k.strip()] = v
    try:
        return {"preset": name, "params": preset_params(name, d).to_dict(), "overrides": d}
    except (TypeError, ValueError) as e:
        raise SystemExit(f"--reflex-set: {e}")


def make_spec(a) -> dict:
    s = {k: getattr(a, k, None) for k in SPEC_KEYS}
    s["script82_sha256"] = SHA82
    rx = reflex_config(a)
    if rx is not None:                                               # 反射が切のときは控えの形を変えない
        s["reflex"] = rx
    return s


def check_spec(a, out: pathlib.Path, spec: dict) -> None:
    p = out / "resume_spec.json"
    if p.is_file():
        old = json.loads(p.read_text(encoding="utf-8"))
        # 制限時間を足す前の控え（キーが無い）は、そのキーを比べない（記録の制限時間は cmd_main が試行の json で照らす）
        diff = {k: (old.get(k), spec.get(k)) for k in spec if k in old and old.get(k) != spec.get(k)}
        if old.get("reflex") != spec.get("reflex"):                  # 反射の入切・引数は、控えに無くても食い違いとして見る
            diff["reflex"] = (old.get("reflex"), spec.get("reflex"))
        if diff and not a.accept_spec_change:
            raise SystemExit(f"{p} の引数の控えと今回の引数が違う（前, 今）: {diff}\n"
                             f"同じ条件の続きなら引数を前に合わせる。意図して変えるなら --accept-spec-change（別の条件名にする方が安全）")
        if diff:
            p = out / f"resume_spec_{time.strftime('%Y%m%d-%H%M%S')}.json"
            print(f"[resume] 引数の控えが違うが --accept-spec-change で進める: {diff}", flush=True)
    elif (out / "run.json").is_file():
        old = json.loads((out / "run.json").read_text(encoding="utf-8"))
        pairs = {"experiment": "experiment", "condition": "condition", "model": "model", "mode": "mode",
                 "exec_interval": "exec_interval", "trials": "trials", "induce": "induce"}
        diff = {k: (old.get(r), spec.get(k)) for k, r in pairs.items() if r in old and old.get(r) != spec.get(k)
                and not (k == "exec_interval" and spec.get(k) is None)}
        if old.get("reflex") != spec.get("reflex"):
            diff["reflex"] = (old.get("reflex"), spec.get("reflex"))
        if diff and not a.accept_spec_change:
            raise SystemExit(f"{out}/run.json と今回の引数が違う（前, 今）: {diff}")
    if not p.exists():
        _write_atomic(p, json.dumps(spec, ensure_ascii=False, indent=1))


# ---------------------------------------------------------------- 試行の本体（eval_trials の cmd_run・cmd_task を写したもの）
class Engine:
    """eval_trials の cmd_run の準備（世界・センサ・make_runtime）を、最初の新しい試行の直前まで遅らせて作る。"""

    def __init__(self, a, v82, cfg, lim, env=None):
        self.a, self.v82, self.cfg, self.lim = a, v82, cfg, lim      # cfg は制限時間を重ねた写し
        self.env = env_brief(env)                                     # 各試行の meta["env"] に写す要約
        self.reflex_cfg = reflex_config(a) if getattr(a, "cmd", None) == "run" else None   # 握り損ねの反射（既定は切）
        self.world = self.suite = None
        self.cache = {}
        self.n_trials = 0

    def new_world(self):
        """世界とセンサのモデルを作る（--reuse-world でなければ試行ごとに作り直す）。eval_trials は 1 つの世界で試行を続けて回すが、
        WorldRig.reset は前の試行のハンドの力の上限のまま落ち着かせるので、2 本目以降は開始の指の開きが 1 本目と違う
        （最大 0.94 mm、96_s4_replay_check.py で確かめた）。作り直せば、方策を通さない部分が試行の順と
        止めて再開したかによらなくなる。方策（GPU のモデル）は self.cache に残して使い回す。"""
        from recovla.harness.sensors import SensorSuite
        from recovla.harness.world import WorldRig
        if self.suite is not None:
            self.suite.close()
        gc = False if getattr(self.a, "diag_no_gravcomp", False) else None
        self.world = WorldRig(render=False, cfg=self.cfg, gravcomp=gc)
        self.suite = SensorSuite(self.world.model, self.cfg)

    def _before_trial(self):
        if self.n_trials and self.a.world_per_trial:
            self.new_world()
        self.n_trials += 1

    def init_run(self):
        a, v82, CFG = self.a, self.v82, self.cfg
        from recovla.common import config
        from recovla.runtime import cue as C
        from recovla.runtime.motion import Motion
        from recovla.runtime.perception import Params, Perception
        from recovla.runtime.policy import SensorPolicy
        from recovla.runtime.runner import PolicyRuntime, disable_rtc_for, enable_rtc_for
        from recovla.runtime.safety import PerceptionSafetyFilter
        self.new_world()                                             # eval_trials と同じ WorldRig(render=False, cfg, gravcomp)・SensorSuite
        rt_cfg, act = CFG["runtime"], CFG["actuation"]
        self.exec_interval = int(a.exec_interval or rt_cfg["exec_interval"])
        cache = self.cache

        def make_runtime(io, setup):                                 # 82 cmd_run の make_runtime と同じ（CFG は ablate 後のもの）
            if "pol" not in cache:
                mo = Motion(setup)
                cache["fk_motion"] = mo
                cache["pol"] = SensorPolicy(config.path(v82.CKPT[a.model]), setup, mo.hand_pose)
                if a.mode == "rtc":
                    enable_rtc_for(cache["pol"], int(rt_cfg["rtc_guidance_horizon"]), rt_cfg["rtc_schedule"],
                                   float(rt_cfg["rtc_max_guidance_weight"]))
                else:
                    disable_rtc_for(cache["pol"])
            pol = cache["pol"]
            pol.setup = setup
            if pol.cue is not None:
                pol.cue = C.TargetCue(C.calibration_from_setup(setup.cameras["overhead"]), pol.cue.thr,
                                      setup.table_z + 0.5 * setup.cube_size, setup.cue_fallback_xy)
            rtv = CFG["runtime_v2"]
            per = Perception(setup, Params.from_config(rtv["perception"]), C.Thresholds.from_dict(config.color_detect(CFG)))
            sf = None
            if not a.no_safety:
                sf = PerceptionSafetyFilter(setup, CFG["safety_filter"], float(rtv["safety_extra_margin_m"] or 0.0))
            gate = dict(rtv.get("gripper_gate") or {})
            if a.grip_gate:
                gate["enabled"] = True
            extra = {}
            if self.reflex_cfg is not None:                          # --reflex のときだけ（切なら eval_trials と同じ呼び方）
                from recovla.runtime.reflex import GraspLossReflex, ReflexParams
                extra["reflex"] = GraspLossReflex(ReflexParams.from_dict(self.reflex_cfg["params"]))
            return PolicyRuntime(io, setup, pol, perception=per, safety=sf, checks=rtv["checks"], gripper_gate=gate, **extra,
                                 tip_offset=float(CFG["sim"]["fingertip_offset"]), mode=a.mode, s=self.exec_interval,
                                 d_init=int(rt_cfg["delay_steps"]), rtc_horizon=int(rt_cfg["rtc_guidance_horizon"]),
                                 motion=Motion(setup, limiter_enabled=not a.no_limiter, margin=float(act["limiter_margin"]),
                                               ik_on=a.diag_ik, xcmd_leash_m=a.xcmd_leash or rtv.get("xcmd_leash_m"),
                                               cart_margin=a.cart_margin or rtv.get("cart_margin")))
        self.make_runtime = make_runtime

    def run_one(self, i, seed, lay, tgt):
        a, v82, CFG = self.a, self.v82, self.cfg
        from recovla.eval import induce as I
        from recovla.harness.loop import run_policy_trial
        self._before_trial()
        if a.induce == "P2S":                                        # 引き抜きの落下（recovla.eval.induce_slip。P2S-v1）
            from recovla.eval.induce_slip import PullOutInducer
            ind = PullOutInducer(seed, lay, tgt, self.world)
        else:
            ind = I.Inducer(a.induce, seed, lay, tgt, self.world) if a.induce else None
        w0 = time.perf_counter()
        meta, arrays, rlog = run_policy_trial(self.world, self.suite, self.make_runtime, lay, tgt, seed, inducer=ind, cfg=CFG,
                                              time_limit_s=self.lim["time_limit_s"])
        if self.reflex_cfg is not None:                              # 記録に足す欄（--reflex のときだけ。発火の事象を含む）
            meta["reflex"] = rlog["runtime"].get("reflex")
        meta["time_limits"] = dict(self.lim)                         # 記録に足す欄（eval_trials と同じ time_limit_s も loop が書く）
        meta["env"] = dict(self.env)                                 # 記録に足す欄（環境の要約。回の始めに読んだ値）
        meta["world_per_trial"] = bool(a.world_per_trial)            # 記録に足す欄（試行ごとに世界を作り直したか）
        meta.update({"trial": i, "experiment": a.experiment, "condition": a.condition, "model": {"name": a.model}, "ablate": a.ablate,
                     "runtime": {"mode": a.mode, "exec_interval": self.exec_interval, "delay_steps": "sampled",
                                 "safety_filter": not a.no_safety,
                                 "gripper_gate": bool(a.grip_gate or (CFG["runtime_v2"].get("gripper_gate") or {}).get("enabled")),
                                 "xcmd_leash_m": a.xcmd_leash or CFG["runtime_v2"].get("xcmd_leash_m"),
                                 "cart_margin": a.cart_margin or CFG["runtime_v2"].get("cart_margin")},
                     "mode": a.mode, "limiter": not a.no_limiter, "safety": not a.no_safety, "wall_s": round(time.perf_counter() - w0, 2)})
        return meta, arrays, rlog

    def init_task(self):
        a, v82, CFG = self.a, self.v82, self.cfg
        from recovla.common import config
        from recovla.runtime import cue as C
        from recovla.runtime.executor import TaskRuntime
        from recovla.runtime.judge import JudgeV2
        from recovla.runtime.motion import Motion
        from recovla.runtime.perception import Params, Perception
        from recovla.runtime.policy import SensorPolicy
        from recovla.runtime.runner import PolicyRuntime, disable_rtc_for
        from recovla.runtime.safety import PerceptionSafetyFilter
        self.new_world()                                             # eval_trials と同じ WorldRig(render=False, cfg)・SensorSuite
        rt_cfg, act, rtv =CFG["runtime"], CFG["actuation"], CFG["runtime_v2"]
        thr = C.Thresholds.from_dict(config.color_detect(CFG))
        cache = self.cache
        decompose_fn, self.planner_info = planner_of(getattr(a, "planner", None) or "s4")   # eval_trials は D.decompose を渡す

        def make(io, setup):                                         # 82 cmd_task の make と同じ（渡す計画役の関数だけ選べる）
            if "pol" not in cache:
                cache["pol"] = SensorPolicy(config.path(v82.CKPT[a.model]), setup, Motion(setup).hand_pose)
                disable_rtc_for(cache["pol"])
            pol = cache["pol"]
            pol.setup = setup
            if pol.cue is not None:
                pol.cue = C.TargetCue(C.calibration_from_setup(setup.cameras["overhead"]), pol.cue.thr,
                                      setup.table_z + 0.5 * setup.cube_size, setup.cue_fallback_xy)
            per = Perception(setup, Params.from_config(rtv["perception"]), thr)
            sf = None if a.no_safety else PerceptionSafetyFilter(setup, CFG["safety_filter"], float(rtv["safety_extra_margin_m"]))
            prt = PolicyRuntime(io, setup, pol, perception=per, safety=sf, checks=rtv["checks"], gripper_gate=rtv.get("gripper_gate"),
                                tip_offset=float(CFG["sim"]["fingertip_offset"]), mode="naive",
                                s=int(a.exec_interval or rt_cfg["exec_interval"]),
                                d_init=int(rt_cfg["delay_steps"]), motion=Motion(setup, margin=float(act["limiter_margin"]),
                                                                                 xcmd_leash_m=rtv.get("xcmd_leash_m"),
                                                                                 cart_margin=rtv.get("cart_margin")))
            judge = JudgeV2(setup, per, thr, rtv["judge"])
            e = CFG["expert"]
            mp = {"gain": float(e["gain_per_s"]), "xy_max": float(e["speed_ref"]["xy"]), "z_max": float(e["speed_ref"]["z"]),
                  "z_tol": float(e["move_tol_m"]), "tol": float(e["phase"]["retreat_tol_m"])}
            return TaskRuntime(io, setup, prt, judge, CFG["planner"], mp, decompose_fn, CFG["convert"]["instruction"])
        self.make_task = make

    def run_one_task(self, i, seed):
        a, CFG = self.a, self.cfg
        from recovla.harness.task_loop import run_task_trial
        from recovla.sim import scene
        lay = scene.sample_layout(seed, "empty", start="home")
        self._before_trial()
        w0 = time.perf_counter()
        meta, arrays, rlog = run_task_trial(self.world, self.suite, self.make_task, lay, a.text, seed, cfg=CFG,
                                            time_limit_s=self.lim["task_time_limit_s"])
        meta["time_limits"] = dict(self.lim)                         # 記録に足す欄（eval_trials の task の記録には制限時間の欄がない）
        meta["env"] = dict(self.env)                                 # 記録に足す欄（環境の要約）
        meta["world_per_trial"] = bool(self.a.world_per_trial)       # 記録に足す欄（試行ごとに世界を作り直したか）
        meta["planner"] = dict(self.planner_info)                    # 記録に足す欄（使った計画役）
        meta.update({"run": i, "model": a.model, "wall_s": round(time.perf_counter() - w0, 1)})
        return meta, arrays, rlog

    def close(self):
        if self.suite is not None:
            self.suite.close()


# ---------------------------------------------------------------- 本体
def plan_trials(a, v82) -> list:
    """[(i, seed, lay or None, tgt or None)]。task はシードだけ（配置は回すときに作る）。"""
    if a.cmd == "run":
        return [(i, seed, lay, tgt) for i, (seed, lay, tgt) in enumerate(v82.trial_list(a.trials))]
    base, n = (int(x) for x in a.trials.split(":"))
    return [(i, seed, None, None) for i, seed in enumerate(range(base, base + n))]


def run_json_text(a, kind, rows, wall_s, exec_interval, lim, env=None, segs=None) -> str:
    if kind == "run":
        d = {"experiment": a.experiment, "condition": a.condition, "model": a.model, "mode": a.mode, "exec_interval": exec_interval,
             "safety": not a.no_safety, "trials": a.trials, "induce": a.induce, "n": len(rows),
             "successes": sum(r["success"] for r in rows), "wall_s": round(wall_s, 1), "written": _now_s()}
    else:
        d = {"n": len(rows), "all_three": sum(r["all_three"] for r in rows), "text": a.text, "model": a.model, "trials": a.trials}
        pl = getattr(a, "planner", None) or "s4"
        d["planner"] = dict(PLANNERS[pl], name=pl)                   # 記録に足す欄（使った計画役）
    d["time_limits"] = dict(lim)                                     # 記録に足す欄
    rx = reflex_config(a) if kind == "run" else None
    if rx is not None:                                               # 握り損ねの反射の引数（--reflex のときだけ）
        d["reflex"] = rx
    d["world_per_trial"] = bool(getattr(a, "world_per_trial", False))  # 記録に足す欄
    d["env"] = env                                                   # 記録に足す欄（この回の環境）
    d["env_segments"] = segs                                         # 完全な記録を環境ごとに分けた並び（長さ 2 以上なら報告で分ける）
    return json.dumps(d, ensure_ascii=False, indent=1)


def row_of(kind, i, seed, tgt, meta) -> dict:
    if kind == "run":
        return {"trial": i, "seed": seed, "target": tgt, "success": meta["success"], "established": meta["induce"].get("established")}
    return {"seed": seed, "all_three": meta["all_three_in_box"]}


def cmd_main(a, v82, ops) -> int:
    kind = a.cmd
    a.world_per_trial = not getattr(a, "reuse_world", False)          # 既定: 試行ごとに世界を作り直す（Engine.new_world の注）
    base_cfg = v82.ablate(v82.CFG, a.ablate.split(",")) if (kind == "run" and a.ablate) else v82.CFG
    lim = limits_of(a, base_cfg)
    if kind == "task":
        a.task_time_limit_s = lim["task_time_limit_s"]               # 控え（resume_spec）に実際の値を残す
    cfg = overlay_limits(base_cfg, lim)
    out = v82.OUT / a.experiment / a.condition
    if a.model not in v82.CKPT:
        raise SystemExit(f"--model {a.model}: {sorted(v82.CKPT)} のどれか")
    if kind == "task":
        if a.planner not in PLANNERS:
            raise SystemExit(f"--planner {a.planner}: {sorted(PLANNERS)} のどれか")
        print(f"[resume] 計画役: {a.planner}（{PLANNERS[a.planner]['module']}、{PLANNERS[a.planner]['model']}）", flush=True)
    trials = plan_trials(a, v82)
    stop_paths = [pathlib.Path(a.stop_file) if a.stop_file else out / "STOP", S4 / "STOP"]
    spec = make_spec(a)
    # 現状の確認
    stamp = time.strftime("%Y%m%d-%H%M%S")
    states = []
    for i, seed, lay, tgt in trials:
        ok, why, meta = check_complete(kind, out, i, {"seed": seed, "target": tgt})
        states.append((ok, why, meta))
    todo = [i for i, st in enumerate(states) if not st[0]]
    n_done = len(trials) - len(todo)
    # 完全な記録の制限時間が今回と違えば、混ぜない（30 秒の記録と 60 秒の記録が 1 つの条件に入るのを防ぐ）
    clash = {}
    for i, st in enumerate(states):
        if st[0]:
            d = limits_conflict(kind, record_limits(kind, st[2], v82.CFG), lim)
            if d:
                clash[i] = d
    print(f"[resume] 制限時間: {lim}", flush=True)
    # 環境（ドライバ・torch・OS・git）。完全な記録・前の回と違えば止める（--accept-env-change で分けて進める）
    log_path = out / "resume_log.json"
    log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.is_file() else {"sessions": []}
    env = read_env(ops)
    env_clash = env_conflicts(env, states, log)
    print(f"[resume] 環境: {env_brief(env)}", flush=True)
    if a.dry_run:
        print(f"[dry-run] {out}: 全 {len(trials)} 本、完全 {n_done} 本（飛ばす）、回す {len(todo)} 本")
        for (i, seed, lay, tgt), (ok, why, _) in zip(trials, states):
            print(f"  {i:4d} seed {seed} {tgt or '':6s} {'完全' if ok else '回す(' + why + ')'}"
                  f"{'  制限時間が違う ' + str(clash[i]) if i in clash else ''}")
        if env_clash:
            print(f"[dry-run] 環境の食い違い: {env_clash}{'（--accept-env-change で分けて進める）' if a.accept_env_change else ''}")
        return 3 if (clash or not env.get("driver") or (env_clash and not a.accept_env_change)) else 0
    if clash:
        first = next(iter(clash.items()))
        raise SystemExit(f"{out}: 完全な記録 {len(clash)} 本の制限時間が今回と違う（例 試行 {first[0]}: (記録, 今回) {first[1]}）。"
                         f"同じ条件に混ぜない。記録に合わせた制限時間で回すか、別の条件名にする")
    if not env.get("driver"):
        raise SystemExit("nvidia-smi からドライバの版を読めない。環境の記録のない実行は報告に使えないので回さない")
    if env_clash and not a.accept_env_change:
        raise SystemExit(f"{out}: 環境が完全な記録・前の回と違う: {env_clash}\n"
                         f"（(前, 今)。unknown_records は環境の記録がない完全な記録）。同じ条件に黙って混ぜない。"
                         f"続けるなら --accept-env-change（この回を別の環境の区切り env_segment として resume_log に書き、報告で分ける）")
    out.mkdir(parents=True, exist_ok=True)
    prog_path = pathlib.Path(a.progress_file) if a.progress_file else out / "progress.json"
    # 同じ条件を 2 つのプロセスで回さない（前の progress.json の pid が生きていて、状態が最終でなければ止める）
    if prog_path.is_file():
        try:
            old = json.loads(prog_path.read_text(encoding="utf-8"))
        except Exception:                        # noqa: BLE001
            old = {}
        if old.get("status") in ops.LIVE_STATUS and old.get("pid") and old.get("pid") != os.getpid() \
                and ops._alive(int(old["pid"]), old.get("proc_create_time")):
            raise SystemExit(f"{prog_path}: pid {old['pid']} がこの条件をまだ回している（status={old.get('status')}）。二重に回さない")
    check_spec(a, out, spec)
    # 書きかけの一時ファイル（*.tmp。原子的な置き換えの途中で切れたもの）は退避する
    for p in out.glob("*.tmp"):
        d = out / f"_incomplete_{stamp}"
        d.mkdir(exist_ok=True)
        shutil.move(str(p), str(d / p.name))
    base = {"experiment": a.experiment, "condition": a.condition, "cmd": kind, "model": a.model, "trials_spec": a.trials,
            "pid": os.getpid(), "host": os.environ.get("COMPUTERNAME"), "started": _now_s(), "status": "starting",
            "total": len(trials), "done": n_done, "skipped_complete": n_done, "ran_this_session": 0, "successes": None,
            "last_trial_wall_s": None, "stop_reason": None, "error": None, "quiet_window": quiet_window_of(a),
            "min_free_gb": a.min_free_gb, "time_limits": lim, "env": env, "env_conflicts": env_clash or None}
    prog = Progress(prog_path, ops, base)
    prog.start_heartbeat()

    def _succ(meta) -> int:
        return int(bool(meta["success"] if kind == "run" else meta["all_three_in_box"]))

    n_succ = sum(_succ(st[2]) for st in states if st[0])

    sess = {"start": _now_s(), "pid": os.getpid(), "skipped_complete": n_done, "to_run": todo, "ran": [], "quarantined": {},
            "end": None, "status": "running", "model_init_s": None, "env": env}
    if env_clash:                                                    # --accept-env-change で進めた: この回から別の環境
        sess["env_segment"] = {"accepted": "--accept-env-change", "conflicts": env_clash, "first_trial": todo[0] if todo else None,
                               "note": "この回で回した試行は、前の環境の試行と別の環境として分けて報告する（試行の meta.env で分けられる）"}
    log["sessions"].append(sess)

    def save_log():
        _write_atomic(log_path, json.dumps(log, ensure_ascii=False, indent=1))

    status, exit_code, eng = "done", 0, None
    try:
        reason = stop_requested(stop_paths)
        if reason:
            raise _Stop(reason)
        # 壊れた・欠けた記録を退避（完全なものには触れない）。古い run.json も退避
        if todo:
            for i in todo:
                moved = quarantine(kind, out, i, stamp)
                if moved:
                    sess["quarantined"][str(i)] = {"why": states[i][1], "files": moved}
            for stale in ("run.json", "G_AUDIT.json"):
                if (out / stale).is_file():
                    d = out / f"_incomplete_{stamp}"
                    d.mkdir(exist_ok=True)
                    shutil.move(str(out / stale), str(d / stale))
        save_log()
        prog.update(status="starting", todo=todo)
        eng = Engine(a, v82, cfg, lim, env)
        ran_new = 0
        for i in todo:
            if a.max_new and ran_new >= a.max_new:
                raise _Stop(f"max_new:{a.max_new}")
            r = gate_before_trial(a, ops, prog, stop_paths)
            if r == "memory_timeout":
                status, exit_code = "memory_timeout", 1
                prog.update(stop_reason="memory_timeout")
                break
            if r:
                raise _Stop(r)
            _, seed, lay, tgt = trials[i]
            if eng.world is None:
                prog.update(status="loading")
                t_init = time.perf_counter()
                (eng.init_run if kind == "run" else eng.init_task)()
                sess["model_init_s"] = round(time.perf_counter() - t_init, 1)
                prog.update(status="running")
            prog.update(current=i, current_seed=seed, current_target=tgt, current_started=_now_s())
            t0 = time.perf_counter()
            if kind == "run":
                meta, arrays, rlog = eng.run_one(i, seed, lay, tgt)
            else:
                meta, arrays, rlog = eng.run_one_task(i, seed)
            write_trial_files(kind, out, i, meta, arrays, rlog, v82)
            wall = round(time.perf_counter() - t0, 2)
            ran_new += 1
            n_succ += _succ(meta)
            g = meta["audit"]
            sess["ran"].append({"i": i, "seed": seed, "target": tgt, "wall_s": wall,
                                "success": meta["success"] if kind == "run" else meta["all_three_in_box"]})
            save_log()
            prog.update(done=n_done + ran_new, ran_this_session=ran_new, last_trial_wall_s=wall, successes=n_succ)
            if kind == "run":
                print(f"[v2] {a.condition} {i:3d} seed {seed} {tgt:5s} success {meta['success']} est {meta['induce'].get('established')} "
                      f"stops {g['g2']['world_stops']} early {g['g2']['early_use']} g3 {g['g3']['total_violations']} "
                      f"wall {meta['wall_s']}", flush=True)
            else:
                print(f"[v2task] {i} seed {seed} plan {(meta['plan'] or {}).get('steps')} final {meta['final_in_box']} "
                      f"stopped {bool(meta['stopped'])} g1 {g['g1']['violations']} g2 {g['g2']['world_stops']}/{g['g2']['early_use']} "
                      f"g3 {g['g3']['total_violations']} wall {meta['wall_s']}", flush=True)
    except _Stop as s:
        status, exit_code = "stopped", 1
        prog.update(stop_reason=s.reason)
        print(f"[resume] 止める合図: {s.reason}", flush=True)
    except KeyboardInterrupt:
        status, exit_code = "interrupted", 1
        prog.update(stop_reason="KeyboardInterrupt")
    except SystemExit:
        raise
    except BaseException as e:                   # noqa: BLE001
        status, exit_code = "error", 2
        prog.update(error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-3000:])
        traceback.print_exc()
    finally:
        if eng is not None:
            try:
                eng.close()
            except Exception:                    # noqa: BLE001
                pass
    # 全部そろっていれば run.json と gate
    complete = True
    rows, metas = [], []
    for i, seed, lay, tgt in trials:
        ok, why, meta = check_complete(kind, out, i, {"seed": seed, "target": tgt})
        complete &= ok
        if ok:
            rows.append(row_of(kind, i, seed, tgt, meta))
            metas.append((i, meta))
    if status == "done" and not complete:
        status, exit_code = "error", 2
        prog.update(error="全部回したのに、完全でない記録が残っている")
    if status == "done":
        wall_total = sum(r["wall_s"] for s_ in log["sessions"] for r in s_.get("ran", []))
        exec_interval = getattr(eng, "exec_interval", None) or int(a.exec_interval or cfg["runtime"]["exec_interval"])
        if todo or not (out / "run.json").is_file():
            _write_atomic(out / "run.json", run_json_text(a, kind, rows, wall_total, exec_interval, lim, env, env_segments(metas)))
        if todo or not (out / "G_AUDIT.json").is_file():
            v82._gate_mark(out)
        prog.update(g_audit_met=json.loads((out / "G_AUDIT.json").read_text(encoding="utf-8")).get("met"))
    sess.update({"end": _now_s(), "status": status, "stop_reason": prog.d.get("stop_reason"),
                 "peak_wset_gb": round(prog.peak["wset_gb"], 2), "peak_pagefile_gb": round(prog.peak["pagefile_gb"], 2),
                 "min_free_phys_gb": prog.min_free, "min_free_commit_gb": prog.min_commit_free})
    save_log()
    prog.stop_heartbeat()
    prog.update(status=status, done=len(rows) if complete else prog.d.get("done"), successes=n_succ, complete=complete)
    print(f"[resume] {a.condition}: {status}（完全 {len(rows)}/{len(trials)}、この回 {len(sess['ran'])} 本、飛ばした {n_done} 本）", flush=True)
    return exit_code


class _Stop(Exception):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "task"):
        p = sub.add_parser(name)
        p.add_argument("--experiment", required=True)
        p.add_argument("--condition", required=True)
        p.add_argument("--model", default="R2" if name == "task" else None, required=name == "run")
        p.add_argument("--trials", required=True, help="run: natural|induced|selection:<種の先頭>:<数>、task: <種の先頭>:<数>")
        p.add_argument("--exec-interval", type=int, default=None)
        p.add_argument("--no-safety", action="store_true")
        if name == "run":
            p.add_argument("--time-limit-s", type=float, default=60.0,
                           help="試行の制限時間 [s]（段階 4 の既定 60。段階 3 の記録は 30）")
            p.add_argument("--mode", default="naive", choices=("naive", "sync", "rtc"))
            p.add_argument("--induce", default=None)
            p.add_argument("--no-limiter", action="store_true")
            p.add_argument("--diag-ik", default="commanded", choices=("commanded", "measured"))
            p.add_argument("--diag-no-gravcomp", action="store_true")
            p.add_argument("--ablate", default=None)
            p.add_argument("--grip-gate", action="store_true")
            p.add_argument("--xcmd-leash", type=float, default=None)
            p.add_argument("--cart-margin", type=float, default=None)
            p.add_argument("--reflex", default=None, choices=("drop_only", "full"),
                           help="握り損ねの反射を、名前のついた設定で入れる（recovla.runtime.reflex.PRESETS。drop_only = 落下だけ"
                                "（実験の主）、full = 落下＋空掴み）。既定は切。切なら動きも記録も前と同じ")
            p.add_argument("--reflex-set", action="append", default=None, metavar="KEY=VALUE",
                           help="反射の引数を設定から変える（例 stop=freeze）。何度でも付けられる。変えたら別の条件名にする")
        else:
            p.add_argument("--text", default="全部片付けて")
            p.add_argument("--step-timeout-s", type=float, default=TASK_STEP_TIMEOUT_DEFAULT,
                           help="1 手順の持ち時間 [s]（既定 30 = 段階 3 と同じ。作者の決定 10/08）。やり直しの回数は configs の retry のまま")
            p.add_argument("--task-time-limit-s", type=float, default=TASK_TIME_LIMIT_DEFAULT,
                           help="試行全体の打ち切り [s]（既定 200 = task_loop の既定。段階 3 と同じ）")
            p.add_argument("--planner", default="s4", choices=sorted(PLANNERS),
                           help="計画役。s4 = decompose_haiku（claude-haiku-5-5、既定）、legacy = 凍結の decompose（Haiku 4.5）")
        g = p.add_argument_group("続きから回すための引数")
        g.add_argument("--dry-run", action="store_true", help="何を飛ばし何を回すかを見るだけ（何も書かない）")
        g.add_argument("--ignore-quiet", action="store_true",
                       help="--quiet-window を無効にする（互換のため残す。既定では窓がないので付けなくてよい）")
        g.add_argument("--quiet-window", default="",
                       help="HH:MM-HH:MM。指定したときだけ、その窓では新しい試行を始めない（既定は空＝窓なし。"
                            "01:45〜02:45 の窓は作者の決定 10/08 で必須から外した）")
        g.add_argument("--min-free-gb", type=float, default=12.0,
                       help="空きの物理メモリがこれ未満なら待つ（運用の決まり 12 GB。並行の本数で割らない）")
        g.add_argument("--min-commit-free-gb", type=float, default=6.0, help="空きのコミット（仮想領域）がこれ未満なら待つ")
        g.add_argument("--mem-timeout-min", type=float, default=120.0, help="メモリ待ちがこの分数を超えたら止まる（0 で無制限）")
        g.add_argument("--max-new", type=int, default=0, help="新しい試行をこの本数回したら止まる（0 で無制限）")
        g.add_argument("--stop-file", default=None, help="止める合図のファイル（省略時は <条件>\\STOP と outputs\\s4\\STOP）")
        g.add_argument("--progress-file", default=None)
        g.add_argument("--accept-spec-change", action="store_true")
        g.add_argument("--reuse-world", action="store_true",
                       help="82 と同じく 1 つの世界で試行を続けて回す（既定は試行ごとに作り直す。止めて再開しても方策を通さない部分が一致する）")
        g.add_argument("--accept-env-change", action="store_true",
                       help="ドライバ・torch・CUDA・OS の版が完全な記録・前の回と違っても進める（この回を env_segment として分ける）")
        g.add_argument("--allow-82-change", action="store_true")
    p = sub.add_parser("score", help="記録だけを読んで、T 秒の採点と成功の時刻の分布を出す（GPU もシミュレーションも使わない）")
    p.add_argument("--experiment", required=True)
    p.add_argument("--condition", required=True, nargs="+")
    p.add_argument("--at", default="30,45,60", help="採点する時刻 [s] のカンマ区切り")
    p.add_argument("--out", default=None, help="結果の json を書く先（省略時は標準出力だけ）")
    return ap


# ---------------------------------------------------------------- score（記録から T 秒の採点）
def score_condition(d: pathlib.Path, ats) -> dict:
    """run: success かつ t_success <= T を T 秒の成功とする（打ち切りの前の経過は制限時間によらない）。
      わざと失敗させる試行（meta["induce"]["kind"] がある）の分母は、失敗が T 秒より前に確定した試行（induce.t_established < T）
      だけ（ちょうど T 秒に確定した試行を入れないことで、60 s の記録の 30 s の採点が 30 s で打ち切った記録と同じ定義になる）。
      確定しなかった・T 秒ちょうど以後に確定した試行は、その T の分母にも分子にも入れない。失敗させない試行は全試行が分母。
    task: 主な指標は「終わりに 3 個とも箱の中（all_three_in_box）」の本数と、全体の打ち切りの本数（timed_out）。
      時間ごとの曲線は、all_three_in_box が真の試行についてだけ、真値の 3 色の成功の時刻の最大 <= T を「T 秒までに 3 個そろった」
      とする（途中で箱から出した試行を曲線で成功に数えない）。分母は全試行。
    at[T] = {"successes": 分子, "n": 分母, "n_trials": 記録の本数, "denominator": 分母の定義}。
    problems: 制限時間が 2 種類以上、環境（ENV_STOP_KEYS）が 2 つ以上（区切りごとに分けて出す）、task で all_three_in_box が
      真なのに真値の時刻がそろわない試行。1 つでもあれば cmd_score は終了コード 1（数は出すが、報告に使う前に分ける）。"""
    runs = sorted(d.glob("trial_[0-9][0-9][0-9][0-9].json"))
    kind = "run"
    if not runs:
        runs, kind = sorted(d.glob("run_[0-9][0-9][0-9][0-9].json")), "task"
    res = {"dir": str(d), "kind": kind, "n": len(runs), "limits_seen": {}, "at": {}, "t_success": []}
    ts, t_est, induced, metas = [], [], [], []
    n_all3 = n_timed = n_all3_no_t = 0
    for p in runs:
        m = json.loads(p.read_text(encoding="utf-8"))
        metas.append((int(p.stem.split("_")[1]), m))
        lim = m.get("time_limits") or ({"time_limit_s": m.get("time_limit_s")} if kind == "run" else {"step_timeout_s": "configs"})
        key = json.dumps({k: v for k, v in lim.items() if k != "source"}, sort_keys=True)
        res["limits_seen"][key] = res["limits_seen"].get(key, 0) + 1
        if kind == "run":
            t = m.get("t_success") if m.get("success") else None
            ind = m.get("induce") or {}
            induced.append(bool(ind.get("kind")))
            te = ind.get("t_established") if ind.get("established") else None
            t_est.append(float(te) if te is not None else None)
        else:
            tt = m.get("truth_success_t") or {}
            all3 = bool(m.get("all_three_in_box"))
            ok_t = len(tt) == 3 and all(v is not None for v in tt.values())
            t = max(tt.values()) if (all3 and ok_t) else None
            n_all3 += all3
            n_all3_no_t += all3 and not ok_t
            n_timed += bool(m.get("timed_out"))
            induced.append(False)
            t_est.append(None)
        ts.append(t)
    n_ind = sum(induced)
    for T in ats:
        # 分母に入る試行: 失敗させない試行は全部、わざと失敗させる試行は T 秒より前に確定したものだけ（ちょうど T は入れない）
        den = [i for i in range(len(ts)) if not induced[i] or (t_est[i] is not None and t_est[i] < T - 1e-9)]
        k = sum(1 for i in den if ts[i] is not None and ts[i] <= T + 1e-9)
        what = (("全試行（分子は all_three_in_box が真で、3 色の真値の時刻の最大 <= T）" if kind == "task" else "全試行")
                if not n_ind else
                "誘発が T 秒より前に成立した試行（induce.t_established < T）" if n_ind == len(ts) else
                "自然の試行は全部、誘発の試行は T 秒より前に成立したものだけ")
        res["at"][f"{T:g}"] = {"successes": k, "n": len(den), "n_trials": len(ts), "denominator": what}
    res["t_success"] = ts
    if kind == "task":
        res["all_three_in_box"] = n_all3                             # 主な指標の本数
        res["timed_out"] = n_timed                                   # 全体の打ち切りが効いた本数（1 件でも事前登録の逸脱として書く）
        res["all_three_without_truth_time"] = n_all3_no_t
    if n_ind:
        res["n_induced"] = n_ind
        res["t_established"] = t_est
    probs = []
    if len(res["limits_seen"]) > 1:
        probs.append(f"制限時間が {len(res['limits_seen'])} 種類ある（同じ条件に混ぜない）: {sorted(res['limits_seen'])}")
    segs = env_segments(metas)
    res["env_segments"] = [{"env": s["env"], "n": len(s["trials"]), "trials": s["trials"]} for s in segs]
    if len(segs) > 1:
        probs.append(f"環境の区切りが {len(segs)} つある（区切りごとに分けて出す。env_segments）")
    if n_all3_no_t:
        probs.append(f"all_three_in_box が真なのに真値の 3 色の時刻がそろわない試行が {n_all3_no_t} 本（曲線の終わりが主な数より低い）")
    if probs:
        res["problems"] = probs
    lims = [json.loads(k) for k in res["limits_seen"]]
    cap = min((x.get("time_limit_s") or 0) for x in lims) if kind == "run" and lims else None
    if cap and any(T > cap + 1e-9 for T in ats):
        res["warning"] = f"制限時間 {cap} s の記録で、それより長い時刻の採点は意味がない"
    return res


def cmd_score(a) -> int:
    ats = [float(x) for x in a.at.split(",") if x.strip()]
    base = ROOT / "outputs" / "v2eval" / a.experiment
    out = {"written": _now_s(), "experiment": a.experiment, "at": ats,
           "conditions": {c: score_condition(base / c, ats) for c in a.condition}}
    text = json.dumps(out, ensure_ascii=False, indent=1)
    print(text)
    if a.out:
        _write_atomic(pathlib.Path(a.out), text)
    bad = {c: r["problems"] for c, r in out["conditions"].items() if r.get("problems")}
    for c, ps in bad.items():
        print(f"[score] {c}: {' / '.join(ps)}", file=sys.stderr)
    return 1 if bad else 0


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:                            # noqa: BLE001
        pass
    a = build_parser().parse_args(argv)
    if a.cmd == "score":
        return cmd_score(a)
    ops = load_ops()
    v82 = load_82(a.allow_82_change)
    try:
        return cmd_main(a, v82, ops)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(e.code, file=sys.stderr, flush=True)
            return 3
        raise


if __name__ == "__main__":
    raise SystemExit(main())
