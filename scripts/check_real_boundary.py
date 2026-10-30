"""実行時のコードがシミュレーションの真の状態に触れていないかを、構文木で調べる（G1 = 実機の境界の検査）。

実行時の判断をするコード（方策の入口・LLM の層・完了判定・安全フィルタ・実行器）が、実機でも得られる
センサの出力だけを使っているかを見る。数える印:
  - import: 世界のモデル・真値・評価・生成プログラムのモジュール（sim.rig・sim.scene・expert・eval など）
  - 属性: 世界の状態（qpos・xpos・contact など）、真値の関数、制御器の内部（gripper_closed・integrator など）
  - 文字列の添字: 記録用の辞書の真値の欄（cube_pos・x_des・fingers など）と、設定の箱の位置（"box"）
  - 名前: rig・SimRig
src/recovla/runtime/ と src/recovla/api/ の直下は違反 0 件で合格。以前の実行系（RUNTIME_FILES）は、
docs/g1_known_violations.json の既知の数より増えず、既知の一覧がタグ g1-baseline の時点から増えていなければ合格。

    .venv\\Scripts\\python.exe scripts\\check_real_boundary.py [--rev HEAD]       # 検査（--rev は git の版から読む）
    .venv\\Scripts\\python.exe scripts\\check_real_boundary.py --write-baseline   # 既知の一覧を作り直す（数が減ったときだけ使う）
"""
import argparse
import ast
import collections
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
GIT = pathlib.Path(__file__).resolve().parents[1] / ".tools" / "git" / "cmd" / "git.exe"
BASELINE = "docs/g1_known_violations.json"
BASELINE_TAG = "g1-baseline"

# 以前の実行系で、実行時の判断をするコード
RUNTIME_FILES = (
    "src/recovla/policy/runner.py", "src/recovla/policy/scene_policy.py", "src/recovla/eval/closed_loop.py",
    "src/recovla/data/vla_observation.py", "src/recovla/data/vla_state.py", "src/recovla/data/convert.py",
    "src/recovla/perception/color.py", "src/recovla/planner/detect.py", "src/recovla/planner/judge.py",
    "src/recovla/planner/executor.py", "src/recovla/planner/decompose.py", "src/recovla/sim/safety.py",
    "src/recovla/eval/scene_trial.py",
)
FORBIDDEN_IMPORTS = ("recovla.sim.rig", "recovla.sim.scene", "recovla.sim.frames", "recovla.sim.contact",
                     "recovla.expert", "recovla.eval", "recovla.record.episode")
FORBIDDEN_ATTRS = {"qpos", "qvel", "qacc", "xpos", "xquat", "xmat", "geom_xpos", "geom_xmat", "cam_xpos", "cam_xmat",
                   "contact", "ctrl", "truth", "cube_ids", "cube_vadr", "cubes_in_box", "gripper_closed", "integrator",
                   "controller", "scratch", "forward_scratch", "finger_qadr", "finger_vadr", "meter", "desired_pos",
                   "x_cmd", "mj_geomDistance"}
FORBIDDEN_KEYS = {"cube_pos", "cube_quat", "cube_linvel", "cube_in_box", "phase", "contact_robot", "contact_cube_cube",
                  "min_dist", "x_des", "tracker_target", "gripper_cmd", "gripper_closed", "fingertip", "fingers",
                  "safety_active", "target", "box"}
FORBIDDEN_NAMES = {"rig", "SimRig"}


# 今の実行系（src/recovla/runtime/ の直下）の厳しい検査。既知の違反は認めない（0 件で合格）
RUNTIME_V2_DIR = "src/recovla/runtime"
ALLOWED_IMPORTS = ("recovla.runtime", "recovla.common", "recovla.data.vla_", "recovla.sim.control", "recovla.sim.controller_ik",
                   "recovla.sim.device", "recovla.policy.schedule", "recovla.policy.runner",
                   "numpy", "scipy", "cv2", "mujoco", "torch", "lerobot", "dataclasses", "math", "itertools", "typing",
                   "functools", "pathlib", "contextlib", "io", "time", "json", "collections", "__future__")
V2_FORBIDDEN_NAMES = {"rig", "SimRig", "WorldRig", "SensorSuite", "SimRobotIO", "Truth", "truth", "world"}
V2_FORBIDDEN_KEYS = {"cube_pos", "cube_quat", "cube_linvel", "cube_in_box", "contact_robot", "contact_cube_cube", "min_dist",
                     "phase", "scene"}
# タスクと感覚の窓口（src/recovla/api/ の直下。真値を使う api/harness/ は対象の外）も同じ厳しい検査。
# import の許可は runtime の一覧に api・yaml・enum を足したもの。API_DENY は許可の一覧にあっても数える
API_DIR = "src/recovla/api"
API_ALLOWED_IMPORTS = ALLOWED_IMPORTS + ("recovla.api", "yaml", "enum")
API_DENY = ("recovla.sim", "recovla.harness", "recovla.eval", "recovla.expert", "recovla.api.harness")
IO_API = {"sense", "command_joints", "gripper_move", "gripper_grasp", "compute", "now"}
MODEL_LOADERS = {"from_xml_path", "from_xml_string", "from_file", "from_binary_path"}


def scan_runtime(src: str, filename: str, allowed: tuple = ALLOWED_IMPORTS, deny: tuple = ()) -> collections.Counter:
    """実行系の 1 ファイルの違反を数える。import は許可した一覧だけ、世界・真値の名前と記録の欄は使わない、
    RobotIO は 6 つの口（IO_API）だけ、mujoco のモデルを読むのは robot_model.py だけ（腕だけの XML）。"""
    c = collections.Counter()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            for a in node.names:
                if not a.name.startswith(allowed) or a.name.startswith(deny):
                    c[f"import:{a.name}"] += 1
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                c["import:relative"] += 1
            elif node.module:
                for a in node.names:                            # from recovla.sim import control などは名前まで見る
                    full = f"{node.module}.{a.name}"
                    if (not node.module.startswith(allowed) and not full.startswith(allowed)) or full.startswith(deny):
                        c[f"import:{full}"] += 1
        elif isinstance(node, ast.Name) and node.id in V2_FORBIDDEN_NAMES:
            c[f"name:{node.id}"] += 1
        elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and node.slice.value in V2_FORBIDDEN_KEYS:
            c[f"key:{node.slice.value}"] += 1
        elif isinstance(node, ast.Attribute):
            base = node.value
            is_io = (isinstance(base, ast.Name) and base.id == "io") or (
                isinstance(base, ast.Attribute) and base.attr == "io")
            if is_io and node.attr not in IO_API:
                c[f"io:{node.attr}"] += 1
            if node.attr in MODEL_LOADERS and not filename.endswith("robot_model.py"):
                c[f"loader:{node.attr}"] += 1
            if node.attr in V2_FORBIDDEN_NAMES:
                c[f"attr:{node.attr}"] += 1
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and ("scene_3cube" in node.value
                                                                                 or "scene_g0" in node.value):
            c["str:scene_xml"] += 1
    return c


def runtime_v2_files(rev, d: str = RUNTIME_V2_DIR) -> list:
    """d の直下の .py（下のフォルダは入れない。api/harness/ が外れる）。"""
    if rev is None:
        return sorted(str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / d).glob("*.py"))
    out = subprocess.run([str(GIT), "-C", str(ROOT), "ls-tree", "--name-only", f"{rev}:{d}/"],
                         capture_output=True, text=True)
    return sorted(f"{d}/{n}" for n in out.stdout.split() if n.endswith(".py"))


def read(path: str, rev):
    if rev is None:
        p = ROOT / path
        return p.read_text(encoding="utf-8") if p.is_file() else None
    out = subprocess.run([str(GIT), "-C", str(ROOT), "show", f"{rev}:{path}"], capture_output=True)
    return out.stdout.decode("utf-8") if out.returncode == 0 else None


def scan(src: str) -> collections.Counter:
    c = collections.Counter()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith(FORBIDDEN_IMPORTS):
                    c[f"import:{a.name}"] += 1
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            for m in mods:
                if m.startswith(FORBIDDEN_IMPORTS):
                    c[f"import:{m}"] += 1
                    break
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            c[f"attr:{node.attr}"] += 1
        elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and node.slice.value in FORBIDDEN_KEYS:
            c[f"key:{node.slice.value}"] += 1
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            c[f"name:{node.id}"] += 1
    return c


def current(rev) -> dict:
    out = {}
    for f in RUNTIME_FILES:
        src = read(f, rev)
        if src is None:
            continue
        hits = scan(src)
        if hits:
            out[f] = dict(sorted(hits.items()))
    return out


def load_baseline(rev) -> dict:
    src = read(BASELINE, rev)
    return {} if src is None else json.loads(src)["violations"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rev", default=None, help="git の版から読む（既定は作業ツリー）")
    ap.add_argument("--write-baseline", action="store_true")
    ap.add_argument("--root", default=None, help="調べるリポジトリ（既定はこのスクリプトのあるリポジトリ）")
    a = ap.parse_args(argv)
    global ROOT
    if a.root:
        ROOT = pathlib.Path(a.root).resolve()
    now = current(a.rev)
    if a.write_baseline:
        doc = {"note": "G1 の既知の違反（scripts/check_real_boundary.py が数える）。減らすだけ。関門 1 で空にする（0106）",
               "violations": now}
        (ROOT / BASELINE).write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"書いた: {BASELINE}（{sum(sum(v.values()) for v in now.values())} 件）")
        return 0
    base = load_baseline(a.rev)
    problems = []
    for f, hits in now.items():
        for k, n in hits.items():
            known = base.get(f, {}).get(k, 0)
            if n > known:
                problems.append(f"{f}: {k} が {n} 件（既知 {known} 件）")
    tag_ok = subprocess.run([str(GIT), "-C", str(ROOT), "rev-parse", "--verify", "--quiet", f"{BASELINE_TAG}^{{commit}}"],
                            capture_output=True).returncode == 0
    if tag_ok:
        first = load_baseline(BASELINE_TAG)
        for f, hits in base.items():
            for k, n in hits.items():
                if n > first.get(f, {}).get(k, 0):
                    problems.append(f"既知の一覧が {BASELINE_TAG} から増えた: {f}: {k}")
    else:
        problems.append(f"タグ {BASELINE_TAG} がない")
    # 今の実行系（runtime/）は既知の違反を認めない
    v2_files = runtime_v2_files(a.rev)
    v2_hits = 0
    for f in v2_files:
        src = read(f, a.rev)
        if src is None:
            continue
        for k, n in scan_runtime(src, f).items():
            v2_hits += n
            problems.append(f"v2 の実行系 {f}: {k} が {n} 件（0 件でなければならない）")
    # タスクと感覚の窓口（src/recovla/api/ の直下）も既知の違反を認めない
    api_files = runtime_v2_files(a.rev, API_DIR)
    for f in api_files:
        src = read(f, a.rev)
        if src is None:
            continue
        for k, n in scan_runtime(src, f, API_ALLOWED_IMPORTS, API_DENY).items():
            problems.append(f"task API の実行系 {f}: {k} が {n} 件（0 件でなければならない）")
    total = sum(sum(v.values()) for v in now.values())
    if problems:
        print(f"G1 の境界の検査: 不合格（{len(problems)} 件）")
        for p in problems:
            print("  " + p)
        return 1
    print(f"G1 の境界の検査: 合格（v2 の実行系 {len(v2_files)} ファイルで違反 0。旧版の実行系の既知の違反 {total} 件は増えていない。"
          f"task API の実行系 {len(api_files)} ファイルで違反 0）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
