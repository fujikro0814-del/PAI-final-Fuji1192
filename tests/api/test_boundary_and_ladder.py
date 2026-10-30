"""task API の G1（実行系は真値を読まない）と入口 scripts/api/ladder.py の確かめ。CPU だけ。
入口の dry-run は評価の本体（scripts/eval_resumable.py）の dry-run まで通す（何も書かない。outputs/・チェックポイント・GPU・API キーは要らない）。"""
import ast
import importlib.util
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
API = ROOT / "src" / "recovla" / "api"
FORBIDDEN = ("recovla.sim", "recovla.harness", "recovla.eval", "recovla.expert", "recovla.api.harness")
RUNTIME_PARTS = ("task_spec", "senses", "perception", "monitors", "executor_monitored", "skills", "skills_script", "planner",
                 "geometry", "estimated", "relations")
KNOWN_DIMS = ("cube_size", "box_outer", "box_wall", "box_nominal_xy", "box_height", "box_floor")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def g1():
    return _load(ROOT / "scripts" / "check_real_boundary.py", "check_g1_boundary_api")


def api_runtime_files():
    return sorted(p for p in API.glob("*.py"))


def test_api_runtime_files_pass_the_strict_scan(g1):
    files = g1.runtime_v2_files(None, g1.API_DIR)
    assert files == [str(p.relative_to(ROOT)).replace("\\", "/") for p in api_runtime_files()]
    assert {f"src/recovla/api/{n}.py" for n in RUNTIME_PARTS} <= set(files)
    assert not any("/harness/" in f for f in files)                         # 真値を使う api/harness/ は対象の外
    for f in files:
        assert not g1.scan_runtime((ROOT / f).read_text(encoding="utf-8"), f, g1.API_ALLOWED_IMPORTS, g1.API_DENY), f


def test_api_runtime_imports_no_truth_modules():
    """構文木で直に数える（検査の道具に頼らない）: import する名前が FORBIDDEN で始まらない。"""
    for p in api_runtime_files():
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                assert not node.level, p.name
                names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            else:
                continue
            for n in names:
                assert not n.startswith(FORBIDDEN), (p.name, n)


def test_importing_the_api_loads_no_truth_module():
    """実際に import して読み込まれた module を見る（runtime/motion が使う IK の部品 sim.control・controller_ik・device は、
    既存の G1 の検査が実行系に許している物。場面・真値・評価の module は読まれない）。"""
    code = ("import sys\n" + "".join(f"import recovla.api.{n}\n" for n in RUNTIME_PARTS)
            + "print('\\n'.join(sorted(m for m in sys.modules if m.startswith('recovla'))))\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, check=True).stdout.split()
    bad = [m for m in out if m.startswith(("recovla.harness", "recovla.eval", "recovla.expert", "recovla.api.harness",
                                           "recovla.record"))]
    assert bad == []
    assert {m for m in out if m.startswith("recovla.sim")} <= {"recovla.sim", "recovla.sim.control",
                                                               "recovla.sim.controller_ik", "recovla.sim.device"}


def _known_dim_uses(path):
    """属性・文字列（getattr の名前）で既知の寸法を読む所 → [(関数の名前, 名前)]。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    parents = {}
    for node in ast.walk(tree):
        for ch in ast.iter_child_nodes(node):
            parents[ch] = node
    for node in ast.walk(tree):
        name = node.attr if isinstance(node, ast.Attribute) else (node.value if isinstance(node, ast.Constant) else None)
        if name in KNOWN_DIMS:
            p = node
            while p in parents and not isinstance(p, (ast.FunctionDef, ast.ClassDef)):
                p = parents[p]
            out.append((getattr(p, "name", "<module>"), name))
    return out


def test_new_ports_do_not_read_known_sizes():
    """推定の版・関係の判定・失敗の検知・スクリプト方策は既知の大きさ・寸法を読まない。読むのは既知の値の版
    （perception.ColorWorldModelFinder と inner_half_of）だけ。"""
    for p in api_runtime_files():
        uses = _known_dim_uses(p)
        if p.name == "perception.py":
            assert {f for f, _ in uses} <= {"from_spec", "inner_half_of"}, uses
        else:
            assert uses == [], (p.name, uses)


def test_strict_scan_catches_truth_access_in_api(g1):
    bad = ("from recovla.api.harness import goal_truth\nimport recovla.sim.frames\nfrom recovla.sim import control\n"
           "from recovla.expert.script import phase_of\nfrom recovla import eval\n"
           "def f(self):\n    world = self.io.world\n    return truth['cube_pos']\n")
    hits = g1.scan_runtime(bad, "src/recovla/api/x.py", g1.API_ALLOWED_IMPORTS, g1.API_DENY)
    for k in ("import:recovla.api.harness.goal_truth", "import:recovla.sim.frames", "import:recovla.sim.control",
              "import:recovla.expert.script.phase_of", "import:recovla.eval", "io:world", "name:truth", "key:cube_pos",
              "name:world"):
        assert hits[k] >= 1, k
    ok = "import yaml\nimport enum\nfrom recovla.api import task_spec\nfrom recovla.runtime.reflex import GraspLossDetector\n"
    assert not g1.scan_runtime(ok, "src/recovla/api/y.py", g1.API_ALLOWED_IMPORTS, g1.API_DENY)
    # runtime の検査（既定の引数）: sim.control は許し、api・yaml は許さない
    assert not g1.scan_runtime("from recovla.sim import control\n", "x.py")
    assert g1.scan_runtime("import yaml\nfrom recovla.api import senses\n", "x.py") == {"import:yaml": 1,
                                                                                          "import:recovla.api.senses": 1}


def test_g1_main_passes_with_api_when_git_and_tag_exist(g1, monkeypatch, capsys):
    """開発用のリポジトリ（タグ g1-baseline がある）なら、api を足した後も main が合格する。"""
    import shutil
    git = shutil.which("git") or (str(g1.GIT) if pathlib.Path(g1.GIT).is_file() else None)   # PATH に無ければ .tools/git
    if git is None or subprocess.run([git, "-C", str(ROOT), "rev-parse", "--verify", "--quiet", "g1-baseline^{commit}"],
                                     capture_output=True).returncode != 0:
        pytest.skip("git かタグ g1-baseline がない（提出用のリポジトリ・浅い clone）")
    monkeypatch.setattr(g1, "GIT", git)
    assert g1.main([]) == 0
    assert "task API の実行系" in capsys.readouterr().out


# ---------------------------------------------------------------- 入口
@pytest.fixture(scope="module")
def ladder():
    return _load(ROOT / "scripts" / "api" / "ladder.py", "api_ladder_for_test")


def test_arm_plans(ladder):
    u0, l2, cl = (ladder.arm_plan(a) for a in ("U0", "L2v2", "CL"))
    assert u0["skill"]["name"] == "vla" and u0["monitors"] is None and u0["replan"] is None and u0["runnable"]
    assert "TaskRuntime" in u0["executor"] and "decompose_haiku" in u0["planner"]
    assert l2["skill"]["name"] == "vla" and [m["name"] for m in l2["monitors"]["active"]] == ["grasp_drop", "held_estimate_drop"]
    assert l2["monitors"]["active"][0]["needs"] == ["gripper", "joints", "camera/overhead"]
    assert l2["monitors"]["params"]["grasp_drop"] == {"confirm_s": 1.0, "camera_confirm": True}
    assert l2["monitors"]["params"]["held_estimate_drop"] == {"on_table_s": 0.5}
    assert u0["finder"] is None and l2["finder"] == cl["finder"] == "estimated"
    assert ladder.arm_plan("CL", finder="color")["finder"] == "color"
    assert "replan_event" in l2["replan"] and "executor_monitored" in l2["executor"]
    assert cl["skill"]["name"] == "script" and cl["monitors"] is None and "TaskRuntime" in cl["executor"]
    assert [r["name"] for r in cl["senses"]] == ["joints", "gripper", "camera/overhead", "camera/wrist"]
    real = ladder.arm_plan("L2v2", "real")                                   # カメラのない実機では失敗の検知を止める
    assert real["monitors"]["active"] == [] and real["monitors"]["skipped"][0]["missing"] == ["camera/overhead"]
    assert [m["name"] for m in real["monitors"]["skipped"]] == ["grasp_drop", "held_estimate_drop"]
    assert not real["runnable"] and real["skill"]["missing"] == ["camera/overhead", "camera/wrist"]


def test_table_command_prints_every_arm(ladder, capsys):
    assert ladder.main(["table"]) == 0
    out = capsys.readouterr().out
    for arm in ("U0", "L2v2", "CL"):
        assert f"[api] 腕 {arm}" in out
    assert "見張り役: grasp_drop" in out and "技能: script" in out and "camera/overhead" in out
    assert ladder.main(["table", "--target", "real", "--arm", "L2v2"]) == 0
    assert "見張り役（止めた）: grasp_drop" in capsys.readouterr().out


def test_files_listed_for_sha_exist(ladder):
    for p in ladder.FILES:
        assert (ROOT / p).is_file(), p


@pytest.mark.parametrize("arm, finder", [("CL", "estimated"), ("L2v2", "color")])
def test_dry_run_prints_the_table_and_writes_nothing(arm, finder, tmp_path):
    cond = f"pytest_api_dry_{arm}"
    out = ROOT / "outputs" / "v2eval" / "APIDRY" / cond
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "api" / "ladder.py"), "run", "--arm", arm, "--finder", finder,
                        "--experiment", "APIDRY", "--condition", cond, "--trials", "193700:2", "--model", "Rreal",
                        "--exec-interval", "6", "--dry-run"], capture_output=True, text=True, cwd=ROOT, timeout=600,
                       encoding="utf-8", errors="replace", env=dict(os.environ, PYTHONUTF8="1"))  # Windows の cp932 で読み損ねない
    text = r.stdout + r.stderr
    assert r.returncode in (0, 3), text                                      # 3 は GPU のドライバが読めない（GPU のない環境）
    assert f"[api] 腕 {arm}" in text and "[dry-run]" in text and "seed 193700" in text, text
    assert f"知覚の口: {finder}" in text
    assert not out.exists()


def test_run_refuses_other_models_and_arms(ladder, capsys):
    assert ladder.main(["run", "--arm", "CL", "--experiment", "X", "--condition", "y", "--trials", "193700:1",
                        "--model", "R2", "--dry-run"]) == 3
    with pytest.raises(SystemExit):
        ladder.main(["run", "--arm", "L1", "--experiment", "X", "--condition", "y", "--trials", "1:1"])
    with pytest.raises(SystemExit):
        ladder.main([])
