"""コードの版。データの生成・学習の起動・評価が記録に残す。

git_commit は <ROOT> の HEAD（PATH に git が無ければ <ROOT>\\.tools\\git の MinGit を使う）、git_dirty はコミットしていない
変更があるか。記録・変換・観測・学習の起動・評価を決めるファイルの SHA-256 も残し、git が無くても版が分かるようにする。
"""
import hashlib
import pathlib
import shutil
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[3]           # <ROOT>
PORTABLE_GIT = ROOT / ".tools" / "git" / "cmd" / "git.exe"
CODE_FILES = (
    "src/recovla/sim/control.py", "src/recovla/sim/controller_ik.py", "src/recovla/sim/device.py",
    "src/recovla/sim/render.py", "src/recovla/record/recorder.py", "src/recovla/record/replay.py",
    "assets/mjcf/scene_g0.xml", "assets/mjcf/panda/panda.xml",
    "src/recovla/data/vla_image_spec.py", "src/recovla/data/vla_state.py", "src/recovla/data/vla_observation.py",
    "src/recovla/data/convert.py", "src/recovla/policy/train_launcher.py", "src/recovla/eval/closed_loop.py",
    "src/recovla/common/code_version.py", "src/recovla/common/config.py", "configs/default.yaml", "configs/g0.yaml",
    # 3 色の場面、生成プログラム、データの生成、記録
    "assets/mjcf/scene_3cube.xml", "src/recovla/common/seeds.py", "src/recovla/sim/frames.py",
    "src/recovla/sim/scene.py", "src/recovla/sim/rig.py", "src/recovla/sim/contact.py",
    "src/recovla/expert/script.py", "src/recovla/expert/generate.py", "src/recovla/record/episode.py",
    "src/recovla/record/snapshot.py",
)


def file_sha256(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def git_executable():
    found = shutil.which("git")
    if found:
        return found
    return str(PORTABLE_GIT) if PORTABLE_GIT.is_file() else None


def _git(root, *args):
    git = git_executable()
    if git is None:
        return None
    try:
        out = subprocess.run([git, "-C", str(root), *args], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def git_commit(root=ROOT):
    """git の HEAD。git が無いかリポジトリでなければ None。"""
    out = _git(root, "rev-parse", "HEAD")
    return out.strip() if out else None


def git_dirty(root=ROOT):
    """HEAD と違うファイル（追跡外を含む、ignore は除く）があれば True。git が無ければ None。"""
    out = _git(root, "status", "--porcelain")
    return None if out is None else bool(out.strip())


def code_version(root=ROOT) -> dict:
    root = pathlib.Path(root)
    files = {f: (file_sha256(root / f) if (root / f).is_file() else None) for f in CODE_FILES}
    combined = hashlib.sha256("\n".join(f"{f} {h}" for f, h in files.items()).encode()).hexdigest()
    commit = git_commit(root)
    return {"git_commit": commit, "git_dirty": git_dirty(root) if commit else None,
            "code_sha256": combined, "files_sha256": files}
