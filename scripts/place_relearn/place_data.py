"""置く位置の入力を足した学習データ（docs/api/r2_protocol.md 6.4 節）: Rdag と同じ 670 本を、置く位置の入力を足して変換し直す
（状態 17 次元 → 19 次元）。選んだエピソード・フレーム・画像・行動・目標位置の入力は Rdag と同じ（同じマニフェストの同じ並び）。

    .venv\\Scripts\\python.exe scripts\\place_relearn\\place_data.py manifest      # outputs/manifests/Rplace_<時刻>.json（記録のフォルダを絶対パスに）
    .venv\\Scripts\\python.exe scripts\\place_relearn\\place_data.py convert --manifest <上の json> [--limit N --out <dir>]

変換の引数は Rdag と同じ（--target-cue --cue-thresholds runtime_v2 --cue-without-flag）に --place-cue を足す。
"""
import argparse
import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
R1 = pathlib.Path(r"C:\PAI\wt_r1")
RDAG_MANIFEST = R1 / "outputs" / "manifests" / "Rdag_20261021-133603.json"
RDAG_MANIFEST_SHA = "1ab4006e180a20ef20ded02436d4fc47752b83d718907826949fd7cf0ad49ea9"
CUE_ARGS = ["--target-cue", "--cue-thresholds", "runtime_v2", "--cue-without-flag", "--place-cue"]


def sha256(p: pathlib.Path) -> str:
    import hashlib
    return hashlib.sha256(p.read_bytes()).hexdigest()


def cmd_manifest(a) -> int:
    if sha256(RDAG_MANIFEST) != RDAG_MANIFEST_SHA:
        sys.exit(f"{RDAG_MANIFEST}: SHA-256 differs from Rdag's")
    m = json.loads(RDAG_MANIFEST.read_text(encoding="utf-8"))
    name = "Rplace_" + time.strftime("%Y%m%d-%H%M%S")
    out = ROOT / "outputs" / "manifests" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    entries = [{"run": str((R1 / e["run"]).resolve()), "key": e["key"]} for e in m["entries"]]
    out.write_text(json.dumps({"name": name, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "base_commit": None,
                               "rule": f"R2 stage P: the {len(entries)} Rdag episodes in the same order "
                                       f"({RDAG_MANIFEST.name}, sha256 {RDAG_MANIFEST_SHA}), run folders made absolute",
                               "from_manifest": str(RDAG_MANIFEST), "from_manifest_sha256": RDAG_MANIFEST_SHA,
                               "entries": entries, "composition": m.get("composition")}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(out)
    return 0


def cmd_convert(a) -> int:
    mp = pathlib.Path(a.manifest)
    m = json.loads(mp.read_text(encoding="utf-8"))
    if a.limit:
        m = dict(m, name=m["name"] + f"_first{a.limit}", entries=m["entries"][:a.limit])
        mp = pathlib.Path(a.out).with_suffix(".manifest.json")
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    ds = pathlib.Path(a.out) if a.out else ROOT / "outputs" / "datasets" / m["name"]
    log = ds.parent / f"convert_{ds.name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as f:
        c1 = subprocess.run([sys.executable, "-m", "recovla.data.convert", "--manifest", str(mp), "--out", str(ds),
                             "--name", ds.name, *CUE_ARGS], stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
        c2 = subprocess.run([sys.executable, "-m", "recovla.data.convert", "--verify", str(ds)],
                            stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT)) if c1.returncode == 0 else None
    print(f"convert rc {c1.returncode}, verify rc {None if c2 is None else c2.returncode}, log {log}")
    return 0 if c1.returncode == 0 and c2 is not None and c2.returncode == 0 else 1


def main(argv=None) -> int:
    import recovla
    if pathlib.Path(recovla.__file__).resolve().parents[1] != ROOT / "src":
        sys.exit(f"recovla is imported from {recovla.__file__}; set PYTHONPATH={ROOT / 'src'}")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("manifest")
    c = sub.add_parser("convert")
    c.add_argument("--manifest", required=True)
    c.add_argument("--out", default=None)
    c.add_argument("--limit", type=int, default=0, help="試し: 最初の N 本だけ")
    a = ap.parse_args(argv)
    return {"manifest": cmd_manifest, "convert": cmd_convert}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
