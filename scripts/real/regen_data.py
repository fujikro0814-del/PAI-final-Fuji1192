"""実機の Panda で測った応答に合わせたシミュレーションで、学習データを作り直す（docs/real/prereg_regen_v1.md 第 5 節）。
CPU だけ（描画は合わせる前の生成と同じ）。

    .venv\\Scripts\\python.exe scripts\\real\\regen_data.py plan                  # 計画 → outputs\\real\\regen\\plan.json
    .venv\\Scripts\\python.exe scripts\\real\\regen_data.py pilot [--workers 6]   # 試しの 30 本: 生成 → 変換と照合 → 点検 → 目で見る PNG
    .venv\\Scripts\\python.exe scripts\\real\\regen_data.py pilot --resume outputs\\gen\\V3R_pilot_<日時>   # 止まった続きから
    .venv\\Scripts\\python.exe scripts\\real\\regen_data.py sheet                 # 目で見る PNG だけ作り直す（pilot.json の回）
    .venv\\Scripts\\python.exe scripts\\real\\regen_data.py gen-data [--workers 6]   # 本番: 生成 → Rreal・Nreal のデータ → 変換と照合 → 確かめ (2) → data.json
    .venv\\Scripts\\python.exe scripts\\real\\regen_data.py gen-data --resume outputs\\gen\\V3R_data_<日時>   # 止まった続きから

計画は scripts\\recovery_data.py の cmd_plan をそのまま呼ぶ（importlib で読み込む。recovery_data.py は書き換えない）。変えるのは
立て直しデータの必要数だけ（A 掴み損ね 60・B 0・C 置き損ね 30）。通常 240 は合わせる前の outputs\\f\\plan.json と同じ指定で
なければ止める。
生成は v3_real の世界（recovla.expert.generate_real.generate、configs\\v3_real.yaml の既定）。
止まっても続きから回せるように、指定を配置の組ごとにまとめて「かたまり」（chunk_<番号>）に分けて生成する。
終わったかたまり（timing.json がある）は飛ばし、途中のかたまりは _incomplete_ を付けて退避してから作り直す。
変換と照合は合わせる前と同じ（--target-cue --cue-thresholds runtime_v2、旗は outputs\\f\\data_cue.json の keep_flag_in_input）。
記録の stage3 の欄は、合わせる前の世界（rig v3、学習データ R1v3・N1v3）の同じ指定の値。
確かめ (1)・(2) は、事前登録（prereg_regen_v1.md 第 5 節）の「データの関門」(1)・(2)。
"""
import argparse
import collections
import contextlib
import dataclasses
import hashlib
import importlib.util
import json
import pathlib
import shutil
import subprocess
import sys
import time

import numpy as np

from recovla.common import config

ROOT = config.ROOT
CFG = config.load()
OUTS = config.path(CFG["paths"]["outputs"])
GEN = OUTS / "gen"
REGEN = OUTS / "real" / "regen"
STAGE3_PLAN = OUTS / "f" / "plan.json"
STAGE3_DATA = OUTS / "f" / "data_v3.json"
DATA_CUE = OUTS / "f" / "data_cue.json"

RECOVERY_NEED = {"A": 60, "B": 0, "C": 30}         # 事前登録 第 5 節（B は外す）
SEED_BAND = {"A": (30000, 30199), "C": (32000, 32099)}   # 立て直しデータのシードの範囲（30000〜30079・32000〜32039 は合わせる前と同じシード）
WORLD = "v3_real"
V3_REAL_SHA256 = "e39185d10c181ee17e99743831a1be0b0730800755944d3d92509c84c9789bc2"   # 事前登録 第 5 節の値（変更 1、prereg_regen_v1_amend1.md）
PILOT_N = {"n": 10, "A": 10, "C": 10}               # 試しの数（事前登録 第 5 節の (1)）
SHEET_PER_KIND = 2                                   # 目で見る PNG に並べる本数（種類ごと）
SHEET_FRAMES = 8                                     # 1 本あたりのフレームの数


def write_json(p: pathlib.Path, obj: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[regen] wrote {p}", flush=True)


def rel(p) -> str:
    return str(pathlib.Path(p).relative_to(ROOT)).replace("\\", "/")


def sha256_file(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def load_f30():
    """scripts\\recovery_data.py を読み込む（関数と定数を借りるだけ。書き換えない）。"""
    spec = importlib.util.spec_from_file_location("f30_for_regen", ROOT / "scripts" / "recovery_data.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------------------------------------------ 計画

def make_plan() -> dict:
    """recovery_data.py の cmd_plan を、必要数だけ替えて呼ぶ。書き出しの関数を差し替えて、outputs\\f には何も書かない。"""
    F = load_f30()
    got = {}
    F.RECOVERY_NEED = dict(RECOVERY_NEED)
    F.write = lambda name, obj: got.update(obj)
    F.cmd_plan(None)
    return got


def check_plan(plan: dict) -> dict:
    """通常 240 が合わせる前と同じか、候補のシードが SEED_BAND の範囲に収まるか。"""
    old = json.loads(STAGE3_PLAN.read_text(encoding="utf-8"))
    same_normal = plan["normal"]["specs"] == old["normal"]["specs"]
    out = {"normal_n": len(plan["normal"]["specs"]), "normal_same_as_stage3": same_normal,
           "stage3_plan_sha256": sha256_file(STAGE3_PLAN), "by_kind": {}}
    old_cands = {(c["seed"], c["color"], c["layout_kind"], c["start"])
                 for cells in old["recovery"].values() for cell in cells.values() for c in cell["candidates"]}
    old_seeds = {c["seed"] for cells in old["recovery"].values() for cell in cells.values() for c in cell["candidates"]}
    ok_band = True
    for kind, cells in plan["recovery"].items():
        cands = [c for cell in cells.values() for c in cell["candidates"]]
        row = {"need": sum(cell["need"] for cell in cells.values()), "candidates": len(cands),
               "cells": {lk: {"need": cell["need"], "candidates": len(cell["candidates"]),
                              "seeds": [cell["candidates"][0]["seed"], cell["candidates"][-1]["seed"]] if cell["candidates"] else None}
                         for lk, cell in cells.items()}}
        if cands:
            seeds = [c["seed"] for c in cands]
            row["seed_range"] = [min(seeds), max(seeds)]
            lo, hi = SEED_BAND[kind]
            row["in_ledger_band"] = all(lo <= s <= hi for s in seeds)
            ok_band &= row["in_ledger_band"]
            row["new_seeds_not_in_stage3_plan"] = sorted(s for s in seeds if s not in old_seeds)
            row["same_spec_as_stage3_candidate"] = sum((c["seed"], c["color"], c["layout_kind"], c["start"]) in old_cands
                                                       for c in cands)
        out["by_kind"][kind] = row
    out["seeds_in_ledger_bands"] = ok_band
    out["ok"] = same_normal and ok_band
    return out


def cmd_plan(a) -> int:
    plan = make_plan()
    chk = check_plan(plan)
    p = REGEN / "plan.json"
    body = {**{k: v for k, v in plan.items() if k not in ("check", "written")}, "recovery_need": RECOVERY_NEED,
            "made_by": "scripts/recovery_data.py の cmd_plan（RECOVERY_NEED だけ替えた）", "regen_check": chk}
    if p.is_file():                                   # 2 回目以降に呼ばれたとき: 中身が同じなら、そのまま使う
        old = json.loads(p.read_text(encoding="utf-8"))
        body = json.loads(json.dumps(body, ensure_ascii=False, default=str))   # タプルを JSON と同じリストにそろえて比べる
        if old["normal"] != body["normal"] or old["recovery"] != body["recovery"]:
            raise SystemExit(f"{p} がすでにあり、今の計画と違う（消さずに止める）")
        print(f"[regen] {p} はすでにあり、中身は同じ", flush=True)
    else:
        write_json(p, {"check": "regen_plan", "written": time.strftime("%Y-%m-%d %H:%M:%S"), **body})
    print(json.dumps({k: chk[k] for k in ("normal_n", "normal_same_as_stage3", "seeds_in_ledger_bands", "ok")}, ensure_ascii=False))
    for k, r in chk["by_kind"].items():
        print(f"  {k}: 必要 {r['need']}、候補 {r['candidates']}、種 {r.get('seed_range')}、段階 3 の候補にない種 "
              f"{len(r.get('new_seeds_not_in_stage3_plan', []))} 個", flush=True)
    if not chk["ok"]:
        raise SystemExit("計画の確かめが通らない（通常 240 が段階 3 と違う、または種が帯の外）")
    return 0


def load_plan() -> dict:
    p = REGEN / "plan.json"
    if not p.is_file():
        raise SystemExit("先に plan を回す")
    plan = json.loads(p.read_text(encoding="utf-8"))
    if not plan["regen_check"]["ok"]:
        raise SystemExit("plan.json の確かめが通っていない")
    return plan


# ------------------------------------------------------------------------------------ 生成（続きから回せる）

def check_world() -> dict:
    """世界の設定が事前登録の値か（違えば止める）。生成で使う入り・切りを返す。"""
    from recovla.harness import real_matched as V
    sha = sha256_file(config.CONFIG_DIR / "v3_real.yaml")
    if sha != V3_REAL_SHA256:
        raise SystemExit(f"configs/v3_real.yaml の SHA-256 が事前登録と違う: {sha}")
    sw = V.load_config(WORLD, generation=True)["v3_real"]["switches"]
    if sw.get("contact_std") or sw.get("b_slip"):
        raise SystemExit(f"v3_real の既定が事前登録と違う（変更 1 で contact_std 切・b_slip 切のはず）: {sw}")
    return {"config": "configs/v3_real.yaml", "sha256": sha, "switches": sw}


def spec_key(s) -> tuple:
    return (s.kind, s.layout_seed, s.color)


def generate_resumable(specs: list, run: pathlib.Path, workers: int, chunk_groups: int) -> tuple:
    """specs を配置の組（generate.generate の仕事の単位と同じ）ごとにまとめ、chunk_groups 組ずつのかたまりで生成する。
    返り値は（結果の列＝指定の順、かたまりの情報）。"""
    from recovla.expert import generate_real as GR
    run.mkdir(parents=True, exist_ok=True)
    rows = [dataclasses.asdict(s) for s in specs]
    sp = run / "regen_specs.json"
    if sp.is_file():
        if json.loads(sp.read_text(encoding="utf-8"))["specs"] != rows:
            raise SystemExit(f"{sp} の指定が今の指定と違う（続きにならない）")
    else:
        write_json(sp, {"world": WORLD, "workers": workers, "chunk_groups": chunk_groups, "specs": rows})
    groups = {}
    for s in specs:                                    # generate.generate と同じまとめ方（組の最初の画像を揃えるため）
        groups.setdefault((s.layout_seed, s.layout_kind, s.kind, s.start), []).append(s)
    glist = list(groups.values())
    chunks = [sum(glist[i:i + chunk_groups], []) for i in range(0, len(glist), chunk_groups)]
    results, info = {}, []
    for i, cs in enumerate(chunks):
        d = run / f"chunk_{i:02d}"
        t0 = time.perf_counter()
        if (d / "timing.json").is_file():              # 終わっている
            rs = [json.loads(ln) for ln in (d / "generation.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
            state = "done_before"
        else:
            if d.exists():                             # 途中で止まった: 退避して作り直す
                dst = run / f"_incomplete_{d.name}_{time.strftime('%Y%m%d-%H%M%S')}"
                shutil.move(str(d), str(dst))
                print(f"[regen] 途中のかたまりを退避 {d.name} → {dst.name}", flush=True)
            print(f"[regen] かたまり {i + 1}/{len(chunks)}: {len(cs)} 本を生成", flush=True)
            rs = GR.generate(cs, d, world=WORLD, workers=workers, render=True)
            state = "generated"
        if sorted(spec_key(s) for s in cs) != sorted((r["kind"], r["layout_seed"], r["color"]) for r in rs):
            raise SystemExit(f"{d} の結果が指定と合わない")
        for r in rs:
            r["_run"] = rel(d)
            results[(r["kind"], r["layout_seed"], r["color"])] = r
        info.append({"chunk": d.name, "specs": len(cs), "state": state, "wall_s": round(time.perf_counter() - t0, 1),
                     "generate_wall_s": json.loads((d / "timing.json").read_text(encoding="utf-8"))["wall_s"]})
    return [results[spec_key(s)] for s in specs], info


def saved_name(r):
    return r["attempts"][-1]["name"] if r["success"] else None


def cue_args() -> list:
    """合わせる前の学習データ（rig v3）と同じ決め方（outputs/f/data_cue.json の keep_flag_in_input）。"""
    keep_flag = json.loads(DATA_CUE.read_text(encoding="utf-8"))["keep_flag_in_input"]
    return ["--target-cue", "--cue-thresholds", "runtime_v2"] + ([] if keep_flag else ["--cue-without-flag"])


def convert_verify(name: str, entries: list, rule: str) -> dict:
    """マニフェストを書き、変換して照合する。終わっていれば飛ばす。"""
    from recovla.data import convert as C
    mpath = OUTS / "manifests" / f"{name}.json"
    if mpath.is_file():
        if json.loads(mpath.read_text(encoding="utf-8"))["entries"] != entries:
            raise SystemExit(f"{mpath} がすでにあり、中身が違う")
    else:
        C.write_manifest(mpath, name, entries, rule)
    ds = OUTS / "datasets" / name
    log = REGEN / f"convert_{name}.log"
    done = log.is_file() and "[verify] PASS" in log.read_text(encoding="utf-8", errors="replace") and (ds / "meta" / "info.json").is_file()
    t0 = time.perf_counter()
    if not done:
        if ds.exists():                                # 途中の変換: 退避してやり直す
            shutil.move(str(ds), str(ds.parent / f"_incomplete_{name}_{time.strftime('%Y%m%d-%H%M%S')}"))
        args = cue_args()
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "w", encoding="utf-8") as f:
            c1 = subprocess.run([sys.executable, "-m", "recovla.data.convert", "--manifest", str(mpath), "--out", str(ds),
                                 "--name", name, *args], stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
            c2 = subprocess.run([sys.executable, "-m", "recovla.data.convert", "--verify", str(ds)],
                                stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
        codes = (c1.returncode, c2.returncode)
    else:
        codes = (0, 0)
    text = log.read_text(encoding="utf-8", errors="replace")
    info = json.loads((ds / "meta" / "info.json").read_text(encoding="utf-8")) if (ds / "meta" / "info.json").is_file() else {}
    return {"manifest": rel(mpath), "manifest_sha256": sha256_file(mpath), "dataset": rel(ds), "episodes": len(entries),
            "frames": info.get("total_frames"), "convert_exit": codes[0], "verify_exit": codes[1],
            "verify_pass": "[verify] PASS" in text, "cue_convert_args": cue_args(),
            "verify_fails": [ln.strip() for ln in text.splitlines() if ln.strip().startswith("FAIL")],
            "wall_s": round(time.perf_counter() - t0, 1), "skipped_done_before": done, "log": rel(log)}


# ------------------------------------------------------------------------------------ 1 本の性質

def _mats(q: np.ndarray) -> np.ndarray:
    """四元数（w, x, y, z）の列 → 回転行列の列。"""
    import mujoco
    out = np.zeros((len(q), 9))
    for k in range(len(q)):
        mujoco.mju_quat2Mat(out[k], np.asarray(q[k], float))
    return out.reshape(-1, 3, 3)


def _angle(R) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


def _rotvec(R) -> np.ndarray:
    a = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    if a < 1e-9:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * np.sin(a))
    return np.degrees(a * w)


LIFTED_M = 0.005                                     # 立方体が机から上がったとみなす高さ（lift の最初のフレームからの上がり）


def _yaw_off(R) -> float:
    """手の座標で見た立方体の、手の z 軸（指の向き）まわりのずれ [deg]。立方体は 90° ごとに同じ形なので ±45° に折り返す。
    0 なら指の当て板と立方体の面が平行（まっすぐ掴んでいる）。"""
    cols = [R[:, k] for k in range(3)]
    ax = max(cols, key=lambda v: abs(v[2]) < 0.7 and np.hypot(v[0], v[1]))   # 手の xy 面に近い立方体の軸
    a = np.degrees(np.arctan2(ax[1], ax[0]))
    return float((a + 45.0) % 90.0 - 45.0)


def episode_props(ep: pathlib.Path, conv_frames: dict = None) -> dict:
    """保存した 1 本の、フレーム数・長さ・掴んだ後の立方体の向きの変化・掴んだ幅。
    記録の phase の lift は「閉じる指令」から始まる（指が閉じる間を含む）。そこで 2 つに分ける:
      閉じる間: lift の最初のフレームから、立方体が LIFTED_M 上がるフレームの前まで（指が立方体を挟んで向きを揃える間）
      運ぶ間:   立方体が LIFTED_M 上がったフレームから、放す（release）の最初のフレームの前まで。立方体が指先から
                expert.phase.drop_dist_m 以上離れたら、そフレームで
    手の中の回り＝運ぶ間の、手の座標で見た立方体の向きの変化（運ぶ間の最初のフレームとの差）。
    手の向きは記録の ee_quat（センサの関節角からの順運動学）、立方体は記録の cube_quat（真値）。
    掴んだ幅＝運ぶ間の記録の指の幅（fingers の和＝センサの値）の中央値。
    斜めの掴み＝運ぶ間の最初のフレームの、指の向きと立方体の面のずれ（_yaw_off）。"""
    from recovla.common.seeds import COLORS
    from recovla.expert.script import Phase
    meta = json.loads((ep / "meta.json").read_text(encoding="utf-8"))
    z = np.load(ep / "data.npz")
    out = {"name": meta["name"], "n_frames": int(meta["n_frames"]), "duration_s": meta["duration_s"],
           "recorded_s": round(float(z["sim_time"][-1] - z["sim_time"][0]), 3),
           "converted_frames": (conv_frames or {}).get(meta["name"])}
    ph = np.asarray(z["phase"])
    lift = np.flatnonzero(ph == int(Phase.lift))
    rel_ = np.flatnonzero(ph == int(Phase.release))
    out["in_hand"] = None
    if not len(lift) or not len(rel_) or rel_[0] <= lift[0]:
        return out
    t = COLORS.index(meta["target"])
    il, i1 = int(lift[0]), int(rel_[0])
    cz = np.asarray(z["cube_pos"][:, t, 2], float)
    up = np.flatnonzero(cz[il:i1] - cz[il] >= LIFTED_M)
    if not len(up):
        return out
    i0 = il + int(up[0])
    drop = float(CFG["expert"]["phase"]["drop_dist_m"])
    dist = np.linalg.norm(z["cube_pos"][:, t] - z["fingertip"], axis=1)
    gone = np.flatnonzero(dist[i0:i1] >= drop)
    i1h = i0 + int(gone[0]) if len(gone) else i1        # 手から離れたら、そフレームで
    Rh, Rc = _mats(z["ee_quat"][il:i1h]), _mats(z["cube_quat"][il:i1h, t])
    Rrel = np.einsum("nji,njk->nik", Rh, Rc)            # 手の座標の立方体の向き（lift の最初のフレームから）
    k0 = i0 - il
    D = np.einsum("nij,kj->nik", Rrel[k0:], Rrel[k0])    # 運ぶ間の最初のフレームからの回り
    ang = np.array([_angle(x) for x in D])
    rv = np.abs(np.array([_rotvec(x) for x in D]))
    w = np.asarray(z["fingers"], float).sum(axis=1)[i0:i1h]
    ts = np.asarray(z["sim_time"]) - float(z["sim_time"][0])
    out["in_hand"] = {"carry_frames": int(i1h - i0), "left_hand_before_release": bool(len(gone)),
                      "t_close_cmd_s": round(float(ts[il]), 2), "t_lifted_s": round(float(ts[i0]), 2),
                      "rot_closing_deg": round(_angle(Rrel[k0] @ Rrel[0].T), 3),
                      "yaw_off_at_close_cmd_deg": round(_yaw_off(Rrel[0]), 2),
                      "yaw_off_lifted_deg": round(_yaw_off(Rrel[k0]), 2),
                      "rot_max_deg": round(float(ang.max()), 3), "rot_end_deg": round(float(ang[-1]), 3),
                      "rot_max_xyz_deg_hand": [round(float(v), 3) for v in rv.max(axis=0)],
                      "width_median_mm": round(float(np.median(w)) * 1e3, 3),
                      "width_range_mm": [round(float(w.min()) * 1e3, 3), round(float(w.max()) * 1e3, 3)]}
    return out


def conversion_frames(ds_rel: str) -> dict:
    p = config.path(ds_rel) / "meta" / "conversion.json"
    if not p.is_file():
        return {}
    return {s["name"]: s["frames"] for s in json.loads(p.read_text(encoding="utf-8"))["sources"]}


def stage3_lookup() -> tuple:
    """合わせる前（outputs\\f\\data_v3.json）の生成の結果を（種類、シード、色）で引けるようにする。変換後のフレーム数は R1v3・N1v3 から。"""
    d3 = json.loads(STAGE3_DATA.read_text(encoding="utf-8"))
    run = config.path(d3["run"])
    by = {}
    for ln in (run / "generation.jsonl").read_text(encoding="utf-8").splitlines():
        if ln.strip():
            r = json.loads(ln)
            by[(r["kind"], r["layout_seed"], r["color"])] = r
    frames = {}
    for k in ("R1", "N1"):
        frames.update(conversion_frames(d3["datasets"][k]["dataset"]))
    return run, by, frames, d3


# ------------------------------------------------------------------------------------ 試しの 30 本

def pilot_specs(plan: dict) -> list:
    """通常は計画の先頭から 10、A・C は計画の候補の並び（枠の順、各枠の候補の順）の先頭から 10 ずつ。"""
    from recovla.expert import generate as G
    specs = [G.EpisodeSpec(s["seed"], s["color"], s["layout_kind"], "n") for s in plan["normal"]["specs"][:PILOT_N["n"]]]
    for kind in ("A", "C"):
        cands = [(lk, c) for lk, cell in plan["recovery"][kind].items() for c in cell["candidates"]][:PILOT_N[kind]]
        specs += [G.EpisodeSpec(c["seed"], c["color"], lk, kind) for lk, c in cands]
    return specs


def attempt_rows(r: dict) -> list:
    return [{"name": x["name"], "success": x["success"], "failure": x["failure"], "duration_s": x["duration_s"],
             "n_frames": x["n_frames"], "inject_status": (x.get("inject") or {}).get("status"),
             "inject_reason": (x.get("inject") or {}).get("reason")} for x in r["attempts"]]


def _stats(v: list) -> dict:
    v = [x for x in v if x is not None]
    if not v:
        return None
    return {"n": len(v), "mean": round(float(np.mean(v)), 3), "median": round(float(np.median(v)), 3),
            "min": round(float(np.min(v)), 3), "max": round(float(np.max(v)), 3)}


def cmd_pilot(a) -> int:
    from recovla.common import code_version
    from recovla.expert import inject
    from recovla.harness import real_matched as V
    world = check_world()
    plan = load_plan()
    specs = pilot_specs(plan)
    if a.resume:
        run = config.path(a.resume)
        if not run.is_dir():
            raise SystemExit(f"{run} がない")
        stamp = run.name.rsplit("_", 1)[-1]
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        run = GEN / f"V3R_pilot_{stamp}"
        if run.exists():
            raise SystemExit(f"{run} がすでにある")
    t0 = time.perf_counter()
    results, chunks = generate_resumable(specs, run, a.workers, a.chunk_groups)
    gen_wall = time.perf_counter() - t0
    name = f"V3R_pilot_{stamp}"
    entries = [{"run": r["_run"], "key": saved_name(r)} for r in results if r["success"]]
    conv = convert_verify(name, entries, "regen pilot (docs/real/prereg_regen_v1.md 5 (1)): normal 10 + A 10 + C 10, world v3_real")
    frames_new = conversion_frames(conv["dataset"])
    run3, by3, frames3, d3 = stage3_lookup()
    cfg_gen = V.load_config(WORLD, generation=True)
    F = load_f30()
    rows = []
    for s, r in zip(specs, results):
        row = {"kind": s.kind, "seed": s.layout_seed, "color": s.color, "layout_kind": s.layout_kind,
               "success": r["success"], "retries": r["retries"], "attempts": attempt_rows(r)}
        if s.kind in ("A", "C"):
            cats = [F.category(x) for x in r["attempts"]]
            row["attempt_categories"] = cats
            row["not_effective_attempts"] = cats.count("not_effective")
        if r["success"]:
            ep = config.path(r["_run"]) / saved_name(r)
            row["new"] = episode_props(ep, frames_new)
            meta = json.loads((ep / "meta.json").read_text(encoding="utf-8"))
            row["world_in_meta"] = (meta.get("world") or {}).get("world")
            if s.kind in ("A", "C"):
                row["inject_status_saved"] = meta["inject"]["status"]
                row["recovery_duration_s"] = meta.get("recovery_duration_s")
                sc = inject.start_condition(dict(np.load(ep / "data.npz")), meta, cfg_gen)
                row["start_condition"] = sc
                row["start_condition_ok"] = all(v for v in sc.values() if isinstance(v, bool))
        r3 = by3.get((s.kind, s.layout_seed, s.color))
        if r3 is not None:
            row["stage3"] = {"success": r3["success"], "retries": r3["retries"],
                             "attempt_failures": [x["failure"] for x in r3["attempts"]]}
            if r3["success"]:
                row["stage3"].update(episode_props(run3 / saved_name(r3), frames3))
        rows.append(row)

    # 種類ごとのまとめ
    summary = {}
    for kind in ("n", "A", "C"):
        rs = [x for x in rows if x["kind"] == kind]
        ok = [x for x in rs if x["success"]]
        pair = [x for x in ok if x.get("stage3", {}).get("n_frames")]
        sm = {"specs": len(rs), "success": len(ok), "retried": sum(x["retries"] > 0 for x in rs),
              "dropped": sum(not x["success"] for x in rs), "attempts": sum(len(x["attempts"]) for x in rs),
              "attempt_failures": dict(collections.Counter(y["failure"] for x in rs for y in x["attempts"] if y["failure"])),
              "stage3_same_specs": {"success": sum(x.get("stage3", {}).get("success", False) for x in rs),
                                    "retried": sum(x.get("stage3", {}).get("retries", 0) > 0 for x in rs)}}
        if kind in ("A", "C"):
            sm["not_effective_attempts"] = sum(x["not_effective_attempts"] for x in rs)
            sm["saved_inject_confirmed"] = sum(x.get("inject_status_saved") == "confirmed" for x in ok)
            sm["start_condition_ok"] = sum(x.get("start_condition_ok", False) for x in ok)
        sm["raw_frames"] = {"new": _stats([x["new"]["n_frames"] for x in pair]), "stage3": _stats([x["stage3"]["n_frames"] for x in pair])}
        sm["converted_frames"] = {"new": _stats([x["new"]["converted_frames"] for x in pair]),
                                  "stage3": _stats([x["stage3"]["converted_frames"] for x in pair])}
        sm["recorded_s"] = {"new": _stats([x["new"]["recorded_s"] for x in pair]), "stage3": _stats([x["stage3"]["recorded_s"] for x in pair])}
        if pair:
            n_new = sum(x["new"]["converted_frames"] or 0 for x in pair)
            n_old = sum(x["stage3"]["converted_frames"] or 0 for x in pair)
            sm["converted_frames_ratio_new_over_stage3"] = round(n_new / n_old, 4) if n_old else None
            sm["paired_episodes"] = len(pair)
        ih = [x for x in pair if x["new"].get("in_hand") and x["stage3"].get("in_hand")]
        sm["in_hand"] = {
            "paired": len(ih),
            "rot_max_deg": {"new": _stats([x["new"]["in_hand"]["rot_max_deg"] for x in ih]),
                            "stage3": _stats([x["stage3"]["in_hand"]["rot_max_deg"] for x in ih])},
            "rot_end_deg": {"new": _stats([x["new"]["in_hand"]["rot_end_deg"] for x in ih]),
                            "stage3": _stats([x["stage3"]["in_hand"]["rot_end_deg"] for x in ih])},
            "rot_closing_deg": {"new": _stats([x["new"]["in_hand"]["rot_closing_deg"] for x in ih]),
                                "stage3": _stats([x["stage3"]["in_hand"]["rot_closing_deg"] for x in ih])},
            "abs_yaw_off_lifted_deg": {"new": _stats([abs(x["new"]["in_hand"]["yaw_off_lifted_deg"]) for x in ih]),
                                       "stage3": _stats([abs(x["stage3"]["in_hand"]["yaw_off_lifted_deg"]) for x in ih])},
            "width_median_mm": {"new": _stats([x["new"]["in_hand"]["width_median_mm"] for x in ih]),
                                "stage3": _stats([x["stage3"]["in_hand"]["width_median_mm"] for x in ih])},
            "left_hand_before_release": {"new": sum(x["new"]["in_hand"]["left_hand_before_release"] for x in ih),
                                         "stage3": sum(x["stage3"]["in_hand"]["left_hand_before_release"] for x in ih)}}
        summary[kind] = sm
    # 合わせる前の全体（R1v3・N1v3）の 1 本あたりの変換後のフレーム数（確かめ (2) の ±20% の比べの元）
    st3_all = {k: round(d3["datasets"][k]["frames"] / d3["datasets"][k]["episodes"], 2) for k in ("R1", "N1")}
    gate1 = {"convert_verify_pass": conv["convert_exit"] == 0 and conv["verify_exit"] == 0 and conv["verify_pass"],
             "all_30_saved": all(x["success"] for x in rows),
             "A_C_inject_confirmed_and_start_condition": all(summary[k]["saved_inject_confirmed"] == summary[k]["success"]
                                                            and summary[k]["start_condition_ok"] == summary[k]["success"]
                                                            for k in ("A", "C")),
             "visual_check": "人が目で見る（outputs/real/regen/pilot_sheet_*.png）。未判定"}
    out = {"check": "regen_pilot", "written": time.strftime("%Y-%m-%d %H:%M:%S"), "prereg": "docs/real/prereg_regen_v1.md 5 (1)",
           "world": world, "run": rel(run), "chunks": chunks, "generation_wall_s": round(gen_wall, 1), "workers": a.workers,
           "specs": [dataclasses.asdict(s) for s in specs], "convert": conv, "summary": summary,
           "stage3_frames_per_episode_all": st3_all, "stage3_run": rel(run3), "gate1_auto": gate1,
           "code_version": code_version.code_version(), "rows": rows,
           "notes": ["手の中の回り（rot_*）: 立方体が 5 mm 上がったこまから放す前まで（運ぶ間）、手の座標で見た立方体の向きの変化（20 Hz のこま）",
                     "rot_closing_deg: 閉じる指令から立方体が 5 mm 上がるまで（指が挟む間）の、手の座標の立方体の向きの変化",
                     "yaw_off_*: 指の向きと立方体の面のずれ（手の z 軸まわり、±45° に折り返し）。0 ならまっすぐ掴んでいる",
                     "掴んだ幅: 運ぶ間の記録の指の幅（センサの値。段階 3 は 10 Hz・0.1 mm 刻み、v3_real は 6.5 Hz で保つ・0.08 s 前の値）の中央値",
                     "段階 3 の比べは同じ（種類、種、色）の段階 3 のエピソード（outputs/f/data_v3.json の回）"]}
    write_json(REGEN / "pilot.json", out)
    for kind, sm in summary.items():
        print(f"{kind}: 成功 {sm['success']}/{sm['specs']}、作り直し {sm['retried']}、捨てた {sm['dropped']}、"
              f"変換後のこま数の比（新/段階 3）{sm.get('converted_frames_ratio_new_over_stage3')}", flush=True)
    print(f"変換 {conv['convert_exit']}・照合 {'PASS' if conv['verify_pass'] else 'FAIL'}", flush=True)
    make_sheets(out)
    return 0 if gate1["convert_verify_pass"] and gate1["all_30_saved"] else 1


# ------------------------------------------------------------------------------------ 本番（Rreal・Nreal のデータ）

DROPPED_STOP = 0.10                                  # 捨てた指定の割合がこれを超えたら止める（合わせる前と同じ）
FRAME_TOL = 0.20                                     # 確かめ (2): フレーム数が合わせる前の ±20% に入る


def production_specs(plan: dict) -> tuple:
    """生成する並び: 通常 240、立て直しの候補の全部、候補と同じ（シード、色）の通常（相手）。"""
    from recovla.expert import generate as G
    normal = [G.EpisodeSpec(s["seed"], s["color"], s["layout_kind"], "n") for s in plan["normal"]["specs"]]
    rec_specs, twin_specs, cells = [], [], []
    for kind, lk_cells in plan["recovery"].items():
        for lk, cell in lk_cells.items():
            if not cell["candidates"]:                 # B（必要数 0）は候補なし
                continue
            cells.append((kind, lk, cell["need"], cell["candidates"]))
            for c in cell["candidates"]:
                rec_specs.append(G.EpisodeSpec(c["seed"], c["color"], lk, kind))
                twin_specs.append(G.EpisodeSpec(c["seed"], c["color"], lk, "n"))
    return normal, rec_specs, twin_specs, cells


def choose(normal, cells, by) -> tuple:
    """採り方: 立て直しと相手の通常の両方が成功した候補を、枠ごとに候補の順に必要数まで。"""
    normal_ok = [by[("n", s.layout_seed, s.color)] for s in normal]
    normal_keys = [(r["_run"], saved_name(r)) for r in normal_ok if r["success"]]
    chosen, cell_rows = [], []
    for kind, lk, need, cands in cells:
        got, examined, dropped = [], 0, 0
        for c in cands:
            if len(got) >= need:
                break
            rr, tr = by[(kind, c["seed"], c["color"])], by[("n", c["seed"], c["color"])]
            examined += 1
            if rr["success"] and tr["success"]:
                got.append({"kind": kind, "layout_kind": lk, "seed": c["seed"], "color": c["color"], "start": c["start"],
                            "recovery": [rr["_run"], saved_name(rr)], "twin": [tr["_run"], saved_name(tr)]})
            else:
                dropped += 1
        chosen += got
        cell_rows.append({"kind": kind, "layout_kind": lk, "need": need, "got": len(got), "examined": examined,
                          "dropped": dropped, "short": need - len(got)})
    return normal_ok, normal_keys, chosen, cell_rows


def _comp(rows) -> dict:
    return {"n": len(rows), "color": dict(collections.Counter(r["color"] for r in rows)),
            "layout_kind": dict(collections.Counter(r["layout_kind"] for r in rows)),
            "start": dict(collections.Counter(r["start"] for r in rows))}


def data_props(normal_ok: list, chosen: list, frames: dict) -> dict:
    """学習データの性質（事前登録 第 3 節）: 通常 240 と採った立て直し・相手の、フレーム数・手の中の回り・掴んだ幅を、合わせる前の
    同じ指定と並べる。"""
    run3, by3, frames3, _ = stage3_lookup()
    groups = {"normal": [r for r in normal_ok if r["success"]],
              "A": [], "C": [], "twin": []}
    for c in chosen:
        groups[c["kind"]].append({"_run": c["recovery"][0], "_name": c["recovery"][1], "kind": c["kind"], "layout_seed": c["seed"], "color": c["color"]})
        groups["twin"].append({"_run": c["twin"][0], "_name": c["twin"][1], "kind": "n", "layout_seed": c["seed"], "color": c["color"]})
    out = {}
    for g, rs in groups.items():
        new, old = [], []
        for r in rs:
            name = r.get("_name") or saved_name(r)
            new.append(episode_props(config.path(r["_run"]) / name, frames))
            r3 = by3.get((r["kind"], r["layout_seed"], r["color"]))
            old.append(episode_props(run3 / saved_name(r3), frames3) if r3 and r3["success"] else None)
        ih = [(a["in_hand"], b["in_hand"] if b else None) for a, b in zip(new, old)]
        row = {"n": len(new), "stage3_same_spec_found": sum(b is not None for b in old)}
        for key in ("converted_frames", "recorded_s"):
            row[key] = {"new": _stats([a[key] for a in new]), "stage3_same_spec": _stats([b[key] for b in old if b])}
        for key in ("rot_max_deg", "rot_closing_deg", "width_median_mm"):
            row[key] = {"new": _stats([a[key] for a, _ in ih if a]), "stage3_same_spec": _stats([b[key] for _, b in ih if b])}
        row["abs_yaw_off_lifted_deg"] = {"new": _stats([abs(a["yaw_off_lifted_deg"]) for a, _ in ih if a]),
                                         "stage3_same_spec": _stats([abs(b["yaw_off_lifted_deg"]) for _, b in ih if b])}
        row["yaw_off_lifted_over_5deg"] = {"new": sum(abs(a["yaw_off_lifted_deg"]) > 5 for a, _ in ih if a),
                                           "stage3_same_spec": sum(abs(b["yaw_off_lifted_deg"]) > 5 for _, b in ih if b)}
        row["left_hand_before_release"] = {"new": sum(a["left_hand_before_release"] for a, _ in ih if a),
                                           "stage3_same_spec": sum(b["left_hand_before_release"] for _, b in ih if b)}
        out[g] = row
    return out


def cmd_gen_data(a) -> int:
    """本番: 生成（続きから回せる）→ 採る → Rreal・Nreal のデータのマニフェスト → 変換と照合 → 確かめ (2) → data.json。"""
    import concurrent.futures
    from recovla.common import code_version
    world = check_world()
    plan = load_plan()
    normal, rec_specs, twin_specs, cells = production_specs(plan)
    specs = normal + rec_specs + twin_specs
    if a.resume:
        run = config.path(a.resume)
        if not run.is_dir():
            raise SystemExit(f"{run} がない")
        stamp = run.name.rsplit("_", 1)[-1]
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        run = GEN / f"V3R_data_{stamp}"
        if run.exists():
            raise SystemExit(f"{run} がすでにある")
    print(f"[regen] 本番の生成: 通常 {len(normal)}・復帰の候補 {len(rec_specs)}・相手の通常 {len(twin_specs)} → {rel(run)}", flush=True)
    t0 = time.perf_counter()
    results, chunks = generate_resumable(specs, run, a.workers, a.chunk_groups)
    gen_wall = time.perf_counter() - t0
    by = {(r["kind"], r["layout_seed"], r["color"]): r for r in results}
    normal_ok, normal_keys, chosen, cell_rows = choose(normal, cells, by)

    # 捨てた割合（合わせる前と同じ: 種類ごとの、調べた候補のうち捨てた数。通常 240 も数える）
    dropped_by_kind = {}
    for k in sorted({c[0] for c in cells}):
        rows_k = [r for r in cell_rows if r["kind"] == k]
        ex, dr = sum(r["examined"] for r in rows_k), sum(r["dropped"] for r in rows_k)
        dropped_by_kind[k] = {"dropped": dr, "examined": ex, "ratio": round(dr / ex, 4) if ex else None}
    n_drop = len(normal) - len(normal_keys)
    dropped_by_kind["normal"] = {"dropped": n_drop, "examined": len(normal), "ratio": round(n_drop / len(normal), 4)}
    stop_drop = any(v["ratio"] is not None and v["ratio"] > DROPPED_STOP for v in dropped_by_kind.values())

    entries_r = [{"run": r, "key": k} for r, k in normal_keys] + [{"run": c["recovery"][0], "key": c["recovery"][1]} for c in chosen]
    entries_n = [{"run": r, "key": k} for r, k in normal_keys] + [{"run": c["twin"][0], "key": c["twin"][1]} for c in chosen]
    rules = {"R": "regen new R (docs/real/prereg_regen_v1.md 5 + amend1): normal 240 (seeds 20000-) + recovery A 60 / C 30 "
                  "(seeds 30000- / 32000-), world v3_real (contact_std off), outputs/real/regen/plan.json",
             "N": "regen new N: the same normal 240 + one-shot normal demos on the same 90 recovery layouts and targets, world v3_real"}
    names = {"R": f"R_v3r_{stamp}", "N": f"N_v3r_{stamp}"}
    datasets = {}
    if stop_drop:
        print(f"[regen] 捨てた割合が {DROPPED_STOP:.0%} を超えた。変換しないで止める: {dropped_by_kind}", flush=True)
    else:
        with concurrent.futures.ThreadPoolExecutor(2) as ex:            # Rreal と Nreal のデータの変換を並べて回す
            futs = {k: ex.submit(convert_verify, names[k], ent, rules[k]) for k, ent in (("R", entries_r), ("N", entries_n))}
            datasets = {k: f.result() for k, f in futs.items()}

    # 確かめ (2)（事前登録 第 5 節）
    d3 = json.loads(STAGE3_DATA.read_text(encoding="utf-8"))
    base = {"R": d3["datasets"]["R1"], "N": d3["datasets"]["N1"]}
    gate = {"discard_rate_le_10pct": not stop_drop, "dropped_by_kind": dropped_by_kind,
            "recovery_90_present": sum(r["got"] for r in cell_rows) == sum(RECOVERY_NEED.values()) and all(r["short"] == 0 for r in cell_rows),
            "recovery_by_kind": {k: sum(1 for c in chosen if c["kind"] == k) for k in ("A", "C")}}
    for k, ds in datasets.items():
        ratio = ds["frames"] / base[k]["frames"] if ds.get("frames") else None
        gate[f"{k}_verify_pass"] = ds["convert_exit"] == 0 and ds["verify_exit"] == 0 and ds["verify_pass"]
        gate[f"{k}_episodes"] = ds["episodes"]
        gate[f"{k}_frames"] = {"new": ds.get("frames"), "stage3": base[k]["frames"], "stage3_dataset": base[k]["dataset"],
                               "ratio": round(ratio, 4) if ratio else None}
        gate[f"{k}_frames_within_20pct"] = ratio is not None and abs(ratio - 1.0) <= FRAME_TOL
    gate["same_layouts_and_targets"] = sorted((c["seed"], c["color"]) for c in chosen) == sorted(
        (int(e["key"].split("_")[1]), e["key"].split("_")[2]) for e in entries_n[len(normal_keys):])
    gate["normal_part_identical"] = entries_r[:len(normal_keys)] == entries_n[:len(normal_keys)]
    gate["episodes_330_each"] = len(entries_r) == 330 and len(entries_n) == 330
    keys = ["discard_rate_le_10pct", "recovery_90_present", "R_verify_pass", "N_verify_pass", "R_frames_within_20pct",
            "N_frames_within_20pct", "same_layouts_and_targets", "normal_part_identical", "episodes_330_each"]
    gate["pass"] = all(gate.get(k) for k in keys)
    gate["checked"] = keys

    props = data_props(normal_ok, chosen, {**conversion_frames(datasets["R"]["dataset"]), **conversion_frames(datasets["N"]["dataset"])}) \
        if datasets else None
    out = {"check": "regen_data", "written": time.strftime("%Y-%m-%d %H:%M:%S"),
           "prereg": ["docs/real/prereg_regen_v1.md 5", "docs/real/prereg_regen_v1_amend1.md"], "world": world,
           "run": rel(run), "chunks": chunks, "generation_wall_s": round(gen_wall, 1),
           "generation_wall_s_sum_chunks": round(sum(c["generate_wall_s"] for c in chunks), 1), "workers": a.workers,
           "n_specs": {"normal": len(normal), "recovery_candidates": len(rec_specs), "twins": len(twin_specs), "total": len(specs)},
           "attempts_total": sum(len(r["attempts"]) for r in results),
           "attempt_failures": dict(collections.Counter(x["failure"] for r in results for x in r["attempts"] if x["failure"])),
           "normal": {"composition": _comp([s for s in plan["normal"]["specs"]
                                            if any(r["layout_seed"] == s["seed"] and r["color"] == s["color"] and r["success"] for r in normal_ok)])},
           "recovery": {"cells": cell_rows, "composition": _comp(chosen),
                        "by_kind": {k: _comp([c for c in chosen if c["kind"] == k]) for k in ("A", "C")}},
           "chosen": chosen, "names": names, "datasets": datasets, "gate2": gate, "data_props": props,
           "dropped_stop_rule": f"捨てた割合が {DROPPED_STOP:.0%} を超えたら止める（段階 3 と同じ）",
           "code_version": code_version.code_version()}
    write_json(REGEN / "data.json", out)
    print(f"[regen] データの関門 (2): {'通過' if gate['pass'] else '外れ'}", flush=True)
    for k in keys:
        print(f"  {k}: {gate.get(k)}", flush=True)
    return 0 if gate["pass"] else 1


# ------------------------------------------------------------------------------------ 目で見る PNG

def make_sheets(pilot: dict) -> list:
    """種類ごとに 2 本。1 本につき俯瞰と手首を、時間に沿って 8 フレームずつ横に並べた 1 枚の PNG。"""
    import cv2
    from recovla.record.recorder import read_png
    made = []
    for kind in ("n", "A", "C"):
        picks, seen = [], set()
        for x in pilot["rows"]:                        # 違う配置から先頭の 2 本
            if x["kind"] == kind and x["success"] and x["seed"] not in seen:
                picks.append(x)
                seen.add(x["seed"])
            if len(picks) >= SHEET_PER_KIND:
                break
        blocks = []
        for x in picks:
            name = x["new"]["name"]
            run = next(e["run"] for e in json.loads((ROOT / pilot["convert"]["manifest"]).read_text(encoding="utf-8"))["entries"]
                       if e["key"] == name)
            ep = config.path(run) / name
            z = np.load(ep / "data.npz")
            ts = np.asarray(z["sim_time"]) - float(z["sim_time"][0])
            idx = np.linspace(0, len(ts) - 1, SHEET_FRAMES).round().astype(int)
            title = np.full((28, 256 * SHEET_FRAMES, 3), 255, np.uint8)
            cv2.putText(title, f"{name}  ({x['layout_kind']})  top: overhead / bottom: wrist", (6, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            rows_ = [title]
            for cam in ("overhead", "wrist"):
                tiles = []
                for i in idx:
                    im = read_png(ep / cam / f"{i:06d}.png").copy()
                    if cam == "overhead":
                        cv2.putText(im, f"t={ts[i]:.1f}s", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                    tiles.append(im)
                rows_.append(np.hstack(tiles))
            blocks.append(np.vstack(rows_))
            blocks.append(np.full((10, 256 * SHEET_FRAMES, 3), 255, np.uint8))
        if not blocks:
            continue
        img = np.vstack(blocks[:-1])
        p = REGEN / f"pilot_sheet_{kind}.png"
        ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        buf.tofile(str(p))
        made.append(rel(p))
        print(f"[regen] wrote {p}（{img.shape[1]}x{img.shape[0]}）", flush=True)
    return made


def cmd_sheet(a) -> int:
    make_sheets(json.loads((REGEN / "pilot.json").read_text(encoding="utf-8")))
    return 0


def main(argv=None) -> int:
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    s = sub.add_parser("pilot")
    s.add_argument("--workers", type=int, default=6)
    s.add_argument("--chunk-groups", type=int, default=12, help="1 つのかたまりに入れる配置の組の数")
    s.add_argument("--resume", default=None, help="止まった回のフォルダ（outputs/gen/V3R_pilot_<日時>）")
    sub.add_parser("sheet")
    s = sub.add_parser("gen-data")
    s.add_argument("--workers", type=int, default=6)
    s.add_argument("--chunk-groups", type=int, default=40, help="1 つのかたまりに入れる配置の組の数")
    s.add_argument("--resume", default=None, help="止まった回のフォルダ（outputs/gen/V3R_data_<日時>）")
    a = ap.parse_args(argv)
    return {"plan": cmd_plan, "pilot": cmd_pilot, "sheet": cmd_sheet, "gen-data": cmd_gen_data}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
