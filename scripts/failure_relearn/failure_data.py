"""自分の失敗から学び直すデータ（docs/api/r1_protocol.md 2・3・4.5 節）を作る: 交代後のデータ・外乱のデータ・箱に 2 個入った
場面の成功データの生成、変換と照合、目で確かめる PNG。

    python scripts/failure_relearn/failure_data.py pilot [--workers 6]                       # 試し: 交代 30 本（GPU）＋ B・D・P 各 5 → 変換と照合 → PNG
    python scripts/failure_relearn/failure_data.py handover --band H_A --quota 100            # 交代後のデータ（PC A）。続きから回せる（同じ --out）
    python scripts/failure_relearn/failure_data.py handover --band H_B --quota 100            # 交代後のデータ（PC B）
    python scripts/failure_relearn/failure_data.py demos --kind B --quota 40 [--workers 6]    # 外乱 B（運ぶ途中の落下）。D・P も同じ
    python scripts/failure_relearn/failure_data.py data [--handover-runs <フォルダ…>]          # 採る → マニフェスト → 変換と照合 → data.json
    python scripts/failure_relearn/failure_data.py sheet --pilot                              # PNG だけ作り直す
    python scripts/failure_relearn/failure_data.py bands                                      # シードの範囲と目標の本数を出すだけ

種類とシードの範囲（BANDS。どれも学習用で 100000 未満）:
  H 交代（VLA の Rreal で動かし、失敗の直後に生成プログラムへ切り替える）  試し 70000〜70099、PC A 70100〜71049・PC B 71050〜71999
  B 運ぶ途中の落下（外乱）               試し 72000〜72019、データ 72020〜72199
  D 掴む前に目標を動かす（外乱）          試し 73000〜73019、データ 73020〜73199
  P 箱に 2 個入った場面から始まる成功データ  試し 74000〜74019、データ 74020〜74199
VLA の実行の設定は評価（scripts/api/ladder.py の U0 の 1 手順）と同じ: 実機の Panda で測った応答に合わせたシミュレーション
（v3_real）の評価の設定、実行の方式 naive（RTC なし）・6 ステップごとに推論、安全フィルタなし、開閉の絞り K=5、遅延は
robot_io.fixed_latency。実行器（TaskRuntime）は使わず PolicyRuntime だけ（サブタスク 1 つ）。
変換と照合は Rreal と同じ引数（--target-cue --cue-thresholds runtime_v2 --cue-without-flag）。
書くもの: outputs/gen/R1_*（生成の回）、outputs/manifests・outputs/datasets（変換）、outputs/r1/*.json・*.png（まとめ）。
"""
import argparse
import collections
import contextlib
import dataclasses
import datetime
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from recovla.common import config  # noqa: E402

CFG = config.load()
os.environ.setdefault("HF_HOME", str(config.path(CFG["paths"]["models_home"])))
os.environ["HF_HUB_OFFLINE"] = "1"
OUTS = config.path(CFG["paths"]["outputs"])
GEN = OUTS / "gen"
R1 = OUTS / "r1"
WORLD = "v3_real"
RREAL = "outputs/real/regen/train/train_Rreals1000_20261014-082146_20261014-082146/checkpoints/020000/pretrained_model"
RREAL_SHA256 = "620eb010333341014e742701356673cdf6362f04d0caccaef05f48780f150c62"     # docs/real/regen_models.md
RREAL_MANIFEST = "outputs/manifests/R_v3r_20261014-055653.json"                       # Rreal の 330 本
RREAL_RUN = "outputs/gen/V3R_data_20261014-055653"
MAIN_REPO = pathlib.Path(r"C:\PAI\recovery_vla")                                       # Rreal の学習データの置き場所（読むだけ）
GRIP_DEBOUNCE = 5
EXEC_INTERVAL = 6
CUE_ARGS = ["--target-cue", "--cue-thresholds", "runtime_v2", "--cue-without-flag"]   # outputs/f/data_cue.json の keep_flag_in_input = false
BANDS = {"H_pilot": (70000, 70099), "H_A": (70100, 71049), "H_B": (71050, 71999),
         "B_pilot": (72000, 72019), "B": (72020, 72199), "D_pilot": (73000, 73019), "D": (73020, 73199),
         "P_pilot": (74000, 74019), "P": (74020, 74199)}
TARGET = {"H": 200, "H_min": 150, "B": 40, "D": 40, "P": 60}
PILOT = {"H": 30, "B": 5, "D": 5, "P": 5}
DEMO_KIND = {"B": "B", "D": "D", "P": "n"}           # 生成の種類（P は 2 個入りから始まる通常の生成 n）
RECOVERY_SHARE_MAX = 0.50
SHEET_FRAMES = 8


def write_json(p: pathlib.Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[r1] wrote {p}", flush=True)


def rel(p) -> str:
    return str(pathlib.Path(p).resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def sha256_file(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def seeds_of(band: str) -> range:
    lo, hi = BANDS[band]
    return range(lo, hi + 1)


def check_band(lo: int, hi: int) -> None:
    """学習用のシード（100000 未満）だけ。評価のシードは使わない。"""
    if not (0 <= lo <= hi < 100000):
        raise SystemExit(f"種 {lo}〜{hi} は学習用の帯（100000 未満）の外")


# ------------------------------------------------------------------------------------ 方策の実行系
def debounce_gate(k: int) -> dict:
    return {"enabled": True, "close": 0.0, "open": 0.0, "min_hold_s": 0.0, "open_steps": int(k)}


def eval_config() -> dict:
    """評価の設定（v3_real の評価の層 + 開閉の絞り）。"""
    from recovla.harness import real_matched as V
    cfg = V.load_config(WORLD, generation=False)
    cfg["runtime_v2"]["gripper_gate"] = debounce_gate(GRIP_DEBOUNCE)
    return cfg


def runtime_factory(ckpt: str, cfg: dict):
    """評価と同じ PolicyRuntime を作る関数（naive・6 ステップごとに推論・安全フィルタなし・開閉の絞り K=5）。"""
    from recovla.runtime import cue as C
    from recovla.runtime.motion import Motion
    from recovla.runtime.perception import Params, Perception
    from recovla.runtime.policy import SensorPolicy
    from recovla.runtime.runner import PolicyRuntime, disable_rtc_for
    rt_cfg, act, rtv = cfg["runtime"], cfg["actuation"], cfg["runtime_v2"]
    thr = C.Thresholds.from_dict(config.color_detect(cfg))
    cache = {}

    def make_runtime(io, setup):
        if "pol" not in cache:
            cache["pol"] = SensorPolicy(config.path(ckpt), setup, Motion(setup).hand_pose)
            disable_rtc_for(cache["pol"])
        pol = cache["pol"]
        pol.setup = setup
        if pol.cue is not None:
            pol.cue = C.TargetCue(C.calibration_from_setup(setup.cameras["overhead"]), pol.cue.thr,
                                  setup.table_z + 0.5 * setup.cube_size, setup.cue_fallback_xy)
        per = Perception(setup, Params.from_config(rtv["perception"]), thr)
        return PolicyRuntime(io, setup, pol, perception=per, safety=None, checks=rtv["checks"],
                             gripper_gate=rtv.get("gripper_gate"), tip_offset=float(cfg["sim"]["fingertip_offset"]),
                             mode="naive", s=EXEC_INTERVAL, d_init=int(rt_cfg["delay_steps"]),
                             motion=Motion(setup, margin=float(act["limiter_margin"]), xcmd_leash_m=rtv.get("xcmd_leash_m"),
                                           cart_margin=rtv.get("cart_margin")))
    return make_runtime


def check_ckpt(ckpt: str) -> str:
    p = config.path(ckpt) / "model.safetensors"
    if not p.is_file():
        raise SystemExit(f"保存点がない: {p}")
    sha = sha256_file(p)
    if sha != RREAL_SHA256:
        raise SystemExit(f"Rreal の保存点の SHA-256 が記録と違う: {sha}")
    return sha


# ------------------------------------------------------------------------------------ 交代後のデータの収集
def read_jsonl(p: pathlib.Path) -> list:
    if not p.is_file():
        return []
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def collect_handover(seed_list, quota: int, run: pathlib.Path, until: str = None) -> dict:
    """交代をシードの順に回す。run があれば、終わったシードを飛ばして続ける。quota 本保存したかシードが尽きたら終わる。"""
    from recovla.common import code_version
    from recovla.expert.handover_real import run_r1_handover
    from recovla.harness import real_matched as V
    check_band(min(seed_list), max(seed_list))
    sha = check_ckpt(RREAL)
    cfg_eval = eval_config()
    make_runtime = runtime_factory(RREAL, cfg_eval)
    run.mkdir(parents=True, exist_ok=True)
    log_p = run / "handover.jsonl"
    done = read_jsonl(log_p)
    seen = {r["seed"] for r in done}
    got = sum(bool(r["saved"]) for r in done)
    sp = run / "start.json"
    start = {"model": "Rreal", "checkpoint": RREAL, "checkpoint_sha256": sha, "world": WORLD, "grip_debounce": GRIP_DEBOUNCE,
             "exec_interval": EXEC_INTERVAL, "mode": "naive", "safety": False, "quota": quota,
             "seeds": [min(seed_list), max(seed_list)], "code_version": code_version.code_version()}
    if sp.is_file():
        old = json.loads(sp.read_text(encoding="utf-8"))
        if {k: old.get(k) for k in ("checkpoint_sha256", "world", "grip_debounce", "seeds")} != \
                {k: start[k] for k in ("checkpoint_sha256", "world", "grip_debounce", "seeds")}:
            raise SystemExit(f"{sp} と今回の設定が違う（続きにならない）")
    else:
        write_json(sp, dict(start, started=time.strftime("%Y-%m-%d %H:%M:%S")))
    deadline = datetime.datetime.strptime(until, "%Y-%m-%d %H:%M") if until else None
    rig = V.V3RealGenRig(V.load_config(WORLD, generation=True))
    t0, n = time.perf_counter(), 0
    try:
        with open(log_p, "a", encoding="utf-8") as log:
            for seed in seed_list:
                if got >= quota:
                    break
                if deadline and datetime.datetime.now() >= deadline:
                    print(f"[r1] 打ち切りの時刻 {until} に達した", flush=True)
                    break
                if seed in seen:
                    continue
                out = run_r1_handover(rig, make_runtime, seed, run, "Rreal", cfg_eval)
                n += 1
                got += int(out["saved"])
                log.write(json.dumps(out, ensure_ascii=False, default=float) + "\n")
                log.flush()
                h = out["handover"]
                print(f"[r1h] {seed} {out['color']:5s} {out['rollout_layout']['stratum']:4s} trigger {h['trigger']}/{h['detail']} "
                      f"t {h['t_handover']:5.1f} saved {out['saved']} discard {out['discard']} got {got}/{quota} "
                      f"wall {out['wall_s']:.0f}s", flush=True)
    finally:
        rig.close()
    rows = read_jsonl(log_p)
    summ = summarize_handover(rows)
    write_json(run / "run.json", {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "attempts_this_call": n,
                                  "wall_s_this_call": round(time.perf_counter() - t0, 1), **summ})
    return summ


def summarize_handover(rows: list) -> dict:
    saved = [r for r in rows if r["saved"]]
    trig_all = collections.Counter(str(r["handover"]["trigger"]) for r in rows)
    trig_saved = collections.Counter(r["handover"]["trigger"] for r in saved)
    walls = [r["wall_s"] for r in rows]
    by_st = {}
    for st in ("pf2", "yaw", "near", "free"):
        rs = [r for r in rows if r["rollout_layout"]["stratum"] == st]
        by_st[st] = {"attempts": len(rs), "triggered": sum(r["handover"]["trigger"] is not None for r in rs),
                     "saved": sum(r["saved"] for r in rs)}
    return {"attempts": len(rows), "saved": len(saved),
            "discards": dict(collections.Counter(r["discard"] for r in rows if not r["saved"])),
            "details": dict(collections.Counter(f"{r['handover']['trigger']}/{r['handover']['detail']}" for r in rows)),
            "triggers_all": dict(trig_all), "triggers_saved": dict(trig_saved), "by_stratum": by_st,
            "wall_s": {"sum": round(sum(walls), 1), "mean": round(float(np.mean(walls)), 1) if walls else None,
                       "median": round(float(np.median(walls)), 1) if walls else None},
            "wall_per_saved_s": round(sum(walls) / len(saved), 1) if saved else None,
            "recorded_frames_20hz": sum(r["n_frames"] for r in saved),
            "recovery_duration_s": {"median": round(float(np.median([r["recovery_duration_s"] for r in saved])), 2)} if saved else None}


def cmd_handover(a) -> int:
    i, n = (int(v) for v in a.shard.split("/"))
    lo = BANDS[a.band][0]
    seeds = [s for s in seeds_of(a.band) if (s - lo) % n == i]      # 1 台で 2 本並べるときは --shard 0/2・1/2
    tag = "" if n == 1 else f"_s{i}of{n}"
    run = config.path(a.out) if a.out else GEN / f"R1H_{a.band}{tag}_{time.strftime('%Y%m%d-%H%M%S')}"
    summ = collect_handover(seeds, a.quota, run, a.until)
    print(json.dumps({k: summ[k] for k in ("attempts", "saved", "triggers_saved", "discards")}, ensure_ascii=False))
    print(run)
    return 0


# ------------------------------------------------------------------------------------ 外乱・2 個入りの生成
def demo_specs(kind: str, band: str) -> list:
    """1 つのシードに 1 本（目標は机上の色からシードで決める）。P は 2 個入り（机上は 1 色）。B・D の配置の種類は既定の割合。"""
    from recovla.expert import generate as G
    from recovla.sim import scene
    out = []
    for s in seeds_of(band):
        lk = "prefilled_2" if kind == "P" else None
        lay = scene.sample_layout(s, lk)
        color = lay.table_colors[s % len(lay.table_colors)]
        out.append(G.EpisodeSpec(int(s), color, lk, DEMO_KIND[kind]))
    return out


def generate_quota(kind: str, band: str, quota: int, run: pathlib.Path, workers: int, chunk: int) -> list:
    """指定を chunk 本ずつ生成し、保存が quota 本になるまで続ける。終わったかたまり（timing.json）は読み直す。"""
    from recovla.expert import generate_real as GR
    specs = demo_specs(kind, band)
    check_band(specs[0].layout_seed, specs[-1].layout_seed)
    run.mkdir(parents=True, exist_ok=True)
    results = []
    for i in range(0, len(specs), chunk):
        if sum(r["success"] for r in results) >= quota:
            break
        d = run / f"chunk_{i // chunk:02d}"
        cs = specs[i:i + chunk]
        if (d / "timing.json").is_file():
            rs = read_jsonl(d / "generation.jsonl")
        else:
            if d.exists():
                shutil.move(str(d), str(run / f"_incomplete_{d.name}_{time.strftime('%Y%m%d-%H%M%S')}"))
            print(f"[r1] {kind}: かたまり {d.name}（種 {cs[0].layout_seed}〜{cs[-1].layout_seed}）", flush=True)
            rs = GR.generate(cs, d, world=WORLD, workers=workers, render=True)
        for r in rs:
            r["_run"] = rel(d)
        results += rs
    return results


def saved_entries(results: list, quota: int) -> list:
    out = [{"run": r["_run"], "key": r["attempts"][-1]["name"]} for r in results if r["success"]]
    return out[:quota]


def demo_summary(results: list) -> dict:
    att = [x for r in results for x in r["attempts"]]
    return {"specs": len(results), "saved": sum(r["success"] for r in results), "attempts": len(att),
            "attempt_failures": dict(collections.Counter(x["failure"] for x in att if x["failure"])),
            "inject_status": dict(collections.Counter((x.get("inject") or {}).get("status") for x in att if x.get("inject"))),
            "disturb_status": dict(collections.Counter((x.get("disturb") or {}).get("status") for x in att if x.get("disturb"))),
            "wall_s_attempts": round(sum(x["wall_s"] for x in att), 1)}


def cmd_demos(a) -> int:
    band = a.band or a.kind
    run = config.path(a.out) if a.out else GEN / f"R1{a.kind}_{band}_{time.strftime('%Y%m%d-%H%M%S')}"
    t0 = time.perf_counter()
    res = generate_quota(a.kind, band, a.quota, run, a.workers, a.chunk)
    summ = dict(demo_summary(res), wall_s=round(time.perf_counter() - t0, 1), kind=a.kind, band=band, quota=a.quota)
    write_json(run / "demos.json", summ)
    print(json.dumps(summ, ensure_ascii=False))
    return 0


# ------------------------------------------------------------------------------------ 変換と照合
def convert_verify(name: str, entries: list, rule: str) -> dict:
    from recovla.data import convert as C
    mpath = OUTS / "manifests" / f"{name}.json"
    if not mpath.is_file():
        C.write_manifest(mpath, name, entries, rule)
    elif json.loads(mpath.read_text(encoding="utf-8"))["entries"] != entries:
        raise SystemExit(f"{mpath} がすでにあり、中身が違う")
    ds = OUTS / "datasets" / name
    log = R1 / f"convert_{name}.log"
    done = log.is_file() and "[verify] PASS" in log.read_text(encoding="utf-8", errors="replace") and (ds / "meta" / "info.json").is_file()
    t0 = time.perf_counter()
    codes = (0, 0)
    if not done:
        if ds.exists():
            shutil.move(str(ds), str(ds.parent / f"_incomplete_{name}_{time.strftime('%Y%m%d-%H%M%S')}"))
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "w", encoding="utf-8") as f:
            c1 = subprocess.run([sys.executable, "-m", "recovla.data.convert", "--manifest", str(mpath), "--out", str(ds),
                                 "--name", name, *CUE_ARGS], stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
            c2 = subprocess.run([sys.executable, "-m", "recovla.data.convert", "--verify", str(ds)],
                                stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
        codes = (c1.returncode, c2.returncode)
    text = log.read_text(encoding="utf-8", errors="replace")
    info = json.loads((ds / "meta" / "info.json").read_text(encoding="utf-8")) if (ds / "meta" / "info.json").is_file() else {}
    return {"manifest": rel(mpath), "manifest_sha256": sha256_file(mpath), "dataset": rel(ds), "episodes": len(entries),
            "frames": info.get("total_frames"), "convert_exit": codes[0], "verify_exit": codes[1],
            "verify_pass": "[verify] PASS" in text, "cue_convert_args": CUE_ARGS,
            "verify_fails": [ln.strip() for ln in text.splitlines() if ln.strip().startswith("FAIL")],
            "wall_s": round(time.perf_counter() - t0, 1), "log": rel(log)}


def frames_by_kind(ds_rel: str) -> dict:
    p = config.path(ds_rel) / "meta" / "conversion.json"
    by, n = collections.Counter(), collections.Counter()
    for s in json.loads(p.read_text(encoding="utf-8"))["sources"]:
        k = s["name"].split("_")[0]
        by[k] += int(s["frames"])
        n[k] += 1
    return {k: {"episodes": n[k], "frames": by[k], "per_episode": round(by[k] / n[k], 1)} for k in by}


RECOVERY_KINDS = ("A", "C", "H", "B")        # 立て直しデータ（失敗の後の状態から始まる）。D・n は通常のデータ


def recovery_share(fk: dict) -> float:
    tot = sum(v["frames"] for v in fk.values())
    return round(sum(v["frames"] for k, v in fk.items() if k in RECOVERY_KINDS) / tot, 4) if tot else None


# ------------------------------------------------------------------------------------ 試し
def cmd_pilot(a) -> int:
    from recovla.common import code_version
    stamp = time.strftime("%Y%m%d-%H%M%S")
    runs = {}
    out = {"check": "r1_pilot", "written": None, "protocol": "docs/api/r1_protocol.md 2・3・4.5", "workers": a.workers}
    # 外乱・2 個入り（CPU）を先に（交代の時間を測るとき CPU を取り合わない）
    for kind in ("B", "D", "P"):
        run = config.path(a.resume_dir) / f"R1{kind}_pilot" if a.resume_dir else GEN / f"R1pilot_{stamp}" / f"R1{kind}_pilot"
        t0 = time.perf_counter()
        res = generate_quota(kind, f"{kind}_pilot", PILOT[kind], run, a.workers, chunk=8)
        runs[kind] = {"run": rel(run), "results": res}
        out[kind] = dict(demo_summary(res), wall_s=round(time.perf_counter() - t0, 1))
        print(f"[r1] 試し {kind}: {out[kind]}", flush=True)
    run_h = config.path(a.resume_dir) / "R1H_pilot" if a.resume_dir else GEN / f"R1pilot_{stamp}" / "R1H_pilot"
    t0 = time.perf_counter()
    h0, hn = (int(v) for v in a.h_seeds.split(":"))
    if not (BANDS["H_pilot"][0] <= h0 and h0 + hn - 1 <= BANDS["H_pilot"][1]):
        raise SystemExit("--h-seeds は試しの帯（70000〜70099）の中")
    summ = collect_handover(list(range(h0, h0 + hn)), hn, run_h)
    out["H"] = dict(summ, wall_s_total=round(time.perf_counter() - t0, 1))
    rows = read_jsonl(run_h / "handover.jsonl")
    entries = [{"run": rel(run_h), "key": r["name"]} for r in rows if r["saved"]]
    for kind in ("B", "D", "P"):
        entries += saved_entries(runs[kind]["results"], PILOT[kind])
    name = f"R1pilot_{(pathlib.Path(a.resume_dir).name.split('_', 1)[-1]) if a.resume_dir else stamp}"
    conv = convert_verify(name, entries, "R1 pilot (docs/api/r1_protocol.md): handover from Rreal + B/D/P demos, world v3_real")
    out["convert"] = conv
    if conv.get("frames"):
        out["frames_by_kind"] = frames_by_kind(conv["dataset"])
    out["projection"] = projection(out)
    out["code_version"] = code_version.code_version()
    out["written"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out["sheets"] = make_sheets(entries, R1 / "pilot_sheet.png")
    write_json(R1 / "pilot.json", out)
    print(json.dumps({"H": {k: out["H"][k] for k in ("attempts", "saved", "triggers_all", "triggers_saved", "discards")},
                      "convert": {k: conv[k] for k in ("episodes", "frames", "verify_pass")}, "projection": out["projection"]},
                     ensure_ascii=False, indent=1))
    return 0 if conv["verify_pass"] and conv["convert_exit"] == 0 else 1


def projection(p: dict) -> dict:
    """試しの数字から、交代後のデータ 200 本（最低 150）に要る時間の見込み（PC 1 台・2 台）。"""
    h = p["H"]
    if not h["saved"]:
        return {"note": "保存 0 本。見込みを出せない"}
    per_saved = h["wall_s"]["sum"] / h["saved"]
    out = {"wall_per_attempt_s": h["wall_s"]["mean"], "saved_per_attempt": round(h["saved"] / h["attempts"], 3),
           "wall_per_saved_s": round(per_saved, 1)}
    for n in (TARGET["H"], TARGET["H_min"]):
        out[f"hours_{n}_one_pc"] = round(n * per_saved / 3600, 1)
        out[f"hours_{n}_two_pcs"] = round(n * per_saved / 3600 / 2, 1)
    return out


# ------------------------------------------------------------------------------------ 学習データ
def ensure_rreal_runs() -> dict:
    """Rreal の学習データ（330 本）をこの作業場所から読めるようにする。無ければ MAIN_REPO の生成の回へのジャンクションを作る。
    MAIN_REPO は読むだけ（書かない）。"""
    dst = ROOT / RREAL_RUN
    if not dst.exists():
        src = MAIN_REPO / RREAL_RUN
        if not src.is_dir():
            raise SystemExit(f"Rreal の生成の回がない: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["cmd", "/c", "mklink", "/J", str(dst), str(src)], check=True, capture_output=True)
    mp = ROOT / RREAL_MANIFEST
    if not mp.is_file():
        src = MAIN_REPO / RREAL_MANIFEST
        mp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, mp)
    m = json.loads(mp.read_text(encoding="utf-8"))
    return {"manifest": RREAL_MANIFEST, "manifest_sha256": sha256_file(mp), "entries": m["entries"]}


def cmd_data(a) -> int:
    """採る: 交代後のデータはシードの順に TARGET["H"] 本まで、外乱・2 個入りはシードの順に目標の本数まで。
    立て直しデータのフレーム数の割合が RECOVERY_SHARE_MAX を超えたら、交代後のデータをシードの大きい方から減らす（変換の前の見込みで
    決め、変換の後に確かめる）。"""
    from recovla.common import code_version
    base = ensure_rreal_runs()
    h_runs = [config.path(r) for r in (a.handover_runs or [])] or sorted(p for p in GEN.glob("R1H_H_*") if p.is_dir())
    rows = sorted([dict(r, _run=rel(d)) for d in h_runs for r in read_jsonl(d / "handover.jsonl") if r["saved"]],
                  key=lambda r: r["seed"])
    demos = {}
    for kind in ("B", "D", "P"):
        runs = [config.path(r) for r in (getattr(a, f"runs_{kind.lower()}") or [])] or \
            sorted(p for p in GEN.glob(f"R1{kind}_{kind}_*") if p.is_dir())
        res = [dict(r, _run=rel(c)) for d in runs for c in sorted(d.glob("chunk_*")) if (c / "timing.json").is_file()
               for r in read_jsonl(c / "generation.jsonl")]
        res.sort(key=lambda r: r["layout_seed"])
        demos[kind] = saved_entries(res, TARGET[kind])
    # 立て直しの割合の見込み（記録の 20 Hz のフレーム数の半分 ≈ 変換後の 10 fps のフレーム数）
    base_fk = {"n": 38099, "A": 8633, "C": 3927}             # Rreal の変換後（R_v3r_20261014-055653 の conversion.json）
    h_take = rows[:TARGET["H"]]
    while True:
        h_frames = sum(r["n_frames"] for r in h_take) / 2
        demo_frames = {k: sum(_raw_frames(e) for e in v) / 2 for k, v in demos.items()}
        rec = base_fk["A"] + base_fk["C"] + h_frames + demo_frames["B"]
        tot = sum(base_fk.values()) + h_frames + sum(demo_frames.values())
        if rec / tot <= RECOVERY_SHARE_MAX or not h_take:
            break
        h_take = h_take[:-1]
    entries = base["entries"] + [{"run": r["_run"], "key": r["name"]} for r in h_take] + demos["B"] + demos["D"] + demos["P"]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = f"Rdag_{stamp}"
    conv = convert_verify(name, entries, f"Rdag (docs/api/r1_protocol.md 4.5): Rreal 330 ({RREAL_MANIFEST}) + handover "
                          f"{len(h_take)} + B {len(demos['B'])} + D {len(demos['D'])} + P {len(demos['P'])}, world v3_real")
    fk = frames_by_kind(conv["dataset"]) if conv.get("frames") else {}
    share = recovery_share(fk)
    steps = training_steps(conv.get("frames"))
    gate = {"verify_pass": conv["verify_pass"] and conv["convert_exit"] == 0 and conv["verify_exit"] == 0,
            "handover_ge_min": len(h_take) >= TARGET["H_min"], "recovery_share_le_50pct": share is not None and share <= RECOVERY_SHARE_MAX,
            "B_D_P_full": all(len(demos[k]) == TARGET[k] for k in ("B", "D", "P"))}
    out = {"check": "r1_data", "written": time.strftime("%Y-%m-%d %H:%M:%S"), "base": base | {"entries": len(base["entries"])},
           "handover_runs": [rel(d) for d in h_runs], "handover_saved_seen": len(rows), "handover_used": len(h_take),
           "handover_last_seed": h_take[-1]["seed"] if h_take else None,
           "triggers_used": dict(collections.Counter(r["handover"]["trigger"] for r in h_take)),
           "demos_used": {k: len(v) for k, v in demos.items()}, "dataset": conv, "frames_by_kind": fk,
           "recovery_share": share, "training": steps, "gate": gate, "pass": all(gate.values()) or (
               gate["verify_pass"] and gate["recovery_share_le_50pct"]),
           "code_version": code_version.code_version()}
    write_json(R1 / "data.json", out)
    print(json.dumps({k: out[k] for k in ("handover_used", "demos_used", "recovery_share", "training", "gate")}, ensure_ascii=False, indent=1))
    return 0 if out["pass"] else 1


def _raw_frames(e: dict) -> int:
    m = json.loads((config.path(e["run"]) / e["key"] / "meta.json").read_text(encoding="utf-8"))
    return int(m["n_frames"])


RREAL_FRAMES = 50659
RREAL_STEPS = 20000
BATCH = 32


def training_steps(frames: int) -> dict:
    """手数はフレーム数に比例（Rreal の 20,000 ステップ × 32 / 50,659 フレーム ≈ 12.6 周にそろえる）。1,000 ステップの単位に切り上げ。"""
    if not frames:
        return None
    epochs = RREAL_STEPS * BATCH / RREAL_FRAMES
    raw = frames * epochs / BATCH
    steps = int(np.ceil(raw / 1000.0) * 1000)
    return {"frames": int(frames), "epochs_rreal": round(epochs, 3), "steps": steps, "epochs": round(steps * BATCH / frames, 3),
            "rule": "steps = ceil(frames × 20000 / 50659 / 1000) × 1000、最後の保存点を使う"}


# ------------------------------------------------------------------------------------ 目で見る PNG
def make_sheets(entries: list, path: pathlib.Path, per_kind: int = 2) -> list:
    """種類ごとに per_kind 本（H は交代のきっかけごとに 1 本ずつ）。1 本につき俯瞰と手首を 8 フレーム。"""
    import cv2
    from recovla.record.recorder import read_png
    picks, seen = [], collections.Counter()
    for e in entries:
        kind = e["key"].split("_")[0]
        meta = json.loads((config.path(e["run"]) / e["key"] / "meta.json").read_text(encoding="utf-8"))
        sub = (meta.get("handover") or {}).get("trigger") if kind == "H" else None
        key = (kind, sub)
        if seen[key] < (1 if kind == "H" else per_kind):
            picks.append((e, meta, sub))
            seen[key] += 1
    blocks = []
    for e, meta, sub in picks:
        ep = config.path(e["run"]) / e["key"]
        z = np.load(ep / "data.npz")
        ts = np.asarray(z["sim_time"]) - float(z["sim_time"][0])
        idx = np.linspace(0, len(ts) - 1, SHEET_FRAMES).round().astype(int)
        title = np.full((28, 256 * SHEET_FRAMES, 3), 255, np.uint8)
        extra = f" trigger {sub}/{meta['handover']['detail']}" if sub else ""
        if meta.get("disturb"):
            extra = f" disturb {meta['disturb'].get('goal', {}).get('dist_m', 0) * 100:.1f} cm"
        cv2.putText(title, f"{e['key']} ({meta.get('layout_kind')}){extra}  top: overhead / bottom: wrist", (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        rows = [title]
        for cam in ("overhead", "wrist"):
            tiles = []
            for i in idx:
                im = read_png(ep / cam / f"{i:06d}.png").copy()
                if cam == "overhead":
                    cv2.putText(im, f"t={ts[i]:.1f}s", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                tiles.append(im)
            rows.append(np.hstack(tiles))
        blocks += [np.vstack(rows), np.full((10, 256 * SHEET_FRAMES, 3), 255, np.uint8)]
    if not blocks:
        return []
    img = np.vstack(blocks[:-1])
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    buf.tofile(str(path))
    print(f"[r1] wrote {path}（{img.shape[1]}x{img.shape[0]}）", flush=True)
    return [rel(path)]


def cmd_sheet(a) -> int:
    p = json.loads((R1 / "pilot.json").read_text(encoding="utf-8"))
    m = json.loads((ROOT / p["convert"]["manifest"]).read_text(encoding="utf-8"))
    make_sheets(m["entries"], R1 / "pilot_sheet.png")
    return 0


def cmd_bands(a) -> int:
    for k, (lo, hi) in BANDS.items():
        print(f"{k:8s} {lo}〜{hi}")
    print(json.dumps({"target": TARGET, "pilot": PILOT}, ensure_ascii=False))
    return 0


def main(argv=None) -> int:
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("pilot")
    s.add_argument("--workers", type=int, default=6)
    s.add_argument("--resume-dir", default=None, help="止まった試しの回（outputs/gen/R1pilot_<日時>）")
    s.add_argument("--h-seeds", default="70000:30", help="引き継ぎの試しの種 <始め>:<数>（試しの帯の中）")
    s = sub.add_parser("handover")
    s.add_argument("--band", required=True, choices=("H_A", "H_B", "H_pilot"))
    s.add_argument("--quota", type=int, default=100)
    s.add_argument("--out", default=None, help="続きから回すときの回のフォルダ")
    s.add_argument("--until", default=None, help="打ち切りの時刻 \"YYYY-MM-DD HH:MM\"")
    s.add_argument("--shard", default="0/1", help="i/n: 帯の種のうち (種 − 帯の始め) mod n = i だけを回す")
    s = sub.add_parser("demos")
    s.add_argument("--kind", required=True, choices=("B", "D", "P"))
    s.add_argument("--band", default=None)
    s.add_argument("--quota", type=int, required=True)
    s.add_argument("--workers", type=int, default=6)
    s.add_argument("--chunk", type=int, default=24)
    s.add_argument("--out", default=None)
    s = sub.add_parser("data")
    s.add_argument("--handover-runs", nargs="*")
    s.add_argument("--runs-b", nargs="*")
    s.add_argument("--runs-d", nargs="*")
    s.add_argument("--runs-p", nargs="*")
    sub.add_parser("sheet")
    sub.add_parser("bands")
    a = ap.parse_args(argv)
    return {"pilot": cmd_pilot, "handover": cmd_handover, "demos": cmd_demos, "data": cmd_data, "sheet": cmd_sheet,
            "bands": cmd_bands}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
