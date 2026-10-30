"""自分の成功も足して学び直すためのデータ: Rdag を走らせ、自分で成功した試行だけを記録する（事前に決めた比較の外の追加の試み）。

    python scripts/success_relearn/selfroll_data.py selfroll --band S_pilot --quota 10           # 試し（学習に使わない）
    python scripts/success_relearn/selfroll_data.py selfroll --band S_A --quota 250 [--shard 0/2]  # PC A。続きから回せる（同じ --out）
    python scripts/success_relearn/selfroll_data.py selfroll --band S_B --quota 250               # PC B
    python scripts/success_relearn/selfroll_data.py bands
    python scripts/success_relearn/selfroll_data.py data [--self-runs <フォルダ…>]             # 採る → マニフェスト → 変換と照合 → 統計量を Rdag に → data.json

シードの範囲（BANDS、学習用）: S 自分の成功（Rdag）  試し 75000〜75099、PC A 75100〜76549、PC B 76550〜77999
データ（docs/api/r2_protocol.md 4 節）: Rdag の学習データ 670 本（そのまま）＋自分の成功。自分の成功は、立て直しを含むものを
全部先に、残りをシードの順に、自分の成功のフレーム数が全体の SELF_SHARE_MAX（35%）を超えない所まで。試しのシード（S_pilot）は使わない。
学習は Rdag から続けるので、データセットの統計量（meta/stats.json）を Rdag のデータセットのものに置き換えた学習用の写しを作る
（正規化を Rdag と同じに保つ）。
VLA の実行の設定は scripts/failure_relearn/failure_data.py の runtime_factory を使う: v3_real の評価の設定、naive・6 ステップごとに推論、安全フィルタなし、
開閉の絞り K=5、遅延は robot_io.fixed_latency。保存の条件は expert/selfroll_real.py。
書くもの: outputs/gen/R2S_*（生成の回）。評価の記録（outputs/v2eval/R1E*・R1D*）と評価のシードには触らない。
"""
import argparse
import collections
import contextlib
import datetime
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "failure_relearn"))

import failure_data as R1D  # noqa: E402
from recovla.common import config  # noqa: E402

GEN = R1D.GEN
RDAG = "outputs/r1/train/train_Rdags1000_20261021-155742_20261021-155742/checkpoints/040000/pretrained_model"
RDAG_SHA256 = "93f74cc86e81a00b6c8610b8fbb5c576e10fa536abda7db646a960e8f0c60233"     # docs/api/r1_protocol.md
RDAG_HOME = pathlib.Path(r"C:\PAI\wt_r1")                                               # Rdag を学習した作業場所（読むだけ）
BANDS = {"S_pilot": (75000, 75099), "S_A": (75100, 76549), "S_B": (76550, 77999)}
R2 = config.path("outputs/r2")
RDAG_DATA = "outputs/r1/data.json"                                                      # Rdag の学習データの記録（RDAG_HOME の下）
SELF_SHARE_MAX = 0.35
# 自分の成功のラベルから、始めの保持と短い開閉を外して変換する（recovla.data.convert の SELF_CLEAN）。外さないと、Rdag の
# チェックポイントでの損失が自分の成功のデータ（S）だけ約 20 倍になる（グリッパ 1.64 と 0.016）
SELF_LABEL_CLEAN = True


def rdag_ckpt(arg: str = None) -> str:
    """Rdag のチェックポイント: 指定、なければこの作業場所、なければ RDAG_HOME の同じ場所。"""
    for p in ([arg] if arg else []) + [str(ROOT / RDAG), str(RDAG_HOME / RDAG)]:
        if (pathlib.Path(p) / "model.safetensors").is_file():
            return p
    raise SystemExit(f"Rdag の保存点がない: {RDAG}")


def check_ckpt(ckpt: str) -> str:
    sha = R1D.sha256_file(pathlib.Path(ckpt) / "model.safetensors")
    if sha != RDAG_SHA256:
        raise SystemExit(f"Rdag の保存点の SHA-256 が記録と違う: {sha}")
    return sha


def collect_selfroll(seed_list, quota: int, run: pathlib.Path, ckpt: str, until: str = None) -> dict:
    """シードの順に回す。run があれば終わったシードを飛ばして続ける。quota 本保存したかシードが尽きたら終わる。"""
    from recovla.common import code_version
    from recovla.expert.selfroll_real import run_r2_selfroll
    from recovla.harness import real_matched as V
    R1D.check_band(min(seed_list), max(seed_list))
    sha = check_ckpt(ckpt)
    cfg_eval = R1D.eval_config()
    make_runtime = R1D.runtime_factory(ckpt, cfg_eval)
    run.mkdir(parents=True, exist_ok=True)
    log_p = run / "selfroll.jsonl"
    done = R1D.read_jsonl(log_p)
    seen = {r["seed"] for r in done}
    got = sum(bool(r["saved"]) for r in done)
    sp = run / "start.json"
    start = {"model": "Rdag", "checkpoint": str(ckpt), "checkpoint_sha256": sha, "world": R1D.WORLD,
             "grip_debounce": R1D.GRIP_DEBOUNCE, "exec_interval": R1D.EXEC_INTERVAL, "mode": "naive", "safety": False,
             "quota": quota, "seeds": [min(seed_list), max(seed_list)], "code_version": code_version.code_version()}
    keys = ("checkpoint_sha256", "world", "grip_debounce", "seeds")
    if sp.is_file():
        old = json.loads(sp.read_text(encoding="utf-8"))
        if {k: old.get(k) for k in keys} != {k: start[k] for k in keys}:
            raise SystemExit(f"{sp} と今回の設定が違う（続きにならない）")
    else:
        R1D.write_json(sp, dict(start, started=time.strftime("%Y-%m-%d %H:%M:%S")))
    deadline = datetime.datetime.strptime(until, "%Y-%m-%d %H:%M") if until else None
    rig = V.V3RealGenRig(V.load_config(R1D.WORLD, generation=True))
    t0, n = time.perf_counter(), 0
    try:
        with open(log_p, "a", encoding="utf-8") as log:
            for seed in seed_list:
                if got >= quota:
                    break
                if deadline and datetime.datetime.now() >= deadline:
                    print(f"[r2] 打ち切りの時刻 {until} に達した", flush=True)
                    break
                if seed in seen:
                    continue
                out = run_r2_selfroll(rig, make_runtime, seed, run, "Rdag", cfg_eval)
                n += 1
                got += int(out["saved"])
                log.write(json.dumps(out, ensure_ascii=False, default=float) + "\n")
                log.flush()
                ev = ",".join(e["trigger"] for e in out["events"]) or "-"
                print(f"[r2s] {seed} {out['color']:5s} {out['rollout_layout']['stratum']:4s} saved {out['saved']} "
                      f"discard {out['discard']} events {ev} t {out['t_end']:5.1f} got {got}/{quota} wall {out['wall_s']:.0f}s",
                      flush=True)
    finally:
        rig.close()
    summ = summarize(R1D.read_jsonl(log_p))
    R1D.write_json(run / "run.json", {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "attempts_this_call": n,
                                      "wall_s_this_call": round(time.perf_counter() - t0, 1), **summ})
    return summ


def summarize(rows: list) -> dict:
    saved = [r for r in rows if r["saved"]]
    walls = [r["wall_s"] for r in rows]
    by_st = {}
    for st in ("pf2", "yaw", "near", "free"):
        rs = [r for r in rows if r["rollout_layout"]["stratum"] == st]
        by_st[st] = {"attempts": len(rs), "saved": sum(r["saved"] for r in rs),
                     "saved_recovered": sum(r["saved"] and r["recovered"] for r in rs)}
    return {"attempts": len(rows), "saved": len(saved), "saved_recovered": sum(r["recovered"] for r in saved),
            "discards": dict(collections.Counter(r["discard"] for r in rows if not r["saved"])),
            "events_saved": dict(collections.Counter(e["trigger"] for r in saved for e in r["events"])),
            "by_stratum": by_st,
            "wall_s": {"sum": round(sum(walls), 1), "mean": round(float(np.mean(walls)), 1) if walls else None},
            "wall_per_saved_s": round(sum(walls) / len(saved), 1) if saved else None,
            "recorded_frames_20hz": sum(r["n_frames"] for r in saved),
            "t_end_s": {"median": round(float(np.median([r["t_end"] for r in saved])), 2)} if saved else None}


def cmd_selfroll(a) -> int:
    i, n = (int(v) for v in a.shard.split("/"))
    lo, hi = BANDS[a.band]
    seeds = [s for s in range(lo, hi + 1) if (s - lo) % n == i]      # 1 台で 2 本並べるときは --shard 0/2・1/2
    tag = "" if n == 1 else f"_s{i}of{n}"
    run = config.path(a.out) if a.out else GEN / f"R2S_{a.band}{tag}_{time.strftime('%Y%m%d-%H%M%S')}"
    summ = collect_selfroll(seeds, a.quota, run, rdag_ckpt(a.ckpt), a.until)
    print(json.dumps({k: summ[k] for k in ("attempts", "saved", "saved_recovered", "discards")}, ensure_ascii=False))
    print(run)
    return 0


def cmd_bands(a) -> int:
    for k, (lo, hi) in BANDS.items():
        print(f"{k:8s} {lo}〜{hi}")
    return 0


def rdag_base() -> dict:
    """Rdag の学習データ（670 本）をこの作業場所から読めるようにする。マニフェストは写して SHA-256 を照らし、生成の回は
    RDAG_HOME（Rreal の回は MAIN_REPO）へのジャンクションを作る。どちらも読むだけ（書かない）。"""
    d = json.loads((RDAG_HOME / RDAG_DATA).read_text(encoding="utf-8"))
    ds = d["dataset"]
    src_m = RDAG_HOME / ds["manifest"]
    dst_m = ROOT / ds["manifest"]
    if not dst_m.is_file():
        dst_m.parent.mkdir(parents=True, exist_ok=True)
        dst_m.write_bytes(src_m.read_bytes())
    if R1D.sha256_file(dst_m) != ds["manifest_sha256"]:
        raise SystemExit(f"{dst_m} の SHA-256 が Rdag の記録と違う")
    m = json.loads(dst_m.read_text(encoding="utf-8"))
    for top in sorted({"/".join(e["run"].split("/")[:3]) for e in m["entries"]}):    # outputs/gen/<回>
        dst = ROOT / top
        if dst.exists():
            continue
        src = (R1D.MAIN_REPO if top == R1D.RREAL_RUN else RDAG_HOME) / top
        if not src.is_dir():
            raise SystemExit(f"Rdag の生成の回がない: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        R1D.subprocess.run(["cmd", "/c", "mklink", "/J", str(dst), str(src)], check=True, capture_output=True)
    return {"manifest": ds["manifest"], "manifest_sha256": ds["manifest_sha256"], "dataset": ds["dataset"],
            "frames": int(ds["frames"]), "entries": m["entries"], "frames_by_kind": d.get("frames_by_kind")}


def pick_self(rows: list, base_frames: int) -> tuple:
    """立て直しを含む成功を全部先に、残りをシードの順に。自分の成功のフレーム数（記録の 20 Hz の半分 ≈ 変換後）が全体の
    SELF_SHARE_MAX を超えない所まで。(採った行, 見込みの割合)。"""
    order = [r for r in rows if r["recovered"]] + [r for r in rows if not r["recovered"]]
    take, frames = [], 0.0
    for r in order:
        f = r["n_frames"] / 2
        if (frames + f) / (base_frames + frames + f) > SELF_SHARE_MAX:
            break
        take.append(r)
        frames += f
    return take, round(frames / (base_frames + frames), 4) if take else 0.0


def stats_copy(ds_rel: str, stats_from: pathlib.Path) -> str:
    """学習用の写し <ds>_statsRdag: data・images はハードリンク（同じ中身）、meta は写して stats.json だけ Rdag のデータセットの
    ものに置き換え、conversion.json に stats_source を書く。"""
    import os
    import shutil
    src = config.path(ds_rel).resolve()
    dst = src.with_name(f"{src.name}_statsRdag")
    if dst.exists():
        return R1D.rel(dst)
    for p in src.rglob("*"):
        q = dst / p.relative_to(src)
        if p.is_dir():
            q.mkdir(parents=True, exist_ok=True)
        elif p.relative_to(src).parts[0] == "meta":
            q.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, q)
        else:
            q.parent.mkdir(parents=True, exist_ok=True)
            os.link(p, q)
    shutil.copy2(stats_from / "meta" / "stats.json", dst / "meta" / "stats.json")
    conv = json.loads((dst / "meta" / "conversion.json").read_text(encoding="utf-8"))
    conv["stats_source"] = {"dataset": str(stats_from), "stats_json_sha256": R1D.sha256_file(stats_from / "meta" / "stats.json"),
                            "own_stats_json_sha256": R1D.sha256_file(src / "meta" / "stats.json"),
                            "note": "R2a: Rdag から続けて学習するので、統計量を Rdag のデータセットのものに固定する学習用の写し"}
    (dst / "meta" / "conversion.json").write_text(json.dumps(conv, ensure_ascii=False, indent=2), encoding="utf-8")
    return R1D.rel(dst)


def cmd_data(a) -> int:
    from recovla.common import code_version
    base = rdag_base()
    runs = [config.path(r) for r in (a.self_runs or [])] or sorted(
        p for p in GEN.glob("R2S_S_[AB]*") if p.is_dir())
    rows = sorted([dict(r, _run=R1D.rel(d)) for d in runs for r in R1D.read_jsonl(d / "selfroll.jsonl") if r["saved"]],
                  key=lambda r: r["seed"])
    if any(BANDS["S_pilot"][0] <= r["seed"] <= BANDS["S_pilot"][1] for r in rows):
        raise SystemExit("試しの帯（S_pilot）の成功が入っている（学習に使わない）")
    if len({r["seed"] for r in rows}) != len(rows):
        raise SystemExit("同じ種の成功が 2 本以上ある（回のフォルダを確かめる）")
    take, share_est = pick_self(rows, base["frames"])
    entries = base["entries"] + [{"run": r["_run"], "key": r["name"]} for r in take]
    name = f"R2a_{time.strftime('%Y%m%d-%H%M%S')}"
    R1D.R1 = R2                                     # 変換のログの置き場所（convert_verify が R1 / convert_<名>.log に書く）
    if SELF_LABEL_CLEAN and "--self-label-clean" not in R1D.CUE_ARGS:
        R1D.CUE_ARGS = R1D.CUE_ARGS + ["--self-label-clean"]
    conv = R1D.convert_verify(name, entries, f"R2a (docs/api/r2_protocol.md 4): Rdag 670 ({base['manifest']}) + self successes "
                              f"{len(take)} (recovered {sum(r['recovered'] for r in take)}), world v3_real")
    fk = R1D.frames_by_kind(conv["dataset"]) if conv.get("frames") else {}
    tot = sum(v["frames"] for v in fk.values())
    share = round(fk.get("S", {}).get("frames", 0) / tot, 4) if tot else None
    ok_conv = conv["verify_pass"] and conv["convert_exit"] == 0 and conv["verify_exit"] == 0
    train_ds = stats_copy(conv["dataset"], RDAG_HOME / base["dataset"]) if ok_conv else None
    gate = {"verify_pass": ok_conv, "self_share_le_max": share is not None and share <= SELF_SHARE_MAX + 0.01,
            "base_frames_same": fk and sum(v["frames"] for k, v in fk.items() if k != "S") == base["frames"]}
    out = {"check": "r2_data", "written": time.strftime("%Y-%m-%d %H:%M:%S"),
           "base": {k: base[k] for k in ("manifest", "manifest_sha256", "dataset", "frames")} | {"entries": len(base["entries"])},
           "self_runs": [R1D.rel(d) for d in runs], "self_saved_seen": len(rows), "self_used": len(take),
           "self_recovered_used": sum(r["recovered"] for r in take), "self_recovered_seen": sum(r["recovered"] for r in rows),
           "self_last_seed": take[-1]["seed"] if take else None,
           "self_by_stratum": dict(collections.Counter(r["rollout_layout"]["stratum"] for r in take)),
           "self_share_estimate": share_est, "self_share": share, "dataset": conv, "frames_by_kind": fk,
           "train_dataset": train_ds, "gate": gate, "pass": all(gate.values()), "code_version": code_version.code_version()}
    R1D.write_json(R2 / "data.json", out)
    print(json.dumps({k: out[k] for k in ("self_used", "self_recovered_used", "self_share", "train_dataset", "gate")},
                     ensure_ascii=False, indent=1))
    return 0 if out["pass"] else 1


def main(argv=None) -> int:
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    dd = sub.add_parser("data")
    dd.add_argument("--self-runs", nargs="*", default=None, help="自分の成功の回のフォルダ（既定は outputs/gen/R2S_S_A*・R2S_S_B*）")
    s = sub.add_parser("selfroll")
    s.add_argument("--band", required=True, choices=tuple(BANDS))
    s.add_argument("--quota", type=int, default=250)
    s.add_argument("--out", default=None, help="続きから回すときの回のフォルダ")
    s.add_argument("--until", default=None, help="打ち切りの時刻 \"YYYY-MM-DD HH:MM\"")
    s.add_argument("--shard", default="0/1", help="i/n: 帯の種のうち (種 − 帯の始め) mod n = i だけを回す")
    s.add_argument("--ckpt", default=None, help="Rdag の保存点のフォルダ（既定は RDAG、無ければ C:\\PAI\\wt_r1 の同じ場所）")
    sub.add_parser("bands")
    a = ap.parse_args(argv)
    return {"selfroll": cmd_selfroll, "bands": cmd_bands, "data": cmd_data}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
