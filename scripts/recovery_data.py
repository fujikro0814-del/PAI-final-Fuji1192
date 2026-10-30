"""立て直しデータの生成の段取り（わざと起こす失敗の値の振り、生成の確かめ、R1・N1 の学習データの計画と生成）。

結果は outputs/f/<項目>.json。scripts/real/regen_data.py が、学習データの計画（cmd_plan）の部品として読む。

    .venv\\Scripts\\python.exe scripts\\recovery_data.py plan                         # R1・N1 の生成の計画 → outputs/f/plan.json
    ... gen-data [--workers 8] [--smoke]             # R1・N1 のデータ → outputs/f/data.json
    ... sweep [--workers 8] [--n 120] [--smoke]      # わざと起こす失敗の値の振り
    ... check-gen / check-eval [--smoke]             # 生成の確かめ（着地・最初のフレーム・再生の一致・映像）→ outputs/f/check.json
    ... review [--smoke]                             # 目視用の映像と一覧 → outputs/f/review/

R1 = 成功データ＋立て直しデータ、N1 = 成功データ＋立て直しと同じ配置の成功データ。失敗の種類は A 掴み損ね・
B 運ぶ途中の落下・C 置き損ね（recovla.expert.inject）。
振り（描画なし、作り直しなし）: A は閉じる高さの上げ幅の中心 2.0〜4.5 cm と横ずらしだけ、B は箱の中心からの距離の下限
0.15〜0.23 m、C は今の値。シードは A 51000〜、B 51200〜、C 51400〜。同じ種類の中ではどの点も同じシードを使う
（対応のある比較。点の違いは値の違いだけになる）。
"""
import argparse
import collections
import copy
import json
import multiprocessing
import os
import pathlib
import subprocess
import sys
import time

from recovla.common import config
from recovla.eval.stats import wilson_interval

CFG = config.load()
OUT = config.path(CFG["paths"]["outputs"]) / "f"
SEED_BASE = {"A": 51000, "B": 51200, "C": 51400}
A_RAISE = [0.020, 0.025, 0.030, 0.035, 0.040, 0.045]
A_HALF = 0.005
B_DMIN = [0.15, 0.17, 0.19, 0.21, 0.23]
CATS = ("not_effective", "landing_invalid", "recovery_failed", "success")


def points() -> list:
    pts = [{"id": f"A_raise_{r * 100:.1f}cm", "kind": "A",
            "overrides": {"A": {"raise_close_m": [round(r - A_HALF, 4), round(r + A_HALF, 4)], "raise_ratio": 1.0}}}
           for r in A_RAISE]
    pts.append({"id": "A_lateral_only", "kind": "A", "overrides": {"A": {"raise_ratio": 0.0}}})
    pts += [{"id": f"B_dmin_{d:.2f}m", "kind": "B", "overrides": {"B": {"min_dist_from_box_m": d}}} for d in B_DMIN]
    pts.append({"id": "C_current", "kind": "C", "overrides": {}})
    return pts


def apply_overrides(cfg: dict, overrides: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    for k, sub in overrides.items():
        cfg["inject"][k].update(sub)
    return cfg


def write(name: str, obj: dict) -> pathlib.Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.json"
    p.write_text(json.dumps({"check": name, "written": time.strftime("%Y-%m-%d %H:%M:%S"), **obj},
                            ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[f] wrote {p}", flush=True)
    return p


# ------------------------------------------------------------------------------------ 並列の作業

_RIG = None


def _init() -> None:
    global _RIG
    from recovla.sim.rig import SimRig
    _RIG = SimRig(render=False)


def _job(job):
    from recovla.expert import generate as G
    from recovla.sim import scene
    pid, kind, overrides, seeds_ = job
    _RIG.cfg = apply_overrides(CFG, overrides)            # run_attempt・わざと起こす失敗は rig.cfg を使う（物理の設定は変えない）
    rows = []
    for seed in seeds_:
        lay = scene.sample_layout(seed)
        color = lay.table_colors[seed % len(lay.table_colors)]
        t0 = time.perf_counter()
        a = G.run_attempt(_RIG, G.EpisodeSpec(seed, color, kind=kind), 0, None, False)
        a.pop("_frames", None)
        a["point"] = pid
        a["wall_attempt_s"] = round(time.perf_counter() - t0, 3)
        a["worker_pid"] = os.getpid()
        rows.append(a)
    return rows


# ------------------------------------------------------------------------------------ 集計

def category(a: dict) -> str:
    if a["success"]:
        return "success"
    if a["failure"] == "inject_not_effective":
        return "not_effective"
    if a["failure"] == "inject_landing_invalid":
        return "landing_invalid"
    if a["inject"]["status"] == "confirmed":
        return "recovery_failed"
    return "excluded"                                      # 失敗させる前に起きた失敗など（件数だけ別に数える）


def landing_reasons(a: dict, min_clear: float) -> set:
    """着地が不適切の内訳: 壁・箱の縁／先客（箱の中）の立方体／机上の他の立方体／その他。
    立方体は「隙間が min_clear 未満」か「記録の前に目標が触れた」のどちらか。"""
    inj = a["inject"]
    info = inj["info"]
    clear = info.get("clearance_m", {})
    out = set()
    if info.get("over_box") or any(v < min_clear for k, v in clear.items() if k.startswith("box_wall")):
        out.add("wall_or_box_edge")
    for k, v in clear.items():
        if not k.startswith("cube_"):
            continue
        c = k[len("cube_"):]
        if v < min_clear or c in inj.get("cube_contact_before", []):
            out.add("prefilled_cube" if c in inj.get("prefilled", []) else "table_cube")
    rest = set(info.get("landing_fail", [])) - {"over_box", "clearance"}
    if inj.get("reason") == "no_rest":
        rest.add("no_rest")
    if rest or not out:
        out.add("other:" + ",".join(sorted(rest)) if rest else "other")
    return out


def wilson(k: int, n: int) -> list:
    lo, hi = wilson_interval(k, n)
    return [None if lo is None else round(lo, 4), None if hi is None else round(hi, 4)]


def summarize(rows: list) -> list:
    split = float(CFG["inject"]["landing_orientation_split_deg"])
    min_clear = float(CFG["inject"]["landing"]["min_clearance_m"])
    by = collections.defaultdict(list)
    for a in rows:
        by[a["point"]].append(a)
    out = []
    for pt in points():
        rs = by.get(pt["id"], [])
        if not rs:
            continue
        n = len(rs)
        cnt = collections.Counter(category(a) for a in rs)
        excluded = collections.Counter(f"{a['failure']}:{a['inject']['reason']}" for a in rs if category(a) == "excluded")
        n_excl = sum(excluded.values())
        valid = cnt["success"] + cnt["recovery_failed"]            # 確定（わざと起こした失敗が confirmed）
        eff = valid + cnt["landing_invalid"]                        # 確定＋着地が不適切（不適切の割合の分母）
        succ = [a for a in rs if a["success"]]
        within = sum(abs(a["inject"]["info"].get("yaw_rel_deg", 0.0)) <= split for a in succ)
        row = {
            "point": pt["id"], "kind": pt["kind"], "overrides": pt["overrides"], "n": n,
            "seeds": [min(a["layout_seed"] for a in rs), max(a["layout_seed"] for a in rs)],
            "counts": {c: cnt[c] for c in CATS}, "excluded": dict(excluded), "excluded_n": n_excl,
            "r0_below_min": sum(v for k, v in excluded.items() if k.endswith(":r0_below_min")),
            "wilson_of_n": {c: wilson(cnt[c], n) for c in CATS},
            # 分母は n − 失敗させる前に起きた失敗（区分の外）
            "not_effective_rate": [cnt["not_effective"], n - n_excl, wilson(cnt["not_effective"], n - n_excl)],
            "confirmed_rate": [valid, n, wilson(valid, n)],
            "landing_invalid_ratio": [cnt["landing_invalid"], eff, wilson(cnt["landing_invalid"], eff)],
            "recovery_success": [cnt["success"], valid, wilson(cnt["success"], valid)],
            "success_yaw_within": [within, len(succ)], "success_yaw_beyond": [len(succ) - within, len(succ)],
            "wall_attempt_s_mean": round(sum(a["wall_attempt_s"] for a in rs) / n, 3),
        }
        if pt["kind"] == "A":
            row["a_mode_counts"] = dict(collections.Counter(a["inject"]["params"].get("a_mode") for a in rs))
            row["not_effective_reason"] = dict(collections.Counter(
                a["inject"]["reason"] for a in rs if category(a) == "not_effective"))
        inv = [a for a in rs if category(a) == "landing_invalid"]
        reasons = collections.Counter(r for a in inv for r in landing_reasons(a, min_clear))
        row["landing_invalid_breakdown"] = dict(reasons)
        row["landing_invalid_examples"] = [
            {"name": a["name"], "reasons": sorted(landing_reasons(a, min_clear)),
             "clearance_m": a["inject"]["info"].get("clearance_m"), "target_pos": a["inject"]["info"].get("target_pos"),
             "cube_contact_before": a["inject"].get("cube_contact_before"), "prefilled": a["inject"].get("prefilled")}
            for a in inv[:8]]
        row["touched_other_cube_but_landing_ok"] = sum(
            1 for a in rs if category(a) in ("success", "recovery_failed") and a["inject"].get("cube_contact_before"))
        out.append(row)
    return out


def rate(x) -> float:
    k, n = x[0], x[1]
    return k / n if n else float("nan")


def decide(rows: list) -> dict:
    """振りの結果から A・B の値を決める（決まりは結果を見る前に固定した）。"""
    out = {}
    # A: 上げる型の点のうち「効かず」の推定値 ≤ 10% の点が続く範囲をつなぐ。raise_ratio は 0.5 のまま
    a_pts = [(r["overrides"]["A"]["raise_close_m"], rate(r["not_effective_rate"]))
             for r in rows if r["kind"] == "A" and r["overrides"]["A"].get("raise_ratio") == 1.0]
    a_pts.sort(key=lambda x: x[0][0])
    runs, cur = [], []
    for rng_, r in a_pts:
        if r <= 0.10:
            cur.append(rng_)
        else:
            if cur:
                runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    if runs:
        best = max(runs, key=lambda run: (len(run), -run[0][0]))
        out["A"] = {"raise_close_m": [best[0][0], best[-1][1]], "raise_ratio": 0.5,
                    "runs": [[run[0][0], run[-1][1]] for run in runs], "rule": "0035 の 1"}
    else:
        out["A"] = {"raise_ratio": 0.0, "runs": [], "rule": "0035 の 1（10% 以下の点がない → 横ずらしだけ）"}
    # B: 「不適切」の推定値 ≤ 20% の最小の下限。20% 以下がない、または「机上の他の立方体」が下限とともに増えたら ask
    b = sorted((r for r in rows if r["kind"] == "B"), key=lambda r: r["overrides"]["B"]["min_dist_from_box_m"])
    table_cube = [r["landing_invalid_breakdown"].get("table_cube", 0) for r in b]
    ok = [r["overrides"]["B"]["min_dist_from_box_m"] for r in b if rate(r["landing_invalid_ratio"]) <= 0.20]
    increasing = len(table_cube) >= 2 and table_cube[-1] > table_cube[0]
    out["B"] = {"table_cube_by_point": dict(zip([r["point"] for r in b], table_cube)),
                "points_at_or_below_20pct": ok, "table_cube_increasing": increasing}
    if not ok or increasing:
        out["B"]["decision"] = "ask"
        out["B"]["why"] = ("20% 以下の点がない" if not ok else "") + (" 机上の他の立方体が下限とともに増えている" if increasing else "")
    else:
        out["B"]["decision"] = {"min_dist_from_box_m": min(ok)}
    return out


def cmd_sweep(a) -> None:
    n = 2 if a.smoke else a.n
    if n > 200:
        raise SystemExit("n は 200 以下（掲示板 0035 の 4。超えると B と C の種の範囲が重なる）")
    jobs = []
    for pt in points():
        base = SEED_BASE[pt["kind"]]
        seeds_ = list(range(base, base + n))
        for i in range(0, n, a.chunk):
            jobs.append((pt["id"], pt["kind"], pt["overrides"], seeds_[i:i + a.chunk]))
    OUT.mkdir(parents=True, exist_ok=True)
    name = "sweep_smoke" if a.smoke else "sweep"
    rows_path = OUT / f"{name}_rows.jsonl"
    t0 = time.perf_counter()
    rows = []
    with open(rows_path, "w", encoding="utf-8") as log:
        if a.workers <= 1:
            _init()
            it = map(_job, jobs)
            pool = None
        else:
            pool = multiprocessing.get_context("spawn").Pool(a.workers, initializer=_init)
            it = pool.imap_unordered(_job, jobs)
        for rs in it:
            for r in rs:
                rows.append(r)
                log.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
            log.flush()
            print(f"[f] {len(rows)}/{len(points()) * n} {time.perf_counter() - t0:.0f}s", flush=True)
        if pool is not None:
            pool.close()
            pool.join()
    pts = summarize(rows)
    write(name, {"n_per_point": n, "workers": a.workers, "wall_s": round(time.perf_counter() - t0, 1),
                 "seed_ranges": {k: [v, v + n - 1] for k, v in SEED_BASE.items()},
                 "rows": str(rows_path.relative_to(config.ROOT)), "points": pts, "decision": decide(pts)})


# ------------------------------------------------------------------------ 生成の確かめ 1〜4
# 1 着地の不適切の割合と立て直しの成功率、2 最初のフレームの条件、3 再生の一致、4 映像

GEN = config.path(CFG["paths"]["outputs"]) / "gen"
CHECK_SEED_BASE = {"A": 52000, "B": 52200, "C": 52400}   # 確かめ用のシード（振りのシードと重ねない）
REPLAY_PER_KIND = 3
VIDEOS_PER_KIND = 5


def check_specs(n: int) -> list:
    from recovla.expert import generate as G
    from recovla.sim import scene
    specs = []
    for kind, base in CHECK_SEED_BASE.items():
        for seed in range(base, base + n):
            lay = scene.sample_layout(seed)
            specs.append(G.EpisodeSpec(seed, lay.table_colors[seed % len(lay.table_colors)], lay.kind, kind))
    return specs


def cmd_check_gen(a) -> None:
    """確かめ用のデータを作る（描画あり、作り直しあり）。学習と同時に回さない（GPU の描画）。"""
    from recovla.expert import generate as G
    n = 2 if a.smoke else a.n
    run = GEN / f"f_check{'' if a.rig == 'v1' else '_' + a.rig}{'_smoke' if a.smoke else ''}_{time.strftime('%Y%m%d-%H%M%S')}"
    G.generate(check_specs(n), run, workers=a.workers, render=True, rig_kind=a.rig)
    print(f"[f] generated {run}", flush=True)


def latest_run(prefix: str) -> pathlib.Path:
    runs = sorted(p for p in GEN.glob(f"{prefix}_2*") if p.is_dir())
    if not runs:
        raise SystemExit(f"no run outputs/gen/{prefix}_* (run check-gen first)")
    return runs[-1]


COND1_RULE = ("決裁 0042: 各指定の最初の試み（作り直しの前）で数える。着地が不自然の割合 = 不自然 ÷（確定＋不自然）"
              " ≤ inject.landing.max_invalid_ratio、立て直しの成功率 = 成功 ÷ 確定 ≥ 0.9。作り直しを含む全部の試みの数と、"
              "捨てた指定の割合は別の指標として出す")


def _kind_counts(rs: list) -> dict:
    split = float(CFG["inject"]["landing_orientation_split_deg"])
    cnt = collections.Counter(category(t) for t in rs)
    valid = cnt["success"] + cnt["recovery_failed"]
    eff = valid + cnt["landing_invalid"]
    succ = [t for t in rs if t["success"]]
    within = sum(abs(t["inject"]["info"].get("yaw_rel_deg", 0.0)) <= split for t in succ)
    return {"attempts": len(rs), "counts": {c: cnt[c] for c in CATS},
            "excluded": dict(collections.Counter(t["failure"] for t in rs if category(t) == "excluded")),
            "recovery_success": [cnt["success"], valid, wilson(cnt["success"], valid)],
            "landing_invalid_ratio": [cnt["landing_invalid"], eff, wilson(cnt["landing_invalid"], eff)],
            "success_yaw_within": [within, len(succ)], "success_yaw_beyond": [len(succ) - within, len(succ)]}


def check_table(results: list) -> dict:
    """種類ごとに、最初の試みの 4 区分（確かめ 1 の判定に使う）と、作り直しを含む全部の試みの 4 区分、捨てた指定。"""
    table = {}
    for kind in CHECK_SEED_BASE:
        rk = [r for r in results if r["kind"] == kind]
        first = _kind_counts([r["attempts"][0] for r in rk])
        n, saved = len(rk), sum(1 for r in rk if r["success"])
        table[kind] = {**first, "all_attempts": _kind_counts([t for r in rk for t in r["attempts"]]),
                       "specs": n, "specs_saved": saved, "specs_dropped": n - saved,
                       "dropped_ratio": [n - saved, n, wilson(n - saved, n)]}
    return table


def cond1_pass(table: dict) -> bool:
    lim = float(CFG["inject"]["landing"]["max_invalid_ratio"])
    return all(t["recovery_success"][1] > 0 and t["recovery_success"][0] / t["recovery_success"][1] >= 0.9
               and (t["landing_invalid_ratio"][1] == 0 or t["landing_invalid_ratio"][0] / t["landing_invalid_ratio"][1] <= lim)
               for t in table.values())


def cmd_check_eval(a) -> None:
    import numpy as np
    from recovla.common import seeds
    from recovla.expert import inject
    from recovla.record import replay as legacy, replay_scene as RS
    from recovla.sim.rig import SimRig
    run = latest_run("f_check_smoke" if a.smoke else "f_check")
    results = [json.loads(l) for l in (run / "generation.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    table = check_table(results)
    if a.table_only:                      # 描画を使わずに 1 の表だけ作り直す（2〜4 は前の測定のまま）
        p = OUT / ("check_smoke.json" if a.smoke else "check.json")
        old = json.loads(p.read_text(encoding="utf-8"))
        old.update(table=table, cond1_pass=cond1_pass(table), cond1_rule=COND1_RULE,
                   table_rewritten=time.strftime("%Y-%m-%d %H:%M:%S"))
        p.write_text(json.dumps(old, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"[f] rewrote table in {p}", flush=True)
        return
    # 2: 保存した全エピソードの最初のフレーム
    saved = sorted(p for p in run.iterdir() if p.is_dir() and not p.name.endswith(".partial"))
    cond2 = []
    for p in saved:
        ep = legacy.load_episode(p)
        c = inject.start_condition(ep.data, ep.meta)
        cond2.append({"episode": p.name, "ok": c["ok"], "checks": c["checks"]})
    # 3: 種類ごとに無作為の 3 本を再生（step はビット一致と画像、window は手先 5 mm 以内）
    rng = seeds.stream(52900, "order")
    rig = SimRig(render=True)
    replay_rows = []
    try:
        for kind in CHECK_SEED_BASE:
            eps = [p for p in saved if p.name.startswith(kind + "_")]
            for i in rng.choice(len(eps), min(REPLAY_PER_KIND, len(eps)), replace=False):
                ep = legacy.load_episode(eps[i])
                row = {"episode": eps[i].name}
                for mode in ("step", "window"):
                    rep = RS.replay(ep, mode, rig, render=True)
                    r = RS.compare(ep, rep, images=True)
                    r["pass"] = RS.verdict(mode, r)
                    row[mode] = r
                replay_rows.append(row)
    finally:
        rig.close()
    # 4: 種類ごとに 5 本の映像（俯瞰と手首を横に並べる）
    import cv2
    import shutil
    import tempfile
    from recovla.record.recorder import read_png
    vdir = OUT / "videos"
    vdir.mkdir(parents=True, exist_ok=True)
    made = []
    for kind in CHECK_SEED_BASE:
        for p in [p for p in saved if p.name.startswith(kind + "_")][:VIDEOS_PER_KIND]:
            n = json.loads((p / "meta.json").read_text(encoding="utf-8"))["n_frames"]
            tmp = pathlib.Path(tempfile.mkdtemp()) / "v.mp4"
            vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), 20, (512, 256))
            for i in range(n):
                im = np.hstack([read_png(p / v / f"{i:06d}.png") for v in CFG["sim"]["cameras"]])
                vw.write(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            vw.release()
            dst = vdir / f"{p.name}.mp4"
            shutil.move(str(tmp), str(dst))
            made.append(str(dst.relative_to(config.ROOT)))
    write("check_smoke" if a.smoke else "check", {
        "run": str(run.relative_to(config.ROOT)), "table": table,
        "cond1_pass": cond1_pass(table), "cond1_rule": COND1_RULE,
        "cond2_episodes": len(cond2), "cond2_pass": bool(cond2) and all(c["ok"] for c in cond2),
        "cond2_failures": [c for c in cond2 if not c["ok"]],
        "cond3_rows": replay_rows,
        "cond3_pass": bool(replay_rows) and all(r["step"]["pass"] and r["window"]["pass"] for r in replay_rows),
        "cond4_videos": made,
        "cond4_pass": all(sum(1 for m in made if pathlib.Path(m).name.startswith(k + "_")) >= VIDEOS_PER_KIND
                          for k in CHECK_SEED_BASE),
        "seed_ranges": {k: [v, v + a.n - 1] for k, v in CHECK_SEED_BASE.items()}, "replay_order_seed": 52900})


# ---------------------------------------------------------------------------- R1・N1 の学習データの計画

# 成功データ 240 本（空の箱 60 配置 x 3 色、先客 1 個 15 配置 x 2 色、先客 2 個 30 配置 x 1 色）は R1・N1 共通。
# 立て直しデータ 90 本（A40・B30・C20）は別のシード（30000〜）で、配置の種類の割合（75・12.5・12.5%）を成功データと揃える。
# N1 の追加 90 本は、立て直しと同じ配置・同じ目標の成功データ（一度で成功するもの）。
NORMAL_SEEDS = {"empty": (20000, 60), "prefilled_1": (20100, 15), "prefilled_2": (20200, 30)}
RECOVERY_NEED = {"A": 40, "B": 30, "C": 20}
RECOVERY_SEED_BASE = {"A": 30000, "B": 31000, "C": 32000}
RECOVERY_KIND_SHARE = {"empty": 0.75, "prefilled_1": 0.125, "prefilled_2": 0.125}
CANDIDATE_FACTOR = 2.0               # 捨てる分を見込んで、各枠の候補をこの倍数だけ先に決めておく（使うのは先頭から）


def split_counts(total: int) -> dict:
    """種類ごとの必要数を配置の種類へ割り振る（最大剰余法。成功データの割合 180:30:30 と同じ）。"""
    raw = {k: total * s for k, s in RECOVERY_KIND_SHARE.items()}
    out = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - out[k], reverse=True)[:total - sum(out.values())]:
        out[k] += 1
    return out


def cmd_plan(a) -> None:
    from recovla.sim import scene
    normal = []
    for lk, (base, n) in NORMAL_SEEDS.items():
        for seed in range(base, base + n):
            lay = scene.sample_layout(seed, lk)
            normal += [{"seed": seed, "color": c, "layout_kind": lk, "start": lay.start} for c in lay.table_colors]
    recovery = {}
    for kind, need in RECOVERY_NEED.items():
        cells = {}
        seed = RECOVERY_SEED_BASE[kind]
        for lk, k in split_counts(need).items():
            cand = []
            for i in range(int(round(k * CANDIDATE_FACTOR))):
                lay = scene.sample_layout(seed, lk)
                cols = lay.table_colors
                cand.append({"seed": seed, "color": cols[i % len(cols)], "layout_kind": lk, "start": lay.start})
                seed += 1
            cells[lk] = {"need": k, "candidates": cand}
        recovery[kind] = cells

    def comp(rows):
        return {"n": len(rows), "color": dict(collections.Counter(r["color"] for r in rows)),
                "layout_kind": dict(collections.Counter(r["layout_kind"] for r in rows)),
                "start": dict(collections.Counter(r["start"] for r in rows))}
    first = [c for cells in recovery.values() for cell in cells.values() for c in cell["candidates"][:cell["need"]]]
    # 所要時間の見積もり: 振りの 1 本の時間（描画なし）と、描画ありの並列の本/時（outputs/d/throughput.json）
    est = {}
    sweep = OUT / "sweep.json"
    if sweep.is_file():
        pts = json.loads(sweep.read_text(encoding="utf-8"))["points"]
        est["recovery_attempt_s_no_render"] = {p["point"]: p["wall_attempt_s_mean"] for p in pts}
    thr = config.path(CFG["paths"]["outputs"]) / "d" / "throughput.json"
    if thr.is_file():
        est["normal_throughput_d"] = {r["workers"]: r["episodes_per_hour"]
                                      for r in json.loads(thr.read_text(encoding="utf-8"))["rows"]}
    write("plan", {"normal": {"specs": normal, "composition": comp(normal), "seed_ranges": NORMAL_SEEDS},
                   "recovery": recovery, "recovery_first_choice_composition": comp(first),
                   "recovery_seed_base": RECOVERY_SEED_BASE, "candidate_factor": CANDIDATE_FACTOR,
                   "n1_extra": "復帰に採った 90 の（種、色）で通常デモ（kind n）を生成する。どちらかが作り直しを含めて失敗したら"
                               "その候補を両方から外し、同じ枠の次の候補を使う",
                   "estimate_inputs": est})


# --------------------------------------------------------------- R1・N1 の学習データの生成

DROPPED_STOP = 0.10            # 捨てた指定の割合がこれを超えたら止める


def layout_targets(entries: list, exclude: list) -> list:
    """マニフェストの名前（<種類>_<シード>_<色>_r<回>）から、exclude 以外の（シード、色）を並べる。"""
    ex = set(exclude)
    out = []
    for e in entries:
        if e["key"] in ex:
            continue
        _, seed, color, _ = e["key"].split("_")
        out.append((int(seed), color))
    return sorted(out)


def cmd_gen_data(a) -> None:
    """plan.json のとおりに生成し、R1・N1 のマニフェストを作って LeRobot 形式に変換する。

    1 回の生成（outputs/gen/F_data_<日時>）に、成功データ 240 の指定、立て直しの候補（枠ごとに必要数の 2 倍）、その候補と
    同じ（シード、色）の成功データの指定をまとめて入れる。立て直しと対の成功データの両方が（作り直しを含めて）成功した
    候補を、枠ごとに候補の順に必要数だけ採る。成功データ 240 で捨てた（シード、色）は R1・N1 の両方から外れる。"""
    from recovla.data import convert as C
    from recovla.expert import generate as G
    plan = json.loads((OUT / "plan.json").read_text(encoding="utf-8"))
    normal = [G.EpisodeSpec(s["seed"], s["color"], s["layout_kind"], "n") for s in plan["normal"]["specs"]]
    rec_specs, twin_specs, cells = [], [], []
    for kind, lk_cells in plan["recovery"].items():
        for lk, cell in lk_cells.items():
            cells.append((kind, lk, cell["need"], cell["candidates"]))
            for c in cell["candidates"]:
                rec_specs.append(G.EpisodeSpec(c["seed"], c["color"], lk, kind))
                twin_specs.append(G.EpisodeSpec(c["seed"], c["color"], lk, "n"))
    specs = normal + rec_specs + twin_specs
    run = GEN / f"F_data{'' if a.rig == 'v1' else '_' + a.rig}{'_smoke' if a.smoke else ''}_{time.strftime('%Y%m%d-%H%M%S')}"
    if a.smoke:                                            # 動作の確認: 各群から少しだけ
        specs = normal[:6] + rec_specs[::30] + twin_specs[::30]
    t0 = time.perf_counter()
    results = G.generate(specs, run, workers=a.workers, render=True, rig_kind=a.rig)   # v2・v3 はセンサのモデルを通して記録する（harness/gen_sensed.py）
    gen_wall = time.perf_counter() - t0
    by = {(r["kind"], r["layout_seed"], r["color"]): r for r in results}

    def saved_name(r):
        return r["attempts"][-1]["name"] if r["success"] else None

    normal_ok = [by[("n", s.layout_seed, s.color)] for s in normal if ("n", s.layout_seed, s.color) in by]
    normal_keys = [saved_name(r) for r in normal_ok if r["success"]]
    chosen, cell_rows = [], []
    for kind, lk, need, cands in cells:
        got, examined, dropped = [], 0, 0
        for c in cands:
            rk, tk = (kind, c["seed"], c["color"]), ("n", c["seed"], c["color"])
            if rk not in by or tk not in by:
                continue                                    # 動作の確認で間引いたもの
            if len(got) >= need:
                break
            examined += 1
            if by[rk]["success"] and by[tk]["success"]:
                got.append({"kind": kind, "layout_kind": lk, "seed": c["seed"], "color": c["color"],
                            "recovery": saved_name(by[rk]), "twin": saved_name(by[tk]), "start": c["start"]})
            else:
                dropped += 1
        chosen += got
        cell_rows.append({"kind": kind, "layout_kind": lk, "need": need, "got": len(got), "examined": examined,
                          "dropped": dropped, "short": need - len(got)})
    runrel = str(run.relative_to(config.ROOT)).replace("\\", "/")
    entries_r1 = [{"run": runrel, "key": k} for k in normal_keys] + [{"run": runrel, "key": c["recovery"]} for c in chosen]
    entries_n1 = [{"run": runrel, "key": k} for k in normal_keys] + [{"run": runrel, "key": c["twin"]} for c in chosen]
    stamp = run.name.rsplit("_", 1)[-1] if a.rig in ("v2", "v3") else run.name.split("_", 2)[-1]
    if a.smoke:
        stamp = "smoke_" + run.name.rsplit("_", 1)[-1]
    # v2・v3: 目標位置の入力の形（旗の採否）は data_cue.json と同じ、閾値は今の実行系の値（runtime_v2）で変換する
    cue_args = []
    if a.rig in ("v2", "v3"):
        keep_flag = json.loads((OUT / "data_cue.json").read_text(encoding="utf-8"))["keep_flag_in_input"]
        cue_args = ["--target-cue", "--cue-thresholds", "runtime_v2"] + ([] if keep_flag else ["--cue-without-flag"])
    suffix = "" if a.rig == "v1" else a.rig
    out = {}
    for name, entries, rule in (
            ("R1", entries_r1, "Step F R1: normal demos (seeds 20000-) + recovery A/B/C (seeds 30000-), plan.json"),
            ("N1", entries_n1, "Step F N1: normal demos (seeds 20000-) + one-shot normal demos on the same recovery layouts")):
        dname = f"{name}{suffix}_{stamp}"
        mpath = config.path(CFG["paths"]["outputs"]) / "manifests" / f"{dname}.json"
        C.write_manifest(mpath, dname, entries, rule + {"": "", "v2": " (goals v2 stage 2: sensor-v1, rig v2)", "v3": " (goals v3 stage 3 round 1: sensor-v1, rig v2 + configs/expert_v3.yaml)"}[suffix])
        ds = config.path(CFG["paths"]["outputs"]) / "datasets" / dname
        log = OUT / f"convert_{dname}.log"
        t1 = time.perf_counter()
        with open(log, "w", encoding="utf-8") as f:
            c1 = subprocess.run([sys.executable, "-m", "recovla.data.convert", "--manifest", str(mpath), "--out",
                                 str(ds), "--name", dname, *cue_args], stdout=f, stderr=subprocess.STDOUT)
            c2 = subprocess.run([sys.executable, "-m", "recovla.data.convert", "--verify", str(ds)],
                                stdout=f, stderr=subprocess.STDOUT)
        text = log.read_text(encoding="utf-8", errors="replace")
        info = json.loads((ds / "meta" / "info.json").read_text(encoding="utf-8")) if (ds / "meta" / "info.json").is_file() else {}
        out[name] = {"manifest": str(mpath.relative_to(config.ROOT)), "dataset": str(ds.relative_to(config.ROOT)),
                     "episodes": len(entries), "frames": info.get("total_frames"), "convert_exit": c1.returncode,
                     "verify_exit": c2.returncode, "verify_pass": "[verify] PASS" in text,
                     "convert_wall_s": round(time.perf_counter() - t1, 1), "log": str(log.relative_to(config.ROOT))}
    # 構成表
    def comp(rows):
        return {"n": len(rows), "color": dict(collections.Counter(r["color"] for r in rows)),
                "layout_kind": dict(collections.Counter(r["layout_kind"] for r in rows)),
                "start": dict(collections.Counter(r["start"] for r in rows))}
    nspec = {(s["seed"], s["color"]): s for s in plan["normal"]["specs"]}
    normal_rows = [nspec[(r["layout_seed"], r["color"])] for r in normal_ok if r["success"]]
    rec_by_kind = {k: comp([c for c in chosen if c["kind"] == k]) for k in RECOVERY_NEED}
    fr = {k: out[k]["frames"] for k in out}
    frame_diff = (abs(fr["R1"] - fr["N1"]) / max(fr["R1"], fr["N1"])) if all(fr.values()) else None
    dropped_by_kind = {}
    for k in RECOVERY_NEED:
        rows_k = [r for r in cell_rows if r["kind"] == k]
        ex, dr = sum(r["examined"] for r in rows_k), sum(r["dropped"] for r in rows_k)
        dropped_by_kind[k] = [dr, ex, wilson(dr, ex) if ex else [None, None]]
    n_dropped = [len(normal) - len(normal_rows), len(normal), wilson(len(normal) - len(normal_rows), len(normal))] \
        if not a.smoke else None
    stop = any(v[1] and v[0] / v[1] > DROPPED_STOP for v in dropped_by_kind.values())
    from recovla.common import code_version
    write(("data_smoke" if a.smoke else "data") + ("" if a.rig == "v1" else "_" + a.rig), {
        "run": runrel, "generation_wall_s": round(gen_wall, 1), "workers": a.workers, "rig": a.rig,
        "cue_convert_args": cue_args,
        # 生成に使ったコミットと設定の値
        "code_version": code_version.code_version(),
        "config_used": {"inject": CFG["inject"], "expert": (G.rig_config(a.rig) or CFG)["expert"], "scene": CFG["scene"], "sim": CFG["sim"]},
        "normal": {"composition": comp(normal_rows), "dropped": n_dropped},
        "recovery": {"by_kind": rec_by_kind, "cells": cell_rows, "dropped_by_kind": dropped_by_kind,
                     "composition": comp(chosen)},
        "chosen": chosen, "datasets": out, "frame_diff_ratio_R1_N1": frame_diff,
        # マニフェストの名前から数え直す。R1 と N1 で、成功データの部分は同じ名前、
        # 残り（R1 の立て直し・N1 の対の成功データ）は（シード、色）の集合が一致する
        "same_layouts_and_targets": layout_targets(entries_r1, normal_keys) == layout_targets(entries_n1, normal_keys),
        "normal_part_identical": sorted(e["key"] for e in entries_r1 if e["key"] in set(normal_keys)) == sorted(
            e["key"] for e in entries_n1 if e["key"] in set(normal_keys)),
        "dropped_stop_rule": f"復帰の種類ごとの捨てた割合が {DROPPED_STOP:.0%} を超えたら止めて諮る（決裁 0042）",
        "stop": stop})


# ------------------------------------------------------ R1・N1 に目標位置の入力を足す

def cmd_convert_cue(a) -> None:
    """同じ生の記録（data.json のマニフェスト）を、目標位置の入力つきで変換し直す（生成し直さない）。
    旗を入力に残すかは、単体検査（scripts/24_cue_check.py --data → outputs/f/cue_check.json）の判定に従う。"""
    d = json.loads((OUT / "data.json").read_text(encoding="utf-8"))
    chk = json.loads((OUT / "cue_check.json").read_text(encoding="utf-8"))
    keep = {k: v["keep_flag_in_input"] for k, v in chk["training_flag0"].items()}
    if len(set(keep.values())) != 1:
        raise SystemExit(f"R1 と N1 で旗の判定が違う {keep}（揃えるかを諮る）")
    keep_flag = next(iter(keep.values()))
    stamp_ = time.strftime("%Y%m%d-%H%M%S")
    procs = {}
    for name in ("R1", "N1"):
        dname = f"{name}cue_{stamp_}"
        ds = config.path(CFG["paths"]["outputs"]) / "datasets" / dname
        log = OUT / f"convert_{dname}.log"
        cmd = [sys.executable, "-m", "recovla.data.convert", "--manifest", str(config.path(d["datasets"][name]["manifest"])),
               "--out", str(ds), "--name", dname, "--target-cue"] + ([] if keep_flag else ["--cue-without-flag"])
        f = open(log, "w", encoding="utf-8")
        procs[name] = (subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT), f, ds, log, dname)
    t0 = time.perf_counter()
    out = {}
    for name, (p, f, ds, log, dname) in procs.items():
        c1 = p.wait()
        f.close()
        with open(log, "a", encoding="utf-8") as f2:
            c2 = subprocess.run([sys.executable, "-m", "recovla.data.convert", "--verify", str(ds)],
                                stdout=f2, stderr=subprocess.STDOUT).returncode
        text = log.read_text(encoding="utf-8", errors="replace")
        info = json.loads((ds / "meta" / "info.json").read_text(encoding="utf-8")) if (ds / "meta" / "info.json").is_file() else {}
        conv = json.loads((ds / "meta" / "conversion.json").read_text(encoding="utf-8")) if (ds / "meta" / "conversion.json").is_file() else {}
        out[name] = {"dataset": str(ds.relative_to(config.ROOT)), "episodes": info.get("total_episodes"),
                     "frames": info.get("total_frames"), "frames_15d": d["datasets"][name]["frames"],
                     "state_names": conv.get("state"), "cue_flag0_frames": sum(s.get("cue_flag0_frames", 0)
                                                                               for s in conv.get("sources", [])),
                     "convert_exit": c1, "verify_exit": c2, "verify_pass": "[verify] PASS" in text,
                     "verify_fails": [l.strip() for l in text.splitlines() if l.strip().startswith("FAIL")],
                     "log": str(log.relative_to(config.ROOT))}
    from recovla.common import code_version
    write("data_cue", {"source": "data.json", "keep_flag_in_input": keep_flag, "flag_decision": chk["training_flag0"],
                       "flag_rule": chk["flag_rule"], "datasets": out, "wall_s": round(time.perf_counter() - t0, 1),
                       "code_version": code_version.code_version(), "thresholds": chk["thresholds"]})


# ------------------------------------------------------------------ 目視用の資料

REVIEW_PER_KIND = 10
REVIEW_PRE_S = 1.0             # 失敗の発動のこの秒数前から写す


def _review_video(rig, spec, retry: int, path: pathlib.Path, saved_meta: dict) -> dict:
    """同じシード・同じ作り直しの回数で最初から回し直し、20 Hz ごとに描いて、失敗の発動の少し前から終わりまでを mp4 にする。
    描画は scratch への複写に対して行うので物理は変わらない。回し直しが保存したものと同じかも確かめる。"""
    import cv2
    import numpy as np
    from recovla.expert import generate as G
    shots = []
    original = rig.pad_read

    def pad_read(vel=None, press=False, on_step=None):
        def hook(r):
            if on_step is not None:
                on_step(r)
            if r.step % r.record_every == 0:
                r.forward_scratch()
                shots.append((r.step * r.timestep, r.render()))
        return original(vel, press, hook)

    rig.pad_read = pad_read
    try:
        a = G.run_attempt(rig, spec, retry, None, False)
    finally:
        rig.pad_read = original
    inj = a["inject"]
    t0 = max(0.0, float(inj["t_fire"]) - REVIEW_PRE_S)
    t_rec = a["t_record_start"]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (512, 256))
    for t, imgs in shots:
        if t < t0 - 1e-9:
            continue
        im = cv2.cvtColor(np.hstack(imgs), cv2.COLOR_RGB2BGR)
        label = "injection" if t < (t_rec or 1e9) else "recorded (recovery)"
        if inj["t_fire"] is not None and t < float(inj["t_fire"]):
            label = "before injection"
        cv2.putText(im, f"{spec.kind} {spec.layout_seed} {spec.color} t={t:5.2f}s {label}", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        vw.write(im)
    vw.release()
    return {"video": str(path.relative_to(config.ROOT)), "t_fire": inj["t_fire"], "t_record_start": t_rec,
            "reproduced": bool(a["success"]) and a["record_start_step"] == saved_meta.get("record_start_step"),
            "a_mode": inj["params"].get("a_mode")}


def cmd_review(a) -> None:
    """R1 の立て直しデータから種類ごとに 10 本の映像（失敗の発動の 1 秒前から立て直しの完了まで）と、90 本の最初のフレームの一覧。"""
    import cv2
    import numpy as np
    from recovla.expert import generate as G
    from recovla.record.recorder import read_png
    from recovla.sim.rig import SimRig
    data = json.loads((OUT / ("data_smoke.json" if a.smoke else "data.json")).read_text(encoding="utf-8"))
    run = config.ROOT / data["run"]
    chosen = data["chosen"]
    rdir = OUT / ("review_smoke" if a.smoke else "review")
    rdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in chosen:
        meta = json.loads((run / c["recovery"] / "meta.json").read_text(encoding="utf-8"))
        rows.append({**c, "meta": meta})
    # 一覧: 種類ごとに 1 枚（俯瞰と手首を 128 px に縮めて横に並べ、種類・シード・色・向きの角を添える）
    sheets = []
    tiles_all = []
    for kind in RECOVERY_NEED:
        tiles = []
        for r in [r for r in rows if r["kind"] == kind]:
            ep = run / r["recovery"]
            im = np.hstack([cv2.resize(read_png(ep / v / "000000.png"), (128, 128), interpolation=cv2.INTER_AREA)
                            for v in CFG["sim"]["cameras"]])
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            bar = np.full((18, 256, 3), 255, np.uint8)
            yaw = r["meta"]["inject"]["info"].get("yaw_rel_deg")
            mode = r["meta"]["inject"]["params"].get("a_mode", "")
            txt = f"{kind} {r['seed']} {r['color']} yaw {yaw:+.0f}" + (f" {mode}" if mode else "")
            cv2.putText(bar, txt, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            tiles.append(np.vstack([bar, im]))
        tiles_all += tiles
        cols = 5
        while len(tiles) % cols:
            tiles.append(np.full_like(tiles[0], 255))
        grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
        p = rdir / f"first_frames_{kind}.png"
        cv2.imwrite(str(p), grid)
        sheets.append(str(p.relative_to(config.ROOT)))
    # 映像: 種類ごとに先頭から 10 本
    rig = SimRig(render=True)
    videos = []
    try:
        for kind in RECOVERY_NEED:
            for r in [r for r in rows if r["kind"] == kind][:REVIEW_PER_KIND]:
                retry = int(r["recovery"].rsplit("_r", 1)[1])
                spec = G.EpisodeSpec(r["seed"], r["color"], r["layout_kind"], kind)
                videos.append({"episode": r["recovery"],
                               **_review_video(rig, spec, retry, rdir / f"{r['recovery']}.mp4", r["meta"])})
    finally:
        rig.close()
    write("review_smoke" if a.smoke else "review", {
        "data": data["run"], "sheets": sheets, "videos": videos,
        "all_reproduced": all(v["reproduced"] for v in videos),
        "first_frames_by_kind": {k: sum(1 for r in rows if r["kind"] == k) for k in RECOVERY_NEED},
        "note": "映像は同じ種・同じ作り直しの回数で最初から回し直して描いたもの（注入の 1 秒前から）。保存したデータは"
                "「recorded (recovery)」の表示の部分だけ"})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sweep")
    s.add_argument("--workers", type=int, default=1)
    s.add_argument("--n", type=int, default=120)
    s.add_argument("--chunk", type=int, default=10)
    s.add_argument("--smoke", action="store_true")
    for name in ("check-gen", "check-eval"):
        s = sub.add_parser(name)
        s.add_argument("--workers", type=int, default=1)
        s.add_argument("--n", type=int, default=60)
        s.add_argument("--smoke", action="store_true")
        s.add_argument("--rig", default="v1", choices=("v1", "v2", "v3"), help="v2: 目標書 v2 の口とセンサの模型で記録する（段階 2）。v3: v2 ＋ 案 1 のエキスパート（configs/expert_v3.yaml、段階 3＝0126）")
        if name == "check-eval":
            s.add_argument("--table-only", action="store_true", help="完了条件 1 の表だけを作り直す（描画しない）")
    sub.add_parser("plan")
    s = sub.add_parser("gen-data")
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--smoke", action="store_true", help="各群から少しだけ（通しの確認）")
    s.add_argument("--rig", default="v1", choices=("v1", "v2", "v3"), help="v2: 目標書 v2 の口とセンサの模型で記録する（段階 2）。v3: v2 ＋ 案 1 のエキスパート（configs/expert_v3.yaml、段階 3＝0126）")
    s = sub.add_parser("review")
    s.add_argument("--smoke", action="store_true", help="data_smoke.json の回から作る")
    sub.add_parser("convert-cue")
    a = ap.parse_args(argv)
    {"sweep": cmd_sweep, "check-gen": cmd_check_gen, "check-eval": cmd_check_eval, "plan": cmd_plan,
     "gen-data": cmd_gen_data, "review": cmd_review, "convert-cue": cmd_convert_cue}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
