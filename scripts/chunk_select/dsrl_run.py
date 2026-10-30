"""行動列の選び方の学習（ノイズ空間の強化学習 DSRL と、成功の見込みの学習。docs/api/r2c_plan.md）の、試行の集めと学び。
事前に決めた比較の外の追加の試み。

    python scripts/chunk_select/dsrl_run.py worker --band C_pilot --max-episodes 2 --out outputs/r2c/smoke     # 試しの実行（学習に使わない）
    python scripts/chunk_select/dsrl_run.py worker --band C_train --shard 0/3 --out outputs/r2c/run1 --actor outputs/r2c/run1/actor.pt
    python scripts/chunk_select/dsrl_run.py learner --out outputs/r2c/run1 [--episodes-glob ...] [--until "2026-10-27 22:00"]

worker: 自分の成功を集めるときと同じ 1 手順の試行（selfroll_real.run_r2_selfroll、v3_real・naive・6 ステップごとに推論・開閉の絞り K=5、
  記録はしない）を、方策に入れる雑音を小さな方策（actor）に選ばせて回す。試行ごとに <out>/episodes/<シード>.npz（特徴・雑音・報酬）
  と .json（要約）。報酬は selfroll の「保存する」判定（成功し、捨てる理由がない）なら 1、それ以外 0。actor.pt が新しくなったら
  次の試行から読み直す。--select では、ふつうの雑音から K 本の行動列を作り、学習した成功の見込み（--scorer）で選ぶ。
  続きから回せる（同じ --out で、終わったシードを飛ばす）。
learner: episodes/ の新しい試行を溜めて SAC を学び、actor.pt を置き換える（1 分ごと）。state.pt から続けられる。
learner-select: 試行の成功を当てる見込みを学び、scorer.pt を置き換える（1 分ごと）。
シードの範囲（BANDS、学習用）: C_train 80000〜89999、C_pilot 79900〜79999（試し、学習に使わない）。
"""
import argparse
import datetime
import glob
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "failure_relearn"))
sys.path.insert(0, str(ROOT / "scripts" / "success_relearn"))

import failure_data as R1D  # noqa: E402
import selfroll_data as R2D  # noqa: E402
from recovla.common import config  # noqa: E402

BANDS = {"C_train": (80000, 89999), "C_pilot": (79900, 79999)}
UTD = 2                  # 新しい遷移 1 つあたりの更新の回数
MIN_TRANSITIONS = 1000   # これだけ溜まるまで学ばない（約 37 試行。少ない記録で何度も学んで Q が外れるのを避ける）
SAVE_EVERY_S = 60.0
STATE_EVERY_S = 600.0


def until_dt(s):
    return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M") if s else None


def cmd_worker(a) -> int:
    from recovla.common import code_version
    from recovla.expert.selfroll_real import run_r2_selfroll
    from recovla.harness import real_matched as V
    from recovla.rl import dsrl
    i, n = (int(v) for v in a.shard.split("/"))
    lo, hi = BANDS[a.band]
    R1D.check_band(lo, hi)
    used = set()                       # 前の回で使ったシードは使い直さない（学習用のシードは 1 回ずつ）
    for d in a.skip_seeds_from or []:
        for p in pathlib.Path(d).glob("*"):
            if p.suffix in (".npz", ".skip") and p.stem.isdigit():
                used.add(int(p.stem))
    seeds = [s for s in range(lo, hi + 1) if (s - lo) % n == i and s not in used]
    if used:
        print(f"[r2c] 前の回で使った種 {len(used)} 個を飛ばす", flush=True)
    out = config.path(a.out)
    epi = out / "episodes"
    epi.mkdir(parents=True, exist_ok=True)
    if a.model == "Rdag":
        ckpt = R2D.rdag_ckpt(a.ckpt)
        sha = R2D.check_ckpt(ckpt)
    else:                              # 土台を R2a にするとき。チェックポイントの SHA-256 は起動の側が記録から渡す
        if not (a.ckpt and a.ckpt_sha256):
            raise SystemExit(f"--model {a.model} には --ckpt と --ckpt-sha256 が要る")
        ckpt = str(config.path(a.ckpt))
        sha = R1D.sha256_file(pathlib.Path(ckpt) / "model.safetensors")
        if sha != a.ckpt_sha256:
            raise SystemExit(f"{a.model} の保存点の SHA-256 が渡された値と違う: {sha}")
    cfg_eval = R1D.eval_config()
    base = R1D.runtime_factory(ckpt, cfg_eval)
    if a.select:                       # ふつうの雑音から K 本の行動列を作り、見込みで選ぶ（recovla.rl.select）
        from recovla.rl import select as S
        sp = None if not a.scorer else (a.scorer if a.scorer == "random" else config.path(a.scorer))
        steer = S.Select(sp, k=a.k, control=a.control)
    else:
        steer = dsrl.Steer(config.path(a.actor) if a.actor else None, deterministic=a.deterministic, seed=lo + i)
    cur = {"seed": None}

    def make_runtime(io, setup):
        rt = base(io, setup)
        if a.select:
            S.attach(rt.policy, steer)
            steer.begin_episode(cur["seed"])
        else:
            dsrl.attach(rt.policy, steer)
            steer.begin_episode()
        return rt

    start = {"model": a.model, "checkpoint": str(ckpt), "checkpoint_sha256": sha, "world": R1D.WORLD, "band": a.band,
             "shard": a.shard, "actor": a.actor, "deterministic": bool(a.deterministic), "lat_dim": dsrl.LAT_DIM,
             "b_w": dsrl.B_W, "select": bool(a.select), "scorer": a.scorer, "k": a.k, "control": a.control,
             "code_version": code_version.code_version(), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    for old in out.glob("start_*.json"):             # 1 つの回に別の土台の試行を混ぜない
        o = json.loads(old.read_text(encoding="utf-8"))
        if o.get("checkpoint_sha256") != sha:
            raise SystemExit(f"{old} は別の保存点（{o.get('model')}）の回。--out を替える")
    R1D.write_json(out / f"start_{a.band}_s{i}of{n}.json", start)
    deadline = until_dt(a.until)
    rig = V.V3RealGenRig(V.load_config(R1D.WORLD, generation=True))
    log_p = out / f"worker_{a.band}_s{i}of{n}.jsonl"
    done_n = 0
    try:
        with open(log_p, "a", encoding="utf-8") as log:
            for seed in seeds:
                if a.max_episodes and done_n >= a.max_episodes:
                    break
                if deadline and datetime.datetime.now() >= deadline:
                    print(f"[r2c] 打ち切りの時刻 {a.until} に達した", flush=True)
                    break
                p = epi / f"{seed}.npz"
                if p.is_file() or p.with_suffix(".skip").is_file():
                    continue
                if (out / "STOP").exists():
                    print("[r2c] STOP を見た", flush=True)
                    break
                steer.reload()
                ver = steer.meta.get("version")
                cur["seed"] = seed
                summ = run_r2_selfroll(rig, make_runtime, seed, None, f"{a.model}+C", cfg_eval)
                done_n += 1
                rows = list(steer.rows)
                reward = 1.0 if summ["saved"] else 0.0
                rec = {"seed": seed, "reward": reward, "discard": summ["discard"], "success": summ["success"],
                       "recovered": summ["recovered"], "stratum": summ["rollout_layout"]["stratum"], "color": summ["color"],
                       "t_end": summ["t_end"], "n_inference": len(rows), "actor_version": ver, "wall_s": summ["wall_s"]}
                if a.select:
                    rec.update(control=bool(steer.control), chose=bool(rows and rows[0]["k"] > 1))
                    arrays = S.episode_arrays(rows, reward, steer.control) if rows else None
                else:
                    arrays = dsrl.episode_arrays(rows, reward) if rows else None
                if rows:
                    dsrl.write_episode(p, arrays, dict(rec, summary=summ))
                else:
                    p.with_suffix(".skip").write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
                log.write(json.dumps(rec, ensure_ascii=False, default=float) + "\n")
                log.flush()
                tag = (" ctrl" if rec["control"] else (" sel" if rec["chose"] else " base")) if a.select else ""
                print(f"[r2c] {seed} {summ['color']:5s} {rec['stratum']:4s} r {reward:.0f} discard {summ['discard']} "
                      f"inf {len(rows)} actor {ver}{tag} t {summ['t_end']:5.1f} wall {summ['wall_s']:.0f}s", flush=True)
    finally:
        rig.close()
    return 0


def cmd_learner(a) -> int:
    import torch
    from recovla.rl import dsrl
    out = config.path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pats = a.episodes_glob or [str(out / "episodes" / "*.npz")]
    sac = dsrl.SAC(seed=a.seed)
    state_p = out / "state.pt"
    if state_p.is_file():
        sac.load_state(torch.load(state_p, map_location="cpu", weights_only=False))
        print(f"[r2c] state.pt から続ける（更新 {sac.n_updates}）", flush=True)
    actor_p = config.path(a.actor) if a.actor else out / "actor.pt"
    loaded, rewards = set(), []
    budget = 0
    t_save = t_state = 0.0
    deadline = until_dt(a.until)
    log = open(out / "learner.jsonl", "a", encoding="utf-8")
    resumed = sac.n_updates > 0                     # 続きのときは、溜め直した試行の分の更新はしない（学び済み）

    def save(kind):
        meta = {"version": sac.n_updates, "episodes": len(loaded), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        dsrl.save_actor(sac.actor, actor_p, meta)
        for m in a.mirror or []:                      # 別の PC の worker が読む写し（共有フォルダ）。失敗しても止めない
            try:
                dsrl.save_actor(sac.actor, m, meta)
            except OSError as e:
                print(f"[r2c-learn] 写しに失敗 {m}: {e}", flush=True)
        if kind == "state":
            buf = out / "state.pt.tmp"
            torch.save(sac.state(), buf)
            buf.replace(state_p)
        return meta

    save("actor")                                   # worker が最初から同じ重みを読む
    while True:
        new = []
        for pat in pats:
            for f in sorted(glob.glob(pat)):
                if f in loaded or f.endswith(".tmp.npz"):
                    continue
                try:
                    d = np.load(f)
                    new.append((f, d["feat"], d["w"], float(d["reward"])))
                except Exception:
                    continue                          # 書きかけ。次の回で読む
        for f, feat, w, r in new:
            sac.add_episode(feat, w, r)
            loaded.add(f)
            rewards.append(r)
            if not resumed:
                budget += UTD * len(feat)
        resumed = False
        info = {}
        if budget > 0 and len(sac) >= max(sac.batch, MIN_TRANSITIONS):
            k = min(budget, 2000)
            info = sac.update(k)
            budget -= k
        now = time.time()
        if new or info:
            if now - t_save >= SAVE_EVERY_S:
                meta = save("actor")
                t_save = now
                last = rewards[-100:]
                row = {**meta, "transitions": len(sac), "success_last100": round(float(np.mean(last)), 3) if last else None,
                       **{k: round(v, 4) for k, v in info.items()}}
                if info and not (abs(info["q_mean"]) <= 2.0 and info["alpha"] <= dsrl.ALPHA_MAX * 1.01):
                    row["warn"] = "q_mean か alpha が範囲の外（Q は 0〜1、alpha は ALPHA_MAX まで）"
                log.write(json.dumps(row, ensure_ascii=False) + "\n")
                log.flush()
                print(f"[r2c-learn] {row}", flush=True)
            if now - t_state >= STATE_EVERY_S:
                save("state")
                t_state = now
        if (out / "STOP").exists() or (deadline and datetime.datetime.now() >= deadline):
            save("state")
            print("[r2c-learn] 終わり", flush=True)
            return 0
        if not new and budget <= 0:
            time.sleep(10)


def cmd_learner_select(a) -> int:
    """試行の成功を当てる見込み（recovla.rl.select.Scorer）を学び、scorer.pt を置き換える（1 分ごと）。
    新しい試行は、学ぶ前の見込みで当てた値を先に記録する（前向きの AUC。学びに使う前の試行で測るので、覗き見にならない）。"""
    import torch
    from recovla.rl import select as S
    out = config.path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pats = a.episodes_glob or [str(out / "episodes" / "*.npz")]
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    sc = S.Scorer()
    opts = S.make_opts(sc)
    scorer_p = config.path(a.scorer) if a.scorer else out / "scorer.pt"
    X, y, grp, eps = [], [], [], []           # grp: 網ごとの重み (E, T)。eps: (シード, 報酬, 対照か, 選んだか, 学ぶ前の見込み)
    loaded = set()
    version, t_save = 0, 0.0
    deadline = until_dt(a.until)
    log = open(out / "learner.jsonl", "a", encoding="utf-8")
    trained = False
    while True:
        new = []
        for pat in pats:
            for f in sorted(glob.glob(pat)):
                if f in loaded or f.endswith(".tmp.npz"):
                    continue
                try:
                    d = np.load(f)
                    new.append((f, d["feat"], d["act"], float(d["reward"]), bool(d["control"]), int(d["k"].max())))
                except Exception:
                    continue
        for f, feat, act, r, ctrl, kmax in new:
            x = np.concatenate([feat, act], 1).astype(np.float32)
            pre = float(torch.sigmoid(sc.logits(torch.from_numpy(x))).mean()) if trained else float("nan")
            sid = int(pathlib.Path(f).stem)
            X.append(x)
            y.append(np.full(len(x), r, np.float32))
            grp.append(np.repeat(S.boot_weights(sid)[:, None], len(x), 1))
            eps.append((sid, r, ctrl, kmax > 1, pre))
            loaded.add(f)
        info = {}
        if new and len(eps) >= S.MIN_EPISODES:
            Xa, ya, ga = np.concatenate(X), np.concatenate(y), np.concatenate(grp, 1)
            with torch.no_grad():
                sc.mu.copy_(torch.from_numpy(Xa.mean(0)))
                sc.sd.copy_(torch.from_numpy(Xa.std(0) + 1e-3))
            n_new = int(sum(len(v[1]) for v in new))
            steps = max(1, int(np.ceil(a.utd * n_new / 512)))        # 覚え込みを避け、新しい分に比例
            if not trained:                                          # 最初（再起動の後は、溜まった全部を読み直す）
                steps = min(max(steps, a.steps), 5000)
            info["loss"] = S.fit(sc, Xa, ya, ga, steps=steps, rng=rng, opts=opts)
            trained = True
            version += 1
        now = time.time()
        if info and now - t_save >= SAVE_EVERY_S:
            meta = {"version": version, "episodes": len(eps), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
            S.save_scorer(sc, scorer_p, meta)
            for m in a.mirror or []:
                try:
                    S.save_scorer(sc, m, meta)
                except OSError as e:
                    print(f"[r2c-learn] 写しに失敗 {m}: {e}", flush=True)
            t_save = now

            def rate(rows):
                return (round(float(np.mean([e[1] for e in rows])), 3), len(rows)) if rows else (None, 0)
            ctrl = [e for e in eps if e[2]][-200:]
            sel = [e for e in eps if e[3]][-200:]
            pre = [e for e in eps if e[4] == e[4]][-300:]
            row = {**meta, "transitions": int(sum(len(v) for v in y)), "loss": round(info["loss"], 4),
                   "control_last200": rate(ctrl), "select_last200": rate(sel),
                   "auc_before_seen": round(S.auc([e[4] for e in pre], [e[1] for e in pre]), 3) if pre else None,
                   "pred_mean_sel": round(float(np.mean([e[4] for e in sel if e[4] == e[4]])), 3) if any(e[4] == e[4] for e in sel) else None}
            log.write(json.dumps(row, ensure_ascii=False) + "\n")
            log.flush()
            print(f"[r2c-learn] {row}", flush=True)
        if (out / "STOP").exists() or (deadline and datetime.datetime.now() >= deadline):
            if trained:
                S.save_scorer(sc, scorer_p, {"version": version, "episodes": len(eps), "time": time.strftime("%Y-%m-%d %H:%M:%S")})
            print("[r2c-learn] 終わり", flush=True)
            return 0
        if not new:
            time.sleep(10)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("worker")
    w.add_argument("--band", choices=sorted(BANDS), required=True)
    w.add_argument("--shard", default="0/1")
    w.add_argument("--out", required=True)
    w.add_argument("--actor")
    w.add_argument("--model", choices=["Rdag", "R2a"], default="Rdag")
    w.add_argument("--ckpt")
    w.add_argument("--ckpt-sha256")
    w.add_argument("--deterministic", action="store_true")
    w.add_argument("--max-episodes", type=int, default=0)
    w.add_argument("--until")
    w.add_argument("--skip-seeds-from", nargs="*", help="前の回の episodes フォルダ（そこにある種を飛ばす）")
    w.add_argument("--select", action="store_true", help="run3: K 本作って見込みで選ぶ（--scorer）")
    w.add_argument("--scorer")
    w.add_argument("--k", type=int, default=8)
    w.add_argument("--control", choices=["seed", "never", "always"], default="seed")
    l_ = sub.add_parser("learner")
    l_.add_argument("--out", required=True)
    l_.add_argument("--actor")
    l_.add_argument("--episodes-glob", nargs="*")
    l_.add_argument("--mirror", nargs="*")
    l_.add_argument("--seed", type=int, default=0)
    l_.add_argument("--until")
    s_ = sub.add_parser("learner-select")
    s_.add_argument("--out", required=True)
    s_.add_argument("--scorer")
    s_.add_argument("--episodes-glob", nargs="*")
    s_.add_argument("--mirror", nargs="*")
    s_.add_argument("--seed", type=int, default=0)
    s_.add_argument("--steps", type=int, default=200, help="最初に学ぶときの、網 1 つあたりの更新の回数")
    s_.add_argument("--utd", type=float, default=8.0, help="その後: 新しい遷移 1 つあたり、何回見るか（1 回の更新は 512 遷移）")
    s_.add_argument("--until")
    a = ap.parse_args(argv)
    return {"worker": cmd_worker, "learner": cmd_learner, "learner-select": cmd_learner_select}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
