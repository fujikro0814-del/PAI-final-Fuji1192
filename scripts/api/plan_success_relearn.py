"""自分の成功も足したモデルと、行動列の候補から選ぶ条件の評価（docs/api/r2_protocol.md 6 節）の実行の計画
（configs/rotate の plan_*.json）を作る。

    .venv\\Scripts\\python.exe scripts\\api\\plan_success_relearn.py --r2a-ckpt <R2a のチェックポイントのフォルダ> [--out configs/rotate] [--dry-run]
    .venv\\Scripts\\python.exe scripts\\api\\plan_success_relearn.py --smoke 193590 --tag a [--r2a-ckpt ...]   # 試しの実行（smoke）の計画（評価のシードを使わない）

条件（3 つ、同じ配置、どれも --arm U0＝LLM の組み直しなし）: U0（Rreal）・Ud（Rdag）・U2a（R2a＝Rdag に自分の成功も足して
学び直したモデル）。
実験: R2E（PC A、シード 171700〜171799）・R2E_B（PC B、171800〜171899）。外乱の評価はしない。
どの条件も開閉の絞り 5・手順の切り替え home・--min-free-gb 8。条件の順番は PC A が ORDER_A、PC B はその逆。
R2a のチェックポイントは学習の後に渡す（無ければ書かない。確かめだけなら --allow-missing）。
書くもの: <out>/plan_r2e_a.json・plan_r2e_b.json（--smoke は plan_r2smoke_<tag>.json）。
Uc（6.1 節。R2a ＋ 行動列の候補から成功の見込みで選ぶ）: --stage-c --scorer-sha256 <値> で plan_r2ce*.json（6 つ。PC A は
  R2CE・R2CE2・R2CE3、PC B は R2CE_B・R2CE_B2・R2CE_B3）。--smoke と合わせると plan_r2csmoke_<tag>.json。
Udc（6.2 節。Rdag ＋ 見込みの重み scorer_run4 で選ぶ）: --stage-d --scorer-sha256 <値> で plan_r2de*.json（6 つ、シードの分け方は
  Uc と同じ）。--smoke と合わせると plan_r2dsmoke_<tag>.json。
Ldc（6.3 節。LLM の組み直しありの Ld（条件 Lf ＋ Rdag）に、見込みの重み scorer_run4 の選びを足す）: --stage-l --scorer-sha256 <値> で
  plan_r2l*.json（30 個、1 つに 1 条件・約 33 本。どの PC が回してもよい）。R2LP1〜6 は Ldc をシード 171400〜171599（Ld は R1E・R1E_B
  の記録と組）、R2LD01〜12 は Ld、R2LC01〜12 は Ldc をシード 171900〜172299。--smoke と合わせると plan_r2lsmoke_<tag>.json。
Lp（6.4 節。Ld の重みを、置く位置の入力つきの Rplace に替える）: --stage-p --rplace-ckpt <チェックポイントのフォルダ> で plan_r2pp*.json・
  plan_r2pn*.json（18 個、シードは Ldc と同じ 600 配置）。--smoke と合わせると試しの実行の計画。
Lr（6.5 節。Lp に RTC を足す。設定は RTC_CANDS）: --stage-r --rplace-ckpt <チェックポイントのフォルダ> で plan_r2r<設定の記号><番号>.json
  （設定ごとに 6 つ、シード 171400〜171599 の 200 を約 33 本ずつ。Lp は R2PP1〜6 の記録と組）。--smoke と合わせると
  plan_r2rsmoke_<tag>.json（設定を 1 本ずつ、シードは SMOKE_R）。
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
RDAG = "outputs/r1/train/train_Rdags1000_20261021-155742_20261021-155742/checkpoints/040000/pretrained_model"
MAIN = {"R2E": (171700, 100), "R2E_B": (171800, 100)}
CONDS = {"U0": ("U0", "Rreal"), "Ud": ("U0", "Rdag"), "U2a": ("U0", "R2a")}
ORDER_A = ("U0", "Ud", "U2a")
ORDER_B = tuple(reversed(ORDER_A))
SMOKE_SEEDS = (193590, 193599)          # 試しの実行用の予備のシード。評価のシードは使わない
# Uc（r2_protocol.md 6.1 節）: U2a と同じ 200 配置を PC 1 台あたり 3 つの実験に分ける（実行の管理は 1 条件に 1 本なので、
# 実験を分けて並行に回す）
MAIN_C = {"R2CE": (171700, 34), "R2CE2": (171734, 33), "R2CE3": (171767, 33),
          "R2CE_B": (171800, 34), "R2CE_B2": (171834, 33), "R2CE_B3": (171867, 33)}
CONDS_C = {"Uc": ("U0", "R2a")}
SCORER = "outputs/r2c/eval/scorer_run3.pt"      # 成功の見込みの重み（学習 run3 の最後の scorer.pt の写し。2 台で同じ道、SHA-256 で照らす）
# Udc（r2_protocol.md 6.2 節）: 土台のモデルを Rdag にし、学習 run4 の見込みで選ぶ。Ud と同じ 200 配置
MAIN_D = {"R2DE": (171700, 34), "R2DE2": (171734, 33), "R2DE3": (171767, 33),
          "R2DE_B": (171800, 34), "R2DE_B2": (171834, 33), "R2DE_B3": (171867, 33)}
CONDS_D = {"Udc": ("U0", "Rdag")}
SCORER_D = "outputs/r2c/eval/scorer_run4.pt"    # 学習 run4 の最後の scorer.pt の写し
SELECT_OF = {"Uc": SCORER, "Udc": SCORER_D, "Ldc": SCORER_D}
# Ldc（r2_protocol.md 6.3 節）: Ld と同じ条件 Lf・Rdag・推定の知覚に、run4 の見込みの選びを足す。
# 1 つの計画に 1 条件・約 33 本（実行の管理は 1 条件に 1 本なので、計画の数で並行に回す。PC は問わない）
def _split(prefix: str, s0: int, n: int, k: int, width: int = 1) -> dict:
    sizes = [n // k + (1 if i < n % k else 0) for i in range(k)]
    out, s = {}, s0
    for i, m in enumerate(sizes, 1):
        out[f"{prefix}{i:0{width}d}"] = (s, m)
        s += m
    return out


MAIN_LP = _split("R2LP", 171400, 200, 6)             # Ldc。Ld は主な結果の評価の記録（R1E・R1E_B）
MAIN_LD = _split("R2LD", 171900, 400, 12, 2)         # Ld、新しいシード 171900〜172299
MAIN_LC = _split("R2LC", 171900, 400, 12, 2)         # Ldc、新しいシード（R2LD と同じ分け方）
CONDS_L = {"Ld": ("Lf", "Rdag"), "Ldc": ("Lf", "Rdag")}
EXPS_L = {**{e: ("Ldc", s) for e, s in MAIN_LP.items()}, **{e: ("Ld", s) for e, s in MAIN_LD.items()},
          **{e: ("Ldc", s) for e, s in MAIN_LC.items()}}
SMOKE_L = {"a": 193596, "b": 193597, "c": 193598}    # 開発用の予備のシード（193590〜193599）
# Lp（r2_protocol.md 6.4 節）: Ld と同じ条件 Lf・推定の知覚で、VLA を置く位置の入力つきの Rplace に替える。
# 配置は Ldc と同じ 600（シード 171400〜171599 の 200 は R2PP、171900〜172299 の 400 は R2PN。Ld は R1E・R1E_B と R2LD の記録と組）
MAIN_PP = _split("R2PP", 171400, 200, 6)
MAIN_PN = _split("R2PN", 171900, 400, 12, 2)
CONDS_P = {"Lp": ("Lf", "Rplace")}
EXPS_P = {**{e: ("Lp", s) for e, s in MAIN_PP.items()}, **{e: ("Lp", s) for e, s in MAIN_PN.items()}}
SMOKE_P = {"a": 193599, "b": 193591}                 # 開発用の予備のシード（193591 は 10/24 に取り消して未使用）
# Lr（6.5 節）: Lp に RTC を足す。設定は原典の soft masking・LeRobot の既定・先頭だけそろえるもの。評価は 6 行ごとに推論する
# （ladder.py の exec_interval 6）、行動列は 50 行なので、原典の soft masking の範囲は 50 − 6 = 44
RTC_CANDS = {"LrE44": {"horizon": 44, "schedule": "EXP", "max_guidance_weight": 5.0},
             "LrL10": {"horizon": 10, "schedule": "LINEAR", "max_guidance_weight": 10.0},
             "LrZ": {"horizon": 44, "schedule": "ZEROS", "max_guidance_weight": 5.0}}
UNIT_R = {"LrE44": "A", "LrL10": "B", "LrZ": "C"}     # 計画の名前の記号（R2RA1〜6 など）
CONDS_R = {c: ("Lf", "Rplace") for c in RTC_CANDS}
EXPS_R = {e: (c, s) for c, x in UNIT_R.items() for e, s in _split(f"R2R{x}", 171400, 200, 6).items()}
SMOKE_R = {"a": 193310, "b": 193311}                 # 開発用の予備のシード（193310〜193319 を段階 R に）


def argv_of(exp: str, cond: str, trials: str, rdag: str, r2a: str, scorer_sha: str = None, rplace: str = None) -> list:
    arm, model = dict(CONDS, **CONDS_C, **CONDS_D, **CONDS_L, **CONDS_P, **CONDS_R)[cond]
    a = ["scripts/api/ladder.py", "run", "--experiment", exp, "--arm", arm, "--condition", cond, "--model", model]
    if model == "Rdag":
        a += ["--model-ckpt", f"Rdag={rdag}"]
    elif model == "Rplace":
        a += ["--model-ckpt", f"Rplace={rplace}"]
    elif model == "R2a":
        a += ["--model-ckpt", f"R2a={r2a}"]
    if cond in SELECT_OF:
        a += ["--select-scorer", SELECT_OF[cond], "--select-scorer-sha256", scorer_sha, "--select-k", "8"]
    a += ["--grip-debounce", "5", "--step-reset", "home", "--trials", trials, "--min-free-gb", "8"]
    a += ["--finder", "estimated"] if arm == "Lf" else []                # R1 の Ld と同じ並び（finder は RTC の前）
    if cond in RTC_CANDS:
        r = RTC_CANDS[cond]
        a += ["--runtime-mode", "rtc", "--rtc-horizon", str(r["horizon"]), "--rtc-schedule", r["schedule"],
              "--rtc-max-guidance", f"{r['max_guidance_weight']:g}"]
    return a


def plan_of(name: str, exp: str, s0: int, n: int, conds, rdag: str, r2a: str, block: int = 10, scorer_sha: str = None,
            rplace: str = None) -> dict:
    return {"name": name, "experiment": exp, "lanes": 4, "block": block, "max_lead_blocks": 3,
            "conditions": {c: {"n": n, "argv": argv_of(exp, c, f"{s0}:{n}", rdag, r2a, scorer_sha, rplace)} for c in conds}}


def plans_p(rplace: str) -> dict:
    return {f"plan_{exp.lower()}.json": plan_of(exp.lower(), exp, s0, n, (cond,), None, None, rplace=rplace)
            for exp, (cond, (s0, n)) in EXPS_P.items()}


def smoke_plans_p(rplace: str, tag: str) -> dict:
    exp = "R2PSMOKE" + ("" if tag == "a" else f"_{tag.upper()}")
    return {f"plan_r2psmoke_{tag}.json": plan_of(f"r2psmoke_{tag}", exp, SMOKE_P[tag], 1, tuple(CONDS_P), None, None, block=1,
                                                  rplace=rplace)}


def plans_r(rplace: str) -> dict:
    return {f"plan_{exp.lower()}.json": plan_of(exp.lower(), exp, s0, n, (cond,), None, None, rplace=rplace)
            for exp, (cond, (s0, n)) in EXPS_R.items()}


def smoke_plans_r(rplace: str, tag: str) -> dict:
    exp = "R2RSMOKE" + ("" if tag == "a" else f"_{tag.upper()}")
    return {f"plan_r2rsmoke_{tag}.json": plan_of(f"r2rsmoke_{tag}", exp, SMOKE_R[tag], 1, tuple(RTC_CANDS), None, None, block=1,
                                                  rplace=rplace)}


def plans_c(rdag: str, r2a: str, scorer_sha: str) -> dict:
    return {f"plan_{exp.lower()}.json": plan_of(exp.lower(), exp, s0, n, tuple(CONDS_C), rdag, r2a, scorer_sha=scorer_sha)
            for exp, (s0, n) in MAIN_C.items()}


def smoke_plans_c(rdag: str, r2a: str, seed: int, tag: str, scorer_sha: str) -> dict:
    exp = "R2CSMOKE" + ("" if tag == "a" else "_B")
    return {f"plan_r2csmoke_{tag}.json": plan_of(f"r2csmoke_{tag}", exp, seed, 1, tuple(CONDS_C), rdag, r2a, block=1,
                                                  scorer_sha=scorer_sha)}


def plans_d(rdag: str, scorer_sha: str) -> dict:
    return {f"plan_{exp.lower()}.json": plan_of(exp.lower(), exp, s0, n, tuple(CONDS_D), rdag, None, scorer_sha=scorer_sha)
            for exp, (s0, n) in MAIN_D.items()}


def smoke_plans_d(rdag: str, seed: int, tag: str, scorer_sha: str) -> dict:
    exp = "R2DSMOKE" + ("" if tag == "a" else "_B")
    return {f"plan_r2dsmoke_{tag}.json": plan_of(f"r2dsmoke_{tag}", exp, seed, 1, tuple(CONDS_D), rdag, None, block=1,
                                                  scorer_sha=scorer_sha)}


def plans_l(rdag: str, scorer_sha: str) -> dict:
    return {f"plan_{exp.lower()}.json": plan_of(exp.lower(), exp, s0, n, (cond,), rdag, None, scorer_sha=scorer_sha)
            for exp, (cond, (s0, n)) in EXPS_L.items()}


def smoke_plans_l(rdag: str, tag: str, scorer_sha: str) -> dict:
    exp = "R2LSMOKE" + ("" if tag == "a" else f"_{tag.upper()}")
    return {f"plan_r2lsmoke_{tag}.json": plan_of(f"r2lsmoke_{tag}", exp, SMOKE_L[tag], 1, tuple(CONDS_L), rdag, None, block=1,
                                                  scorer_sha=scorer_sha)}


def ordered(tag: str) -> tuple:
    return ORDER_A if tag == "a" else ORDER_B


def plans(rdag: str, r2a: str) -> dict:
    return {f"plan_r2e_{tag}.json": plan_of(f"r2e_{tag}", exp, s0, n, ordered(tag), rdag, r2a)
            for (exp, (s0, n)), tag in zip(MAIN.items(), ("a", "b"))}


def smoke_plans(rdag: str, r2a: str, seed: int, tag: str) -> dict:
    exp = "R2SMOKE" + ("" if tag == "a" else "_B")
    return {f"plan_r2smoke_{tag}.json": plan_of(f"r2smoke_{tag}", exp, seed, 1, ordered(tag), rdag, r2a, block=1)}


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:                                    # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rdag-ckpt", default=RDAG)
    ap.add_argument("--r2a-ckpt", default=None)
    ap.add_argument("--out", default=str(ROOT / "configs" / "rotate"))
    ap.add_argument("--allow-missing", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", type=int, default=None, help=f"smoke の種（{SMOKE_SEEDS[0]}〜{SMOKE_SEEDS[1]}）")
    ap.add_argument("--tag", default="a", choices=("a", "b", "c"))
    ap.add_argument("--stage-c", action="store_true", help=f"段階 C の腕 Uc の計画（{SCORER} の SHA-256 を --scorer-sha256 で渡す）")
    ap.add_argument("--stage-d", action="store_true", help=f"段階 C の腕 Udc の計画（Rdag ＋ {SCORER_D}、6.2 節）")
    ap.add_argument("--stage-l", action="store_true", help=f"段階 L の腕 Ld・Ldc の計画（Ld ＋ {SCORER_D}、6.3 節）")
    ap.add_argument("--scorer-sha256", default=None)
    ap.add_argument("--stage-p", action="store_true", help="段階 P の腕 Lp の計画（Ld の重みを Rplace に替える、6.4 節）")
    ap.add_argument("--rplace-ckpt", default=None, help="--stage-p・--stage-r: Rplace の最後のチェックポイントの pretrained_model")
    ap.add_argument("--stage-r", action="store_true", help="段階 R の条件 Lr の計画（Lp ＋ RTC、6.5 節）")
    a = ap.parse_args(argv)
    if a.stage_r:                                      # 段階 P と同じ Rplace。smoke のシードは PC ごと（SMOKE_R）
        if a.smoke is not None and a.smoke != SMOKE_R.get(a.tag):
            print(f"--stage-r の smoke のシードは {SMOKE_R}（--tag で決まる）", file=sys.stderr)
            return 3
    if a.stage_p or a.stage_r:                                      # Lp は Rplace だけを読む。smoke のシードは PC ごと（SMOKE_P）
        if not (a.rplace_ckpt and ((ROOT / a.rplace_ckpt).is_dir() or pathlib.Path(a.rplace_ckpt).is_dir())) and not a.allow_missing:
            print(f"--rplace-ckpt {a.rplace_ckpt} が無い", file=sys.stderr)
            return 3
        if not a.stage_r and a.smoke is not None and a.smoke != SMOKE_P.get(a.tag):
            print(f"--stage-p の smoke の種は {SMOKE_P}（--tag で決まる）", file=sys.stderr)
            return 3
        if a.stage_r:
            todo = smoke_plans_r(a.rplace_ckpt, a.tag) if a.smoke is not None else plans_r(a.rplace_ckpt)
        else:
            todo = smoke_plans_p(a.rplace_ckpt, a.tag) if a.smoke is not None else plans_p(a.rplace_ckpt)
        for name, p in todo.items():
            txt = json.dumps(p, ensure_ascii=False, indent=1) + "\n"
            if a.dry_run:
                print(f"--- {name}\n{txt}")
            else:
                pathlib.Path(a.out).mkdir(parents=True, exist_ok=True)
                (pathlib.Path(a.out) / name).write_text(txt, encoding="utf-8", newline="\n")
                print(pathlib.Path(a.out) / name)
        return 0
    r2a = a.r2a_ckpt or "<R2A>"
    for what, p in (("--rdag-ckpt", a.rdag_ckpt), ("--r2a-ckpt", a.r2a_ckpt)):
        if what == "--rdag-ckpt" and a.stage_c:        # Uc は R2a だけを読む
            continue
        if what == "--r2a-ckpt" and (a.stage_d or a.stage_l):   # Udc・Ld・Ldc は Rdag だけを読む
            continue
        if not (p and ((ROOT / p).is_dir() or pathlib.Path(p).is_dir())) and not a.allow_missing:
            print(f"{what} {p} が無い（確かめだけなら --allow-missing）", file=sys.stderr)
            return 3
    if (a.stage_c or a.stage_d or a.stage_l) and not (a.scorer_sha256 and len(a.scorer_sha256) == 64):
        print("--stage-c・--stage-d・--stage-l には --scorer-sha256 <64 桁> が要る", file=sys.stderr)
        return 3
    if a.tag == "c" and not a.stage_l:
        print("--tag c は --stage-l の smoke だけ", file=sys.stderr)
        return 3
    if a.stage_l:                                      # Ldc の試しの実行のシードは PC ごとに決めてある（SMOKE_L）
        todo = (smoke_plans_l(a.rdag_ckpt, a.tag, a.scorer_sha256.lower()) if a.smoke is not None
                else plans_l(a.rdag_ckpt, a.scorer_sha256.lower()))
        if a.smoke is not None and a.smoke != SMOKE_L[a.tag]:
            print(f"--stage-l の smoke の種は {SMOKE_L}（--tag で決まる）", file=sys.stderr)
            return 3
    elif a.smoke is not None:
        if not SMOKE_SEEDS[0] <= a.smoke <= SMOKE_SEEDS[1]:
            print(f"--smoke は {SMOKE_SEEDS[0]}〜{SMOKE_SEEDS[1]} の種だけ（評価の種は使わない）", file=sys.stderr)
            return 3
        todo = (smoke_plans_d(a.rdag_ckpt, a.smoke, a.tag, a.scorer_sha256.lower()) if a.stage_d
                else smoke_plans_c(a.rdag_ckpt, r2a, a.smoke, a.tag, a.scorer_sha256.lower()) if a.stage_c
                else smoke_plans(a.rdag_ckpt, r2a, a.smoke, a.tag))
    else:
        todo = (plans_d(a.rdag_ckpt, a.scorer_sha256.lower()) if a.stage_d
                else plans_c(a.rdag_ckpt, r2a, a.scorer_sha256.lower()) if a.stage_c else plans(a.rdag_ckpt, r2a))
    for name, p in todo.items():
        txt = json.dumps(p, ensure_ascii=False, indent=1) + "\n"
        if a.dry_run:
            print(f"--- {name}\n{txt}")
        else:
            pathlib.Path(a.out).mkdir(parents=True, exist_ok=True)
            (pathlib.Path(a.out) / name).write_text(txt, encoding="utf-8", newline="\n")
            print(pathlib.Path(a.out) / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
