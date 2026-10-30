"""自分の失敗から学び直す評価（docs/api/r1_protocol.md 6 節）の実行の計画（configs/rotate の plan_*.json）を作る。

    .venv\\Scripts\\python.exe scripts\\api\\plan_failure_relearn.py [--rdag-ckpt <Rdag のチェックポイントのフォルダ>] [--out configs/rotate] [--dry-run]
    .venv\\Scripts\\python.exe scripts\\api\\plan_failure_relearn.py --smoke 193998 --tag a      # 試しの実行（smoke）の計画（評価のシードを使わない）

条件（6 つ、同じ配置）: N0（Nreal、組み直しなし）・U0（Rreal、なし）・Ud（Rdag、なし）・Lf（Rreal、組み直しあり）・
Ld（Rdag、あり）・Ln（Nreal、あり。参考の 2×2 のため）。
実験: R1E（PC A、シード 171400〜171499）・R1E_B（PC B、171500〜171599）。外乱の評価: R1D（PC A、171600〜171649）・R1D_B
（PC B、171650〜171699）、条件 Lf・Ld に --disturb push。
どの条件も開閉の絞り 5・手順の切り替え home・--min-free-gb 8、Lf・Ld・Ln は --finder estimated。
Rdag のチェックポイントの既定は学習の最後のチェックポイント（RDAG。途中のチェックポイントから選ばない。手順書 4.5 節）。フォルダが無ければ書かない
（--allow-missing で書く）。
条件の順番（scripts/real/rotate_jobs.pick の同点の決まり。手順書 6 節）: PC A は ORDER_A、PC B はその逆（ORDER_B）。
書くもの: <out>/plan_r1e_a.json・plan_r1e_b.json・plan_r1d_a.json・plan_r1d_b.json（--dry-run は表示だけ）。
--smoke <シード>: 試しの実行の実験 R1SMOKE・R1DSMOKE（--tag b なら R1SMOKE_B・R1DSMOKE_B）を、6 条件と外乱の 2 条件 ×
  そのシード 1 本で書く（plan_r1smoke_<tag>.json・plan_r1dsmoke_<tag>.json）。評価のシード（170000 以上）は受けない
  （予備の 193000〜193999 だけ）。--smoke-run 2 なら R1SMOKE2・R1DSMOKE2（同じ PC でもう一度回すとき、記録の場所を分ける）。
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
NREAL = "outputs/real/regen/train/train_Nreals1000_20261014-104429_20261014-104429/checkpoints/020000/pretrained_model"
RDAG = "outputs/r1/train/train_Rdags1000_20261021-155742_20261021-155742/checkpoints/040000/pretrained_model"
MAIN = {"R1E": (171400, 100), "R1E_B": (171500, 100)}
DIST = {"R1D": (171600, 50), "R1D_B": (171650, 50)}
CONDS = {"N0": ("U0", "Nreal"), "U0": ("U0", "Rreal"), "Ud": ("U0", "Rdag"), "Lf": ("Lf", "Rreal"), "Ld": ("Lf", "Rdag"),
         "Ln": ("Lf", "Nreal")}                  # Ln: 参考の 2×2（立て直しデータ × LLM の組み直し）の 4 つ目の組
DIST_CONDS = ("Lf", "Ld")
ORDER_A = ("N0", "U0", "Ud", "Lf", "Ld", "Ln")
ORDER_B = tuple(reversed(ORDER_A))
SMOKE_BAND = (193000, 193999)


def argv_of(exp: str, cond: str, trials: str, rdag: str, disturb: bool = False) -> list:
    arm, model = CONDS[cond]
    a = ["scripts/api/ladder.py", "run", "--experiment", exp, "--arm", arm, "--condition", cond, "--model", model]
    if model == "Nreal":
        a += ["--model-ckpt", f"Nreal={NREAL}"]
    elif model == "Rdag":
        a += ["--model-ckpt", f"Rdag={rdag}"]
    a += ["--grip-debounce", "5", "--step-reset", "home", "--trials", trials, "--min-free-gb", "8"]
    if arm == "Lf":
        a += ["--finder", "estimated"]
    if disturb:
        a += ["--disturb", "push"]
    return a


def plan_of(name: str, exp: str, s0: int, n: int, conds, rdag: str, disturb: bool, block: int = 10) -> dict:
    return {"name": name, "experiment": exp, "lanes": 4, "block": block, "max_lead_blocks": 3,
            "conditions": {c: {"n": n, "argv": argv_of(exp, c, f"{s0}:{n}", rdag, disturb)} for c in conds}}


def ordered(conds, tag: str) -> tuple:
    order = ORDER_A if tag == "a" else ORDER_B
    return tuple(c for c in order if c in conds)


def plans(rdag: str) -> dict:
    out = {}
    for (exp, (s0, n)), tag in zip(MAIN.items(), ("a", "b")):
        out[f"plan_r1e_{tag}.json"] = plan_of(f"r1e_{tag}", exp, s0, n, ordered(CONDS, tag), rdag, False)
    for (exp, (s0, n)), tag in zip(DIST.items(), ("a", "b")):
        out[f"plan_r1d_{tag}.json"] = plan_of(f"r1d_{tag}", exp, s0, n, ordered(DIST_CONDS, tag), rdag, True)
    return out


def smoke_plans(rdag: str, seed: int, tag: str, run: str = "") -> dict:
    """run: 同じ PC で試しの実行をもう一度回すときの印（例 "2" → R1SMOKE2・R1DSMOKE2。記録の場所を分ける）。"""
    sfx = run + ("" if tag == "a" else "_B")
    return {f"plan_r1smoke_{tag}.json": plan_of(f"r1smoke{run}_{tag}", f"R1SMOKE{sfx}", seed, 1, ordered(CONDS, tag), rdag,
                                                 False, block=1),
            f"plan_r1dsmoke_{tag}.json": plan_of(f"r1dsmoke{run}_{tag}", f"R1DSMOKE{sfx}", seed, 1, ordered(DIST_CONDS, tag),
                                                  rdag, True, block=1)}


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:                                    # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rdag-ckpt", default=RDAG)
    ap.add_argument("--out", default=str(ROOT / "configs" / "rotate"))
    ap.add_argument("--allow-missing", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", type=int, default=None, help="開発の smoke の種（193000〜193999）")
    ap.add_argument("--tag", default="a", choices=("a", "b"))
    ap.add_argument("--smoke-run", default="", help="smoke をもう一度回すときの印（数字 1 文字。実験の名前に足す）")
    a = ap.parse_args(argv)
    if not (ROOT / a.rdag_ckpt).is_dir() and not pathlib.Path(a.rdag_ckpt).is_dir() and not a.allow_missing:
        print(f"--rdag-ckpt {a.rdag_ckpt} が無い（学習の後に渡す。確かめだけなら --allow-missing）", file=sys.stderr)
        return 3
    if a.smoke is not None:
        if not SMOKE_BAND[0] <= a.smoke <= SMOKE_BAND[1]:
            print(f"--smoke は開発の予備 {SMOKE_BAND[0]}〜{SMOKE_BAND[1]} の種だけ（評価の種は使わない）", file=sys.stderr)
            return 3
        if a.smoke_run and not (len(a.smoke_run) == 1 and a.smoke_run.isdigit()):
            print("--smoke-run は数字 1 文字", file=sys.stderr)
            return 3
        todo = smoke_plans(a.rdag_ckpt, a.smoke, a.tag, a.smoke_run)
    else:
        todo = plans(a.rdag_ckpt)
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
