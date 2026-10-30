"""v3_real の評価（scripts/real/eval_real_matched.py の包み）に、実行器の開閉の指令の絞りを足して回す。eval_real_matched.py と
評価の入口・本体（scripts/eval_resumable.py・eval_trials.py）は書き換えない（eval_real_matched.py は確かめの道具が SHA-256 で固めている）。

    python scripts/real/eval_real_matched_grip.py --grip-debounce 5 --world v3_real run ...     # eval_resumable.py の run と同じ引数
    python scripts/real/eval_real_matched_grip.py --grip-debounce 5 --world v3_real task ...    # eval_resumable.py の task と同じ引数
    python scripts/real/eval_real_matched_grip.py --model-ckpt Rreal=<チェックポイントのフォルダ> ...       # チェックポイントの表（eval_trials の CKPT）に名前を足す（前に置く）

--grip-debounce K（1 以上）: runtime/runner.py の gripper_gate を、閾値 0・最短の保持 0・open_steps K にする。方策の開閉の値が
  「開」（< 0）を K 手続けて出したときだけ開き、閉はすぐ通す。運ぶ途中の短い開の跳ね（docs/real/grip_debounce.md）を通さない。
  付けなければ eval_real_matched.py と同じ動き。記録: 引数の控え（resume_spec.json）と試行の json の "world" に "grip_debounce"、
  runtime の trace に "gripper_gate"（open_steps・開かずに保った手の数）。
"""
import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def _load_ev():
    spec = importlib.util.spec_from_file_location("eval_v3_real_for_grip", ROOT / "scripts" / "real" / "eval_real_matched.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["eval_v3_real_for_grip"] = m
    spec.loader.exec_module(m)
    return m


def debounce_gate(k: int) -> dict:
    return {"enabled": True, "close": 0.0, "open": 0.0, "min_hold_s": 0.0, "open_steps": int(k)}


def split(argv: list) -> tuple:
    """(K または None, {名前: チェックポイント}, 残り)。"""
    deb, ck, rest = None, {}, []
    it = iter(argv)
    for x in it:
        if x == "--grip-debounce":
            deb = int(next(it))
            if deb < 1:
                raise SystemExit("--grip-debounce は 1 以上の手数")
        elif x == "--model-ckpt":
            k, _, v = next(it).partition("=")
            if not k or not v:
                raise SystemExit("--model-ckpt 名前=保存点のフォルダ")
            ck[k] = v
        else:
            rest.append(x)
    return deb, ck, rest


def patch(m96, deb, ck: dict):
    """eval_real_matched.patch の後の eval_resumable に、絞りとチェックポイントの名前を足す。"""
    orig_load_82, orig_spec = m96.load_82, m96.make_spec

    def load_82(allow_change):
        v82 = orig_load_82(allow_change)
        if deb is not None:
            v82.CFG["runtime_v2"]["gripper_gate"] = debounce_gate(deb)
        for k, v in ck.items():
            v82.CKPT[k] = v
        return v82

    def make_spec(a):
        s = orig_spec(a)
        if deb is not None:
            s["world"] = dict(s["world"], grip_debounce=debounce_gate(deb))
        return s

    m96.load_82, m96.make_spec = load_82, make_spec
    if deb is not None:
        m96.V3REAL_WORLD = dict(m96.V3REAL_WORLD, grip_debounce=debounce_gate(deb))
        orig_run_one, orig_run_task = m96.Engine.run_one, m96.Engine.run_one_task

        def run_one(self, *args):
            meta, arrays, rlog = orig_run_one(self, *args)
            meta["world"]["grip_debounce"] = debounce_gate(deb)
            return meta, arrays, rlog

        def run_one_task(self, *args):
            meta, arrays, rlog = orig_run_task(self, *args)
            meta["world"]["grip_debounce"] = debounce_gate(deb)
            return meta, arrays, rlog
        m96.Engine.run_one, m96.Engine.run_one_task = run_one, run_one_task
    return m96


def main(argv=None) -> int:
    deb, ck, rest = split(list(sys.argv[1:] if argv is None else argv))
    if deb is not None and "--grip-gate" in rest:
        raise SystemExit("--grip-debounce と --grip-gate は一緒に使わない")
    ev = _load_ev()
    world, switches, rest96 = ev.split_argv(rest)
    if world != "v3_real":
        raise SystemExit("--world v3_real を付ける")
    m = patch(ev.patch(ev.load_96(), world, switches or None), deb, ck)
    print(f"[v3_real] 世界 {world}、入り・切り {m.V3REAL_WORLD['switches']}、開閉の絞り "
          f"{'切' if deb is None else f'開が {deb} 手続いたときだけ開く'}、保存点の名前 {sorted(ck) or 'なし'}", flush=True)
    return m.main(rest96)


if __name__ == "__main__":
    raise SystemExit(main())
