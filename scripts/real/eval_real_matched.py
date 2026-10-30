"""実機の Panda で測った応答に合わせたシミュレーション（v3_real）で評価を回す。評価の入口 scripts/eval_resumable.py を包む
（eval_resumable と評価の本体 scripts/eval_trials.py は書き換えない）。方策の推論は GPU を使う。

    python scripts/real/eval_real_matched.py --world v3_real run --experiment V3REAL --condition R1v3_nat --model R1v3 \\
        --trials natural:<シード>:<数> --mode naive --exec-interval 6 --no-safety        # eval_resumable の run と同じ引数
    python scripts/real/eval_real_matched.py --world v3_real task ...                       # eval_resumable の task と同じ引数
    python scripts/real/eval_real_matched.py --world v3_real --switch width_sensor=off run ...   # 入り・切りの上書き（何度でも）

eval_resumable を importlib で読み込み、次の 4 つだけを差し替える:
- load_82: eval_trials の CFG を v3_real の評価の設定（harness/real_matched.py の load_config(generation=False) = v2 の層 + configs/v3_real.yaml）にする
- Engine.new_world: WorldRig・SensorSuite の代わりに make_eval_world（腕のサーボ・ハンドの遅延・6.5 Hz の幅のセンサ）
- make_spec・check_spec: 引数の控え（resume_spec.json）に "world" を足し、違う世界の続きを止める
- Engine.run_one・run_one_task・run_json_text: 試行の json と run.json に "world"（入り・切りと値）を足す
試行の並び・記録の形・制限時間・環境の記録は eval_resumable のまま。失敗の意図的な失敗・反射も eval_resumable のまま使える。
"""
import importlib.util
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def load_96():
    spec = importlib.util.spec_from_file_location("s4_resume_v3real", ROOT / "scripts" / "eval_resumable.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["s4_resume_v3real"] = m
    spec.loader.exec_module(m)
    return m


def patch(m, world: str, switches: dict = None):
    """読み込んだ 96（m）を v3_real の世界で回すように差し替える。差し替えた m を返す。"""
    from recovla.harness import real_matched as V
    cfg_world = V.load_config(world, generation=False, switches=switches)
    info = V.world_info(cfg_world)
    orig_load_82, orig_spec, orig_check = m.load_82, m.make_spec, m.check_spec
    orig_run_one, orig_run_task, orig_run_json = m.Engine.run_one, m.Engine.run_one_task, m.run_json_text

    def load_82(allow_change):
        v82 = orig_load_82(allow_change)
        v82.CFG = V.apply_overlay(v82.CFG, world, switches)          # eval_trials の CFG（v2 の層）に重ねる
        return v82

    def new_world(self):
        if self.suite is not None:
            self.suite.close()
        gc = False if getattr(self.a, "diag_no_gravcomp", False) else None
        self.world, self.suite = V.make_eval_world(self.cfg, gravcomp=gc)

    def make_spec(a):
        s = orig_spec(a)
        s["world"] = info
        return s

    def check_spec(a, out, spec):
        p = out / "resume_spec.json"
        if p.is_file():
            old = json.loads(p.read_text(encoding="utf-8")).get("world")
            if old != spec.get("world"):
                raise SystemExit(f"{p}: 世界が違う（前 {old}、今 {spec.get('world')}）。別の条件名にする")
        return orig_check(a, out, spec)

    def run_one(self, *args):
        meta, arrays, rlog = orig_run_one(self, *args)
        meta["world"] = dict(info, actuation=getattr(self.world, "v3_real_actuation", None),
                             contact=getattr(self.world, "v3_real_contact", None))
        return meta, arrays, rlog

    def run_one_task(self, *args):
        meta, arrays, rlog = orig_run_task(self, *args)
        meta["world"] = dict(info, actuation=getattr(self.world, "v3_real_actuation", None),
                             contact=getattr(self.world, "v3_real_contact", None))
        return meta, arrays, rlog

    def run_json_text(*args, **kw):
        d = json.loads(orig_run_json(*args, **kw))
        d["world"] = info
        return json.dumps(d, ensure_ascii=False, indent=1)

    m.load_82, m.make_spec, m.check_spec, m.run_json_text = load_82, make_spec, check_spec, run_json_text
    m.Engine.new_world, m.Engine.run_one, m.Engine.run_one_task = new_world, run_one, run_one_task
    m.V3REAL_WORLD = info
    return m


def split_argv(argv: list) -> tuple:
    """--world と --switch を取り出し、残りを eval_resumable に渡す。"""
    world, switches, rest = None, {}, []
    it = iter(argv)
    for x in it:
        if x == "--world":
            world = next(it)
        elif x.startswith("--world="):
            world = x.split("=", 1)[1]
        elif x == "--switch":
            k, _, v = next(it).partition("=")
            if v not in ("on", "off"):
                raise SystemExit("--switch 名前=on|off")
            switches[k] = v == "on"
        else:
            rest.append(x)
    return world, switches, rest


def main(argv=None) -> int:
    world, switches, rest = split_argv(list(sys.argv[1:] if argv is None else argv))
    if world != "v3_real":
        raise SystemExit("--world v3_real を付ける（既定の段階 3 の世界は scripts/eval_resumable.py をそのまま使う）")
    m = patch(load_96(), world, switches or None)
    print(f"[v3_real] 世界 {world}、入り・切り {m.V3REAL_WORLD['switches']}", flush=True)
    return m.main(rest)


if __name__ == "__main__":
    raise SystemExit(main())
