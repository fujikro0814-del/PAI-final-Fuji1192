"""「全部片付けて」の評価を、タスクと感覚の窓口（recovla.api）を通して回す入口。
評価の枠（scripts/eval_resumable.py ほか）を importlib で読み、実行器・知覚・失敗の検知を差し替えて使う（元のファイルは書き換えない）。

    python scripts/api/ladder.py run --arm U0|L2v2|Lf|CL|CL2 --experiment <名> --condition <名> --trials <シード>:<数> --model Rreal \\
        [--disturb none|push] \\
        [--finder estimated|color] [--grip-debounce K] [--step-reset home|rebuild|home_rebuild] [--dry-run] [eval_resumable task の引数]
        [--task tidy3_cubes|cup_plate|<yaml>] [--vla-text fixed|task] [--task-planner spec|s4] [--model-ckpt <名>=<チェックポイント>]
    python scripts/api/ladder.py table [--target sim|real]        # 条件ごとに使う口・needs・感覚の表だけ（何も読まない・書かない）

条件（--arm）:
  U0    LLM の組み直しなし（VLA だけ）。実行器は runtime/executor.TaskRuntime と PolicyRuntime。
  Lf    LLM の組み直しあり。実行器は api/executor_improved（場面の要約の直し・人への通知・見失いの見直し）、組み直しは
        planner/replan_improved（実行器のやり直しと色ごとの試行回数を分けた指示文、claude-sonnet-5-5）。失敗の検知（見張り）は
        GraspDropMonitor・HeldEstimateDropMonitor・api/monitors_in_hand.ExpectedInHandMonitor。出来事は dropped・timeout。
        --finder estimated だけ。Ud・Ld は U0・Lf を --model Rdag --model-ckpt Rdag=<チェックポイント> で回す。
  L2v2  組み直しの前の版（api/executor_monitored・planner/replan_event。失敗の検知は上の 2 つ）。
  CL    手で書いたスクリプト方策（api/skills.ScriptSkill）を、組み直しなしの実行器で回す。
  CL2 ステップで書いたスクリプト方策の自立版（api/skills_script_self.CL2Skill と api/executor_script.CL2TaskRuntime）。高さの地図・推定した
        大きさ・自分の完了の判定で動き、実行器の完了の判定・戻す動き・やり直し・手順の切り替え時の待機位置への戻し（--step-reset）
        を使わない。--finder estimated だけ。試行の json に "cl2"（落下の記録・手順の終わり方）を足す。
起動時の手順の分解はどの条件も planner/decompose_haiku（claude-haiku-5-5）。api/planner の口は使わない（キャッシュの鍵を変えないため）。
--finder（U0 以外の知覚）: estimated（既定。RGB-D からの推定で、既知の寸法を読まない。Perception に GeometryPerception を被せ、
  WorldModel の計算・実行器・完了判定は同じ）か color（比較用。既知の値を使う ColorWorldModelFinder）。U0 は使わない（記録は None）。
失敗の検知の引数は ReflexParams の既定・confirm_s=1.0・カメラでの確かめあり。
評価の配置・本数・比較は手順書（docs/api/r1_protocol.md など）で事前に決める。この入口は --trials を受けるだけ。
世界は実機の Panda で測った応答に合わせたシミュレーション（v3_real）、チェックポイントは Rreal、--exec-interval 6・安全フィルタなし、
1 手順 30 s・やり直し 1 回・全体 200 s。
記録: 評価の枠と同じ ＋ 試行の json に "replans"・"early_events"・"u4"（条件と実行の設定）・"api"（条件・使った口・失敗の検知の
出来事・使ったファイルの SHA-256）・"world"。--dry-run は条件の表を出してから、何も書かずに確かめる。
--task（既定 tidy3_cubes。名前なら configs/tasks/<名>.yaml）: 別のタスク（カップを皿に置く cup_plate）も回せる。既定のタスクの
  動き・記録・再開の鍵（resume_spec）は変わらない。cup_plate では世界は harness/world_cup_plate（成功の真値 all_on）、知覚は
  api/estimated_place.PlaceGeometryFinder、指示文の既定は書式の instructions の最初（--text で変えられる）。条件は U0・L2v2・Lf・
  CL2（CL は箱専用なので断る。--finder color も断る）。VLA の条件の旧い知覚・完了判定・ハンドの grasp・方策の目標位置の入力は既知の値で
  動き（cube_size = カップの外径、箱の欄に皿、目標位置の入力の平面 = カップの中心。harness/world_cup_plate の注釈）、起動時の箱の位置の
  ずれの確かめは切る（皿の位置は試行ごとに変わるため）。どれも試行の json の "api"・"u4" と run.json に残す。
  --task-planner: spec（既定）= 書式から作った指示で Haiku 5.5 に分けさせる（planner/decompose_task）、s4 = decompose_haiku のまま
  （立方体と箱の指示）。--vla-text（VLA の指示文）: fixed（既定）= "put the {color} cube in the box"、task = 書式の
  policy_instruction（"put the {color} cup on the plate"）。
--disturb push（既定 none。立方体のタスクだけ）: 評価の外乱（eval/disturb_task.PushDropDisturber）。目標を持ち上げてから
  0.5〜2 s 後（シードごと）に横から押して手から落とす。試行の json に "disturb"、再開の鍵に api_disturb（none なら足さない）。
--model-ckpt <名>=<チェックポイントのフォルダ>（何度でも）: --model に Rreal 以外の名前を使えるように、チェックポイントの表（CKPT）に足す。
  Rreal 以外のときだけ "u4" にチェックポイントの道を足す。
--select-scorer <scorer.pt> --select-scorer-sha256 <値> [--select-k 8]（条件 U0 と Lf。Udc と、Ld に選びを足した Ldc。
  選びは方策の推論だけを替え、LLM・直しは同じ）: 推論ごとに、ふつうの雑音から K 本の行動列を作り、学習した成功の見込み
  （recovla.rl.select）が最も高い 1 本を選ぶ（重みは読み直さない）。候補 0 は選ばない条件と同じ乱数。推論の番号は手順・
  やり直しごとに 0 から（学習の 1 手順の試行と同じ）。試行の json に "select"（手順ごとの推論の数・候補 0 以外を選んだ数・
  選んだ候補の見込みの平均）、"u4" に重みの SHA-256、再開の鍵に api_select。
置く位置の入力つきの重み（学習の conversion.json に place_cue があるもの。Lp）: 手順・やり直しの始めに、推定した箱を 2×2 に
  分けたマスのうち箱の中の物からいちばん遠いマスを方策に渡す（api/place_cue.free_cell）。--finder estimated だけ。試行の json に
  "place_cue"（決めた値と決め方）。
--runtime-mode rtc --rtc-horizon H --rtc-schedule EXP|LINEAR|ZEROS|ONES --rtc-max-guidance W（条件 U0 と Lf。既定 naive。
  docs/api/r2_protocol.md 6.5 節）: 方策の実行を RTC（新しい行動列を前の行動列の残りに寄せる非同期実行）にする。PolicyRuntime を
  mode="rtc"、寄せる範囲を H にし、方策に lerobot の RTCConfig（execution_horizon H・prefix_attention_schedule・
  max_guidance_weight W）を入れる。推論の時間は configs/latency_v1.json の policy_rtc の分布。試行の json の "u4" に "runtime"
  （方式と設定）、再開の鍵に api_rtc（naive なら足さない）。
"""
import argparse
import importlib.util
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
RREAL = "outputs/real/regen/train/train_Rreals1000_20261014-082146_20261014-082146/checkpoints/020000/pretrained_model"
ARMS = {"U0": "立て直しなし（VLA）", "L2v2": "見張り役の出来事で立て直し（VLA、Sonnet 5.5）", "CL": "古典の台本（立て直しなし）",
        "CL2": "自力の先生（立て直しなし）", "Lf": "R1 の直しを入れた立て直し（L2v2 ＋ 指示文・場面の要約・見張り役）"}
VLA_ARMS = ("U0", "L2v2", "Lf")
REPLAN_ARMS = ("L2v2", "Lf")
# Lf のときだけ記録に足す
R1_FILES = ("src/recovla/api/executor_improved.py", "src/recovla/api/scene_summary.py", "src/recovla/api/monitors_in_hand.py",
            "src/recovla/api/judge_improved.py", "src/recovla/planner/replan_improved.py")
DISTURB_FILES = ("src/recovla/eval/disturb_task.py",)    # --disturb のときだけ記録に足す
DISTURBS = ("none", "push")
REPLAN_MODEL = "claude-sonnet-5-5"
TASK = "configs/tasks/tidy3_cubes.yaml"
FILES = ("scripts/api/ladder.py", TASK, "src/recovla/api/task_spec.py", "src/recovla/api/senses.py",
         "src/recovla/api/perception.py", "src/recovla/api/monitors.py", "src/recovla/api/executor_monitored.py",
         "src/recovla/api/skills.py", "src/recovla/api/skills_script.py", "src/recovla/api/geometry.py",
         "src/recovla/api/estimated.py", "src/recovla/api/relations.py",
         "src/recovla/runtime/executor.py", "src/recovla/runtime/executor_replan.py", "src/recovla/runtime/executor_replan_early.py",
         "src/recovla/runtime/reflex.py", "src/recovla/runtime/runner.py", "src/recovla/runtime/perception.py",
         "src/recovla/planner/decompose_haiku.py", "src/recovla/planner/replan_base.py", "src/recovla/planner/replan_multi.py",
         "src/recovla/planner/replan_event.py", "scripts/eval_resumable.py", "scripts/real/eval_real_matched.py",
         "scripts/real/eval_real_matched_grip.py", "src/recovla/diag/step_reset.py", "src/recovla/diag/return_pose.py",
         "configs/default.yaml", "configs/v3_real.yaml", "configs/real_v1.yaml")
CL2_FILES = ("src/recovla/api/skills_script_self.py", "src/recovla/api/executor_script.py", "src/recovla/api/heightmap.py",
             "configs/api/cl2.yaml")                  # CL2 のときだけ記録に足す
# 推定の知覚（--finder estimated・CL2）のときだけ記録に足す: 作りで通る finder_class
EST_FILES = ("src/recovla/api/estimated_place.py",)
# 知覚の遅延（compute_latency_s）はどの条件にも効くので、どの条件の記録にも足す
LAT_FILES = ("configs/runtime_v2.yaml", "src/recovla/harness/robot_io.py")
SELECT_FILES = ("src/recovla/rl/select.py", "src/recovla/runtime/policy.py")   # --select-scorer（Udc）のときだけ
TASK_FILES = ("configs/scenes/cup_plate.yaml", "assets/mjcf/scene_cup_plate.xml", "src/recovla/sim/scene.py",
              "src/recovla/sim/rig.py", "src/recovla/harness/world_cup_plate.py", "src/recovla/harness/task_loop.py",
              "src/recovla/api/geometry_place.py", "src/recovla/api/estimated_place.py", "src/recovla/api/task_spec.py",
              "src/recovla/api/harness/goal_truth.py", "src/recovla/api/planner.py",
              "src/recovla/planner/decompose_task.py")  # 既定でないタスクのときだけ記録に足す（そのタスクの書式）
VLA_TEXTS = ("fixed", "task")
RTC_SCHEDULES = ("EXP", "LINEAR", "ZEROS", "ONES")
TASK_PLANNERS = ("spec", "s4")


def task_path(v: str) -> str:
    """--task の値 → 書式のファイル（名前なら configs/tasks/<名>.yaml）。"""
    v = str(v)
    return v if ("/" in v or "\\" in v or v.endswith(".yaml")) else f"configs/tasks/{v}.yaml"


def is_default_task(task: str) -> bool:
    return task_path(task) == TASK


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def _is_api_error(e: BaseException) -> bool:
    return any(c.__module__.split(".")[0] == "anthropic" for c in type(e).__mro__)


# ---------------------------------------------------------------- 条件ごとの口（表）
FINDERS = ("estimated", "color")


def replan_wording(spec, default: bool):
    """組み直しの LLM に渡すタスクの言葉（planner/replan_event.make_wording）。既定のタスクは None（立方体と箱の文のまま）。
    物・場所の呼び方は書式の words の最初、色は目標の物の finder の色、置き方は目標の種類（all_in・all_on）。"""
    if default:
        return None
    from recovla.api import task_spec as TS
    from recovla.planner import replan_event as RL
    objs = [spec.object(n) for n in spec.goal.objects]
    colors = [TS.finder_kind(o.finder)[1] for o in objs]
    return RL.make_wording(objs[0].words[0], spec.place(spec.goal.place).words[0], colors, spec.goal.kind)


def make_monitors(arm: str, registry=None):
    """失敗の検知（見張り）の組。L2v2・Lf だけで、ほかは None。どれかの dropped で組み直しを頼む。"""
    from recovla.api import monitors as M
    from recovla.runtime.reflex import ReflexParams
    if arm not in REPLAN_ARMS:
        return None
    ms = [M.GraspDropMonitor(ReflexParams(), confirm_s=M.CONFIRM_S, camera_confirm=True),
          M.HeldEstimateDropMonitor(ReflexParams(), on_table_s=M.ON_TABLE_S)]
    if arm == "Lf":                                          # 持っていたはずの物が手の中にない（場所を問わない）
        from recovla.api import monitors_in_hand as MR
        ms.append(MR.ExpectedInHandMonitor(ReflexParams(), confirm_s=MR.CONFIRM_S))
    return M.MonitorSet(ms, registry)


def perception_factory(spec, finder: str):
    """知覚を作る関数（setup, params, thr）→ Perception。estimated なら GeometryPerception を被せる（step_reset の作り直しも同じ）。"""
    from recovla.runtime.perception import Perception

    def make(setup, params, thr):
        per = Perception(setup, params, thr)
        if finder == "estimated":
            from recovla.api.estimated import GeometryPerception
            from recovla.api.estimated_place import finder_class      # 立方体と箱は EstimatedGeometryFinder のまま
            GeometryPerception.attach(per, finder_class(spec).from_spec(spec, per, setup))
        return per
    return make


def script_rules(cfg: dict):
    """手で書いたスクリプト方策（CL）の決まり（configs の expert と scene の置き場所）。"""
    from recovla.api.skills_script import ScriptRules
    e = cfg["expert"]
    return ScriptRules.from_config(e, e["phase"])


def arm_plan(arm: str, target: str = "sim", finder: str = "estimated", task: str = TASK, vla_text: str = "fixed",
             task_planner: str = "spec") -> dict:
    """条件が使う口・needs・感覚。タスクの書式のほかは何も読まない。既定のタスクでは tests で固定した辞書、
    ほかのタスクでは task_file・instruction・place_finder・vla_known を足し、計画役を書き替える。"""
    from recovla.api import senses as AS
    from recovla.api import skills as SK
    from recovla.api import task_spec as TS
    spec = TS.load_task(task_path(task))
    reg = AS.standard_registry(target)
    if arm == "CL2":
        from recovla.api.skills_script_self import CL2_INFO
        skill = CL2_INFO
    else:
        skill = SK.SCRIPT_INFO if arm == "CL" else SK.VLA_INFO
    ms = make_monitors(arm, reg)
    needs = list(skill.needs) + ([n for m in ms.monitors for n in m.needs] if ms else [])
    return {"arm": arm, "arm_ja": ARMS[arm], "task": spec.name, "target": target, "finder": None if arm == "U0" else finder,
            "executor": "api/executor_monitored.MonitoredReplanTaskRuntime（L2 の包み）" if arm == "L2v2"
            else ("api/executor_script.CL2TaskRuntime（台本の自分の完了の判定、JudgeV2 は記録だけ）" if arm == "CL2"
                  else ("api/executor_improved.R1ReplanTaskRuntime（L2v2 の包み ＋ 場面の要約の直し・skip_notify・見失いの見直し）"
                        if arm == "Lf"
                        else "runtime/executor.TaskRuntime")),
            "planner": "planner/decompose_haiku（claude-haiku-5-5）",
            "replan": (f"planner/replan_event（{REPLAN_MODEL}）、event dropped・timeout" if arm == "L2v2"
                       else (f"planner/replan_improved（{REPLAN_MODEL}）、event dropped・timeout" if arm == "Lf" else None)),
            "skill": {"name": skill.name, "needs": list(skill.needs), "note": skill.note,
                      "missing": reg.missing(skill.needs)},
            "monitors": None if ms is None else {
                "active": [{"name": m.name, "needs": list(m.needs)} for m in ms.active], "skipped": ms.skipped,
                "params": {m.name: {k: getattr(m, a) for k, a in (("confirm_s", "confirm_s"), ("camera_confirm", "camera"),
                                                                     ("on_table_s", "on_table_s")) if hasattr(m, a)}
                           for m in ms.monitors}},
            "senses": [r for r in reg.table() if r["name"] in dict.fromkeys(needs)],
            "runnable": not reg.missing(needs)} | ({} if is_default_task(task) else task_plan(spec, task, arm, vla_text,
                                                                                               task_planner, finder))


def task_plan(spec, task: str, arm: str, vla_text: str, task_planner: str, finder: str) -> dict:
    """既定でないタスクの arm_plan の足し分。"""
    from recovla.api import task_spec as TS
    from recovla.common import config
    fixed = config.load()["convert"]["instruction"]
    vla = arm in VLA_ARMS
    fmt = TS.policy_format(spec, vla_text, fixed) if vla else fixed
    out = {"task_file": task_path(task),
           "planner": ("planner/decompose_task（claude-haiku-5-5、書式から作った指示）" if task_planner == "spec"
                       else "planner/decompose_haiku（claude-haiku-5-5、立方体と箱の指示のまま）"),
           "instruction": {"mode": vla_text if vla else "fixed", "fmt": fmt,
                           "example": fmt.format(color=TS.cue_of(spec, TS.StepSpec("pick_place", spec.goal.objects[0],
                                                                                   spec.goal.place)))},
           "place_finder": None if arm == "U0" else (f"api/estimated_place.PlaceGeometryFinder（{finder}）"),
           "vla_known": ("旧い知覚・JudgeV2・ハンドの grasp・方策の手がかりは既知の値で動く（cube_size = カップの外径 6 cm、箱の欄に"
                         "皿、手がかりの平面 = 机 + 4 cm（カップの中心））。起動の箱の位置のずれの確かめは切る") if vla else None}
    return out


def format_plan(p: dict) -> str:
    rows = [f"[api] 腕 {p['arm']}（{p['arm_ja']}）、タスク {p['task']}、{p['target']} の感覚で "
            f"{'動ける' if p['runnable'] else '足りない感覚がある'}",
            f"  実行器: {p['executor']}", f"  計画役: {p['planner']}", f"  立て直し: {p['replan'] or 'なし'}",
            f"  知覚の口: {p['finder'] or '使わない（今の PolicyRuntime のまま）'}",
            f"  技能: {p['skill']['name']}  needs {', '.join(p['skill']['needs'])}"
            + (f"  足りない {p['skill']['missing']}" if p['skill']['missing'] else "")]
    if p["monitors"] is None:
        rows.append("  見張り役: なし")
    else:
        for m in p["monitors"]["active"]:
            prm = p["monitors"]["params"][m["name"]]
            rows.append(f"  見張り役: {m['name']}  needs {', '.join(m['needs'])}  "
                        + "  ".join(f"{k} {v}" for k, v in prm.items()) + "（仮）")
        for m in p["monitors"]["skipped"]:
            rows.append(f"  見張り役（止めた）: {m['name']}  足りない {m['missing']}")
    if "task_file" in p:
        rows.append(f"  タスクの書式: {p['task_file']}  指示文 {p['instruction']['mode']}「{p['instruction']['example']}」")
        if p.get("place_finder"):
            rows.append(f"  置き場所・物の推定: {p['place_finder']}")
        if p.get("vla_known"):
            rows.append(f"  VLA 側の既知の値: {p['vla_known']}")
    rows.append("  感覚:")
    for r in p["senses"]:
        rows.append(f"    {r['name']:16s} 実機 {'有' if r['real'] else '無'}  模型 {'有' if r['sim_model'] else '無'}  "
                    f"{r['rate_hz']:.1f} Hz  遅れ {r['latency_s']:.3f} s  {'○' if r['usable'] else '×'}")
    return "\n".join(rows)


def cmd_table(argv) -> int:
    ap = argparse.ArgumentParser(prog="ladder.py table")
    ap.add_argument("--target", default="sim", choices=("sim", "real"))
    ap.add_argument("--arm", default=None, choices=sorted(ARMS))
    ap.add_argument("--finder", default="estimated", choices=FINDERS)
    a = ap.parse_args(argv)
    for arm in ([a.arm] if a.arm else list(ARMS)):
        print(format_plan(arm_plan(arm, a.target, a.finder)))
    return 0


# ---------------------------------------------------------------- 実行
def target_color(ex):
    """今の手順の目標の色（実行器の計画と手順の番号から。外乱が読む）。"""
    plan = None if ex is None else getattr(ex, "plan", None)
    j = None if ex is None else getattr(ex, "j", None)
    if not plan or j is None or not 0 <= j < len(plan.get("steps") or []):
        return None
    return plan["steps"][j]


def make_engine(r96, arm: str, info: dict, step_reset=None, finder: str = "estimated", task: str = TASK,
                vla_text: str = "fixed", task_planner: str = "spec", disturb: str = "none", select: dict = None,
                rtc: dict = None):
    from recovla.api import executor_monitored as XL
    from recovla.api import skills as SK
    from recovla.api import task_spec as TS
    from recovla.diag import step_reset as SR
    from recovla.planner import replan_event as RL
    from recovla.planner import replan_base as R4
    from recovla.runtime.perception import Perception
    spec = TS.load_task(task_path(task))
    new_per = perception_factory(spec, finder)
    default = is_default_task(task)
    sel = None
    if select:                                   # Udc: 推論ごとに K 本の行動列から見込みで 1 本を選ぶ
        from recovla.rl import select as S
        sel = S.Select(select["scorer"], k=select["k"], control="never")
        if sel.sc is None:
            raise SystemExit(f"--select-scorer {select['scorer']} を読めない")
    sel_log = []                                 # 試行の中の手順ごとの選びの記録（start_trial のたびに区切る）
    place_log = []                               # Lp: 試行の中の start_trial ごとに決めた置く位置
    place_prt = []                               # Lp: いまの試行の実行系（閉包に評価の枠 eng を入れると、実行系から世界に届かないことの検査に掛かる）

    def sel_flush():
        if sel is not None and sel.rows:
            ps = [r["p"] for r in sel.rows if r["p"] is not None]
            sel_log.append({"seed": sel.seed, "n_infer": len(sel.rows), "n_moved": sum(1 for r in sel.rows if r["j"] != 0),
                            "p_chosen_mean": sum(ps) / len(ps) if ps else None})
            sel.rows = []

    class LadderEngine(r96.Engine):
        def new_world(self):
            if default:
                return super().new_world()
            from recovla.harness.world_cup_plate import make_cup_plate_eval_world
            if self.suite is not None:
                self.suite.close()
            gc = False if getattr(self.a, "diag_no_gravcomp", False) else None
            self.world, self.suite = make_cup_plate_eval_world(self.cfg, gravcomp=gc, spec=spec)

        def _task_cfg(self):
            """既定でないタスク: VLA の条件の指示文の書式（--vla-text）と、起動の箱の位置のずれの確かめを切った写し。"""
            import copy
            cfg = copy.deepcopy(self.cfg)
            if arm in VLA_ARMS:
                cfg["convert"]["instruction"] = TS.policy_format(spec, vla_text, cfg["convert"]["instruction"])
            ch = cfg["runtime_v2"]["checks"]
            ch["box_dev_m"], ch["box_dev_deg"] = 1e9, 1e9
            return cfg

        def _script_make(self):
            """CL の実行器を作る関数（方策の代わりに ScriptSkill。方策・GPU は読まない）。"""
            a, CFG = self.a, self.cfg
            from recovla.common import config
            from recovla.runtime import cue as C
            from recovla.runtime.executor import TaskRuntime
            from recovla.runtime.judge import JudgeV2
            from recovla.runtime.motion import Motion
            from recovla.runtime.perception import Params
            act, rtv = CFG["actuation"], CFG["runtime_v2"]
            thr = C.Thresholds.from_dict(config.color_detect(CFG))
            decompose_fn, _ = r96.planner_of(getattr(a, "planner", None) or "s4")
            rules = script_rules(CFG)

            def make(io, setup):
                per = new_per(setup, Params.from_config(rtv["perception"]), thr)
                skill = SK.ScriptSkill(io, setup, rules, spec, CFG["convert"]["instruction"], perception=per, finder=finder,
                                       checks=rtv["checks"], tip_offset=float(CFG["sim"]["fingertip_offset"]),
                                       motion=Motion(setup, margin=float(act["limiter_margin"]),
                                                     xcmd_leash_m=rtv.get("xcmd_leash_m"), cart_margin=rtv.get("cart_margin")))
                judge = JudgeV2(setup, per, thr, rtv["judge"])
                e = CFG["expert"]
                mp = {"gain": float(e["gain_per_s"]), "xy_max": float(e["speed_ref"]["xy"]), "z_max": float(e["speed_ref"]["z"]),
                      "z_tol": float(e["move_tol_m"]), "tol": float(e["phase"]["retreat_tol_m"])}
                return TaskRuntime(io, setup, skill, judge, CFG["planner"], mp, decompose_fn, CFG["convert"]["instruction"])
            return make

        def _cl2_make(self):
            """CL2: 今の Perception に推定の版と高さの地図を被せ、CL2Skill と CL2TaskRuntime を作る（方策・GPU は読まない）。
            JudgeV2 は記録だけに作る。"""
            a, CFG = self.a, self.cfg
            from recovla.api import executor_script as XC
            from recovla.api import skills_script_self as CL2
            from recovla.common import config
            from recovla.runtime import cue as C
            from recovla.runtime.judge import JudgeV2
            from recovla.runtime.motion import Motion
            from recovla.runtime.perception import Params
            act, rtv = CFG["actuation"], CFG["runtime_v2"]
            thr = C.Thresholds.from_dict(config.color_detect(CFG))
            decompose_fn, _ = r96.planner_of(getattr(a, "planner", None) or "s4")
            e = CFG["expert"]
            rules = CL2.CL2Rules.from_config(e, e["phase"])
            fmt = CFG["convert"]["instruction"]

            def make(io, setup):
                per = CL2.make_perception(Perception(setup, Params.from_config(rtv["perception"]), thr), spec, setup, rules)
                skill = CL2.CL2Skill(io, setup, rules, spec, fmt, perception=per, checks=rtv["checks"],
                                     tip_offset=float(CFG["sim"]["fingertip_offset"]),
                                     motion=Motion(setup, margin=float(act["limiter_margin"]),
                                                   xcmd_leash_m=rtv.get("xcmd_leash_m"), cart_margin=rtv.get("cart_margin")))
                judge = JudgeV2(setup, per, thr, rtv["judge"])
                return XC.CL2TaskRuntime(io, setup, skill, judge, CFG["planner"], decompose_fn, fmt, spec, rules)
            return make

        def init_task(self):
            if not default:
                self.cfg = self._task_cfg()
            super().init_task()                                  # 世界・計画役（make はまだ呼ばないので方策は読まない）
            if arm == "CL2":
                orig = self._cl2_make()
            else:
                orig = self._script_make() if arm == "CL" else self.make_task
            self.u4_last = None
            eng = self

            def make(io, setup):
                ex = orig(io, setup)
                if rtc is not None:                              # 方策の実行を RTC に（方策は試行をまたいで同じもの）
                    from recovla.runtime.runner import enable_rtc_for
                    prt0 = ex.prt
                    prt0.mode, prt0.E = "rtc", int(rtc["horizon"])
                    if getattr(prt0.policy, "_rtc_set", None) != rtc:
                        enable_rtc_for(prt0.policy, int(rtc["horizon"]), rtc["schedule"], float(rtc["max_guidance_weight"]))
                        prt0.policy._rtc_set = dict(rtc)
                if sel is not None:                              # 方策は試行をまたいで同じもの（attach は 1 回だけ替える）
                    pol = ex.prt.policy
                    S.attach(pol, sel)
                    if not getattr(pol, "_select_start_wrapped", False):
                        pol._select_start_wrapped = True
                        start0 = pol.start_trial

                        def start_trial(seed, _start0=start0):   # 手順・やり直しごと: 学習（1 手順の試行）と同じく推論の番号を 0 から
                            _start0(seed)
                            sel_flush()
                            sel.begin_episode(seed)
                        pol.start_trial = start_trial
                if arm in VLA_ARMS and not default:
                    pol = getattr(ex.prt, "policy", None)          # 方策の目標位置の入力の平面をカップの中心へ（VLA 側の既知の値）
                    if pol is not None and getattr(pol, "cue", None) is not None:
                        from recovla.runtime import cue as C
                        pol.cue = C.TargetCue(pol.cue.calib, pol.cue.thr, setup.table_z + eng.world.cue_height(),
                                              setup.cue_fallback_xy)
                if arm in REPLAN_ARMS and not default and finder == "estimated":
                    from recovla.api.estimated import GeometryPerception
                    from recovla.api.estimated_place import finder_class
                    per = ex.prt.perception                      # executor_monitored は geom の無い知覚にだけ被せる（先に被せる）
                    GeometryPerception.attach(per, finder_class(spec).from_spec(spec, per, ex.setup))
                if arm == "L2v2":
                    ex = XL.install(ex, RL.make_replan_step(REPLAN_MODEL, wording=replan_wording(spec, default)),
                                    make_monitors(arm), spec, max_replans=R4.MAX_REPLANS, finder=finder)
                elif arm == "Lf":
                    from recovla.api import executor_improved as XR
                    from recovla.planner import replan_improved as RR
                    ex = XR.install(ex, RR.make_replan_step(REPLAN_MODEL, wording=replan_wording(spec, default)),
                                    make_monitors(arm), spec,
                                    max_replans=R4.MAX_REPLANS, finder=finder)
                if arm != "CL2":                                 # CL2 は手順の切り替え時の待機位置への戻しを掛けない
                    ex = SR.install(ex, step_reset, Perception if arm == "U0" else new_per)
                pol = getattr(ex.prt, "policy", None)
                if getattr(pol, "place_cue", False):             # Lp: 手順・やり直しの始めに置く位置を決めて渡す
                    if finder != "estimated":
                        raise SystemExit("置く場所の手がかりの重みは --finder estimated だけ（推定した箱から選ぶ）")
                    place_prt[:] = [ex.prt]
                    if not getattr(pol, "_place_start_wrapped", False):
                        pol._place_start_wrapped = True
                        start1 = pol.start_trial

                        def start_trial_p(seed, _start1=start1, _pol=pol):
                            _start1(seed)
                            from recovla.api import place_cue as PCUE
                            prt = place_prt[0]
                            wm = getattr(prt, "wm", None)
                            est = getattr(wm, "geometry", None)
                            color = getattr(prt, "color", None)
                            target = next((o.name for o in spec.objects if TS.finder_kind(o.finder) == ("color", color)), None)
                            xy, pinfo = PCUE.free_cell(est, spec.goal.place, target) if est is not None else (None, {"source": "no_estimate"})
                            _pol.set_place(xy)
                            place_log.append({"t": None if wm is None else round(float(wm.t), 2), "color": color,
                                              "xy": None if xy is None else [round(float(v), 4) for v in xy], **pinfo})
                        pol.start_trial = start_trial_p
                eng.u4_last = ex
                return ex
            self.make_task = make

        def _before_trial(self):
            super()._before_trial()
            self.disturber = None
            if disturb == "push":                            # 評価の外乱（評価の枠の側。実行系には知らせない）
                from recovla.eval import disturb_task as DT
                self.disturber = DT.PushDropDisturber(self._seed)
                DT.attach(self.world, self.disturber, lambda: target_color(self.u4_last))

        def run_one_task(self, i, seed):
            retry = None
            self._seed, self.u4_last = int(seed), None
            sel_log.clear()
            place_log.clear()
            if sel is not None:
                sel.rows = []
            try:
                meta, arrays, rlog = super().run_one_task(i, seed)
            except Exception as e:                           # noqa: BLE001
                if not _is_api_error(e):
                    raise
                retry = {"n": 1, "error": f"{type(e).__name__}: {e}"[:500], "at": time.strftime("%Y-%m-%d %H:%M:%S")}
                print(f"[api] 試行 {i} 種 {seed}: 計画役の API が失敗したので同じ種で 1 回だけ回し直す（{retry['error']}）", flush=True)
                self.u4_last = None
                sel_log.clear()
                place_log.clear()
                if sel is not None:
                    sel.rows = []
                meta, arrays, rlog = super().run_one_task(i, seed)
            if sel is not None:
                sel_flush()
                steps = [x for x in sel_log if x["n_infer"]]
                meta["select"] = dict(select, control="never", steps=steps, n_infer=sum(x["n_infer"] for x in steps),
                                      n_moved=sum(x["n_moved"] for x in steps))
            if place_log:
                meta["place_cue"] = list(place_log)
            if disturb != "none":
                meta["disturb"] = self.disturber.record()
            ex = self.u4_last
            rec = ex.record() if ex is not None else {}
            meta["replans"] = rec.get("replans", [])
            meta["early_events"] = rec.get("early_events", [])
            meta["step_reset"] = rec.get("step_reset")
            meta["u4"] = dict(info, arm=arm, step_reset=step_reset, api_retry=retry, ended=rec.get("ended"),
                              interventions=rec.get("interventions", {"plan_change": 0, "replan_requests": 0, "llm_calls": 0}),
                              max_replans=R4.MAX_REPLANS if arm in REPLAN_ARMS else 0,
                              max_runs_per_color=R4.MAX_RUNS_PER_COLOR if arm in REPLAN_ARMS else None,
                              replan_model=REPLAN_MODEL if arm in REPLAN_ARMS else None)
            if arm == "Lf":
                meta["u4"]["notify"] = rec.get("notify")
            if rtc is not None:
                prt1 = getattr(ex, "prt", None)
                meta["u4"]["runtime"] = dict(rtc, mode=getattr(prt1, "mode", None),
                                             rtc_config=None if prt1 is None else _rtc_config_of(prt1.policy))
            if arm in REPLAN_ARMS and not default:
                meta["u4"]["replan_wording"] = replan_wording(spec, default)
            meta["api"] = {"arm": arm, "plan": arm_plan(arm, finder=finder, task=task, vla_text=vla_text,
                                                        task_planner=task_planner), "monitors": rec.get("monitors"),
                           "files_sha256": info["files_sha256"]}
            if not default:
                meta["api"].update(task=spec.name, task_file=task_path(task), vla_text=vla_text, task_planner=task_planner,
                                   instruction_fmt=self.cfg["convert"]["instruction"] if arm in VLA_ARMS else None)
            if arm == "CL2":
                meta["cl2"] = rec.get("cl2")
            return meta, arrays, rlog
    return LadderEngine


def _rtc_config_of(pol) -> dict:
    """方策に入っている lerobot の RTCConfig（記録用。入っていなければ None）。"""
    from recovla.policy.runner import _base_policy
    c = getattr(_base_policy(pol.policy).config, "rtc_config", None)
    if c is None:
        return None
    return {"enabled": bool(c.enabled), "prefix_attention_schedule": c.prefix_attention_schedule.name,
            "max_guidance_weight": float(c.max_guidance_weight), "execution_horizon": int(c.execution_horizon)}


def cmd_run(argv) -> int:
    ap = argparse.ArgumentParser(prog="ladder.py run", add_help=False, allow_abbrev=False)   # --model を --model-ckpt と取らない
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--grip-debounce", type=int, default=None)
    ap.add_argument("--step-reset", default=None, choices=("home", "rebuild", "home_rebuild"))
    ap.add_argument("--finder", default="estimated", choices=FINDERS)
    ap.add_argument("--task", default=TASK)
    ap.add_argument("--vla-text", default="fixed", choices=VLA_TEXTS)
    ap.add_argument("--task-planner", default="spec", choices=TASK_PLANNERS)
    ap.add_argument("--model-ckpt", action="append", default=[], metavar="NAME=DIR")
    ap.add_argument("--disturb", default="none", choices=DISTURBS)
    ap.add_argument("--select-scorer", default=None, help="R2 の段階 C の腕 Uc・Udc・Ldc: 見込みの重み（scorer.pt）。U0・Lf の腕だけ")
    ap.add_argument("--select-scorer-sha256", default=None)
    ap.add_argument("--select-k", type=int, default=8)
    ap.add_argument("--runtime-mode", default="naive", choices=("naive", "rtc"))
    ap.add_argument("--rtc-horizon", type=int, default=None)
    ap.add_argument("--rtc-schedule", default=None, choices=RTC_SCHEDULES)
    ap.add_argument("--rtc-max-guidance", type=float, default=None)
    mine, rest = ap.parse_known_args(argv)
    arm = mine.arm
    rtc = None
    if mine.runtime_mode == "rtc":
        if arm not in ("U0", "Lf") or None in (mine.rtc_horizon, mine.rtc_schedule, mine.rtc_max_guidance):
            print("--runtime-mode rtc は条件 U0・Lf だけで、--rtc-horizon・--rtc-schedule・--rtc-max-guidance を全部渡す", file=sys.stderr)
            return 3
        if not 1 <= mine.rtc_horizon <= 50 or mine.rtc_max_guidance <= 0:
            print("--rtc-horizon は 1〜50（行動列の長さ）、--rtc-max-guidance は正の値", file=sys.stderr)
            return 3
        rtc = {"horizon": int(mine.rtc_horizon), "schedule": mine.rtc_schedule, "max_guidance_weight": float(mine.rtc_max_guidance)}
    elif any(v is not None for v in (mine.rtc_horizon, mine.rtc_schedule, mine.rtc_max_guidance)):
        print("--rtc-* は --runtime-mode rtc のときだけ", file=sys.stderr)
        return 3
    select = None
    if mine.select_scorer:
        p = pathlib.Path(mine.select_scorer)
        p = p if p.is_absolute() else ROOT / p
        if arm not in ("U0", "Lf") or mine.disturb != "none" or not mine.select_scorer_sha256 or mine.select_k < 2:
            print("--select-scorer は腕 U0・Lf・外乱なしだけで、--select-scorer-sha256 と 2 以上の --select-k が要る", file=sys.stderr)
            return 3
        if not p.is_file():
            print(f"--select-scorer {p} が無い", file=sys.stderr)
            return 3
        s98_ = _load(ROOT / "scripts" / "eval_replan_task.py", "s4_u4_for_api_ladder")
        sha = s98_.sha256_file(p)
        if sha != mine.select_scorer_sha256.lower():
            print(f"--select-scorer の SHA-256 が渡された値と違う: {sha}", file=sys.stderr)
            return 3
        select = {"scorer": mine.select_scorer, "scorer_sha256": sha, "k": int(mine.select_k)}
    if arm == "Lf" and mine.finder != "estimated":
        print("Lf は --finder estimated だけ（場面の要約の直しは推定の形を使う）", file=sys.stderr)
        return 3
    if mine.disturb != "none" and not is_default_task(task_path(mine.task)):
        print("--disturb は立方体のタスク（既定）だけ", file=sys.stderr)
        return 3
    if arm == "CL2" and (mine.finder != "estimated" or mine.step_reset is not None):
        print("CL2 は --finder estimated だけで、--step-reset は掛けない（自分で決め直す）", file=sys.stderr)
        return 3
    task = task_path(mine.task)
    default = is_default_task(task)
    if not (ROOT / task).is_file():
        print(f"--task {mine.task}: {task} が無い", file=sys.stderr)
        return 3
    if not default and (arm == "CL" or mine.finder != "estimated"):
        print("既定でないタスクは腕 U0・L2v2・Lf・CL2 と --finder estimated だけ（CL は箱だけの台本、color は箱の既知の寸法の知覚）",
              file=sys.stderr)
        return 3
    ckpts = {}
    for x in mine.model_ckpt:
        k, _, v = x.partition("=")
        if not k or not v or k == "Rreal":
            print("--model-ckpt <名>=<保存点のフォルダ>（名前は Rreal 以外）", file=sys.stderr)
            return 3
        ckpts[k] = v
    if not default:
        from recovla.api import task_spec as TS
        spec = TS.load_task(task)
        if "--text" not in rest and not any(x.startswith("--text=") for x in rest) and spec.instructions:
            rest = rest + ["--text", spec.instructions[0]]
    plan = arm_plan(arm, finder=mine.finder, task=task, vla_text=mine.vla_text, task_planner=mine.task_planner)
    print(format_plan(plan), flush=True)
    s98 = _load(ROOT / "scripts" / "eval_replan_task.py", "s4_u4_for_api_ladder")
    ev = _load(ROOT / "scripts" / "real" / "eval_real_matched.py", "eval_v3_real_for_api_ladder")
    gr = _load(ROOT / "scripts" / "real" / "eval_real_matched_grip.py", "eval_v3_real_grip_for_api_ladder")
    r96 = gr.patch(ev.patch(_load(ROOT / "scripts" / "eval_resumable.py", "s4_resume_api_ladder"), "v3_real", None),
                   mine.grip_debounce, dict({"Rreal": RREAL}, **ckpts))
    a = r96.build_parser().parse_args(["task"] + rest)
    if a.model != "Rreal" and a.model not in ckpts:
        print("この入口は新 R（--model Rreal）か、--model-ckpt で足した名前だけを回す（CL は方策を読まないが、記録の形をそろえる"
              "ため同じ名前を付ける）", file=sys.stderr)
        return 3
    if a.exec_interval not in (None, 6) or a.planner != "s4":
        print("--exec-interval 6・計画役 s4 だけ（ほかの値は渡さない）", file=sys.stderr)
        return 3
    a.exec_interval, a.no_safety = 6, True
    if arm in REPLAN_ARMS and not a.dry_run:
        from recovla.planner import decompose as D
        if not D._api_key():
            print("L2v2 は ANTHROPIC_API_KEY が要る。--dry-run なら要らない", file=sys.stderr)
            return 3
    extra = LAT_FILES + (CL2_FILES if arm == "CL2" else ()) + (() if default else TASK_FILES + (task,))
    if arm != "U0" and mine.finder == "estimated":
        extra += tuple(p for p in EST_FILES if p not in extra)
    extra += (R1_FILES if arm == "Lf" else ()) + (DISTURB_FILES if mine.disturb != "none" else ())
    extra += SELECT_FILES if select else ()
    info = {"script": "scripts/api/ladder.py", "condition": a.condition, "grip_debounce": mine.grip_debounce,
            "step_reset": mine.step_reset, "task": task, "finder": None if arm == "U0" else mine.finder,
            "files_sha256": {p: s98.sha256_file(ROOT / p) for p in FILES + extra}}
    if not default:
        info.update(vla_text=mine.vla_text, task_planner=mine.task_planner)
    if mine.disturb != "none":
        info["disturb"] = mine.disturb
    if select:
        info["select"] = select
    if rtc:
        info["runtime"] = dict(rtc, mode="rtc")
    if a.model != "Rreal":
        info["model_ckpt"] = ckpts[a.model]
    a.api_arm = arm
    a.api_grip_debounce = mine.grip_debounce
    a.api_step_reset = mine.step_reset
    a.api_finder = None if arm == "U0" else mine.finder
    a.api_files = json.dumps(info["files_sha256"], sort_keys=True)
    r96.SPEC_KEYS = tuple(r96.SPEC_KEYS) + ("api_arm", "api_grip_debounce", "api_step_reset", "api_finder", "api_files")
    if not default:                                          # 既定のタスクの控え（resume_spec）の鍵は変えない
        a.api_task, a.api_vla_text, a.api_task_planner = task, mine.vla_text, mine.task_planner
        r96.SPEC_KEYS = tuple(r96.SPEC_KEYS) + ("api_task", "api_vla_text", "api_task_planner")
        if mine.task_planner == "spec":
            from recovla.planner import decompose_task as DT
            fn = DT.make_decompose(spec)
            r96.planner_of = lambda name, _fn=fn: (_fn, dict(_fn.info, asked=name))
    if a.model != "Rreal":
        a.api_model_ckpt = ckpts[a.model]
        r96.SPEC_KEYS = tuple(r96.SPEC_KEYS) + ("api_model_ckpt",)
    if mine.disturb != "none":                               # 外乱なしの控え（resume_spec）の鍵は変えない
        a.api_disturb = mine.disturb
        r96.SPEC_KEYS = tuple(r96.SPEC_KEYS) + ("api_disturb",)
    if select:                                               # 選ばない条件の控え（resume_spec）の鍵は変えない
        a.api_select = json.dumps(select, sort_keys=True)
        r96.SPEC_KEYS = tuple(r96.SPEC_KEYS) + ("api_select",)
    if rtc:                                                  # naive の再開の鍵は変えない
        a.api_rtc = json.dumps(rtc, sort_keys=True)
        r96.SPEC_KEYS = tuple(r96.SPEC_KEYS) + ("api_rtc",)
    r96.Engine = make_engine(r96, arm, info, mine.step_reset, mine.finder, task, mine.vla_text, mine.task_planner,
                             mine.disturb, select, rtc)
    orig_text = r96.run_json_text

    def run_json_text(a_, kind, rows, *x, **kw):
        d = json.loads(orig_text(a_, kind, rows, *x, **kw))
        d["u4"] = dict(info, arm=arm, arm_ja=ARMS[arm])
        d["api"] = {"arm": arm, "plan": plan, "files_sha256": info["files_sha256"]}
        if not default:
            d["api"].update(task=task, vla_text=mine.vla_text, task_planner=mine.task_planner)
        return json.dumps(d, ensure_ascii=False, indent=1)
    r96.run_json_text = run_json_text
    print(f"[api] 腕 {arm}: {ARMS[arm]}、世界 v3_real、開閉の絞り {mine.grip_debounce or '切'}、手順の切り替え "
          f"{mine.step_reset or 'そのまま'}、指示文「{a.text}」"
          + ("" if default else f"、タスク {task}（VLA の指示文 {mine.vla_text}、計画役 {mine.task_planner}）")
          + ("" if a.model == "Rreal" else f"、保存点 {a.model}={ckpts[a.model]}")
          + ("" if not select else f"、K {select['k']} 本から見込みで選ぶ（{select['scorer']}、{select['scorer_sha256'][:12]}…）")
          + ("" if not rtc else f"、RTC（範囲 {rtc['horizon']}・{rtc['schedule']}・重みの上限 {rtc['max_guidance_weight']:g}）"),
          flush=True)
    ops = r96.load_ops()
    v82 = r96.load_82(a.allow_82_change)
    try:
        return r96.cmd_main(a, v82, ops)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(e.code, file=sys.stderr, flush=True)
            return 3
        raise


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:                                    # noqa: BLE001
        pass
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "table":
        return cmd_table(argv[1:])
    if not argv or argv[0] != "run":
        raise SystemExit("使い方: ladder.py run --arm U0|L2v2|CL|CL2 ...  または  ladder.py table [--target sim|real]")
    return cmd_run(argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
