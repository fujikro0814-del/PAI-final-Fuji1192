"""評価の前に加えた実行系の改良（docs/api/r1_protocol.md 5 節）の確かめ: LLM の組み直しの指示文（planner/replan_improved）、
場面の要約（api/scene_summary）、失敗の検知（api/monitors_in_hand）、実行器の包み（api/executor_improved）、入口の条件 Lf（scripts/api/ladder.py）。
ほかの条件（組み直しの前の版 L2v2・組み直しなしの U0）の指示文・キャッシュの鍵・条件の表が変わらないことも固定する。
CPU だけ（偽の io・知覚・判定・LLM）。"""
import json

import numpy as np
import pytest

from recovla.api import executor_improved as XR
from recovla.api import monitors as M
from recovla.api import monitors_in_hand as MR
from recovla.api import perception as AP
from recovla.api import scene_summary as SS
from recovla.planner import replan_event as RL
from recovla.planner import replan_improved as RR
from recovla.planner import replan_base as R4

from .kin_world import box_region
from .test_monitors import BOX, DROP_RED, RED, SIG, SPEC, T, _load, est_of, feed, grasp_seq, make_ex, ROOT

# 前の版 L2v2 の試行（シード 171201）で 1 回目の組み直しに渡した入力と、その記録のキャッシュの鍵（以前の指示文が変わっていない）
CTX_171201 = {"instruction": "全部片付けて", "plan": ["red", "green", "blue"],
              "steps": [{"color": "red", "result": "success", "attempts": 1}, {"color": "green", "result": "success", "attempts": 1},
                        {"color": "blue", "result": "timeout", "attempts": 2}],
              "perception": {"table": ["blue"], "in_box": ["red", "green"], "in_hand": [], "lost": []},
              "failed": {"step": 2, "color": "blue", "event": "timeout"}, "replans_done": 0}
KEY_171201 = "d12dbf1a458c2d54f2fc8a951c2d07d9029d839e7f699dc537634ed18424a10c"


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.setattr(R4, "CACHE", tmp_path / "llm_cache")


def _rr_ctx(c=CTX_171201, **per):
    p = dict(c["perception"], **per)
    f = c["failed"]
    return RR.make_context(c["instruction"], c["plan"], c["steps"], p, f["step"], f["color"], c["replans_done"], f["event"])


# ---------------------------------------------------------------- 1. 指示文
def test_old_prompt_path_is_unchanged():
    c, f = CTX_171201, CTX_171201["failed"]
    ctx = RL.make_context(c["instruction"], c["plan"], c["steps"], c["perception"], f["step"], f["color"], c["replans_done"], f["event"])
    assert RL.cache_key("claude-sonnet-5-5", R4.user_message(ctx)) == KEY_171201
    assert "max_runs_per_color" in R4.user_message(ctx) and '"attempts": 2' in R4.user_message(ctx)


def test_new_prompt_separates_tries_and_runs():
    ctx = _rr_ctx()
    msg = json.loads(RR.user_message(ctx))
    assert "attempts" not in json.dumps(msg) and "max_runs_per_color" not in msg
    assert msg["steps"][2] == {"color": "blue", "result": "timeout", "tries_in_this_run": 2, "tries_limit_per_run": 2}
    assert msg["runs_per_color"] == {"blue": {"runs_used": 1, "runs_limit": 2, "runs_left": 1}}
    assert msg["candidates"] == ["blue"] and "skip_notify" in msg["actions"]
    assert "tries" in RR.SYSTEM and "runs_left" in RR.SYSTEM and "色ごとの上限ではありません" in RR.SYSTEM
    assert RR.cache_key("claude-sonnet-5-5", RR.user_message(ctx)) != KEY_171201 and RR.KIND != RL.KIND


def test_stacked_is_a_candidate_and_skip_notify_is_checked():
    ctx = _rr_ctx(in_box=["red"], stacked_in_box=["green"])
    assert RR.candidates(ctx) == ["blue"]                          # 緑は成功の手順があるので候補にならない
    c2 = dict(CTX_171201, steps=CTX_171201["steps"][:1] + [{"color": "green", "result": "timeout", "attempts": 2}],
              failed={"step": 1, "color": "green", "event": "timeout"})
    ctx = _rr_ctx(c2, table=["blue"], in_box=["red"], stacked_in_box=["green"])
    assert RR.candidates(ctx) == ["green", "blue"]
    assert RR.check({"action": "next", "color": "green", "order": []}, ctx) == []
    assert RR.check({"action": "skip_notify", "color": "green", "order": []}, ctx) == []
    assert RR.check({"action": "skip_notify", "color": "blue", "order": []}, ctx)
    assert RR.apply({"action": "skip_notify", "color": "green"}, ctx) == ["blue"]
    rep = RR.template_report("skip_notify", ctx)
    assert "後に回し" in rep and not RR.RM.check_report(rep)
    lost = _rr_ctx(c2, table=[], in_box=["red", "blue"], lost=["green"])
    assert RR.apply({"action": "skip_notify", "color": "green"}, lost) == []
    assert not RR.RM.check_report(RR.template_report("skip_notify", lost))


def test_new_replan_step_with_fake_client_and_fallbacks():
    cli = T.FakeClient(T.raw("skip_notify", "blue", [], report="青は取り出せません。ここで終えてお知らせします。"))
    out = RR.replan_step(text="x", plan_order=["red", "green", "blue"], steps=CTX_171201["steps"],
                         perception={"table": [], "in_box": ["red", "green"], "lost": ["blue"]}, failed_step=2,
                         failed_color="blue", replans_done=0, event="timeout", cli=cli, use_cache=False)
    assert out["decision"] == {"action": "skip_notify", "color": "blue", "order": []} and out["accepted"]
    assert out["next_order"] == [] and out["plan_change"] and out["kind"] == "replan_r1"
    assert cli.calls[0]["system"] == RR.SYSTEM and "skip_notify" in json.dumps(cli.calls[0]["output_config"])
    bad = RR.replan_step(text="x", plan_order=["red", "green", "blue"], steps=CTX_171201["steps"],
                         perception={"table": ["blue"], "in_box": ["red", "green"]}, failed_step=2, failed_color="blue",
                         replans_done=0, cli=T.FakeClient(T.raw("next", "red", [])), use_cache=False)
    assert bad["decision"]["action"] == "stop" and bad["fallback"] and bad["rejected"]


# ---------------------------------------------------------------- 2. 場面の要約
def _obj(pos, size=0.04, ext=None, status="seen", near=None, in_box=False):
    e = np.full(3, size) if ext is None else np.asarray(ext, float)
    return AP.ObjState(np.asarray(pos, float), 0.0, status, {"box": in_box}, near, 0.0, e, np.eye(3), SIG ** 2 * np.eye(3))


def _est(**objs):
    return AP.WorldEstimate(0.0, objs, {"box": BOX}, 0.0, "estimated")


BASE = {"table": [], "in_box": ["red", "green"], "in_hand": [], "lost": []}


def test_stacked_cube_is_not_in_box():
    # 箱の底は z 0.01（kin_world.box_region）。緑は赤の上（下の端が底より 4 cm 上）
    est = _est(red=_obj([0.45, 0.25, 0.03]), green=_obj([0.452, 0.25, 0.07]))
    out = SS.refine(BASE, est)
    assert out["in_box"] == ["red"] and out["stacked_in_box"] == ["green"]
    # 推定が下の面（箱の底）まで延ばされた形（高さ 8 cm）でも、下の物の上面との差で見分ける
    est = _est(red=_obj([0.45, 0.25, 0.03]), green=_obj([0.452, 0.25, 0.05], ext=[0.04, 0.04, 0.08]))
    assert SS.refine(BASE, est)["stacked_in_box"] == ["green"]
    # 2 つ並んで底にある: 重なりでない
    est = _est(red=_obj([0.43, 0.25, 0.03]), green=_obj([0.48, 0.25, 0.03]))
    out = SS.refine(BASE, est)
    assert out["in_box"] == ["red", "green"] and out["stacked_in_box"] == []
    # 箱の大きさ・物の大きさを 2 倍にしても同じ答え（既知の寸法を使わない、相対の閾値）
    big = AP.PlaceState(np.array([0.45, 0.25]), 0.0, True, None, box_region(inner=0.16, rim=0.12, floor_t=0.02))
    est = AP.WorldEstimate(0.0, {"red": _obj([0.45, 0.25, 0.06], 0.08), "green": _obj([0.454, 0.25, 0.14], 0.08)},
                           {"box": big}, 0.0, "estimated")
    assert SS.refine(BASE, est)["stacked_in_box"] == ["green"]


def test_near_wall_and_not_in_hand_after_hold():
    base = {"table": ["blue"], "in_box": [], "in_hand": [], "lost": []}
    wall = _est(blue=_obj([0.45 + 0.085 + 0.025, 0.25, 0.02]))           # 外の縁から 2.5 cm（触れると 2 cm）
    far = _est(blue=_obj([0.45, -0.10, 0.02]))
    assert SS.refine(base, wall)["near_box_wall"] == ["blue"] and SS.refine(base, far)["near_box_wall"] == []
    tip = np.array([0.45, -0.10, 0.20])
    assert SS.refine(base, far, held={"blue"}, tip=tip)["not_in_hand_after_hold"] == ["blue"]
    near = _est(blue=_obj([0.45, -0.10, 0.19]))
    assert SS.refine(base, near, held={"blue"}, tip=tip)["not_in_hand_after_hold"] == []
    inbox = {"table": [], "in_box": ["blue"], "in_hand": [], "lost": []}
    assert SS.refine(inbox, _est(blue=_obj([0.45, 0.25, 0.03])), held={"blue"}, tip=tip)["not_in_hand_after_hold"] == []
    # 推定が無い（色の知覚の版）: 欄を足すだけで中身は変えない
    assert SS.refine(base, None) == dict(base, stacked_in_box=[], near_box_wall=[], not_in_hand_after_hold=[])


# ---------------------------------------------------------------- 失敗の検知
def _after(where):
    def f(t, hand):
        tip = hand + [0, 0, -0.1034]
        if t < 3.0 - 1e-9:
            return est_of(tip)
        if where == "stacked":
            return _est(red=_obj([0.45, 0.25, 0.07]), green=_obj([0.45, 0.25, 0.03], in_box=True))
        if where == "inbox":
            return est_of([0.47, 0.27, 0.03], in_box=True)
        if where == "rim":
            return est_of([0.45 + 0.08, 0.25, 0.08])
        return est_of(tip)
    return f


@pytest.mark.parametrize("where,fires", [("stacked", True), ("rim", True), ("inbox", False), ("hand", False)])
def test_expected_in_hand_monitor(where, fires):
    seq = grasp_seq(lambda t: (0.039, True))
    evs = feed(MR.ExpectedInHandMonitor(), seq, _after(where))
    assert bool(evs) == fires
    if fires:
        assert evs[0][0] == "dropped" and evs[0][1] == pytest.approx(4.0)
    # HeldEstimateDropMonitor は縁・重なりの落下では出さない（ここを補う）
    if where in ("stacked", "rim"):
        assert feed(M.HeldEstimateDropMonitor(), seq, _after(where)) == []


# ---------------------------------------------------------------- 3. 実行器: skip_notify で残りを先に、最後に知らせる
def test_executor_skip_notify_continues_then_notifies():
    cli = T.FakeClient(T.raw("skip_notify", "red", [], report="赤は取り出せません。緑と青を先に入れて、最後にお知らせします。"))
    rp = T.recording(lambda **kw: RR.replan_step(**kw, cli=cli, model=RR.DEFAULT_MODEL, use_cache=False))
    rp.takes_event = True
    ex, io, prt = make_ex("L2v2", {}, rp, M.MonitorSet([M.LegacyInHandDropMonitor()]), step_timeout=0.4)
    ex = XR.install(ex, rp, M.MonitorSet([M.LegacyInHandDropMonitor()]), SPEC, max_replans=2, finder="color",
                    fix_judge=False)                              # 偽の判定のまま（JudgeR1 は下で別に確かめる）
    prt.ex = ex
    rec = T.run(ex)
    assert [r["action"] for r in rec["replans"]] == ["skip_notify"]
    assert [s["color"] for s in rec["steps"]] == ["red", "green", "blue"] and rec["stopped"] is None
    assert ex.ended["kind"] == "notify" and ex.ended["deferred"] == ["red"] and ex.ended["left"] == ["red"]
    assert rec["notify"][0]["color"] == "red"
    kw = rp.calls[0]
    assert all(s["tries_limit"] == 2 for s in kw["steps"])
    assert set(kw["perception"]) == set(SS.KEYS + SS.EXTRA)


# ---------------------------------------------------------------- 3''. 見失った色を黙って外さない（r1_protocol.md 5 節の 4）
RETREAT = np.array([0.4, 0.0, 0.3])            # make_ex の setup.retreat_pose
OVER_TABLE = np.array([0.45, -0.10, 0.08])     # 手順の間の腕（机の上。青を隠す）
SKIP_RED = T.raw("skip", "red", [], reason="赤をあきらめて残りを続ける。", report="赤は入れられませんでした。残りを続けます。")


def _lost_run(arm="Lf", show_at_retreat=True, fix_lost=True, sched=None, step_timeout=0.4):
    """赤は終わらない。青は最初は見えるが、手順が始まると腕に隠れて lost。待機位置（手順の外）では見えるか（show_at_retreat）。"""
    mod = RR if arm == "Lf" else RL
    cli = T.FakeClient(SKIP_RED)
    rp = T.recording(lambda **kw: mod.replan_step(**kw, cli=cli, model=mod.DEFAULT_MODEL, use_cache=False))
    rp.takes_event = True
    if arm == "Lf":
        rp.lost_notice = RR.make_replan_step(RR.DEFAULT_MODEL).lost_notice
    ms = M.MonitorSet([M.LegacyInHandDropMonitor()])
    ex, io, prt = make_ex("L2v2", sched or {}, rp, ms, step_timeout=step_timeout)
    if arm == "Lf":
        ex = XR.install(ex, rp, ms, SPEC, max_replans=2, finder="color", fix_judge=False, fix_lost=fix_lost)
    prt.ex = ex
    base = prt.tick

    def tick():
        if ex.steps and ex.phase == "run":
            prt.motion.x_cmd = OVER_TABLE.copy()
        base()
        at = bool(np.linalg.norm(prt.motion.x_cmd - RETREAT) < 0.02) and ex.phase != "run"
        prt.wm.cubes["blue"].status = "seen" if not ex.steps or (show_at_retreat and at) else "lost"
    prt.tick = tick
    return T.run(ex), rp, ex


def test_lost_untried_color_is_seen_again_after_moving_the_arm():
    rec, rp, ex = _lost_run(show_at_retreat=True)
    r = rec["replans"][0]
    lk = r["look"]
    assert lk["colors"] == ["blue"] and lk["seen_again"] == ["blue"] and lk["still_lost"] == [] and lk["arrived"]
    assert lk["t_begin"] == pytest.approx(rec["steps"][0]["t_end"]) and lk["t_arrive"] > lk["t_begin"]
    assert r["t_request"] == pytest.approx(lk["t_end"]) and lk["t_end"] - lk["t_arrive"] >= XR.LOOK_WAIT_S - 1e-6
    kw = rp.calls[0]
    assert "blue" in kw["perception"]["table"] and kw["perception"]["lost"] == [] and kw["event"] == "timeout"
    assert r["out"]["next_order"] == ["green", "blue"] and r["untried_lost"] == []
    assert [s["color"] for s in rec["steps"]] == ["red", "green", "blue"]
    assert rec["notify"] == [] and rec["ended"] is None and rec["stopped"] is None
    json.dumps(rec["replans"], ensure_ascii=False)


@pytest.mark.parametrize("sched,event,timeout", [(None, "timeout", 0.4), (DROP_RED, "dropped", 5.0)])
def test_lost_untried_color_still_lost_is_notified_at_the_end(sched, event, timeout):
    rec, rp, ex = _lost_run(show_at_retreat=False, sched=sched, step_timeout=timeout)
    r = rec["replans"][0]
    assert r["look"]["still_lost"] == ["blue"] and r["look"]["seen_again"] == []
    assert rp.calls[0]["perception"]["lost"] == ["blue"] and rp.calls[0]["event"] == event and r["event"] == event
    assert r["untried_lost"] == ["blue"] and r["out"]["next_order"] == ["green"]
    assert [s["color"] for s in rec["steps"]] == ["red", "green"] and rec["stopped"] is None
    assert [(n["color"], n["kind"]) for n in rec["notify"]] == [("blue", "lost")]
    assert ex.ended["kind"] == "notify" and ex.ended["deferred"] == ["blue"] and "blue" in ex.ended["left"]
    assert ex.ended["report"] == "青の立方体が見つからなかったので、箱に入れていません。"
    assert not RR.RM.check_report(ex.ended["report"])


def test_lost_notice_comes_from_the_task_wording(ladder):
    from recovla.api import task_spec as TS
    want = "青の立方体が見つからなかったので、箱に入れていません。"            # 既定（立方体と箱）の文
    assert RR.lost_notice("blue") == want and RR.make_replan_step(RR.DEFAULT_MODEL).lost_notice("blue") == want
    assert ladder.replan_wording(TS.load_task(), True) is None
    cup = TS.load_task(ROOT / ladder.task_path("cup_plate"))
    w = ladder.replan_wording(cup, False)
    step = RR.make_replan_step(RR.DEFAULT_MODEL, wording=w)
    assert step.lost_notice("red") == "赤のカップが見つからなかったので、皿に置いていません。"
    assert step.wording == w and not RR.RM.check_report(step.lost_notice("red"))
    # 中へ入れる形のタスクの言葉は「に入れていません」（既定と同じ組み立て）
    assert RR.lost_notice("green", RL.make_wording("袋", "かご", ["green"], "all_in")) == "緑の袋が見つからなかったので、かごに入れていません。"
    # 実行器は文を持たない（計画役の lost_notice が無ければ文なし）
    assert not hasattr(XR, "JA")


def test_lost_fix_off_and_frozen_l2v2_drop_without_looking():
    out = {}
    for arm, fix in (("Lf", False), ("L2v2", None)):
        rec, rp, ex = _lost_run(arm=arm, show_at_retreat=True, fix_lost=bool(fix))
        r = rec["replans"][0]
        assert "look" not in r and "untried_lost" not in r and ex.phase != "look"
        assert r["t_request"] == pytest.approx(rec["steps"][0]["t_end"])            # 腕を動かさずにその場で頼む
        assert rp.calls[0]["perception"]["lost"] == ["blue"] and r["out"]["next_order"] == ["green"]
        assert rec.get("notify", []) == [] and rec["ended"] is None
        out[arm] = ([s["color"] for s in rec["steps"]], r["t_request"], [x["kind"] for x in rec["returns"]])
    assert out["Lf"] == out["L2v2"]


def test_frozen_plan_builders_still_drop_lost_colors():
    steps = [{"color": "red", "result": "timeout", "attempts": 2}]
    per = {"table": ["red", "green"], "in_box": [], "in_hand": [], "lost": ["blue"]}
    c4 = R4.make_context("全部片付けて", ["red", "green", "blue"], steps, per, 0, "red", 0)
    assert R4.candidates(c4) == ["red", "green"] and R4.apply({"action": "next", "color": "red"}, c4) == ["red", "green"]
    cr = RR.make_context("全部片付けて", ["red", "green", "blue"], steps, per, 0, "red", 0, "timeout")
    assert RR.apply({"action": "next", "color": "red"}, cr) == ["red", "green"]       # 知らせるのは実行器の役
    from recovla.api import executor_monitored as executor_l2v2
    from recovla.runtime import executor_replan_early as executor_l2, executor_replan as executor_u4
    for cls in (executor_u4.ReplanTaskRuntime, executor_l2.EarlyReplanTaskRuntime, executor_l2v2.MonitoredReplanTaskRuntime):
        assert not hasattr(cls, "_look_tick") and not hasattr(cls, "look_lost")


# ---------------------------------------------------------------- 3'. 完了判定の直し（JudgeR1）
P_JUDGE = {"box_min_pixels": 69, "wrist_min_pixels": 52, "wrist_min_fraction": 0.8}


@pytest.mark.parametrize("box,wrist,frac,depth,want", [
    (62, 900, 0.99, True, True),          # 俯瞰だけ足りない（奥の壁ぎわ・手前に別の立方体）: 通す
    (30, 900, 0.99, True, False),         # 俯瞰が閾値の半分より少ない: 通さない
    (62, 900, 0.99, False, False),        # 深度の条件（重なり・縁）が偽: 通さない
    (62, 40, 0.99, True, False),          # 手首の画素が足りない: 通さない
    (62, 900, 0.5, True, False),          # 手首の割合が足りない: 通さない
    (90, 900, 0.99, True, True),          # 元の条件で通る: そのまま
])
def test_judge_r1_relaxes_only_the_overhead_pixels(monkeypatch, box, wrist, frac, depth, want):
    from recovla.api import judge_improved as JR
    from recovla.runtime import judge as RJ

    def base(self, *a):
        ib = box >= 69 and wrist >= 52 and frac >= 0.8 and depth
        return {"box_pixels": box, "wrist_in_pixels": wrist, "wrist_fraction": frac, "depth_ok": depth, "in_box": ib}
    monkeypatch.setattr(RJ.JudgeV2, "conditions", base)
    j = object.__new__(JR.JudgeR1)
    j.p = P_JUDGE
    out = j.conditions(None, None, None, None)
    assert out["in_box"] is want or out["in_box"] == want
    assert out["overhead_relaxed"] == (want and box < 69)


# ---------------------------------------------------------------- 4. 入口: ほかの条件は同じ、Lf を足しただけ
@pytest.fixture(scope="module")
def ladder():
    return _load(ROOT / "scripts" / "api" / "ladder.py", "api_ladder_for_r1")


def test_ladder_lf_arm_and_old_arms(ladder):
    ref = json.loads((ROOT / "tests" / "api" / "fixtures" / "ladder_plans_main_7629a01.json").read_text(encoding="utf-8"))
    for key, v in ref.items():
        if not key.startswith("_"):
            arm, target, finder = key.split("|")
            assert json.loads(json.dumps(ladder.arm_plan(arm, target, finder), ensure_ascii=False)) == v["plan"], key
    assert list(ladder.FILES) == ref["_FILES"]
    p = ladder.arm_plan("Lf")
    assert "executor_improved" in p["executor"] and "replan_improved" in p["replan"]
    assert [m["name"] for m in p["monitors"]["active"]] == ["grasp_drop", "held_estimate_drop", "expected_in_hand"]
    assert [m.name for m in ladder.make_monitors("L2v2").monitors] == ["grasp_drop", "held_estimate_drop"]
    assert ladder.make_monitors("U0") is None


def test_r1_plan_matches_the_clc_arguments():
    P = _load(ROOT / "scripts" / "api" / "plan_failure_relearn.py", "api_r1_plan_for_test")
    clc = json.loads((ROOT / "configs" / "rotate" / "plan_clc_a.json").read_text(encoding="utf-8"))["conditions"]
    ps = P.plans("outputs/x/rdag")
    main = ps["plan_r1e_a.json"]["conditions"]

    def norm(a, exp, cond, trials):
        a = list(a)
        for k, v in (("--experiment", exp), ("--condition", cond), ("--trials", trials)):
            a[a.index(k) + 1] = v
        return a
    for c in ("N0", "U0"):                                        # plan_clc_a と同じ引数（実験・条件・シードだけ違う）
        assert norm(clc[c]["argv"], "R1E", c, "171400:100") == main[c]["argv"]
    u = norm(main["U0"]["argv"], "R1E", "Ud", "171400:100")
    assert main["Ud"]["argv"] == [x.replace("Rreal", "Rdag") for x in u][:10] + ["--model-ckpt", "Rdag=outputs/x/rdag"] + u[10:]
    assert "--finder" in main["Lf"]["argv"] and "Lf" in main["Ld"]["argv"] and "Rdag=outputs/x/rdag" in main["Ld"]["argv"]
    # Ln: Lf と同じ実行のしかたで方策だけ Nreal（N0 と同じチェックポイント）
    ln, lf = main["Ln"]["argv"], main["Lf"]["argv"]
    nreal = clc["N0"]["argv"][clc["N0"]["argv"].index("--model-ckpt") + 1]
    assert ln[ln.index("--arm") + 1] == "Lf" and ln[ln.index("--model") + 1] == "Nreal" and nreal in ln
    want = norm(lf, "R1E", "Ln", "171400:100")
    want[want.index("--model") + 1] = "Nreal"
    i = want.index("--model") + 2
    assert ln == want[:i] + ["--model-ckpt", f"Nreal={nreal.split('=', 1)[1]}"] + want[i:]
    assert set(ps["plan_r1d_a.json"]["conditions"]) == {"Lf", "Ld"}
    d = ps["plan_r1d_b.json"]
    assert set(d["conditions"]) == {"Lf", "Ld"} and all("push" in v["argv"] for v in d["conditions"].values())
    assert d["conditions"]["Lf"]["argv"][d["conditions"]["Lf"]["argv"].index("--trials") + 1] == "171650:50"
    # 条件の順番: 1 台目の PC（_a）は N0→Ln、2 台目（_b）は逆
    assert list(main) == ["N0", "U0", "Ud", "Lf", "Ld", "Ln"]
    assert list(ps["plan_r1e_b.json"]["conditions"]) == ["Ln", "Ld", "Lf", "Ud", "U0", "N0"]
    assert P.RDAG.endswith("checkpoints/040000/pretrained_model")      # 最後のチェックポイント（選ばない）


def test_r1_plan_smoke_uses_only_the_dev_band():
    P = _load(ROOT / "scripts" / "api" / "plan_failure_relearn.py", "api_r1_plan_for_test2")
    s = P.smoke_plans("outputs/x/rdag", 193998, "a")
    m, d = s["plan_r1smoke_a.json"], s["plan_r1dsmoke_a.json"]
    assert m["experiment"] == "R1SMOKE" and d["experiment"] == "R1DSMOKE" and m["block"] == 1
    assert all(v["argv"][v["argv"].index("--trials") + 1] == "193998:1" for v in m["conditions"].values())
    assert set(d["conditions"]) == {"Lf", "Ld"} and all("push" in v["argv"] for v in d["conditions"].values())
    assert P.smoke_plans("outputs/x/rdag", 193999, "b")["plan_r1smoke_b.json"]["experiment"] == "R1SMOKE_B"
    assert P.main(["--smoke", "171400", "--allow-missing", "--dry-run"]) == 3


def test_ladder_refuses_lf_with_color_finder_and_disturb_on_other_task(ladder, capsys):
    base = ["run", "--experiment", "X", "--condition", "y", "--trials", "193990:1", "--model", "Rreal", "--dry-run"]
    assert ladder.main(base + ["--arm", "Lf", "--finder", "color"]) == 3
    assert ladder.main(base + ["--arm", "U0", "--disturb", "push", "--task", "cup_plate"]) == 3


# ---------------------------------------------------------------- 5. タスクの言葉（ほかのタスクにも同じ仕組み）
SYSTEM_SHA_DEFAULT = "5d3f1828d71e710d822eefcc05515be19067220d8794bbaa562ccc66aa6b14d2"   # 言葉を足す前の SYSTEM
KEY_DEFAULT = "f5fe462f53fedba220394e2ec44d6336fbb137c715f9070fce9b95815206fdbe"          # 同じく、下の入力の鍵
REPORTS_DEFAULT = {
    "next": "緑の立方体を決められた時間の中で箱に入れられませんでした。先に青を入れます。",
    "reorder": "緑の立方体を決められた時間の中で箱に入れられませんでした。青・緑の順に入れ直します。",
    "skip": "緑の立方体を決められた時間の中で箱に入れられませんでした。緑は飛ばして、青を続けます。",
    "skip_notify": "緑の立方体を決められた時間の中で箱に入れられませんでした。緑は後に回し、青を先に入れて、最後にお知らせします。",
    "finish": "緑の立方体を決められた時間の中で箱に入れられませんでした。箱には赤が入っており、ここで終えます。",
    "stop": "緑の立方体を決められた時間の中で箱に入れられませんでした。箱には赤が入っており、ここで止まって知らせます。"}


def _ctx_default():
    return RR.make_context("全部片付けて", ["red", "green", "blue"],
                           [{"color": "red", "result": "success", "attempts": 1}, {"color": "green", "result": "timeout", "attempts": 2}],
                           {"table": ["green", "blue"], "in_box": ["red"], "lost": []}, 1, "green", 0, "timeout")


def _args(a):
    return ("blue" if a == "next" else ("green" if a in ("skip", "skip_notify") else None),
            ["blue", "green"] if a == "reorder" else None)


def test_default_wording_keeps_the_prompt_key_and_reports_byte_identical():
    import hashlib
    ctx = _ctx_default()
    msg = RR.user_message(ctx)
    assert hashlib.sha256(RR.SYSTEM.encode("utf-8")).hexdigest() == SYSTEM_SHA_DEFAULT
    assert RR.system_for(None) is RR.SYSTEM and RR.build_params(RR.DEFAULT_MODEL, msg)["system"] is RR.SYSTEM
    assert RR.cache_key("claude-sonnet-5-5", msg) == RR.cache_key("claude-sonnet-5-5", msg, None) == KEY_DEFAULT
    for a, want in REPORTS_DEFAULT.items():
        c, o = _args(a)
        assert RR.template_report(a, ctx, c, o) == want == RR.template_report(a, ctx, c, o, None)
    step = RR.make_replan_step(RR.DEFAULT_MODEL)
    assert step.wording is None


def _cup_wording(ladder):
    from recovla.api import task_spec as TS
    return ladder.replan_wording(TS.load_task(ROOT / ladder.task_path("cup_plate")), False)


def test_cup_plate_wording_rewords_the_r1_prompt_and_reports(ladder):
    w = _cup_wording(ladder)
    s = RR.system_for(w)
    assert "立方体" not in s and "箱" not in s and "\"green\"" not in s and "（赤・緑・青）" not in s
    for part in ("机の上に赤のカップがあり、ロボットはカップを 1 個ずつ皿に置きます。", "in_box は皿の上、",
                 "stacked_in_box は皿の上の別のカップの上に重なっている色です。皿にきちんと載っていないので、取り出して置き直せます",
                 "near_box_wall は皿の縁の上にある色です。", "（見失った・皿の縁にあって掴めない・runs_left が 0）",
                 "残りの色を先に置き、", "手の中にカップが残っている", "使える色の名前は \"red\"（赤）だけです。", "色は日本語（赤）で書き",
                 "\"dropped\" はその色のカップを運ぶ途中で手から離れ、カップが皿の外にあること"):
        assert part in s, part
    ctx = RR.make_context("カップを皿に置いて", ["red"], [{"color": "red", "result": "timeout", "attempts": 2}],
                          {"table": ["red"], "in_box": [], "lost": []}, 0, "red", 0, "timeout")
    msg = RR.user_message(ctx)
    assert RR.cache_key(RR.DEFAULT_MODEL, msg, w) != RR.cache_key(RR.DEFAULT_MODEL, msg)
    for a in ("next", "skip_notify", "finish", "stop"):
        rep = RR.template_report(a, ctx, "red", None, w)
        assert "立方体" not in rep and "箱" not in rep and "皿" in rep and not RR.RM.check_report(rep), rep
    # 計画役の口: 言葉を変えた指示文を送り、記録に wording を残す
    cli = T.FakeClient(T.raw("next", "red", [], report="赤のカップを置けませんでした。もう一度置きます。"))
    out = RR.replan_step(text="カップを皿に置いて", plan_order=["red"], steps=ctx["steps"], perception={"table": ["red"]},
                         failed_step=0, failed_color="red", replans_done=0, event="timeout", cli=cli, use_cache=False,
                         wording=w)
    assert cli.calls[0]["system"] == s and out["wording"] == w and out["next_order"] == ["red"]
    # 中へ入れる形のほかのタスクでも置き換わる
    s2 = RR.system_for(RL.make_wording("袋", "かご", ["red", "green"], "all_in"))
    assert "立方体" not in s2 and "箱" not in s2 and "かごにきちんと入っていない" in s2 and "かごの壁ぎわ・縁の上" in s2


def _plate():
    from recovla.api.geometry_place import PlateGeom
    g = PlateGeom(np.array([0.45, 0.25, 0.02]), np.eye(3), 0.09, 0.02, 0.0, 0.0, {"center": 0.001}, 100, 36)
    return AP.PlaceState(np.array([0.45, 0.25]), 0.0, True, None, g)


def test_scene_summary_works_on_a_plate():
    pl = _plate()
    cup = dict(size=0.07, ext=[0.07, 0.07, 0.09])
    base = {"table": [], "in_box": ["red", "green"], "in_hand": [], "lost": []}
    on = AP.WorldEstimate(0.0, {"red": _obj([0.45, 0.25, 0.065], **cup), "green": _obj([0.49, 0.25, 0.065], **cup)},
                          {"plate": pl}, 0.0, "estimated")
    out = SS.refine(base, on, place="plate")
    assert out["in_box"] == ["red", "green"] and out["stacked_in_box"] == []          # 皿の上に並んで置いた
    up = AP.WorldEstimate(0.0, {"red": _obj([0.45, 0.25, 0.065], **cup), "green": _obj([0.452, 0.25, 0.155], **cup)},
                          {"plate": pl}, 0.0, "estimated")
    assert SS.refine(base, up, place="plate")["stacked_in_box"] == ["green"]          # カップの上に重なった
    rim = AP.WorldEstimate(0.0, {"red": _obj([0.45 + 0.09 + 0.03, 0.25, 0.045], **cup)}, {"plate": pl}, 0.0, "estimated")
    far = AP.WorldEstimate(0.0, {"red": _obj([0.45, -0.10, 0.045], **cup)}, {"plate": pl}, 0.0, "estimated")
    tb = {"table": ["red"], "in_box": [], "in_hand": [], "lost": []}
    assert SS.refine(tb, rim, place="plate")["near_box_wall"] == ["red"] and SS.refine(tb, far, place="plate")["near_box_wall"] == []
    # 箱の形は箱の式のまま（BoxGeom は floor・outer を持つ）
    assert SS.floor_z(BOX.region) == pytest.approx(float((BOX.region.center + BOX.region.R @ [0, 0, BOX.region.floor])[2]))


def test_ladder_allows_lf_on_cup_plate_with_the_wording(ladder):
    p = ladder.arm_plan("Lf", task="cup_plate")
    assert "executor_improved" in p["executor"] and p["task_file"] == "configs/tasks/cup_plate.yaml" and p["place_finder"]
    src = (ROOT / "scripts" / "api" / "ladder.py").read_text(encoding="utf-8")
    assert "RR.make_replan_step(REPLAN_MODEL, wording=replan_wording(spec, default))" in src
    assert "RL.make_replan_step(REPLAN_MODEL, wording=replan_wording(spec, default))" in src     # L2v2 にも同じ配線
    w = _cup_wording(ladder)
    assert RR.make_replan_step(RR.DEFAULT_MODEL, wording=w).lost_notice("red") == "赤のカップが見つからなかったので、皿に置いていません。"


def test_lf_cup_plate_dry_run_passes_the_entry():
    import os
    import subprocess
    import sys
    out = ROOT / "outputs" / "v2eval" / "APIDRY" / "pytest_api_dry_cup_Lf"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "api" / "ladder.py"), "run", "--experiment", "APIDRY",
                        "--condition", "pytest_api_dry_cup_Lf", "--trials", "193900:2", "--exec-interval", "6", "--dry-run",
                        "--arm", "Lf", "--task", "cup_plate", "--vla-text", "task", "--model", "Rcp",
                        "--model-ckpt", "Rcp=outputs/real/cup_plate/train/x/pretrained_model"],
                       capture_output=True, text=True, cwd=ROOT, timeout=600, encoding="utf-8", errors="replace",
                       env=dict(os.environ, PYTHONUTF8="1"))
    text = r.stdout + r.stderr
    assert r.returncode in (0, 3), text
    assert "[api] 腕 Lf" in text and "executor_improved" in text and "タスク cup_plate" in text and "[dry-run]" in text, text
    assert not out.exists()
