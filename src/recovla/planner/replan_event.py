"""LLM による手順の組み直し（失敗の検知の種類 event つき）。replan_base・replan_multi に「何が起きたか」を入力として足す。

    from recovla.planner import replan_event as RL
    step = RL.make_replan_step("claude-sonnet-5-5")       # 実行器（runtime/executor_replan_early.install）に渡す関数
    out = step(text=..., plan_order=..., steps=..., perception=..., failed_step=..., failed_color=..., replans_done=...,
               event="dropped")                          # event: "dropped"（運ぶ途中で手から離れた）か "timeout"（時間切れ）

replan_base との違い（次の手の種類・規則の検査・候補・同じ色の上限・組み直しの上限 MAX_REPLANS・報告の検査は同じ）:
- SYSTEM: 組み直しを頼む場面に「運ぶ途中で手から離れ、箱の外にある」を足し、入力の failed.event で場面を伝える
- 入力: replan_base.make_context の辞書の failed に event を足す
- 定型の報告（LLM の報告を弾いたとき・安全な手にしたとき）: event が dropped なら「運ぶ途中で落としました」で始める
- キャッシュのキー: KIND "replan_event"（replan_base・replan_multi と混ざらない）
タスクの言葉（wording。docs/api/cup_plate_protocol.md）: 既定（None）は立方体と箱の文のまま。make_wording(物の呼び方,
  場所の呼び方, 色, 目標の種類) を渡すと、SYSTEM と定型の報告の「立方体・箱・入れる」を、タスクの言葉に文字列で置き換える。
  入力の perception.in_box は「場所の上・中に見えた色」のまま渡し、そのことを SYSTEM に書き足す。
モデルごとの引数は replan_multi.settings（温度は送らない）。
シミュレーション・評価・真値のモジュールは import しない（入力は実行器の記録と知覚の色の状態だけ）。
"""
import hashlib
import json
import pathlib
import time

from recovla.planner import replan_multi as RM
from recovla.planner import replan_base as R4

KIND = "replan_l2"
PROMPT_VERSION = 1
EVENTS = ("dropped", "timeout")
DEFAULT_MODEL = RM.MODELS["sonnet"]
SYSTEM = R4.SYSTEM.replace(
    "人の日本語の指示から作った手順（色の列）の途中で、ある色の手順が決められた時間と回数の中で終わりませんでした。",
    "人の日本語の指示から作った手順（色の列）の途中で、ある色の手順がうまくいきませんでした。入力の failed.event が場面です:\n"
    "\"dropped\" はその色の立方体を運ぶ途中で手から離れ、立方体が箱の外にあること、"
    "\"timeout\" はその色の手順が決められた時間と回数の中で終わらなかったことです。")
assert SYSTEM != R4.SYSTEM
JA_COLOR = {"red": "赤", "green": "緑", "blue": "青"}


def make_wording(obj_word: str, place_word: str, colors, goal_kind: str) -> dict:
    """タスクの言葉。goal_kind は "all_in"（中へ入れる）か "all_on"（上に置く）。colors は物の色（"red" など、物の並びの順）。"""
    if goal_kind not in ("all_in", "all_on"):
        raise ValueError(f"goal_kind {goal_kind!r} は all_in か all_on")
    cs = [str(c) for c in colors]
    return {"obj": str(obj_word), "place": str(place_word), "colors": cs, "goal_kind": goal_kind,
            "objs": "・".join(JA_COLOR.get(c, c) for c in cs) + "の" + str(obj_word)}


def _reword_pairs(w: dict) -> list:
    """置き換えの順（長いものから）。all_in は「入れる」、all_on は「置く」。"""
    o, p = w["obj"], w["place"]
    on = w["goal_kind"] == "all_on"
    put, put_te, putable, put_neg = ("置く", "置き", "置け", "置かず") if on else ("入れる", "入れ", "入れられ", "入れず")
    inside, on_word, have = ("の上", "に", "載って") if on else ("の中", "へ", "入って")
    return [
        ("机の上に赤・緑・青の立方体があり、ロボットは立方体を 1 個ずつ箱へ入れます。",
         f"机の上に{w['objs']}があり、ロボットは{o}を 1 個ずつ{p}{on_word}{put_te}ます。"
         f"入力の perception.in_box は{p}{inside}に見えた色です。"),
        ("立方体を運ぶ途中で手から離れ、立方体が箱の外にあること", f"{o}を運ぶ途中で手から離れ、{o}が{p}の外にあること"),
        ("次に入れる色", f"次に{put}色"), ("残りの色を入れる順", f"残りの色を{put}順"),
        ("これ以上は入れずに終える。箱にある色を報告する", f"これ以上は{put_neg}に終える。{p}{inside}にある色を報告する"),
        ("すでに箱の中にある色", f"すでに{p}{inside}にある色"), ("立方体を見失った", f"{o}を見失った"),
        ("箱に何が入っているか", f"{p}{inside}に何が{have}いるか"),
        ("の立方体を決められた時間の中で箱に入れられませんでした", f"の{o}を決められた時間の中で{p}{on_word}{putable}ませんでした"),
        ("の立方体を運ぶ途中で落としました", f"の{o}を運ぶ途中で落としました"),
        ("箱には", f"{p}{inside}には"), ("が入っており", f"が{have}おり"), ("まだ何も入っておらず", f"まだ何も{have}おらず"),
        ("を入れます", f"を{put_te}ます"), ("の順に入れ直します", f"の順に{put_te}直します"),
    ]


def reword(text: str, wording=None) -> str:
    if wording is None:
        return text
    for old, new in _reword_pairs(wording):
        text = text.replace(old, new)
    return text


def system_for(wording=None) -> str:
    return reword(SYSTEM, wording)


def variant(model: str) -> str:
    short = next((k for k, v in RM.MODELS.items() if v == model), model)
    return f"s4_replan_l2_{short}55"


def make_context(text, plan_order, steps, perception, failed_step, failed_color, replans_done, event: str) -> dict:
    if event not in EVENTS:
        raise ValueError(f"event {event!r} は {EVENTS} のどれか")
    ctx = R4.make_context(text, plan_order, steps, perception, failed_step, failed_color, replans_done)
    ctx["failed"]["event"] = event
    return ctx


def template_report(action: str, ctx: dict, color: str = None, order=None, wording=None) -> str:
    rep = R4.template_report(action, ctx, color, order)
    if ctx["failed"].get("event") == "dropped":
        failed = ctx["failed"]["color"]
        old = f"{R4.JA.get(failed, failed)}の立方体を決められた時間の中で箱に入れられませんでした。"
        rep = rep.replace(old, f"{R4.JA.get(failed, failed)}の立方体を運ぶ途中で落としました。")
    return reword(rep, wording)


def build_params(model: str, msg: str, wording=None) -> dict:
    p = RM.build_params(model, msg)
    p["system"] = system_for(wording)
    return p


def cache_key(model: str, msg: str, wording=None) -> str:
    src = json.dumps({"kind": KIND, "model": model, "settings": RM.settings(model), "temperature": None, "system": system_for(wording),
                      "schema": R4.SCHEMA, "user": msg, "prompt_version": PROMPT_VERSION}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(src.encode("utf-8")).hexdigest()


def _out(base, decision, **kw):
    return dict(base, decision=decision, **kw)


def decide(ctx: dict, data: dict, model: str, llm: dict, wording=None) -> dict:
    """replan_multi.decide と同じ規則（定型の報告だけ template_report に替える）。"""
    base = {"replanner_variant": variant(model), "kind": KIND, "candidates": R4.candidates(ctx), "llm": llm}
    stop = {"action": R4.SAFE_ACTION, "color": "none", "order": []}
    rejected = R4.check(data, ctx)
    if rejected:
        return _out(base, stop, reason="", report=template_report(R4.SAFE_ACTION, ctx, wording=wording), accepted=False,
                    rejected=rejected, fallback=True, report_source="template", report_rejected=[], llm_output=data)
    a = data["action"]
    color = data["color"] if a in ("next", "skip") else "none"
    order = list(data["order"]) if a == "reorder" else []
    rep_why = RM.check_report(data["report"])
    report = data["report"].strip() if not rep_why else template_report(a, ctx, color, order, wording=wording)
    return _out(base, {"action": a, "color": color, "order": order}, reason=data["reason"].strip()[:200], report=report,
                accepted=True, rejected=[], fallback=False, report_source="llm" if not rep_why else "template",
                report_rejected=rep_why, llm_output=data)


def replan(ctx: dict, model: str = DEFAULT_MODEL, cli=None, use_cache: bool = True, cache_dir=None, wording=None) -> dict:
    """例外は外に出さない（駄目なら stop にする）。流れは replan_multi.replan と同じ。"""
    RM.settings(model)
    stop = {"action": R4.SAFE_ACTION, "color": "none", "order": []}
    if ctx["replans_done"] >= R4.MAX_REPLANS:
        return {"decision": stop, "reason": "", "report": template_report(R4.SAFE_ACTION, ctx, wording=wording), "accepted": False,
                "rejected": [f"replan_limit:{ctx['replans_done']}"], "fallback": True, "report_source": "template",
                "report_rejected": [], "candidates": R4.candidates(ctx), "llm": {"called": False, "calls": 0},
                "replanner_variant": variant(model), "kind": KIND}
    msg = R4.user_message(ctx)
    key = cache_key(model, msg, wording)
    path = pathlib.Path(cache_dir or R4.CACHE) / f"{key}.json"
    llm = {"called": False, "calls": 0, "errors": [], "cache": key, "from_cache": False, "usage": [],
           "requested_model": model, "temperature": None, "settings": RM.settings(model)}
    data = None
    if use_cache and path.is_file():
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
            data, why = RM.parse_message(rec)
        except (ValueError, KeyError, TypeError, OSError) as e:
            rec, data, why = None, None, f"{type(e).__name__}"
        if data is None:
            llm["errors"].append(f"cache_unreadable:{why}")
        else:
            llm.update(from_cache=True, model=rec.get("model"), stop_reason=rec.get("stop_reason"))
    if data is None:
        for k in range(1 + R4.API_RETRIES):
            llm["called"] = True
            try:
                cli = cli or R4.client()
                llm["calls"] += 1
                t0 = time.perf_counter()
                out = dict(RM.message_fields(cli.messages.create(**build_params(model, msg, wording))),
                           wall_s=round(time.perf_counter() - t0, 3))
            except Exception as e:                         # noqa: BLE001  API の例外・鍵がない、など
                llm["errors"].append(f"api:{type(e).__name__}: {e}"[:300])
                continue
            llm["usage"].append(out["usage"])
            llm.update(model=out["model"], stop_reason=out["stop_reason"], request_id=out["request_id"])
            data, why = RM.parse_message(out)
            if data is None:
                llm["errors"].append(f"parse:{why}")
                continue
            rec = RM.cache_record(key, model, msg, out, k, "sync", None,
                                  extra={"kind": KIND, "replanner_variant": variant(model),
                                         "request": {"system_sha256": hashlib.sha256(system_for(wording).encode()).hexdigest(), "user": msg,
                                                     "prompt_version": PROMPT_VERSION, "requested_model": model,
                                                     "temperature": None, "settings": RM.settings(model)}})
            if use_cache:
                RM.write_cache(path, rec)
            break
    llm["retried"] = llm["calls"] > 1
    if data is None:
        base = {"replanner_variant": variant(model), "kind": KIND, "candidates": R4.candidates(ctx), "llm": llm}
        return _out(base, stop, reason="", report=template_report(R4.SAFE_ACTION, ctx, wording=wording), accepted=False,
                    rejected=["llm_failed"] + llm["errors"][-1:], fallback=True, report_source="template",
                    report_rejected=[], llm_output=None)
    return decide(ctx, data, model, llm, wording)


def finish_step(ctx: dict, out: dict, wording=None) -> dict:
    out = RM.finish_step(ctx, out)
    if out.get("fallback") and "empty_order" in (out.get("rejected") or []):
        out["report"] = template_report(R4.SAFE_ACTION, ctx, wording=wording)
    if wording is not None:
        out["wording"] = dict(wording)
    return out


def replan_step(text, plan_order, steps, perception, failed_step, failed_color, replans_done, event: str = "timeout",
                cli=None, use_cache: bool = True, model: str = DEFAULT_MODEL, cache_dir=None, wording=None) -> dict:
    ctx = make_context(text, plan_order, steps, perception, failed_step, failed_color, replans_done, event)
    return finish_step(ctx, replan(ctx, model=model, cli=cli, use_cache=use_cache, cache_dir=cache_dir, wording=wording),
                       wording)


def make_replan_step(model: str = DEFAULT_MODEL, wording=None):
    RM.settings(model)

    def step(**kw):
        return replan_step(**kw, model=model, wording=wording)
    step.model = model
    step.takes_event = True
    step.wording = wording
    return step
