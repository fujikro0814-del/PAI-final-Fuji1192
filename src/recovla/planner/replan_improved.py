"""LLM による手順の組み直し（指示文と場面の要約を直した版。手順書 docs/api/r1_protocol.md 5 節）。
呼び方は replan_event（Sonnet 5.5、event つき）と同じで、指示文・入力の欄・手の種類だけが違う。replan_event は組み直しの前の版（L2v2）。

    from recovla.planner import replan_improved as RR
    step = RR.make_replan_step("claude-sonnet-5-5")       # 実行器（api/executor_improved.install）に渡す関数

replan_event との違い:
1. 2 つの回数を分けて書く（前の版では、実行器のやり直しの回数を色ごとの上限と読み違えて、まだ組めるのに止めることがあった）。
   - steps[].tries_in_this_run・tries_limit_per_run: その色の手順を 1 回実行する中での、実行器のやり直し
   - runs_per_color: {色: {runs_used, runs_limit, runs_left}}: その色の手順を組める回数（max_runs_per_color）
   SYSTEM にも、tries は上限ではないこと、runs_left が 1 以上なら同じ色をもう一度組めることを書く。
2. 場面の要約の欄を足す（api/scene_summary）: stacked_in_box（箱の中の立方体の上に重なった。箱の中に数えない）・
   near_box_wall（箱の壁ぎわ・縁の上）・not_in_hand_after_hold（持っていたはずが手の中にない）。
   stacked_in_box の色は、机の上の色と同じく候補になる（取り出して入れ直せる）。
3. 手 skip_notify: 取り返せない色（見失った・壁ぎわで掴めない・回数を使い切った）を後に回し、片付けられる残りを先に入れて、
   最後に人に知らせる。color は失敗した色。実行器は残りを続け、終わりに ended の kind "notify" で知らせる。
タスクの言葉（wording、replan_event.make_wording と同じ形）: 既定（None）は立方体と箱の文のまま。渡すと、この指示文だけの言葉
  （_reword_pairs）を置き換えた後に replan_event の置き換えを掛ける。
キャッシュのキー: KIND "replan_improved"（replan_event と混ざらない）。弾く規則・報告の検査は replan_base と同じ
（skip_notify は skip と同じく失敗した色だけ）。
シミュレーション・評価・真値のモジュールは import しない（入力は実行器の記録と場面の要約だけ）。
"""
import hashlib
import json
import pathlib
import time

from recovla.common.seeds import COLORS
from recovla.planner import replan_event as RL
from recovla.planner import replan_multi as RM
from recovla.planner import replan_base as R4

KIND = "replan_r1"
PROMPT_VERSION = 1
DEFAULT_MODEL = RM.MODELS["sonnet"]
EVENTS = RL.EVENTS
ACTIONS = ("next", "reorder", "skip", "skip_notify", "finish", "stop")
PLAN_CHANGE_ACTIONS = ("next", "reorder", "skip", "skip_notify", "finish")
PER_KEYS = ("table", "in_box", "stacked_in_box", "in_hand", "lost", "near_box_wall", "not_in_hand_after_hold")
TRIES_LIMIT = 2                 # 1 回の実行の中の試みの上限の既定（実行器の retry + 1。実行器が渡せばその値）
JA = R4.JA

SYSTEM = """あなたはロボットの作業の立て直しの係です。机の上に赤・緑・青の立方体があり、ロボットは立方体を 1 個ずつ箱へ入れます。
人の日本語の指示から作った手順（色の列）の途中で、ある色の手順がうまくいきませんでした。入力の failed.event が場面です:
"dropped" はその色の立方体を運ぶ途中で手から離れ、立方体が箱の外にあること、"timeout" はその色の手順が決められた時間の中で終わらなかったことです。
入力の記録とカメラで見た様子の要約を読み、次の手を 1 つ選び、人に向けた短い状況報告を書いてください。

回数は 2 種類あります。混ぜないでください:
- tries（steps[].tries_in_this_run と tries_limit_per_run）: その色の手順を 1 回実行する中で、ロボットが待機位置に戻ってやり直した数です。
  1 回の実行は tries_limit_per_run 回試すと終わります。これは色ごとの上限ではありません
- runs（runs_per_color の runs_used・runs_limit・runs_left。まだ終わっていない色だけ）: その色の手順を組んだ数と、組める残りの数です。
  runs_left が 1 以上の色は、candidates にあれば、もう 1 度組めます（next で選ぶ）。runs_left が 0 の色はもう組めません

使える色の名前は "red"（赤）・"green"（緑）・"blue"（青）だけです。色を使わない手では color を "none"、order を空の列にします。

次の手（action）:
- "next": 次に入れる色を 1 つ color に書く。失敗した色をもう 1 度試してもよい（runs_left が 1 以上で candidates にあるとき）
- "reorder": 残りの色を入れる順を order に書く（例: 失敗した色を最後に回す）。order は残りの候補の色を全部、重複なく並べる
- "skip": 失敗した色をあきらめ、残りの色を続ける。color に失敗した色を書く
- "skip_notify": 失敗した色を取り返せないとき（見失った・箱の壁ぎわや縁にあって掴めない・runs_left が 0）、その色を後に回して、
  片付けられる残りの色を先に入れ、最後に人に知らせる。color に失敗した色を書く
- "finish": これ以上は入れずに終える。箱にある色を報告する
- "stop": すぐに止まって人に知らせる

様子の要約（perception）:
- table は机の上、in_box は箱の中、in_hand は手の中、lost は見失った色です
- stacked_in_box は箱の中の別の立方体の上に重なっている色です。箱にきちんと入っていないので、取り出して入れ直せます（candidates に入ります）
- near_box_wall は箱の壁ぎわ・縁の上にある色です。掴みにくいことがあります
- not_in_hand_after_hold は、持っていたはずなのに手の中にない色です

決まり:
- 色を選べるのは、入力の candidates（残りの候補）にある色だけです。机の上にない色、すでに箱の中にある色は選びません
- 失敗した色が取り返せなくても、ほかに candidates の色が残っていれば、"stop" ではなく "skip_notify" で残りを先に片付けます
- "stop" は、手の中に立方体が残っているなど、続けると危ないときだけ選びます
- reason は選んだ理由を日本語で短く 1 文で書きます
- report は利用者が読んで分かる日本語で 2 文以内にします。何が終わらなかったか、箱に何が入っているか、次に何をするかを書きます。
  色は日本語（赤・緑・青）で書き、英語の色の名前・記号・番号の略号は使いません
"""
SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": list(ACTIONS)},
        "color": {"type": "string", "enum": list(COLORS) + ["none"]},
        "order": {"type": "array", "items": {"type": "string", "enum": list(COLORS)}},
        "reason": {"type": "string"},
        "report": {"type": "string"},
    },
    "required": ["action", "color", "order", "reason", "report"],
    "additionalProperties": False,
}


def _reword_pairs(w: dict) -> list:
    """この指示文だけにある立方体・箱の言葉の置き換え（replan_event の置き換えの前に掛ける）。w は replan_event.make_wording の形。"""
    o, p = w["obj"], w["place"]
    on = w["goal_kind"] == "all_on"
    put_te, inside, not_in = ("置き", "の上", "きちんと載っていない") if on else ("入れ", "の中", "きちんと入っていない")
    edge, edge_on = ("縁", "縁の上") if on else ("壁ぎわや縁", "壁ぎわ・縁の上")
    names = "・".join(f"\"{c}\"（{RL.JA_COLOR.get(c, c)}）" for c in w["colors"])
    ja = "・".join(RL.JA_COLOR.get(c, c) for c in w["colors"])
    return [
        ("使える色の名前は \"red\"（赤）・\"green\"（緑）・\"blue\"（青）だけです。", f"使える色の名前は {names}だけです。"),
        ("色は日本語（赤・緑・青）で書き", f"色は日本語（{ja}）で書き"),
        ("（見失った・箱の壁ぎわや縁にあって掴めない・runs_left が 0）", f"（見失った・{p}の{edge}にあって掴めない・runs_left が 0）"),
        ("残りの色を先に入れ、", f"残りの色を先に{put_te}、"),
        ("table は机の上、in_box は箱の中、", f"table は机の上、in_box は{p}{inside}、"),
        ("stacked_in_box は箱の中の別の立方体の上に重なっている色です。箱にきちんと入っていないので、取り出して入れ直せます",
         f"stacked_in_box は{p}{inside}の別の{o}の上に重なっている色です。{p}に{not_in}ので、取り出して{put_te}直せます"),
        ("near_box_wall は箱の壁ぎわ・縁の上にある色です。", f"near_box_wall は{p}の{edge_on}にある色です。"),
        ("手の中に立方体が残っている", f"手の中に{o}が残っている"),
        ("を先に入れて、", f"を先に{put_te}て、"),
    ]


def reword(text: str, wording=None) -> str:
    """タスクの言葉で置き換える。既定（None）は元の文のまま。"""
    if wording is None:
        return text
    for old, new in _reword_pairs(wording):
        text = text.replace(old, new)
    return RL.reword(text, wording)


def system_for(wording=None) -> str:
    return reword(SYSTEM, wording)


def variant(model: str) -> str:
    short = next((k for k, v in RM.MODELS.items() if v == model), model)
    return f"s4_replan_r1_{short}55"


def _colors(cs) -> list:
    return sorted(dict.fromkeys(c for c in cs if c in COLORS), key=COLORS.index)


def make_context(text, plan_order, steps, perception, failed_step, failed_color, replans_done, event: str = "timeout") -> dict:
    """steps は [{"color", "result", "attempts"(, "tries_limit")}]（実行器の step_summary）。perception は PER_KEYS の色の列
    （無い欄は空）。中では replan_base と同じキー（attempts）で持ち、LLM に送る文（user_message）で名前を分ける。"""
    if event not in EVENTS:
        raise ValueError(f"event {event!r} は {EVENTS} のどれか")
    st = [{"color": str(s["color"]), "result": str(s["result"]), "attempts": int(s["attempts"]),
           "tries_limit": int(s.get("tries_limit", TRIES_LIMIT))} for s in steps]
    per = {k: _colors(perception.get(k, ())) for k in PER_KEYS}
    return {"instruction": str(text), "plan": [str(c) for c in plan_order], "steps": st, "perception": per,
            "failed": {"step": int(failed_step), "color": str(failed_color), "event": event}, "replans_done": int(replans_done)}


def runs_of(ctx: dict) -> dict:
    return R4.runs_of(ctx)


def candidates(ctx: dict) -> list:
    """残りの候補: replan_base.candidates と同じ決まりに、箱の中で重なった色（stacked_in_box）を机の上と同じに足す。"""
    done = {s["color"] for s in ctx["steps"] if s["result"] in ("success", "skipped")}
    per, runs = ctx["perception"], runs_of(ctx)
    seen = set(per["table"]) | set(per["stacked_in_box"])
    return [c for c in dict.fromkeys(ctx["plan"])
            if c in COLORS and c not in done and c in seen and c not in per["in_box"] and c not in per["in_hand"]
            and c not in per["lost"] and runs[c] < R4.MAX_RUNS_PER_COLOR]


def remaining_planned(ctx: dict) -> list:
    return R4.remaining_planned(ctx)


def user_message(ctx: dict) -> str:
    """LLM に送る入力。回数の名前を分ける（tries と runs）。"""
    runs = runs_of(ctx)
    lim = R4.MAX_RUNS_PER_COLOR
    body = {"instruction": ctx["instruction"], "plan": ctx["plan"],
            "steps": [{"color": s["color"], "result": s["result"], "tries_in_this_run": s["attempts"],
                       "tries_limit_per_run": s["tries_limit"]} for s in ctx["steps"]],
            "runs_per_color": {c: {"runs_used": runs[c], "runs_limit": lim, "runs_left": max(0, lim - runs[c])}
                               for c in remaining_planned(ctx) if c in COLORS},
            "perception": ctx["perception"], "failed": ctx["failed"], "replans_done": ctx["replans_done"],
            "replans_limit": R4.MAX_REPLANS, "candidates": candidates(ctx), "actions": list(ACTIONS)}
    return json.dumps(body, ensure_ascii=False, sort_keys=True)


def check(d: dict, ctx: dict) -> list:
    """replan_base.check と同じ決まり（候補は candidates）。skip_notify は skip と同じく失敗した色だけ。"""
    a = d["action"]
    if a not in ACTIONS:
        return [f"unknown_action:{a}"]
    if a == "skip_notify":
        return [] if d["color"] == ctx["failed"]["color"] else [f"skip_notify:not_failed_color:{d['color']}:{ctx['failed']['color']}"]
    cand, per, runs = candidates(ctx), ctx["perception"], runs_of(ctx)
    why = []

    def color_reasons(c, tag):
        if c not in COLORS:
            return [f"{tag}:unknown_color:{c}"]
        if c in per["in_box"]:
            return [f"{tag}:in_box:{c}"]
        if c not in per["table"] and c not in per["stacked_in_box"]:
            return [f"{tag}:not_on_table:{c}"]
        if runs[c] >= R4.MAX_RUNS_PER_COLOR:
            return [f"{tag}:runs_limit:{c}:{runs[c]}"]
        if c not in cand:
            return [f"{tag}:not_candidate:{c}"]
        return []

    if a == "next":
        why += color_reasons(d["color"], "next")
    elif a == "reorder":
        order = list(d["order"])
        if not order:
            why.append("reorder:empty")
        if len(order) != len(set(order)):
            why.append("reorder:duplicates")
        for c in dict.fromkeys(order):
            why += color_reasons(c, "reorder")
        if not why and sorted(order, key=COLORS.index) != sorted(cand, key=COLORS.index):
            why.append(f"reorder:not_permutation_of_candidates:{order}:{cand}")
    elif a == "skip" and d["color"] != ctx["failed"]["color"]:
        why.append(f"skip:not_failed_color:{d['color']}:{ctx['failed']['color']}")
    return why


def apply(decision: dict, ctx: dict) -> list:
    """通った手から、これから実行する色の並び（空なら終える）。"""
    a = decision["action"]
    rest = [c for c in remaining_planned(ctx) if c in candidates(ctx)]
    if a == "next":
        c = decision["color"]
        return [c] + [x for x in rest if x != c]
    if a == "reorder":
        return list(decision["order"])
    if a in ("skip", "skip_notify"):
        return [x for x in rest if x != decision["color"]]
    return []


def _ctx_r4(ctx: dict) -> dict:
    """replan_event.template_report に渡す形（知覚は 4 つの欄だけ）。"""
    return dict(ctx, perception={k: ctx["perception"][k] for k in R4_KEYS})


R4_KEYS = ("table", "in_box", "in_hand", "lost")


def template_report(action: str, ctx: dict, color: str = None, order=None, wording=None) -> str:
    if action != "skip_notify":
        return RL.template_report(action, _ctx_r4(ctx), color, order, wording=wording)
    failed = ctx["failed"]["color"]
    head = RL.template_report("stop", _ctx_r4(ctx), wording=wording).split("。")[0] + "。"
    rest = [c for c in apply({"action": "skip_notify", "color": failed}, ctx)]
    if rest:
        return head + reword(f"{JA[failed]}は後に回し、{R4._ja(rest)}を先に入れて、最後にお知らせします。", wording)
    return f"{head}{JA[failed]}は取り出せないので、ここで終えてお知らせします。"


def build_params(model: str, msg: str, wording=None) -> dict:
    s = RM.settings(model)
    oc = dict(s.get("output_config") or {})
    oc["format"] = {"type": "json_schema", "schema": SCHEMA}
    p = {"model": model, "max_tokens": int(s["max_tokens"]), "system": system_for(wording),
         "messages": [{"role": "user", "content": msg}], "output_config": oc}
    if s.get("thinking") is not None:
        p["thinking"] = dict(s["thinking"])
    return p


def cache_key(model: str, msg: str, wording=None) -> str:
    src = json.dumps({"kind": KIND, "model": model, "settings": RM.settings(model), "temperature": None, "system": system_for(wording),
                      "schema": SCHEMA, "user": msg, "prompt_version": PROMPT_VERSION}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(src.encode("utf-8")).hexdigest()


def parse(fields: dict):
    """(data, 理由)。形は replan_base.parse と同じ（手の enum だけ違う。手の名前は check で見る）。"""
    if fields.get("stop_reason") == "refusal":
        return None, "refusal"
    return R4.parse(fields.get("raw"))


def decide(ctx: dict, data: dict, model: str, llm: dict, wording=None) -> dict:
    base = {"replanner_variant": variant(model), "kind": KIND, "candidates": candidates(ctx), "llm": llm}
    stop = {"action": R4.SAFE_ACTION, "color": "none", "order": []}
    rejected = check(data, ctx)
    if rejected:
        return dict(base, decision=stop, reason="", report=template_report(R4.SAFE_ACTION, ctx, wording=wording),
                    accepted=False, rejected=rejected, fallback=True, report_source="template", report_rejected=[], llm_output=data)
    a = data["action"]
    color = data["color"] if a in ("next", "skip", "skip_notify") else "none"
    order = list(data["order"]) if a == "reorder" else []
    rep_why = RM.check_report(data["report"])
    report = data["report"].strip() if not rep_why else template_report(a, ctx, color, order, wording)
    return dict(base, decision={"action": a, "color": color, "order": order}, reason=data["reason"].strip()[:200],
                report=report, accepted=True, rejected=[], fallback=False,
                report_source="llm" if not rep_why else "template", report_rejected=rep_why, llm_output=data)


def replan(ctx: dict, model: str = DEFAULT_MODEL, cli=None, use_cache: bool = True, cache_dir=None, wording=None) -> dict:
    """例外は外に出さない（駄目なら stop にする）。流れは replan_event.replan と同じ。"""
    RM.settings(model)
    stop = {"action": R4.SAFE_ACTION, "color": "none", "order": []}
    if ctx["replans_done"] >= R4.MAX_REPLANS:
        return {"decision": stop, "reason": "", "report": template_report(R4.SAFE_ACTION, ctx, wording=wording), "accepted": False,
                "rejected": [f"replan_limit:{ctx['replans_done']}"], "fallback": True, "report_source": "template",
                "report_rejected": [], "candidates": candidates(ctx), "llm": {"called": False, "calls": 0},
                "replanner_variant": variant(model), "kind": KIND}
    msg = user_message(ctx)
    key = cache_key(model, msg, wording)
    path = pathlib.Path(cache_dir or R4.CACHE) / f"{key}.json"
    llm = {"called": False, "calls": 0, "errors": [], "cache": key, "from_cache": False, "usage": [],
           "requested_model": model, "temperature": None, "settings": RM.settings(model)}
    data = None
    if use_cache and path.is_file():
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
            data, why = parse(rec)
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
            data, why = parse(out)
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
        base = {"replanner_variant": variant(model), "kind": KIND, "candidates": candidates(ctx), "llm": llm}
        return dict(base, decision=stop, reason="", report=template_report(R4.SAFE_ACTION, ctx, wording=wording),
                    accepted=False, rejected=["llm_failed"] + llm["errors"][-1:], fallback=True, report_source="template",
                    report_rejected=[], llm_output=None)
    return decide(ctx, data, model, llm, wording)


def finish_step(ctx: dict, out: dict, wording=None) -> dict:
    out["context"] = ctx
    out["next_order"] = apply(out["decision"], ctx)
    if out["decision"]["action"] in ("next", "reorder") and not out["next_order"]:
        out.update(decision={"action": R4.SAFE_ACTION, "color": "none", "order": []}, accepted=False,
                   rejected=list(out.get("rejected") or []) + ["empty_order"], fallback=True,
                   report=template_report(R4.SAFE_ACTION, ctx, wording=wording), report_source="template")
    out["plan_change"] = out["decision"]["action"] in PLAN_CHANGE_ACTIONS
    if wording is not None:
        out["wording"] = dict(wording)
    return out


def replan_step(text, plan_order, steps, perception, failed_step, failed_color, replans_done, event: str = "timeout",
                cli=None, use_cache: bool = True, model: str = DEFAULT_MODEL, cache_dir=None, wording=None) -> dict:
    ctx = make_context(text, plan_order, steps, perception, failed_step, failed_color, replans_done, event)
    return finish_step(ctx, replan(ctx, model=model, cli=cli, use_cache=use_cache, cache_dir=cache_dir, wording=wording),
                       wording)


def lost_notice(color: str, wording=None) -> str:
    """見失って手順から外した色の知らせ（api/executor_improved が終わりに出す）。wording は replan_event.make_wording の形。
    既定（None）は立方体と箱の文。"""
    if wording is None:
        o, p, put = "立方体", "箱", "入れて"
    else:
        o, p, put = wording["obj"], wording["place"], "置いて" if wording["goal_kind"] == "all_on" else "入れて"
    return f"{JA.get(color, color)}の{o}が見つからなかったので、{p}に{put}いません。"


def make_replan_step(model: str = DEFAULT_MODEL, cache_dir=None, wording=None):
    """wording はタスクの言葉（replan_event.make_wording。指示文・定型の報告・知らせの文。既定 None は立方体と箱のまま）。"""
    RM.settings(model)

    def step(**kw):
        return replan_step(**kw, model=model, cache_dir=cache_dir, wording=wording)
    step.model = model
    step.takes_event = True
    step.kind = KIND
    step.wording = None if wording is None else dict(wording)
    step.lost_notice = lambda c: lost_notice(c, wording)
    return step
