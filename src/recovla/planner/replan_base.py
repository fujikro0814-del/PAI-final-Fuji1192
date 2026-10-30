"""LLM による手順の組み直し（Claude Haiku 5.5）。手順が失敗して実行器が止まる所で、次の手を LLM に選ばせ、日本語で状況を報告させる。

    from recovla.planner import replan_base as R4
    out = R4.replan(ctx)              # ctx は R4.make_context(...) が作る辞書（実行系が実機で得られる情報だけ）
    # {"decision": {"action", "color", "order"}, "reason", "report", "accepted", "rejected": [...], "fallback": 真偽,
    #  "report_source": "llm"|"template", "report_rejected": [...], "llm": {...呼び出しの記録...}, "replanner_variant": ...}

入力（make_context）: 指示文、計画（色の列。今の並び）、各手順の結果（成功・時間切れ・失敗・飛ばした、試みの回数）、
  場面の要約（机の上の色・箱の中の色・手の中の色・見失った色。runtime の Perception の WorldModel から作る。
  シミュレーションの真値は使わない）、失敗した手順（何番目・色）、これまでの組み直しの回数。
出力（構造化出力の SCHEMA）: action（next・reorder・skip・finish・stop のどれか）、color、order、reason（短い文）、
  report（利用者向けの日本語の状況報告、2 文以内）。
手の意味:
  next     次の色を 1 つ選ぶ（失敗した色をもう 1 度でもよい）。残りの並びは、選んだ色を先頭に出した順
  reorder  残りの色の並びを付け直す（例: 失敗した色を後に回す）
  skip     失敗した色を飛ばし、残りの並びで続ける
  finish   箱にある色を報告して終える
  stop     止まって人に知らせる（検査に通らなかったときの安全な手もこれ）
弾く規則（check。1 つでも落ちたら stop にし、理由を rejected に残す）:
  - 応答が読めない・手が 5 つ以外
  - next・reorder・skip の色が、残りの候補（計画に残っている色のうち、机の上に見えていて、箱の中になく、成功していない、
    同じ色の手順が MAX_RUNS_PER_COLOR 回に達していないもの）でない。机の上にない色・箱の中の色・同じ色の 3 回目以上はここで弾く
  - next・reorder の color・order に重複がある、reorder の order が残りの候補の並べ替えになっていない
  - skip の color が失敗した色でない
  - 組み直しの回数が MAX_REPLANS を超える（この場合は LLM を呼ばずに stop）
日本語の報告の検査（check_report）: 空でない、2 文以内、REPORT_MAX_CHARS 字以内、仮名か漢字を含む、FORBIDDEN の語
  （内部の用語・記号）・英語の色の名前を含まない。落ちたら手はそのまま使い、報告だけ template_report（記録と知覚だけから
  決まる文）に替え、理由を report_rejected に残す。
API の呼び出しは decompose_haiku.py と同じ: 温度は送らない、thinking は disabled、output_config の json_schema。
  キャッシュのキーは model・温度（None）・thinking・SYSTEM・SCHEMA・入力・PROMPT_VERSION・KIND。読めた応答だけを
  outputs/llm_cache/<キー>.json に残す。API の例外・読めない応答（拒否を含む）は 1 回だけ呼び直し、それでも駄目なら stop にする。
"""
import hashlib
import json
import re
import time

from recovla.common.seeds import COLORS
from recovla.planner import decompose as _legacy
from recovla.planner.decompose import CACHE, JA, client  # noqa: F401

DEFAULT_MODEL = "claude-haiku-5-5"
REPLANNER_VARIANT = "s4_replan_haiku55"
KIND = "replan_s4"
PROMPT_VERSION = 1
THINKING = {"type": "disabled"}
ACTIONS = ("next", "reorder", "skip", "finish", "stop")
SAFE_ACTION = "stop"
PLAN_CHANGE_ACTIONS = ("next", "reorder", "skip", "finish")   # 「手順の変更」に数える手（stop は数えない）
RESULTS = ("success", "timeout", "failed", "skipped", "pending")
MAX_RUNS_PER_COLOR = 2          # 同じ色の手順は 2 回まで（最初の 1 回＋組み直しでの 1 回）
MAX_REPLANS = 2                 # 1 試行の中の組み直しの上限。超えたら LLM を呼ばずに stop
REPORT_MAX_CHARS = 90
API_RETRIES = 1                 # 同じ入力での呼び直しの回数

SYSTEM = """あなたはロボットの作業の立て直しの係です。机の上に赤・緑・青の立方体があり、ロボットは立方体を 1 個ずつ箱へ入れます。
人の日本語の指示から作った手順（色の列）の途中で、ある色の手順が決められた時間と回数の中で終わりませんでした。
入力の記録とカメラで見た様子の要約を読み、次の手を 1 つ選び、人に向けた短い状況報告を書いてください。

使える色の名前は "red"（赤）・"green"（緑）・"blue"（青）だけです。色を使わない手では color を "none"、order を空の列にします。

次の手（action）:
- "next": 次に入れる色を 1 つ color に書く。失敗した色をもう 1 度試してもよい
- "reorder": 残りの色を入れる順を order に書く（例: 失敗した色を最後に回す）。order は残りの候補の色を全部、重複なく並べる
- "skip": 失敗した色をあきらめ、残りの色を続ける。color に失敗した色を書く
- "finish": これ以上は入れずに終える。箱にある色を報告する
- "stop": 止まって人に知らせる

決まり:
- 色を選べるのは、入力の candidates（残りの候補）にある色だけです。机の上にない色、すでに箱の中にある色は選びません
- 同じ色を何度も試しません。立方体を見失った、手の中に残っているなど、続けると危ない様子なら "stop" を選びます
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

# 報告に使わない語（内部の用語・記号）。scripts/paper_pdf.py と同じ並びで、tests/test_replan.py が照合する
FORBIDDEN = ["\u584a", "\u3053\u307e", "\u7a2e(?!\u985e)", "\u5e2f", "\u53f0\u672c", "\u8a98\u767a", "\u6210\u7acb",
             "\u7acb\u3061\u76f4", "\u624b\u304c\u304b\u308a", "\u7d99\u304e\u76ee", "\u9045\u308c",
             "\u6d41\u308c\u306e\u4e00\u81f4", "\u4fdd\u5b58\u70b9", "\u81ea\u7136", "\u901a\u3057",
             "\u4e3b\u306a\u691c\u5b9a", "\u526f\u306e\u6307\u6a19", "(?<!\u4fe1\u983c)\u533a\u9593",
             "[\u7532\u4e59\u4e19]", "\u672c\u7dda", "\u652f\u7dda", "\u76e3\u7763", "\u6c7a\u88c1",
             "\u63b2\u793a\u677f", "\u6d41\u7528\u5143", "\u5352\u7814", "\u5352\u696d\u7814\u7a76", "C:\\\\\x56LA",
             "\u5b66\u751f", "\u4e88\u5099\u5b9f\u9a13", "(\\d|\u4e07)\\s*\u624b(?![\u5148\u9996\u9806\u6cd5])",
             "\u6a21\u578b",
             "(?<![A-Za-z0-9])[RNP][123](?![0-9])", "R1\\+", "[RN][12]v[23]", "(?<![A-Za-z])E(?:[1-9]|10)(?![0-9])", "(?<![A-Za-z])G[1-6](?![0-9])",
             "\u6ce8\u5165", "\u7269\u7406\u6f14\u7b97.{0,6}(\u505c\u6b62|\u6b62\u307e|\u6b62\u3081)",
             "(\u505c\u6b62|\u6b62\u307e|\u6b62\u3081).{0,6}\u7269\u7406\u6f14\u7b97",
             "\\d\\s*\u5bfe(?!\u5fdc)", "\u5bfe[\u306f\u304c\u3092]", "\u8a2d\u7f6e\u60c5\u5831", "\u30ad\u30e5\u30fc", "\u7279\u6a29\u60c5\u5831",
             "\u504f\u4f4d", "\u89e3\u653e", "\u30d3\u30c3\u30c8\u5358\u4f4d", "\u4ee5\u524d\u306e\u7248"]
# 報告だけに足す語: 英語の色の名前（利用者向けの文は日本語の色で書く）と、方策・モデルの名前
# 英字の前後は (?<![A-Za-z])…(?![A-Za-z]) で区切る（\\b は日本語の前後で効かない。「redの立方体」「LLMが」も弾く）
REPORT_EXTRA = ["(?i)(?<![A-Za-z])(red|green|blue)(?![A-Za-z])", "(?i)(?<![A-Za-z])smolvla(?![A-Za-z])",
                "(?i)(?<![A-Za-z])vla(?![A-Za-z])", "(?i)(?<![A-Za-z])haiku(?![A-Za-z])",
                "(?i)(?<![A-Za-z])claude(?![A-Za-z])", "(?i)(?<![A-Za-z])llm(?![A-Za-z])"]
_JP = re.compile("[\u3040-\u30ff\u4e00-\u9fff]")


def _sorted_colors(cs) -> list:
    return sorted(dict.fromkeys(c for c in cs if c in COLORS), key=COLORS.index)


def make_context(text: str, plan_order, steps, perception: dict, failed_step: int, failed_color: str,
                 replans_done: int) -> dict:
    """組み直しの入力。steps は [{"color", "result", "attempts"}]（実行した手順の順）、perception は
    {"table": [...], "in_box": [...], "in_hand": [...], "lost": [...]}（知覚の WorldModel から作った色の名前だけ）。"""
    steps = [{"color": str(s["color"]), "result": str(s["result"]), "attempts": int(s["attempts"])} for s in steps]
    per = {k: _sorted_colors(perception.get(k, ())) for k in ("table", "in_box", "in_hand", "lost")}
    return {"instruction": str(text), "plan": [str(c) for c in plan_order], "steps": steps, "perception": per,
            "failed": {"step": int(failed_step), "color": str(failed_color)}, "replans_done": int(replans_done)}


def runs_of(ctx: dict) -> dict:
    out = {c: 0 for c in COLORS}
    for s in ctx["steps"]:
        if s["color"] in out and s["result"] != "skipped":
            out[s["color"]] += 1
    return out


def candidates(ctx: dict) -> list:
    """残りの候補: 今の計画で成功していない・飛ばしていない色のうち、机の上に見えていて、箱の中になく、手の中・見失いでなく、
    同じ色の手順が MAX_RUNS_PER_COLOR 回に達していないもの（計画の順）。"""
    done = {s["color"] for s in ctx["steps"] if s["result"] in ("success", "skipped")}
    per, runs = ctx["perception"], runs_of(ctx)
    out = []
    for c in dict.fromkeys(ctx["plan"]):
        if (c in COLORS and c not in done and c in per["table"] and c not in per["in_box"] and c not in per["in_hand"]
                and c not in per["lost"] and runs[c] < MAX_RUNS_PER_COLOR):
            out.append(c)
    return out


def remaining_planned(ctx: dict) -> list:
    """今の計画で、まだ成功も飛ばしもしていない色（候補に入らない色を含む。計画の順）。"""
    done = {s["color"] for s in ctx["steps"] if s["result"] in ("success", "skipped")}
    return [c for c in dict.fromkeys(ctx["plan"]) if c not in done]


def user_message(ctx: dict) -> str:
    body = dict(ctx, candidates=candidates(ctx), actions=list(ACTIONS), max_runs_per_color=MAX_RUNS_PER_COLOR)
    return json.dumps(body, ensure_ascii=False, sort_keys=True)


def cache_key(model: str, msg: str) -> str:
    key_src = json.dumps({"kind": KIND, "model": model, "temperature": None, "thinking": THINKING, "system": SYSTEM,
                          "schema": SCHEMA, "user": msg, "prompt_version": PROMPT_VERSION}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(key_src.encode("utf-8")).hexdigest()


def parse(raw: str):
    """(data, 理由)。data は SCHEMA の形に合う辞書。合わなければ (None, 理由)。"""
    try:
        d = json.loads(raw)
    except (ValueError, TypeError):
        return None, "json_unreadable"
    if not isinstance(d, dict) or set(d) != set(SCHEMA["required"]):
        return None, "schema_keys"
    if not all(isinstance(d[k], str) for k in ("action", "color", "reason", "report")) or not isinstance(d["order"], list):
        return None, "schema_types"
    if not all(isinstance(c, str) for c in d["order"]):
        return None, "schema_types"
    return d, ""


def check(d: dict, ctx: dict) -> list:
    """弾く理由の列（空なら通す）。d は parse を通った辞書。"""
    a, color, order = d["action"], d["color"], list(d["order"])
    cand, failed = candidates(ctx), ctx["failed"]["color"]
    per, runs = ctx["perception"], runs_of(ctx)
    why = []
    if a not in ACTIONS:
        return [f"unknown_action:{a}"]

    def color_reasons(c, tag):
        r = []
        if c not in COLORS:
            r.append(f"{tag}:unknown_color:{c}")
        elif c in per["in_box"]:
            r.append(f"{tag}:in_box:{c}")
        elif c not in per["table"]:
            r.append(f"{tag}:not_on_table:{c}")
        elif runs[c] >= MAX_RUNS_PER_COLOR:
            r.append(f"{tag}:runs_limit:{c}:{runs[c]}")
        elif c not in cand:
            r.append(f"{tag}:not_candidate:{c}")
        return r

    if a == "next":
        why += color_reasons(color, "next")
    elif a == "reorder":
        if not order:
            why.append("reorder:empty")
        if len(order) != len(set(order)):
            why.append("reorder:duplicates")
        for c in dict.fromkeys(order):
            why += color_reasons(c, "reorder")
        if not why and sorted(order, key=COLORS.index) != sorted(cand, key=COLORS.index):
            why.append(f"reorder:not_permutation_of_candidates:{order}:{cand}")
    elif a == "skip":
        if color != failed:
            why.append(f"skip:not_failed_color:{color}:{failed}")
    return why


def check_report(text: str) -> list:
    t = (text or "").strip()
    if not t:
        return ["report:empty"]
    why = []
    n_sent = len([s for s in re.split("[。！？!?]", t) if s.strip()])
    if n_sent > 2:
        why.append(f"report:sentences:{n_sent}")
    if len(t) > REPORT_MAX_CHARS:
        why.append(f"report:too_long:{len(t)}")
    if not _JP.search(t):
        why.append("report:not_japanese")
    for p in FORBIDDEN + REPORT_EXTRA:
        m = re.search(p, t)
        if m:
            why.append(f"report:forbidden:{m.group(0)}")
    return why


def _ja(cs) -> str:
    return "・".join(JA[c] for c in cs)


def template_report(action: str, ctx: dict, color: str = None, order=None) -> str:
    """記録と知覚だけから決まる報告（LLM の報告を弾いたとき・stop にしたときに使う）。2 文以内。"""
    failed = ctx["failed"]["color"]
    box = ctx["perception"]["in_box"]
    head = f"{JA.get(failed, failed)}の立方体を決められた時間の中で箱に入れられませんでした。"
    have = f"箱には{_ja(box)}が入っており" if box else "箱にはまだ何も入っておらず"
    if action == "next":
        nxt = f"もう一度{JA[color]}を入れます" if color == failed else f"先に{JA[color]}を入れます"
        return f"{head}{nxt}。"
    if action == "reorder":
        return f"{head}{_ja(order)}の順に入れ直します。"
    if action == "skip":
        rest = [c for c in remaining_planned(ctx) if c != failed]
        tail = f"{JA[failed]}は飛ばして、{_ja(rest)}を続けます。" if rest else f"{JA[failed]}は飛ばして終えます。"
        return f"{head}{tail}"
    if action == "finish":
        return f"{head}{have}、ここで終えます。"
    return f"{head}{have}、ここで止まって知らせます。"


def decide_without_llm(ctx: dict, why: str) -> dict:
    """LLM を呼ばずに stop にする（組み直しの上限を超えた、など）。"""
    return {"decision": {"action": SAFE_ACTION, "color": "none", "order": []}, "reason": "",
            "report": template_report(SAFE_ACTION, ctx), "accepted": False, "rejected": [why], "fallback": True,
            "report_source": "template", "report_rejected": [], "candidates": candidates(ctx),
            "llm": {"called": False, "calls": 0}, "replanner_variant": REPLANNER_VARIANT, "kind": KIND}


def _call(cli, model: str, msg: str):
    p = _legacy._CFG["planner"]
    t0 = time.perf_counter()
    # 温度は送らない。思考は disabled を明示する
    r = cli.messages.create(model=model, max_tokens=int(p["max_tokens"]),
                            system=SYSTEM, messages=[{"role": "user", "content": msg}],
                            output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
                            thinking=dict(THINKING))
    raw = next((b.text for b in r.content if getattr(b, "type", None) == "text"), "")
    return {"model": r.model, "raw": raw, "stop_reason": r.stop_reason, "request_id": getattr(r, "_request_id", None),
            "usage": {"input_tokens": r.usage.input_tokens, "output_tokens": r.usage.output_tokens},
            "wall_s": round(time.perf_counter() - t0, 3)}


def replan(ctx: dict, cli=None, use_cache: bool = True, model: str = DEFAULT_MODEL) -> dict:
    """ctx（make_context の戻り値）から次の手を決める。例外は外に出さない（駄目なら stop にする）。"""
    if ctx["replans_done"] >= MAX_REPLANS:
        return decide_without_llm(ctx, f"replan_limit:{ctx['replans_done']}")
    msg = user_message(ctx)
    key = cache_key(model, msg)
    path = CACHE / f"{key}.json"
    llm = {"called": False, "calls": 0, "errors": [], "cache": key, "from_cache": False, "usage": [],
           "requested_model": model, "temperature": None, "thinking": dict(THINKING)}
    data, rec = None, None
    if use_cache and path.is_file():
        try:                                               # 壊れたキャッシュは例外にせず、読み損ねとして記録して呼び直す
            rec = json.loads(path.read_text(encoding="utf-8"))
            data, why = parse(rec["raw"])
        except (ValueError, KeyError, TypeError, OSError) as e:
            rec, data, why = None, None, f"{type(e).__name__}"
        if data is None:                                   # 読めないキャッシュは使わない
            llm["errors"].append(f"cache_unreadable:{why}")
            rec = None
        else:
            llm.update(from_cache=True, model=rec.get("model"), stop_reason=rec.get("stop_reason"))
    if data is None:
        for k in range(1 + API_RETRIES):
            llm["called"] = True
            try:
                cli = cli or client()
                llm["calls"] += 1
                out = _call(cli, model, msg)
            except Exception as e:                         # noqa: BLE001  API の例外・鍵がない、など
                llm["errors"].append(f"api:{type(e).__name__}: {e}"[:300])
                continue
            llm["usage"].append(out["usage"])
            llm.update(model=out["model"], stop_reason=out["stop_reason"], request_id=out["request_id"])
            data, why = parse(out["raw"]) if out["stop_reason"] != "refusal" else (None, "refusal")
            if data is None:
                llm["errors"].append(f"parse:{why}")
                continue
            rec = {"key": key, "kind": KIND, "model": out["model"],
                   "request": {"system_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(), "user": msg,
                               "prompt_version": PROMPT_VERSION, "requested_model": model, "temperature": None,
                               "thinking": dict(THINKING)},
                   "raw": out["raw"], "stop_reason": out["stop_reason"], "request_id": out["request_id"],
                   "usage": out["usage"], "wall_s": out["wall_s"], "written": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "attempt": k, "from_cache": False, "replanner_variant": REPLANNER_VARIANT}
            if use_cache:
                CACHE.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
            break
    llm["retried"] = llm["calls"] > 1
    base = {"replanner_variant": REPLANNER_VARIANT, "kind": KIND, "candidates": candidates(ctx), "llm": llm}
    if data is None:
        return dict(base, decision={"action": SAFE_ACTION, "color": "none", "order": []}, reason="",
                    report=template_report(SAFE_ACTION, ctx), accepted=False,
                    rejected=["llm_failed"] + llm["errors"][-1:], fallback=True, report_source="template",
                    report_rejected=[], llm_output=None)
    rejected = check(data, ctx)
    if rejected:
        return dict(base, decision={"action": SAFE_ACTION, "color": "none", "order": []}, reason="",
                    report=template_report(SAFE_ACTION, ctx), accepted=False, rejected=rejected, fallback=True,
                    report_source="template", report_rejected=[], llm_output=data)
    a = data["action"]
    color = data["color"] if a in ("next", "skip") else "none"
    order = list(data["order"]) if a == "reorder" else []
    rep_why = check_report(data["report"])
    report = data["report"].strip() if not rep_why else template_report(a, ctx, color, order)
    return dict(base, decision={"action": a, "color": color, "order": order}, reason=data["reason"].strip()[:200],
                report=report, accepted=True, rejected=[], fallback=False,
                report_source="llm" if not rep_why else "template", report_rejected=rep_why, llm_output=data)


def apply(decision: dict, ctx: dict) -> list:
    """通った手から、これから実行する色の並びを作る（空なら終える）。stop・finish は空。"""
    a = decision["action"]
    rest = [c for c in remaining_planned(ctx) if c in candidates(ctx)]
    if a == "next":
        c = decision["color"]
        return [c] + [x for x in rest if x != c]
    if a == "reorder":
        return list(decision["order"])
    if a == "skip":
        return [x for x in rest if x != decision["color"]]
    return []


def replan_step(text: str, plan_order, steps, perception: dict, failed_step: int, failed_color: str, replans_done: int,
                cli=None, use_cache: bool = True, model: str = DEFAULT_MODEL) -> dict:
    """実行器（runtime/executor_replan.py）に渡す関数。入力を make_context で整え、replan の結果に、使った入力 "context" と
    これから実行する色の並び "next_order"（apply）を足して返す。実行器は planner を import しない（真値との境界を保つ）ので、
    この関数を引数で受け取る。"""
    ctx = make_context(text, plan_order, steps, perception, failed_step, failed_color, replans_done)
    out = replan(ctx, cli=cli, use_cache=use_cache, model=model)
    out["context"] = ctx
    out["next_order"] = apply(out["decision"], ctx)
    if out["decision"]["action"] in ("next", "reorder") and not out["next_order"]:
        # 続ける手なのに並びが空（check を通れば起きない。念のため）: empty_order として stop にする
        out.update(decision={"action": SAFE_ACTION, "color": "none", "order": []}, accepted=False,
                   rejected=list(out.get("rejected") or []) + ["empty_order"], fallback=True,
                   report=template_report(SAFE_ACTION, ctx), report_source="template")
    out["plan_change"] = out["decision"]["action"] in PLAN_CHANGE_ACTIONS
    return out
