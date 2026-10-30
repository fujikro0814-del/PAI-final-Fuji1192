"""LLM の手順の口。TaskSpec から、起動時の分解と手順の組み直しの指示（system・入力・スキーマ・検査）を作る。
物の名前・呼び方は書式から入れ、色の名前を埋め込まない。

評価では使わない（評価の条件は planner/decompose_haiku・planner/replan_event などを使う。LLM のキャッシュの鍵が変わるため）。
口と単体テストだけ。LLM は呼ばない（呼ぶのは planner の client を使う側）。

    sys_ = decompose_system(spec)
    msg = decompose_message(spec, "全部片付けて", table=["red", "blue"], in_place=["green"])
    schema = decompose_schema(spec)
    check_steps(spec, steps, table, in_place)                 # decompose.check と同じ 4 つの検査
    ctx = replan_context(spec, text, plan, steps, perception, failed_step, failed_obj, replans_done, event=None)
"""
import json

PROMPT_VERSION = 1
EVENTS = ("dropped", "timeout")


def _label(spec, name: str) -> str:
    o = spec.object(name)
    return f'"{name}"（{o.words[0]}）' if o.words else f'"{name}"'


def _sorted(spec, names) -> list:
    order = spec.object_names
    return sorted(dict.fromkeys(n for n in names if n in order), key=order.index)


def decompose_system(spec) -> str:
    """起動時の分解の system（decompose.SYSTEM と同じ決まりを、書式の物・場所の名前で書いたもの）。場所の種類が surface
    （皿など）なら「〜へ入れる」を「〜の上に置く」と書く（container の文は変えない）。"""
    place = spec.place(spec.goal.place)
    pw = place.words[0] if place.words else place.name
    if place.kind == "surface":
        put, putting, inside, already = f"{pw}の上に置きます", f"{pw}の上に置く", f"{pw}の上", f"{pw}の上にすでにある"
    else:
        put, putting, inside, already = f"{pw}へ入れます", f"{pw}へ入れる", f"{pw}の中", f"{pw}の中にすでにある"
    verb, verb_d = ("置き", "置く") if place.kind == "surface" else ("入れ", "入れる")
    objs = "・".join(o.words[0] if o.words else o.name for o in spec.objects)
    names = "・".join(_label(spec, n) for n in spec.object_names)
    order = ", ".join(spec.object_names)
    return (f"あなたはロボットの作業計画の係です。机の上に{objs}があり、ロボットは物を 1 個ずつ{put}。\n"
            f"人の日本語の指示を読み、{putting}物の名前を、{verb_d}順に並べた列にしてください。\n\n"
            f"使える物の名前は {names} だけです。\n"
            f"入力には、いま机の上にある物（table_objects）と、すでに{inside}にある物（in_place_objects）が書かれています。\n\n"
            "決まり:\n"
            f"- 手順に入れてよいのは、机の上にある物だけです。{already}物は入れません。同じ物を 2 回入れません\n"
            f"- 「全部」「残り全部」なら、机の上の物を全部{verb}ます。順番の指定がなければ {order} の順にします\n"
            "- 順番の指定（「先に」「最後に」「〜してから」など）があれば、その順にします\n"
            f"- 机の上にない物を頼まれた、何をすればよいか分からない、物を{putting}以外の頼みのときは、steps を空の列にして、"
            "reply で人に分かりやすく尋ねるか、できない理由を伝えてください\n"
            "- 一部だけ実行できる場合は、実行せずに steps を空にして reply で確かめてください\n"
            f"- reply は日本語の短い文にします。実行するときは、何をどの順で{verb_d}かを伝えます\n")


def decompose_schema(spec) -> dict:
    return {"type": "object",
            "properties": {"steps": {"type": "array", "items": {"type": "string", "enum": list(spec.object_names)}},
                           "reply": {"type": "string"}},
            "required": ["steps", "reply"], "additionalProperties": False}


def decompose_message(spec, text: str, table, in_place) -> str:
    """起動時の分解の入力（decompose.user_message と同じ並び・同じ値。欄の名前だけ物の名前の一般形）。"""
    return json.dumps({"instruction": text, "table_objects": _sorted(spec, table), "in_place_objects": _sorted(spec, in_place)},
                      ensure_ascii=False)


def check_steps(spec, steps, table, in_place) -> dict:
    """decompose.check と同じ 4 つの検査（机の上の物だけ・場所の中の物を入れない・重複なし・知っている名前だけ）。"""
    return {"subset_of_table": all(s in table for s in steps),
            "no_box_colors": not any(s in in_place for s in steps),
            "no_duplicates": len(steps) == len(set(steps)),
            "known_colors": all(s in spec.object_names for s in steps)}


def replan_context(spec, text: str, plan_order, steps, perception: dict, failed_step: int, failed_obj: str,
                   replans_done: int, event: str = None) -> dict:
    """手順の組み直しの入力（replan_base.make_context と同じ形。steps は [{"obj", "result", "attempts"}]、perception は
    {"table", "in_place", "in_hand", "lost"} の物の名前）。event を渡すと failed に足す（replan_event と同じ）。"""
    if event is not None and event not in EVENTS:
        raise ValueError(f"event {event!r} は {EVENTS} のどれか")
    st = [{"obj": str(s["obj"]), "result": str(s["result"]), "attempts": int(s["attempts"])} for s in steps]
    per = {k: _sorted(spec, perception.get(k, ())) for k in ("table", "in_place", "in_hand", "lost")}
    failed = {"step": int(failed_step), "obj": str(failed_obj)}
    if event is not None:
        failed["event"] = event
    return {"instruction": str(text), "plan": [str(c) for c in plan_order], "steps": st, "perception": per,
            "failed": failed, "replans_done": int(replans_done)}
