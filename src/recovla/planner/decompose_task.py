"""タスクの書式（configs/tasks/*.yaml）から作った指示文で分ける、指示の分解（例: カップを皿に）。

モデル・呼び方・キャッシュは decompose_haiku と同じで、指示文・入力・スキーマ・検査を api/planner が書式の物・場所の名前から作る。

    fn = make_decompose(spec)                       # 呼び方 fn(text, table_colors, box_colors) は decompose_haiku と同じ
    out = fn("カップを皿に置いて", ["red"], [])     # steps は色、llm_steps は物の名前

実行器は色で手順を持つので、入力の色を物の名前に、答えの物の名前を色に引き直す（api/task_spec の step_of_cue・cue_of）。
キャッシュのキーにはタスクの名前と api/planner の PROMPT_VERSION を入れる。立方体 3 色のタスクは decompose_haiku を使う。
"""
import hashlib
import json
import time

from recovla.api import planner as AP
from recovla.api import task_spec as TS
from recovla.planner import decompose_haiku as D4

PLANNER_VARIANT = "task_s4_haiku55"


def cache_key(spec, model: str, system: str, schema: dict, msg: str) -> str:
    src = json.dumps({"model": model, "temperature": None, "thinking": D4.THINKING, "system": system, "schema": schema,
                      "user": msg, "prompt_version": AP.PROMPT_VERSION, "task": spec.name}, ensure_ascii=False,
                     sort_keys=True)
    return hashlib.sha256(src.encode("utf-8")).hexdigest()


def _names(spec, colors) -> list:
    out = []
    for c in colors:
        try:
            out.append(TS.step_of_cue(spec, c).obj)
        except KeyError:                                       # タスクに無い色（関係ない物）は入れない
            pass
    return out


def fallback_reply(spec, steps, table, in_place) -> str:
    known = [s for s in steps if s in spec.object_names]
    word = {n: (spec.object(n).words[0] if spec.object(n).words else n) for n in spec.object_names}
    place = spec.place(spec.goal.place)
    pw = place.words[0] if place.words else place.name
    parts = []
    there = [s for s in dict.fromkeys(known) if s in in_place]
    absent = [s for s in dict.fromkeys(known) if s not in table and s not in in_place]
    if there:
        parts.append("・".join(word[s] for s in there) + f"はすでに{pw}にあります。")
    if absent:
        parts.append("・".join(word[s] for s in absent) + "は机の上にありません。")
    if len(known) != len(set(known)):
        parts.append("同じ物は 1 回しか運べません。")
    return "".join(parts) or "指示をうまく読み取れませんでした。もう一度教えてください。"


def make_decompose(spec, model: str = D4.DEFAULT_MODEL):
    system, schema = AP.decompose_system(spec), AP.decompose_schema(spec)

    def decompose(text: str, table_colors, box_colors, cli=None, use_cache: bool = True) -> dict:
        table, in_place = _names(spec, table_colors), _names(spec, box_colors)
        msg = AP.decompose_message(spec, text, table, in_place)
        key = cache_key(spec, model, system, schema, msg)
        path = D4.CACHE / f"{key}.json"
        rec = D4._read_cache(path, key) if use_cache else None
        if rec is not None:
            rec["from_cache"] = True
        else:
            cli = cli or D4.client()
            t0 = time.perf_counter()
            r = cli.messages.create(model=model, max_tokens=int(D4._legacy._CFG["planner"]["max_tokens"]),
                                    system=system, messages=[{"role": "user", "content": msg}],
                                    output_config={"format": {"type": "json_schema", "schema": schema}},
                                    thinking=dict(D4.THINKING))
            raw = next((b.text for b in r.content if b.type == "text"), "")
            rec = {"key": key, "model": r.model, "request": {"system_sha256": hashlib.sha256(system.encode()).hexdigest(),
                                                              "user": msg, "prompt_version": AP.PROMPT_VERSION,
                                                              "requested_model": model, "temperature": None,
                                                              "thinking": dict(D4.THINKING), "task": spec.name},
                   "raw": raw, "stop_reason": r.stop_reason, "request_id": getattr(r, "_request_id", None),
                   "usage": {"input_tokens": r.usage.input_tokens, "output_tokens": r.usage.output_tokens},
                   "wall_s": round(time.perf_counter() - t0, 3), "written": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "from_cache": False, "planner_variant": PLANNER_VARIANT}
            rec = D4._store_cache(path, rec, overwrite=not use_cache)
        try:
            data = json.loads(rec["raw"])
            steps, reply = list(data["steps"]), str(data["reply"])
            parsed = True
        except (ValueError, KeyError, TypeError):
            steps, reply, parsed = [], "指示をうまく読み取れませんでした。もう一度教えてください。", False
        checks = {"parsed": parsed, **AP.check_steps(spec, steps, table, in_place)}
        valid = all(checks.values())
        colors = [TS.cue_of(spec, TS.StepSpec("pick_place", s, spec.goal.place)) for s in steps] if valid else []
        return {"steps": colors, "reply": reply if valid else fallback_reply(spec, steps, table, in_place),
                "llm_reply": reply, "llm_steps": steps, "valid": valid, "checks": checks, "cache": rec["key"],
                "from_cache": rec["from_cache"], "usage": rec.get("usage"), "stop_reason": rec.get("stop_reason"),
                "planner_variant": PLANNER_VARIANT, "model": rec.get("model"), "task": spec.name}
    decompose.info = {"name": "task_s4", "model": model, "variant": PLANNER_VARIANT, "task": spec.name,
                      "prompt": "api/planner.decompose_system", "prompt_version": AP.PROMPT_VERSION}
    return decompose
