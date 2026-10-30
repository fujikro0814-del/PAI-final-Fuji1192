"""LLM による手順の組み直し（replan_base）を、モデルを選んで呼ぶ（Haiku 5.5・Sonnet 5.5・Opus 5.5）。

    from recovla.planner import replan_multi as RM
    out = RM.replan(ctx, model="claude-sonnet-5-5")          # ctx は replan_base.make_context(...) の戻り値。戻り値の形は replan_base.replan と同じ
    step = RM.make_replan_step("claude-sonnet-5-5")          # 実行器（runtime/executor_replan.install）に渡す関数（replan_base.replan_step と同じ引数）

指示文・スキーマ・検査・定型の報告などは replan_base のものを使う。
モデルごとの引数（どれも温度は送らない）:
  claude-haiku-5-5   replan_base と同じ: max_tokens は configs の planner.max_tokens、thinking disabled、output_config は format だけ
  claude-sonnet-5-5  max_tokens 2048、thinking between_tools、output_config.effort low
  claude-opus-5-5    max_tokens 4096、thinking は送らない（切れない）、output_config.effort low
キャッシュ: キー = SHA-256(KIND・モデル・設定・温度・SYSTEM・SCHEMA・入力・PROMPT_VERSION・key_extra)。置き場は cache_dir
  （既定は outputs/llm_cache/）。読めた応答だけを残す。API の例外・読めない応答（拒否を含む）は 1 回だけ呼び直し、
  それでも駄目なら stop にする。
報告の検査: replan_base.check_report に、モデルの名前（Sonnet・Opus）を弾く語を足す（MODEL_NAME_EXTRA）。
シミュレーション・評価・真値のモジュールは import しない（入力は replan_base.make_context の辞書だけ）。
"""
import hashlib
import json
import pathlib
import re
import time

from recovla.planner import decompose as _legacy
from recovla.planner import replan_base as R4

MODELS = {"haiku": "claude-haiku-5-5", "sonnet": "claude-sonnet-5-5", "opus": "claude-opus-5-5"}
DEFAULT_MODEL = MODELS["haiku"]
KIND = "replan_multi"
# Sonnet・Opus の引数（vlm/ask.py と同じ値）。Haiku は replan_base と同じ要求にするため settings で作る
MODEL_SETTINGS = {
    "claude-sonnet-5-5": {"max_tokens": 2048, "thinking": {"type": "between_tools"}, "output_config": {"effort": "low"}},
    "claude-opus-5-5": {"max_tokens": 4096, "output_config": {"effort": "low"}},
}
MODEL_NAME_EXTRA = ["(?i)(?<![A-Za-z])(sonnet|opus)(?![A-Za-z])"]


def variant(model: str) -> str:
    short = next((k for k, v in MODELS.items() if v == model), model)
    return f"s4_replan_multi_{short}55"


def settings(model: str) -> dict:
    """送る引数のうち、入力によらない部分（output_config の format は build_params で足す）。"""
    if model == MODELS["haiku"]:
        return {"max_tokens": int(_legacy._CFG["planner"]["max_tokens"]), "thinking": dict(R4.THINKING),
                "output_config": {}}
    if model not in MODEL_SETTINGS:
        raise ValueError(f"使えないモデル: {model}（{sorted(MODELS.values())} のどれか）")
    return json.loads(json.dumps(MODEL_SETTINGS[model]))


def build_params(model: str, msg: str) -> dict:
    """messages.create（Batch では params）に渡す引数。温度は送らない。"""
    s = settings(model)
    oc = dict(s.get("output_config") or {})
    oc["format"] = {"type": "json_schema", "schema": R4.SCHEMA}
    p = {"model": model, "max_tokens": int(s["max_tokens"]), "system": R4.SYSTEM,
         "messages": [{"role": "user", "content": msg}], "output_config": oc}
    if s.get("thinking") is not None:
        p["thinking"] = dict(s["thinking"])
    return p


def cache_key(model: str, msg: str, key_extra=None) -> str:
    src = json.dumps({"kind": KIND, "model": model, "settings": settings(model), "temperature": None, "system": R4.SYSTEM,
                      "schema": R4.SCHEMA, "user": msg, "prompt_version": R4.PROMPT_VERSION, "extra": key_extra},
                     ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(src.encode("utf-8")).hexdigest()


def check_report(text: str) -> list:
    why = R4.check_report(text)
    t = (text or "").strip()
    for p in MODEL_NAME_EXTRA:
        m = re.search(p, t)
        if m:
            why.append(f"report:forbidden:{m.group(0)}")
    return why


def usage_of(r) -> dict:
    u = getattr(r, "usage", None)
    if u is None:
        return {}
    g = (lambda k: u.get(k)) if isinstance(u, dict) else (lambda k: getattr(u, k, None))
    return {k: int(g(k)) for k in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
            if isinstance(g(k), (int, float))}


def message_fields(r) -> dict:
    """応答（Message）から記録に残す欄。思考のアクションチャンクは数だけ数え、中身は残さない。"""
    content = list(getattr(r, "content", None) or [])
    raw = next((b.text for b in content if getattr(b, "type", None) == "text"), "")
    return {"model": getattr(r, "model", None), "raw": raw, "stop_reason": getattr(r, "stop_reason", None),
            "request_id": getattr(r, "_request_id", None), "usage": usage_of(r),
            "thinking_blocks": sum(1 for b in content if getattr(b, "type", None) in ("thinking", "redacted_thinking"))}


def _call(cli, model: str, msg: str) -> dict:
    t0 = time.perf_counter()
    r = cli.messages.create(**build_params(model, msg))
    return dict(message_fields(r), wall_s=round(time.perf_counter() - t0, 3))


def decide_without_llm(ctx: dict, why: str, model: str) -> dict:
    out = R4.decide_without_llm(ctx, why)
    out.update(replanner_variant=variant(model), kind=KIND)
    return out


def fallback_out(ctx: dict, model: str, llm: dict, errors=()) -> dict:
    """LLM が答えられなかった（2 回とも駄目）ときの安全な手 stop。"""
    base = {"replanner_variant": variant(model), "kind": KIND, "candidates": R4.candidates(ctx), "llm": llm}
    return dict(base, decision={"action": R4.SAFE_ACTION, "color": "none", "order": []}, reason="",
                report=R4.template_report(R4.SAFE_ACTION, ctx), accepted=False,
                rejected=["llm_failed"] + list(errors)[-1:], fallback=True, report_source="template",
                report_rejected=[], llm_output=None)


def decide(ctx: dict, data: dict, model: str, llm: dict) -> dict:
    """parse を通った応答 data から手と報告を決める（replan_base.replan の後半と同じ規則）。"""
    base = {"replanner_variant": variant(model), "kind": KIND, "candidates": R4.candidates(ctx), "llm": llm}
    rejected = R4.check(data, ctx)
    if rejected:
        return dict(base, decision={"action": R4.SAFE_ACTION, "color": "none", "order": []}, reason="",
                    report=R4.template_report(R4.SAFE_ACTION, ctx), accepted=False, rejected=rejected, fallback=True,
                    report_source="template", report_rejected=[], llm_output=data)
    a = data["action"]
    color = data["color"] if a in ("next", "skip") else "none"
    order = list(data["order"]) if a == "reorder" else []
    rep_why = check_report(data["report"])
    report = data["report"].strip() if not rep_why else R4.template_report(a, ctx, color, order)
    return dict(base, decision={"action": a, "color": color, "order": order}, reason=data["reason"].strip()[:200],
                report=report, accepted=True, rejected=[], fallback=False,
                report_source="llm" if not rep_why else "template", report_rejected=rep_why, llm_output=data)


def parse_message(fields: dict):
    """(data, 理由)。拒否の stop_reason は読めない応答に数える（replan_base と同じ）。"""
    if fields.get("stop_reason") == "refusal":
        return None, "refusal"
    return R4.parse(fields.get("raw"))


def cache_record(key: str, model: str, msg: str, fields: dict, attempt: int, mode: str, key_extra=None, extra=None) -> dict:
    rec = {"key": key, "kind": KIND, "model": fields.get("model"),
           "request": {"system_sha256": hashlib.sha256(R4.SYSTEM.encode()).hexdigest(), "user": msg,
                       "prompt_version": R4.PROMPT_VERSION, "requested_model": model, "temperature": None,
                       "settings": settings(model), "key_extra": key_extra},
           "raw": fields.get("raw"), "stop_reason": fields.get("stop_reason"), "request_id": fields.get("request_id"),
           "usage": fields.get("usage"), "thinking_blocks": fields.get("thinking_blocks"), "wall_s": fields.get("wall_s"),
           "written": time.strftime("%Y-%m-%d %H:%M:%S"), "attempt": attempt, "mode": mode, "from_cache": False,
           "replanner_variant": variant(model)}
    rec.update(extra or {})
    return rec


def write_cache(path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def replan(ctx: dict, model: str = DEFAULT_MODEL, cli=None, use_cache: bool = True, cache_dir=None, key_extra=None,
           client_factory=None, on_record=None) -> dict:
    """ctx（replan_base.make_context の戻り値）から次の手を決める。例外は外に出さない（駄目なら stop にする）。
    on_record は書いたキャッシュの記録を受ける関数（任意）。"""
    settings(model)                                        # 使えないモデルはここで止める（例外）
    if ctx["replans_done"] >= R4.MAX_REPLANS:
        return decide_without_llm(ctx, f"replan_limit:{ctx['replans_done']}", model)
    msg = R4.user_message(ctx)
    key = cache_key(model, msg, key_extra)
    path = pathlib.Path(cache_dir or R4.CACHE) / f"{key}.json"
    llm = {"called": False, "calls": 0, "errors": [], "cache": key, "from_cache": False, "usage": [],
           "requested_model": model, "temperature": None, "settings": settings(model)}
    data = None
    if use_cache and path.is_file():
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
            data, why = parse_message(rec)
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
                cli = cli or (client_factory or R4.client)()
                llm["calls"] += 1
                out = _call(cli, model, msg)
            except Exception as e:                         # noqa: BLE001  API の例外・鍵がない、など
                llm["errors"].append(f"api:{type(e).__name__}: {e}"[:300])
                continue
            llm["usage"].append(out["usage"])
            llm.update(model=out["model"], stop_reason=out["stop_reason"], request_id=out["request_id"])
            data, why = parse_message(out)
            if data is None:
                llm["errors"].append(f"parse:{why}")
                continue
            rec = cache_record(key, model, msg, out, k, "sync", key_extra)
            if use_cache:
                write_cache(path, rec)
            if on_record is not None:
                on_record(rec)
            break
    llm["retried"] = llm["calls"] > 1
    if data is None:
        return fallback_out(ctx, model, llm, llm["errors"])
    return decide(ctx, data, model, llm)


def replan_step(text: str, plan_order, steps, perception: dict, failed_step: int, failed_color: str, replans_done: int,
                cli=None, use_cache: bool = True, model: str = DEFAULT_MODEL, cache_dir=None, key_extra=None) -> dict:
    """replan_base.replan_step と同じ引数・同じ戻り値の形（モデルを選べる）。"""
    ctx = R4.make_context(text, plan_order, steps, perception, failed_step, failed_color, replans_done)
    out = replan(ctx, model=model, cli=cli, use_cache=use_cache, cache_dir=cache_dir, key_extra=key_extra)
    return finish_step(ctx, out)


def finish_step(ctx: dict, out: dict) -> dict:
    """replan の結果に、使った入力 "context" とこれから実行する色の並び "next_order" を足す（replan_base.replan_step の後半と同じ）。"""
    out["context"] = ctx
    out["next_order"] = R4.apply(out["decision"], ctx)
    if out["decision"]["action"] in ("next", "reorder") and not out["next_order"]:
        out.update(decision={"action": R4.SAFE_ACTION, "color": "none", "order": []}, accepted=False,
                   rejected=list(out.get("rejected") or []) + ["empty_order"], fallback=True,
                   report=R4.template_report(R4.SAFE_ACTION, ctx), report_source="template")
    out["plan_change"] = out["decision"]["action"] in R4.PLAN_CHANGE_ACTIONS
    return out


def make_replan_step(model: str):
    """runtime/executor_replan.install に渡す関数（モデルを固定した replan_step）。"""
    settings(model)

    def step(**kw):
        return replan_step(**kw, model=model)
    step.model = model
    return step
