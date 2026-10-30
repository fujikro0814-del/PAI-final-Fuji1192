"""LLM で日本語の指示を、立方体を箱へ入れる色の列（サブタスク）に分ける。

    out = decompose("全部片付けて", table_colors=["red", "blue"], box_colors=["green"])
    # {"steps": ["red", "blue"], "reply": "...", "valid": True, "checks": {...}, "cache": "<key>", ...}

- モデルは configs の planner.model、温度 0。画像は送らず、色の判定の結果だけを渡す
- JSON スキーマの構造化出力で {"steps", "reply"} を受け、規則で検査する（机上の色の部分集合か、箱の中の色を含まないか、
  重複がないか）。通らなければ手順を空にして人に尋ねる
- 応答は outputs/llm_cache/<キー>.json に保存し、同じ入力では再利用する（キーはモデル名・プロンプト・入力の SHA-256）
- API キーは環境変数 ANTHROPIC_API_KEY（無ければ Windows のユーザー環境変数）。.local/ca_bundle_proxy.pem があれば
  TLS の検証に使う
"""
import hashlib
import json
import os
import pathlib
import time

from recovla.common import config
from recovla.common.seeds import COLORS

_CFG = config.load()
PROMPT_VERSION = 1
SYSTEM = """あなたはロボットの作業計画の係です。机の上に赤・緑・青の立方体があり、ロボットは立方体を 1 個ずつ箱へ入れます。
人の日本語の指示を読み、箱へ入れる立方体の色を、入れる順に並べた列にしてください。

使える色の名前は "red"（赤）・"green"（緑）・"blue"（青）だけです。
入力には、いま机の上にある色（table_colors）と、すでに箱の中にある色（box_colors）が書かれています。

決まり:
- 手順に入れてよいのは、机の上にある色だけです。箱の中にすでにある色は入れません。同じ色を 2 回入れません
- 「全部」「残り全部」なら、机の上の色を全部入れます。順番の指定がなければ red, green, blue の順にします
- 順番の指定（「先に」「最後に」「〜してから」など）があれば、その順にします
- 机の上にない色を頼まれた、何をすればよいか分からない、立方体を箱へ入れる以外の頼み（取り出す、積む、並べるなど）の
  ときは、steps を空の列にして、reply で人に分かりやすく尋ねるか、できない理由を伝えてください
- 一部だけ実行できる場合（例: 2 色を頼まれて 1 色が机の上にない）は、実行せずに steps を空にして reply で確かめてください
- reply は日本語の短い文にします。実行するときは、何をどの順で入れるかを伝えます
"""
SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {"type": "array", "items": {"type": "string", "enum": list(COLORS)}},
        "reply": {"type": "string"},
    },
    "required": ["steps", "reply"],
    "additionalProperties": False,
}
CACHE = config.path(_CFG["paths"]["outputs"]) / "llm_cache"
CA_BUNDLE = config.ROOT / ".local" / "ca_bundle_proxy.pem"


def _api_key():
    k = os.environ.get("ANTHROPIC_API_KEY")
    if not k and os.name == "nt":                     # Windows のユーザー環境変数から読む
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as h:
                k = winreg.QueryValueEx(h, "ANTHROPIC_API_KEY")[0]
        except OSError:
            k = None
    return k


def client():
    import anthropic
    kw = {"api_key": _api_key(), "timeout": float(_CFG["planner"]["timeout_s"]), "max_retries": 2}
    if kw["api_key"] is None:
        raise RuntimeError("ANTHROPIC_API_KEY が設定されていない")
    if CA_BUNDLE.is_file():
        import ssl
        kw["http_client"] = anthropic.DefaultHttpxClient(verify=ssl.create_default_context(cafile=str(CA_BUNDLE)))
    return anthropic.Anthropic(**kw)


def user_message(text: str, table_colors, box_colors) -> str:
    return json.dumps({"instruction": text, "table_colors": sorted(table_colors, key=COLORS.index),
                       "box_colors": sorted(box_colors, key=COLORS.index)}, ensure_ascii=False)


def check(steps, table_colors, box_colors) -> dict:
    return {"subset_of_table": all(s in table_colors for s in steps),
            "no_box_colors": not any(s in box_colors for s in steps),
            "no_duplicates": len(steps) == len(set(steps)),
            "known_colors": all(s in COLORS for s in steps)}


JA = {"red": "赤", "green": "緑", "blue": "青"}


def fallback_reply(steps, table_colors, box_colors) -> str:
    """検査に通らなかったときの返答。LLM の返答の文は使わず、検査の結果だけから作る。"""
    known = [s for s in steps if s in COLORS]
    in_box = [s for s in dict.fromkeys(known) if s in box_colors]
    absent = [s for s in dict.fromkeys(known) if s not in table_colors and s not in box_colors]
    ok = [s for s in dict.fromkeys(known) if s in table_colors]
    parts = []
    if in_box:
        parts.append("・".join(JA[s] for s in in_box) + "はすでに箱の中にあります。")
    if absent:
        parts.append("・".join(JA[s] for s in absent) + "は机の上にありません。")
    if len(known) != len(set(known)):
        parts.append("同じ色は 1 回しか入れられません。")
    if not known or not parts:
        return "指示をうまく読み取れませんでした。どの色の立方体を箱に入れるか、もう一度教えてください。"
    if ok:
        return "".join(parts) + "・".join(JA[s] for s in ok) + "だけ入れますか。"
    return "".join(parts) + "どの色の立方体を箱に入れるか、もう一度教えてください。"


def decompose(text: str, table_colors, box_colors, cli=None, use_cache: bool = True) -> dict:
    p = _CFG["planner"]
    msg = user_message(text, table_colors, box_colors)
    key_src = json.dumps({"model": p["model"], "temperature": p["temperature"], "system": SYSTEM, "schema": SCHEMA,
                          "user": msg, "prompt_version": PROMPT_VERSION}, ensure_ascii=False, sort_keys=True)
    key = hashlib.sha256(key_src.encode("utf-8")).hexdigest()
    path = CACHE / f"{key}.json"
    if use_cache and path.is_file():
        rec = json.loads(path.read_text(encoding="utf-8"))
        rec["from_cache"] = True
    else:
        cli = cli or client()
        t0 = time.perf_counter()
        # SDK の引数に temperature が無いため extra_body で渡す
        r = cli.messages.create(model=p["model"], max_tokens=int(p["max_tokens"]),
                                system=SYSTEM, messages=[{"role": "user", "content": msg}],
                                output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
                                extra_body={"temperature": float(p["temperature"])})
        raw = next((b.text for b in r.content if b.type == "text"), "")
        rec = {"key": key, "model": r.model, "request": {"system_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
                                                          "user": msg, "prompt_version": PROMPT_VERSION},
               "raw": raw, "stop_reason": r.stop_reason, "request_id": getattr(r, "_request_id", None),
               "usage": {"input_tokens": r.usage.input_tokens, "output_tokens": r.usage.output_tokens},
               "wall_s": round(time.perf_counter() - t0, 3), "written": time.strftime("%Y-%m-%d %H:%M:%S"),
               "from_cache": False}
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        data = json.loads(rec["raw"])
        steps, reply = list(data["steps"]), str(data["reply"])
        parsed = True
    except (ValueError, KeyError, TypeError):
        steps, reply, parsed = [], "指示をうまく読み取れませんでした。もう一度教えてください。", False
    checks = {"parsed": parsed, **check(steps, table_colors, box_colors)}
    valid = all(checks.values())
    return {"steps": steps if valid else [], "reply": reply if valid else fallback_reply(steps, table_colors, box_colors),
            "llm_reply": reply,
            "llm_steps": steps, "valid": valid, "checks": checks, "cache": rec["key"], "from_cache": rec["from_cache"],
            "usage": rec.get("usage"), "stop_reason": rec.get("stop_reason")}
