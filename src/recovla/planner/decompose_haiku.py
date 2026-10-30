"""指示の分解（Claude Haiku 5.5）。指示文・スキーマ・検査は decompose.py と同じで、呼び出し方だけが違う。

    from recovla.planner import decompose_haiku as D4
    out = D4.decompose("全部片付けて", table_colors=["red", "blue"], box_colors=["green"])
    # 戻り値は decompose.decompose と同じ形に "planner_variant" と "model" を足したもの

decompose.py との違い:
- モデルは引数 model（既定 claude-haiku-5-5）
- 温度を送らない（このモデルは temperature を受け付けない）。同じ入力の答えはキャッシュで固定する
- 思考は {"type": "disabled"} を明示する
- キャッシュのキーに model・"temperature": None・thinking を入れる（decompose.py のキャッシュと混ざらない）
キャッシュ（outputs/llm_cache/）は一時ファイルに書いてから、先に書いた方を残す。読めない写しは
<キー>.json.broken_<時刻>_<pid> へ退避して取り直す。
"""
import hashlib
import json
import os
import threading
import time

from recovla.planner import decompose as _legacy
from recovla.planner.decompose import (CACHE, PROMPT_VERSION, SCHEMA, SYSTEM, check, client, fallback_reply,  # noqa: F401
                                       user_message)

DEFAULT_MODEL = "claude-haiku-5-5"
PLANNER_VARIANT = "s4_haiku55"
THINKING = {"type": "disabled"}


def cache_key(model: str, msg: str) -> str:
    """キャッシュのキー。decompose.py の項目に、温度（None）と思考の設定を足す。"""
    key_src = json.dumps({"model": model, "temperature": None, "thinking": THINKING, "system": SYSTEM, "schema": SCHEMA,
                          "user": msg, "prompt_version": PROMPT_VERSION}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(key_src.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- キャッシュの読み書き
# 読み: 読めない写し（書きかけ・0 埋め・形の違うもの）は退避して、無いものとして取り直す（例外にしない）。
# 書き: プロセスごとの一時ファイルに書き、os.link で先に書いた方を残す。並列の実行が同じキーを同時に呼んだときは、
#   後の方は先の写しを使う（同じ入力の答えを 1 つに保つ）。use_cache=False のときだけ上書きする。

def _cache_ok(rec, key: str) -> bool:
    return isinstance(rec, dict) and rec.get("key") == key and isinstance(rec.get("raw"), str)


def _quarantine_cache(path):
    dst = path.with_name(f"{path.name}.broken_{time.strftime('%Y%m%d-%H%M%S')}_{os.getpid()}")
    try:
        os.replace(path, dst)
        return dst
    except OSError:
        return None


def _read_cache(path, key: str):
    """キャッシュの写し（dict）か None（無い・読めない）。読めない写しは退避する。開けないとき（ほかのプロセスが
    掴んでいる）は少し待って読み直し、それでも開けなければ例外にする（中身は壊れていないかもしれないので退避しない）。"""
    for k in range(4):
        try:
            rec = json.loads(path.read_text(encoding="utf-8-sig"))
            break
        except FileNotFoundError:
            return None
        except ValueError:                              # JSONDecodeError・UnicodeDecodeError（書きかけ・0 埋め）
            _quarantine_cache(path)
            return None
        except OSError:
            if k == 3:
                raise
            time.sleep(0.3 * (k + 1))
    if not _cache_ok(rec, key):
        _quarantine_cache(path)
        return None
    return rec


def _replace(tmp, path, tries: int = 8) -> None:
    for k in range(tries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:                         # 読んでいる最中の相手がいる（Windows）。少し待つ
            time.sleep(0.2 * (k + 1))
    raise OSError(f"計画役のキャッシュ {path} を置き換えられない（ほかのプロセスが開いている）")


def _store_cache(path, rec: dict, overwrite: bool) -> dict:
    """rec を置き、実際にキャッシュにある写しを返す（先客がいれば先客）。"""
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        if overwrite:
            _replace(tmp, path)
            return rec
        for _ in range(2):
            try:
                os.link(tmp, path)                      # 置き先があれば FileExistsError（先に書いた者が勝つ）
                return rec
            except FileExistsError:
                other = _read_cache(path, rec["key"])   # 読めない先客は退避されるので、もう 1 回置く
                if other is not None:
                    other["from_cache"] = True
                    return other
            except OSError:                             # ハードリンクを作れないファイルシステム: 有無を見てから置き換える
                other = _read_cache(path, rec["key"])
                if other is not None:
                    other["from_cache"] = True
                    return other
                _replace(tmp, path)
                return rec
        raise OSError(f"計画役のキャッシュ {path} を置けない（読めない写しを退避できない）")
    finally:
        try:
            tmp.unlink()
        except OSError:                                 # 置き換え済み（無い）か、読み手が開いている。*.tmp は読まれない
            pass


def decompose(text: str, table_colors, box_colors, cli=None, use_cache: bool = True, model: str = DEFAULT_MODEL) -> dict:
    p = _legacy._CFG["planner"]
    msg = user_message(text, table_colors, box_colors)
    key = cache_key(model, msg)
    path = CACHE / f"{key}.json"
    rec = _read_cache(path, key) if use_cache else None
    if rec is not None:
        rec["from_cache"] = True
    else:
        cli = cli or client()
        t0 = time.perf_counter()
        # 温度は送らない（extra_body も使わない）。思考は disabled を明示する
        r = cli.messages.create(model=model, max_tokens=int(p["max_tokens"]),
                                system=SYSTEM, messages=[{"role": "user", "content": msg}],
                                output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
                                thinking=dict(THINKING))
        raw = next((b.text for b in r.content if b.type == "text"), "")
        rec = {"key": key, "model": r.model, "request": {"system_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
                                                          "user": msg, "prompt_version": PROMPT_VERSION,
                                                          "requested_model": model, "temperature": None,
                                                          "thinking": dict(THINKING)},
               "raw": raw, "stop_reason": r.stop_reason, "request_id": getattr(r, "_request_id", None),
               "usage": {"input_tokens": r.usage.input_tokens, "output_tokens": r.usage.output_tokens},
               "wall_s": round(time.perf_counter() - t0, 3), "written": time.strftime("%Y-%m-%d %H:%M:%S"),
               "from_cache": False, "planner_variant": PLANNER_VARIANT}
        rec = _store_cache(path, rec, overwrite=not use_cache)
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
            "usage": rec.get("usage"), "stop_reason": rec.get("stop_reason"),
            "planner_variant": PLANNER_VARIANT, "model": rec.get("model")}
