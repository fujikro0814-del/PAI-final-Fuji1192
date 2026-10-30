"""指示の分解を「色で分ける」いまの版と「物に番号を振って分ける」版で比べる（LLM だけ、ロボットは動かさない）。

    .venv\\Scripts\\python.exe scripts\\api\\check_object_decompose.py          # 20 の場面を両方の版で 1 回ずつ分ける
    → docs/api/object_decompose_check.md（場面ごとの答えと正否）

場面は机の上の物（種類・色・位置）と箱の中の物を書いたもの。同じ色が複数・立方体以外の物・位置で指す指示を含む。
- 色の版: いまの実行系の decompose_haiku。入力は知覚と同じく色の集合（赤・緑・青だけ。物の種類は見えない）。
  答えの色は、その色の立方体のどれか 1 個に当てる（VLA への指示が "put the {c} cube in the box" のため）。当て方のどれかで
  期待に合えば正しいとする（色の版に甘い数え方）。
- 番号の版: 物ごとに番号・種類・色・位置を渡し、番号の列で答えさせる。規則の検査（机の上の番号だけ・重複なし）に通らなければ
  聞き返す扱い。モデル・呼び方は色の版と同じ（Claude Haiku 5.5、温度なし、思考なし、JSON スキーマ）。
応答は outputs/llm_cache/ に残し、同じ入力では再利用する。
"""
import hashlib
import itertools
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from recovla.planner import decompose_haiku as D4  # noqa: E402

OUT = ROOT / "docs" / "api" / "object_decompose_check.md"
MODEL = D4.DEFAULT_MODEL
PROMPT_VERSION = 1
SYSTEM = """あなたはロボットの作業計画の係です。ロボットは机の上の物を 1 個ずつ箱へ入れます。
入力には、カメラで見つけた物の一覧（objects）があります。各物は番号（id）・種類（kind）・色（color）・位置（x_cm, y_cm）・
場所（where: table は机の上、box は箱の中）を持ちます。x_cm はロボットからの前方への距離（小さいほど手前）、
y_cm は左右（大きいほど左）です。
人の日本語の指示を読み、箱へ入れる物の番号を、入れる順に並べた列（steps）にしてください。

決まり:
- 手順に入れてよいのは、机の上（table）にある物の番号だけです。箱の中の物は入れません。同じ番号を 2 回入れません
- 「全部」「残り全部」なら、机の上の物を全部入れます。順番の指定がなければ、どの順でもかまいません
- 順番・種類・色・位置の指定があれば、それに従います
- 指示に合う物が机の上にない、何をすればよいか分からない、箱へ入れる以外の頼み（積む、並べるなど）のときは、
  steps を空の列にして、reply で人に分かりやすく尋ねるか、できない理由を伝えてください
- 一部だけ実行できる場合（例: 2 個頼まれて 1 個しかない）は、実行せずに steps を空にして reply で確かめてください
- reply は日本語の短い文にします
"""
SCHEMA = {"type": "object",
          "properties": {"steps": {"type": "array", "items": {"type": "integer"}}, "reply": {"type": "string"}},
          "required": ["steps", "reply"], "additionalProperties": False}
COLOR_EN = {"赤": "red", "緑": "green", "青": "blue"}


def O(i, kind, color, x, y, where="table"):
    return {"id": i, "kind": kind, "color": color, "x_cm": x, "y_cm": y, "where": where}


# (指示, 物, 期待)。期待: set=入れる物の集まり、first=最初の物、seq=順番どおり、one_of=どれか 1 個、ask=聞き返す
CASES = [
    ("全部片付けて", [O(1, "立方体", "赤", 45, 10), O(2, "立方体", "緑", 50, -5), O(3, "立方体", "青", 40, -15)], {"set": [1, 2, 3]}),
    ("全部片付けて", [O(1, "立方体", "赤", 45, 10), O(2, "立方体", "赤", 50, -5), O(3, "立方体", "青", 40, -15)], {"set": [1, 2, 3]}),
    ("全部片付けて", [O(1, "立方体", "赤", 45, 10), O(2, "立方体", "赤", 50, -5), O(3, "立方体", "赤", 40, -15)], {"set": [1, 2, 3]}),
    ("全部片付けて", [O(1, "立方体", "赤", 45, 10), O(2, "立方体", "緑", 50, -5), O(3, "立方体", "青", 60, 25, "box")], {"set": [1, 2]}),
    ("全部片付けて", [O(1, "立方体", "赤", 45, 10), O(2, "立方体", "赤", 50, -5), O(3, "立方体", "赤", 60, 25, "box")], {"set": [1, 2]}),
    ("赤いのを全部片付けて", [O(1, "立方体", "赤", 45, 10), O(2, "立方体", "赤", 50, -5), O(3, "立方体", "緑", 40, -15)], {"set": [1, 2]}),
    ("赤い立方体を 1 個だけ箱に入れて", [O(1, "立方体", "赤", 45, 10), O(2, "立方体", "赤", 50, -5), O(3, "立方体", "青", 40, -15)], {"one_of": [1, 2]}),
    ("青を先に、そのあと残りを全部片付けて", [O(1, "立方体", "赤", 45, 10), O(2, "立方体", "緑", 50, -5), O(3, "立方体", "青", 40, -15)], {"set": [1, 2, 3], "first": 3}),
    ("立方体だけ片付けて", [O(1, "カップ", "白", 45, 15), O(2, "立方体", "赤", 50, -5), O(3, "立方体", "緑", 40, -15)], {"set": [2, 3]}),
    ("全部片付けて", [O(1, "カップ", "白", 45, 15), O(2, "立方体", "赤", 50, -5), O(3, "立方体", "緑", 40, -15)], {"set": [1, 2, 3]}),
    ("カップを片付けて", [O(1, "カップ", "赤", 45, 15), O(2, "立方体", "赤", 50, -5), O(3, "立方体", "青", 40, -15)], {"set": [1]}),
    ("カップを 2 つとも片付けて", [O(1, "カップ", "白", 45, 15), O(2, "カップ", "青", 50, -5), O(3, "立方体", "緑", 40, -15)], {"set": [1, 2]}),
    ("ボールを箱に入れて", [O(1, "ボール", "黄", 45, 15), O(2, "立方体", "赤", 50, -5)], {"set": [1]}),
    ("黄色い立方体を片付けて", [O(1, "立方体", "赤", 45, 10), O(2, "立方体", "緑", 50, -5), O(3, "立方体", "青", 40, -15)], {"ask": True}),
    ("立方体を積んで", [O(1, "立方体", "赤", 45, 10), O(2, "立方体", "緑", 50, -5)], {"ask": True}),
    ("全部片付けて", [O(1, "立方体", "赤", 60, 25, "box")], {"ask": True}),
    ("左の赤を片付けて", [O(1, "立方体", "赤", 45, 15), O(2, "立方体", "赤", 45, -15), O(3, "立方体", "青", 55, 0)], {"set": [1]}),
    ("手前から順に全部片付けて", [O(1, "立方体", "赤", 50, 10), O(2, "立方体", "緑", 35, -5), O(3, "立方体", "青", 60, -15)], {"seq": [2, 1, 3]}),
    ("緑と青を片付けて", [O(1, "立方体", "赤", 45, 10), O(2, "立方体", "赤", 50, -5), O(3, "立方体", "緑", 40, -15), O(4, "立方体", "青", 35, 15)], {"set": [3, 4]}),
    ("赤を 2 つ入れて", [O(1, "立方体", "赤", 45, 10), O(2, "立方体", "青", 50, -5)], {"ask": True}),
]


def meets(ids, exp) -> bool:
    if exp.get("ask"):
        return not ids
    if not ids or len(ids) != len(set(ids)):
        return False
    if "one_of" in exp:
        return len(ids) == 1 and ids[0] in exp["one_of"]
    if "seq" in exp:
        return list(ids) == exp["seq"]
    return set(ids) == set(exp["set"]) and ("first" not in exp or ids[0] == exp["first"])


def color_arm(text, objs, exp):
    """いまの版。知覚と同じく色の集合だけを渡し、答えの色をその色の立方体に当てる。"""
    table = sorted({COLOR_EN[o["color"]] for o in objs if o["where"] == "table" and o["color"] in COLOR_EN})
    box = sorted({COLOR_EN[o["color"]] for o in objs if o["where"] == "box" and o["color"] in COLOR_EN})
    out = D4.decompose(text, table, box)
    steps = out["steps"]
    pools = [[o["id"] for o in objs if o["where"] == "table" and o["kind"] == "立方体" and COLOR_EN.get(o["color"]) == c]
             for c in steps]
    ok = (not steps and meets([], exp)) or any(len(set(p)) == len(p) and meets(list(p), exp) for p in itertools.product(*pools))
    return {"answer": "・".join(steps) if steps else "（聞き返す）", "reply": out["reply"], "ok": bool(ok)}


def object_arm(text, objs, exp, cli=None):
    msg = json.dumps({"instruction": text, "objects": objs}, ensure_ascii=False)
    key = hashlib.sha256(json.dumps({"model": MODEL, "thinking": D4.THINKING, "system": SYSTEM, "schema": SCHEMA, "user": msg,
                                     "prompt_version": PROMPT_VERSION, "task": "object_decompose"},
                                    ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    path = D4.CACHE / f"{key}.json"
    rec = D4._read_cache(path, key)
    if rec is None:
        cli = cli or D4.client()
        r = cli.messages.create(model=MODEL, max_tokens=512, system=SYSTEM, messages=[{"role": "user", "content": msg}],
                                output_config={"format": {"type": "json_schema", "schema": SCHEMA}}, thinking=dict(D4.THINKING))
        rec = {"key": key, "model": r.model, "raw": next((b.text for b in r.content if b.type == "text"), ""),
               "request_id": getattr(r, "_request_id", None), "written": time.strftime("%Y-%m-%d %H:%M:%S")}
        rec = D4._store_cache(path, rec, overwrite=False)
    try:
        data = json.loads(rec["raw"])
        steps, reply = [int(s) for s in data["steps"]], str(data["reply"])
    except (ValueError, KeyError, TypeError):
        steps, reply = [], ""
    on_table = {o["id"] for o in objs if o["where"] == "table"}
    if any(s not in on_table for s in steps) or len(steps) != len(set(steps)):     # 規則の検査に落ちたら聞き返す扱い
        steps = []
    return {"answer": "・".join(map(str, steps)) if steps else "（聞き返す）", "reply": reply, "ok": meets(steps, exp)}


def main():
    rows, n_c, n_o = [], 0, 0
    for k, (text, objs, exp) in enumerate(CASES, 1):
        c, o = color_arm(text, objs, exp), object_arm(text, objs, exp)
        n_c += c["ok"]
        n_o += o["ok"]
        scene = "、".join(f"{x['id']}:{x['color']}の{x['kind']}" + ("（箱）" if x["where"] == "box" else "") for x in objs)
        rows.append(f"| {k} | {text} | {scene} | {c['answer']} | {'○' if c['ok'] else '×'} | {o['answer']} | {'○' if o['ok'] else '×'} |")
        print(f"{k:2d} 色 {'o' if c['ok'] else 'x'} {c['answer']:<14} 番号 {'o' if o['ok'] else 'x'} {o['answer']}")
    head = [
        "# 指示の分解: 色で分ける版と、物に番号を振って分ける版（LLM だけの確かめ）",
        "",
        f"`scripts/api/check_object_decompose.py` が {time.strftime('%Y-%m-%d')} に書いた。モデルは両方とも {MODEL}、各場面 1 回。"
        "ロボットは動かしていない（分解の答えが指示に合うかだけを見た）。",
        "",
        f"- 色で分ける版（いまの実行系）: **{n_c}/{len(CASES)}**。入力は色の集合だけで、答えの色はその色の立方体のどれかに当てて、"
        "合う当て方が 1 つでもあれば正しいとした。",
        f"- 物に番号を振って分ける版: **{n_o}/{len(CASES)}**。入力は物ごとの番号・種類・色・位置。",
        "- 番号の版を実際に動かすには、知覚（物を 1 個ずつ見分ける）、完了の判定と組み直しの決まり（物ごとに数える）、"
        "VLA への指示（色の言葉の代わりに物を指す言い方）も変える必要がある。",
        "",
        "| # | 指示 | 場面（番号:物） | 色の版の答え | 正否 | 番号の版の答え | 正否 |",
        "|---|---|---|---|---|---|---|",
    ]
    OUT.write_text("\n".join(head + rows) + "\n", encoding="utf-8", newline="\n")
    print(f"色 {n_c}/{len(CASES)}、番号 {n_o}/{len(CASES)} → {OUT}")


if __name__ == "__main__":
    main()
