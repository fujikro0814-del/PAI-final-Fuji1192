"""提出用の動画（1920×1080・30 fps、読み上げつき）を組み立てる。

    .venv\\Scripts\\python.exe scripts\\video_final.py build     # paper/build/動画_PAI最終課題_<アカウント名>.mp4
    .venv\\Scripts\\python.exe scripts\\video_final.py check     # 画面の文字に使わない語・長さ・倍速
    .venv\\Scripts\\python.exe scripts\\video_final.py stills --times 0,5,10

場面の並び・長さ・読み上げの文は configs/demo/video_v5.yaml。文字・映像・左右比較・横棒・ブロック線図の部品は
scripts/video_scenes.py を読み込んで使う。このファイルが持つのは、冒頭の 1 本、立て直しデータと自分の失敗から学び直す図、
配置位置の図、数の 1 画面、出来上がった場面の mp4・静止画の差し込み、今後の課題、終わりの場面と、並べ方・読み上げ。
見た目は video_scenes と同じ（白地・左揃え・青＝うまくいった、赤＝失敗、同時に出す文字は 3 つまで、倍速は 2 倍まで・画面に書く）。
"""
import argparse
import importlib.util
import json
import pathlib
import re

import yaml
from PIL import Image, ImageDraw

ROOT = pathlib.Path(__file__).resolve().parents[1]
_s = importlib.util.spec_from_file_location("video_s4_for_v5", ROOT / "scripts" / "video_scenes.py")
S = importlib.util.module_from_spec(_s)
_s.loader.exec_module(S)
V, text, header, note, wrap = S.V, S.text, S.header, S.note, S.wrap
W, H, FPS, M = S.W, S.H, S.FPS, S.M
PAPER, INK, SUB, BLUE, RED = S.PAPER, S.INK, S.SUB, S.BLUE, S.RED
SPEC = ROOT / "configs" / "demo" / "video_v5.yaml"
OUT_NAME = S.OUT_NAME                                # 動画_PAI最終課題_<アカウント名>.mp4
TEXTS_NAME = "video_texts.json"                      # 画面の文字と読み上げの一覧（check と check_submission.py が読む）
BUILD = S.BUILD
# 読み上げ（ずんだもん）。手元の VOICEVOX のエンジン（127.0.0.1:50021）で作り、文ごとに outputs/demo_v4/_voice_v5 に
# 取っておく。VOICEVOX の利用規約に従い、動画と概要欄に「VOICEVOX:ずんだもん」と書く。
VOICE_URL = "http://127.0.0.1:50021"
SPEAKER, SPEED, SR = 3, 1.1, 24000                 # ずんだもん ノーマル、少し速め
VOICE_DIR = ROOT / "outputs" / "demo_v4" / "_voice_v5"
VOICE_CREDIT = "音声: VOICEVOX:ずんだもん"


def tts(s: str):
    """文 1 つを読み上げた音（float32、SR Hz）。同じ文・声・速さなら取っておいたものを使う。"""
    import hashlib
    import io
    import urllib.parse
    import urllib.request
    import wave

    import numpy as np
    VOICE_DIR.mkdir(parents=True, exist_ok=True)
    p = VOICE_DIR / (hashlib.sha1(f"{SPEAKER}|{SPEED}|{s}".encode("utf-8")).hexdigest()[:16] + ".wav")
    if not p.is_file():
        q = urllib.parse.urlencode({"text": s, "speaker": SPEAKER})
        aq = json.load(urllib.request.urlopen(urllib.request.Request(f"{VOICE_URL}/audio_query?{q}", method="POST")))
        aq["speedScale"], aq["outputSamplingRate"], aq["outputStereo"] = SPEED, SR, False
        req = urllib.request.Request(f"{VOICE_URL}/synthesis?speaker={SPEAKER}", data=json.dumps(aq).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
        p.write_bytes(urllib.request.urlopen(req).read())
    with wave.open(io.BytesIO(p.read_bytes())) as w:
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    return a


def voice_track(order):
    """場面ごとの読み上げ（yaml の voice: [{t: 場面の中の秒, text}]）を 1 本の音にする。はみ出し・重なりは problems に。"""
    import numpy as np
    total = sum(float(sc["dur_s"]) for sc in order)
    track = np.zeros(int(round(total * SR)) + SR, np.float32)
    problems, t0 = [], 0.0
    for sc in order:
        end_prev = 0.0
        for ln in sc.get("voice") or []:
            a = tts(ln["text"])
            t, dur = float(ln["t"]), len(a) / SR
            if t < end_prev + 0.2:
                problems.append(f"場面 {sc['id']} の読み上げが重なる（{t:.1f} s、前の文は {end_prev:.1f} s まで）")
            if t + dur > float(sc["dur_s"]) + 1e-6:
                problems.append(f"場面 {sc['id']} の読み上げ「{ln['text'][:12]}…」が {t + dur:.1f} s まで（場面は {sc['dur_s']} s）")
            i = int(round((t0 + t) * SR))
            track[i:i + len(a)] += a
            end_prev = t + dur
            V.TEXTS.append(ln["text"])
        t0 += float(sc["dur_s"])
    return np.clip(track, -1, 1), problems


def load_spec():
    return yaml.safe_load(SPEC.read_text(encoding="utf-8"))


def n_frames(sc) -> int:
    return int(round(float(sc["dur_s"]) * FPS))


# ------------------------------------------------------------------ 新しい場面
def single(sc, title):
    """1 本の映像を大きく（左揃え）。上に題と倍速、映像の上にモデルの名前、下に経過の一言。成功したら青の枠。"""
    clip = V.Clip(sc["clip"])
    at = S.timing(sc, [clip])
    y0, y1, x0, x1 = sc.get("crop") or V.CROP
    vw, vh = 1344, 756
    py = 196

    def f(j):
        t, sp = at(j)
        im = Image.new("RGB", (W, H), PAPER)
        im.paste(Image.fromarray(clip.at(t)[y0:y1, x0:x1]).resize((vw, vh)), (M, py))
        d = ImageDraw.Draw(im)
        header(d, title, sp)
        text(d, (M, 146), sc["label"], 34, INK, True, maxw=vw)
        if sc.get("done") is not None and t >= float(sc["done"]):
            S.frame(d, M, py, vw, vh, BLUE)
        e = V._events(sc.get("events"), t)
        if e:
            col = {"ok": BLUE, "ng": RED}.get(e.get("kind"), INK)
            text(d, (M + vw + 48, py + 20), e["text"], 48, col, True, maxw=W - 2 * M - vw - 48)
        note(d)
        return im
    return n_frames(sc), f


class Movie:
    """出来上がった場面の mp4 を、フレームの順に読む。映像より長い場面は最後のフレームで止める。"""

    def __init__(self, path):
        import av
        self.c = av.open(str(path))
        self.it = self.c.decode(video=0)
        self.last = None

    def next(self):
        try:
            fr = next(self.it)
            self.last = fr.to_image().convert("RGB").resize((W, H))
        except StopIteration:
            pass
        return self.last


def movie(sc):
    mv = Movie(sc["path"])
    return n_frames(sc), lambda j: mv.next().copy()


def image(sc):
    im0 = Image.open(sc["path"]).convert("RGB").resize((W, H))
    return n_frames(sc), lambda j: im0.copy()


def steps_card(sc, title, boxes, rec_label, foot):
    """4 つの箱と矢印（最後が青）、その下に記録する範囲を示す線と一言。立て直しデータと学び直しの図で同じ形にそろえる。"""
    bw, gap, by, bh = 410, 50, 220, 300

    def f(j):
        im = Image.new("RGB", (W, H), PAPER)
        d = ImageDraw.Draw(im)
        header(d, title)
        for k, (head, body) in enumerate(boxes):
            x = M + k * (bw + gap)
            last = k == len(boxes) - 1
            d.rectangle((x, by, x + bw, by + bh), fill=BLUE if last else PAPER, outline=BLUE if last else INK, width=3)
            col = PAPER if last else INK
            text(d, (x + 24, by + 24), head, 40, col, True, maxw=bw - 48)
            for i, s in enumerate(body):
                text(d, (x + 24, by + 110 + i * 54), s, 31, col, False, maxw=bw - 40)
            if not last:
                ax, ay = x + bw + 8, by + bh // 2
                d.rectangle((ax, ay - 3, ax + gap - 30, ay + 3), fill=INK)
                d.polygon([(ax + gap - 30, ay - 14), (ax + gap - 16, ay), (ax + gap - 30, ay + 14)], fill=INK)
        rec_x = M + 3 * (bw + gap)
        ty = by + bh + 60
        for x in range(M, rec_x - 20, 36):                       # 記録しない範囲（破線）
            d.rectangle((x, ty + 30, x + 18, ty + 33), fill=S.BAR0)
        text(d, (M, ty - 8), "記録しない", 32, SUB, False)
        d.rectangle((rec_x, ty, W - M, ty + 70), fill=BLUE)
        text(d, (rec_x + 20, ty + 16), rec_label, 36, PAPER, True, maxw=W - M - rec_x - 40)
        y = ty + 190
        for s in foot:
            text(d, (M, y), s, 40, INK, False, maxw=W - 2 * M)
            y += 66
        return im
    return n_frames(sc), f


def recipe_card(sc):
    """立て直しデータの作り方の図。"""
    return steps_card(sc, "＋失敗から立て直す学習データも足す: 生成プログラムにわざと失敗させる", [
        ("ふつうに運ぶ", ["生成プログラムが", "掴んで", "箱へ運ぶ"]),
        ("わざと失敗", ["掴む位置か高さを", "ずらして閉じる・", "机の上で放す"]),
        ("失敗の確定", ["持ち上がらない・", "机の上で止まった", "ことを確かめる"]),
        ("立て直して箱へ", ["生成プログラムが", "掴み直して", "箱に入れるまで"])],
        "立て直しの学習データとして記録",
        ["生成プログラム: シミュレーションの正確な位置を使って動く（学習なし）",
         "成功 240 本＋立て直し 90 本の学習データで VLA を学習（人の操作なし）"])


def r1_card(sc):
    """自分の失敗から学び直す流れの図（立て直しデータの図と同じ形）。"""
    return steps_card(sc, "＋自分の失敗から学び直す: VLA が失敗した所から生成プログラムが立て直す", [
        ("VLA を動かす", ["学習用の初期状態で", "VLA を走らせる", ""]),
        ("VLA の失敗", ["掴み損ねた・", "落とした直後に", "止める"]),
        ("プログラムに交代", ["生成プログラムが", "その場から", "立て直す"]),
        ("立て直して箱へ", ["箱に入れるまで", "", ""])],
        "立て直しの学習データとして記録",
        ["VLA が実際に失敗した場面からの立て直しを学習データに足して、VLA を学び直す（人の操作なし）"])


def place_card(sc):
    """配置位置による目標条件付けの図。左に残った失敗（箱の中の物の上に置いて載ったまま）、右に 2×2 のセルと選んだセル。
    箱を上から見た模式図で、立方体は灰色（色は意味のある色だけ）。右は reveal_s 秒から出す。"""
    bs, cs = 380, 92                                   # 箱の一辺・立方体の一辺（画素）
    reveal = float(sc.get("reveal_s", 6.0))
    lx, rx, by = M + 40, 1010 + 40, 330

    def cube(d, cx, cy, out=INK):
        d.rectangle((cx - cs // 2, cy - cs // 2, cx + cs // 2, cy + cs // 2), fill=(190, 190, 190), outline=out, width=4)

    def f(j):
        t = j / FPS
        im = Image.new("RGB", (W, H), PAPER)
        d = ImageDraw.Draw(im)
        header(d, "＋配置位置による目標条件付け: 箱の空きセルを VLA に与えて学び直す")
        # 左: 残った失敗
        text(d, (M, 170), "残った失敗 30 本のうち 18 本（600 通りの初期状態）", 34, INK, True, maxw=900)
        d.rectangle((lx, by, lx + bs, by + bs), outline=INK, width=5)
        c0 = (lx + bs // 4, by + 3 * bs // 4)
        cube(d, *c0)
        cube(d, c0[0] + 28, c0[1] - 26, RED)              # 先の物に重ねて置いた
        text(d, (lx + bs + 30, by + 40), "収納済みの物体の", 34, RED, True, maxw=480)
        text(d, (lx + bs + 30, by + 92), "上で放し、", 34, RED, True, maxw=480)
        text(d, (lx + bs + 30, by + 144), "載ったまま静止する", 34, RED, True, maxw=480)
        text(d, (lx, by + bs + 24), "箱を上から見た図", 26, SUB)
        if t >= reveal:
            text(d, (1010, 170), "空きセルを配置位置として与える", 34, INK, True, maxw=W - M - 1010)
            d.rectangle((rx, by, rx + bs, by + bs), outline=INK, width=5)
            for x in range(rx, rx + bs, 24):                  # 2×2 のセルの境（破線）
                d.rectangle((x, by + bs // 2 - 1, x + 12, by + bs // 2 + 1), fill=S.BAR0)
            for y in range(by, by + bs, 24):
                d.rectangle((rx + bs // 2 - 1, y, rx + bs // 2 + 1, y + 12), fill=S.BAR0)
            cube(d, rx + bs // 4, by + 3 * bs // 4)
            gx, gy = rx + 3 * bs // 4, by + bs // 4          # 入っている物から最も遠いセル
            d.rectangle((gx - cs // 2, gy - cs // 2, gx + cs // 2, gy + cs // 2), outline=BLUE, width=8)
            d.ellipse((gx - 12, gy - 12, gx + 12, gy + 12), fill=BLUE)
            ax = rx + bs + 20
            d.rectangle((gx + cs // 2 + 8, gy - 3, ax + 60, gy + 3), fill=BLUE)
            d.polygon([(ax + 60, gy - 14), (ax + 76, gy), (ax + 60, gy + 14)], fill=BLUE)
            d.rectangle((ax + 84, gy - 50, ax + 290, gy + 50), outline=BLUE, width=4)
            text(d, (ax + 120, gy - 26), "VLA", 44, BLUE, True)
            text(d, (ax + 84, gy + 70), "配置位置", 34, BLUE, True, maxw=300)
            text(d, (ax + 84, gy + 118), "（収納済みの物体", 30, INK, False, maxw=320)
            text(d, (ax + 84, gy + 160), "から最も遠いセル）", 30, INK, False, maxw=320)
            y = by + bs + 90
            for s in ["実行時: 推定した箱を 2×2 のセルに分け、収納済みの物体から最も遠いセルを与える",
                      "学習: データは追加せず（同じ 670 本）、物体が最終的に静止した位置を配置位置とする"]:
                text(d, (M, y), s, 36, INK, False, maxw=W - 2 * M)
                y += 60
        return im
    return n_frames(sc), f


def limits_card(sc):
    """最後の場面: 今後の課題（重要な順）。読み上げの表記（VOICEVOX の利用規約）も下に書く。"""
    n, f0 = S.list_card(sc, "今後の課題（重要な順）", [
        "実機での評価（今は、シミュレーションで出した腕の動きを実機で再生しただけ）",
        "手首の向きを変える動き（斜めに置かれた物に弱い）",
        "色以外での物の見分け（同じ色の物が複数・立方体以外の物）",
        "布や食品のような形の変わる物の学習データの作り方"], numbered=True)

    def f(j):
        im = f0(j)
        text(ImageDraw.Draw(im), (M, H - 70), VOICE_CREDIT, 28, SUB)
        return im
    return n, f


def numbers_card(sc):
    """数の 1 画面（横棒）。yaml: headline、rows [{name, value, total, shown, kind: ok|ng|plain（灰）, note}]、
    bar_w（棒の欄の幅、既定は全幅）、note_head（棒の右の note の見出し）、lines（棒の下の文）、foot（下の灰色の注）。"""
    col = {"ok": BLUE, "ng": RED}
    rows = [(r["name"], float(r["value"]), col.get(r.get("kind"), S.BAR0), r.get("shown", str(r["value"]))) for r in sc["rows"]]
    total = float(sc["rows"][0]["total"])
    bw = int(sc.get("bar_w", W - 2 * M))
    size, gap, num = (40, 160, 56) if len(rows) <= 2 else (34, 130, 46)
    y0, nx = 180, M + bw + 60

    def f(j):
        im = Image.new("RGB", (W, H), PAPER)
        d = ImageDraw.Draw(im)
        header(d, sc["headline"])
        S.bars(d, M, y0, bw, rows, total, size=size, gap=gap, num=num)
        if sc.get("note_head"):
            text(d, (nx, y0), sc["note_head"], 30, SUB, True, maxw=W - M - nx)
        for k, r in enumerate(sc["rows"]):
            if r.get("note"):
                ng = r.get("kind") == "ng"
                yy = y0 + k * gap + size + 14 + (int(num * 1.1) - 34) // 2 - 4
                text(d, (nx, yy), r["note"], 34, RED if ng else INK, ng, maxw=W - M - nx)
        y = y0 + gap * len(rows) + 50
        for s in sc.get("lines") or []:
            text(d, (M, y), s, 40, INK, True, maxw=W - 2 * M)
            y += 72
        if sc.get("foot"):
            text(d, (M, H - 58), sc["foot"], 26, SUB)
        return im
    return n_frames(sc), f


def diagram_with_note(sc):
    """ブロック線図の場面（video_scenes の diagram_card）に、図の全体は説明資料にある旨の一行を右下に足す。"""
    n, f0 = S.diagram_card(sc)
    note_s = sc.get("note", "図の全体は説明資料（図 2）に掲載")

    def f(j):
        im = f0(j)
        text(ImageDraw.Draw(im), (W - M, H - 58), note_s, 28, SUB, False, "ra")
        return im
    return n, f


def closing_card(sc, title, img):
    im0 = img.resize((900, 506))

    def f(j):
        im = Image.new("RGB", (W, H), PAPER)
        im.paste(im0, (W - M - 900, 250))
        d = ImageDraw.Draw(im)
        a, b = title.split("、", 1) if "、" in title else (title, "")
        text(d, (M + 16, 250), a + ("、" if b else ""), 68, INK, True, maxw=860)
        if b:
            text(d, (M + 16, 340), b, 68, INK, True, maxw=860)
        text(d, (M + 16, 500), "指示「全部片付けて」で、立方体を箱へ（シミュレーション）", 32, SUB, False, maxw=860)
        text(d, (M + 16, H - 90), VOICE_CREDIT, 28, SUB, False)
        return im
    return n_frames(sc), f


# ------------------------------------------------------------------ 並べる
def scenes(spec):
    title = spec["title"]
    order = spec["scenes"]
    by = {sc["id"]: sc for sc in order}
    cover = S.still(by["llm"]["clip_with"], 86.0, by["llm"].get("crop"))
    make = {
        "single": lambda sc: single(sc, title),
        "movie": movie,
        "image": image,
        "recipe_v5": recipe_card,
        "r1": r1_card,
        "place": place_card,
        "compare": S.compare,
        "numbers": numbers_card,
        "diagram": diagram_with_note,
        "limits": limits_card,
        "closing": lambda sc: closing_card(sc, title, cover),
    }
    return [make[sc["kind"]](sc) for sc in order], order


def cmd_build(a) -> None:
    import av
    import numpy as np
    spec = load_spec()
    sc, order = scenes(spec)
    track, vprob = voice_track(order)
    if vprob:
        raise SystemExit("読み上げの時間が合わない:\n" + "\n".join(vprob))
    total = sum(n for n, _ in sc)
    out = BUILD / "video_v5.mp4"
    box = av.open(str(out), mode="w")
    st = box.add_stream("libx264", rate=FPS)
    st.width, st.height, st.pix_fmt = W, H, "yuv420p"
    st.options = {"crf": "22", "preset": "medium", "threads": "4"}
    au = box.add_stream("aac", rate=SR)
    au.layout = "mono"
    FS = 1024                                        # AAC の 1 フレームの標本数
    apos = 0

    def push_audio(upto):
        nonlocal apos
        while apos + FS <= upto:
            fr = av.AudioFrame.from_ndarray(track[apos:apos + FS].reshape(1, -1), format="fltp", layout="mono")
            fr.sample_rate, fr.pts = SR, apos
            for p in au.encode(fr):
                box.mux(p)
            apos += FS

    t, k = 0.0, 0
    for (n, f), s in zip(sc, order):
        print(f"{t:6.1f} s  {s['id']}", flush=True)
        for j in range(n):
            for p in st.encode(av.VideoFrame.from_image(f(j))):
                box.mux(p)
            k += 1
            push_audio(int(k * SR / FPS))                # 映像と音を交互に書く
        t += n / FPS
    push_audio(int(total * SR / FPS))
    for p in st.encode():
        box.mux(p)
    for p in au.encode():
        box.mux(p)
    box.close()
    (BUILD / OUT_NAME).write_bytes(out.read_bytes())
    (BUILD / TEXTS_NAME).write_text(json.dumps({"seconds": total / FPS, "texts": sorted(set(V.TEXTS))},
                                               ensure_ascii=False, indent=1), encoding="utf-8")
    print(BUILD / OUT_NAME, f"{total / FPS:.1f} s", f"{out.stat().st_size / 1e6:.1f} MB")


def cmd_check(a) -> None:
    s = importlib.util.spec_from_file_location("paper", ROOT / "scripts" / "paper_pdf.py")
    pm = importlib.util.module_from_spec(s)
    s.loader.exec_module(pm)
    info = json.loads((BUILD / TEXTS_NAME).read_text(encoding="utf-8"))
    problems = [f"使わない語 {m.group(0)!r}（{p}）: {s[:30]}" for p in pm.FORBIDDEN for s in info["texts"]
                for m in re.finditer(p, s)]                  # 文ごとに見る（つなぐと、前の文の末尾と次の文の頭で誤って当たる）
    if not 60 <= info["seconds"] <= 180:
        problems.append(f"長さ {info['seconds']:.0f} s（1〜3 分の外）")
    order = load_spec()["scenes"]
    problems += voice_track(order)[1]
    want = sum(float(sc["dur_s"]) for sc in order)
    if abs(info["seconds"] - want) > 0.5:
        problems.append(f"長さ {info['seconds']:.1f} s が場面の表の合計 {want:.1f} s と違う")
    print(json.dumps({"seconds": round(info["seconds"], 1), "texts": len(info["texts"]), "problems": problems},
                     ensure_ascii=False, indent=1))
    raise SystemExit(1 if problems else 0)


def cmd_stills(a) -> None:
    import av
    out = ROOT / "outputs" / "demo_v4" / "_review_v5"
    out.mkdir(parents=True, exist_ok=True)
    want = sorted(float(x) for x in a.times.split(","))
    c = av.open(str(BUILD / OUT_NAME))
    k = 0
    for i, fr in enumerate(c.decode(video=0)):
        while k < len(want) and i >= int(round(want[k] * FPS)):
            p = out / f"v5_{want[k]:05.1f}s.png"
            fr.to_image().save(p)
            print(p)
            k += 1
        if k >= len(want):
            break
    c.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    sub.add_parser("check")
    p = sub.add_parser("stills")
    p.add_argument("--times", default="0,5,10,15,20,25")
    a = ap.parse_args(argv)
    {"build": cmd_build, "check": cmd_check, "stills": cmd_stills}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
