"""YouTube のサムネイル（1280×720）を作る。見た目は動画と同じ（白地に黒の文字、青＝うまくいった、赤＝失敗）。

映像は動画の場面 s318_L2v2_c（落下 → LLM の組み直し → 3 個とも箱へ）、数は configs/demo/video_s4.yaml から読む。

    python scripts\\thumbnail.py            # paper/build/サムネ_A.png・_B.png・_C.png・_C_rl.png

案 A: 左に言葉、右に「落とした」「やり直して 3 個とも箱へ」の 2 フレームを縦に
案 B: 上に言葉、下に 2 フレームを横に並べる
案 C: 左に言葉と、3 個とも箱に入った数の棒、右に 3 個とも入ったフレーム（_C_rl は技術の言葉だけ替えた版）
"""
import importlib.util
import pathlib
import sys

from PIL import Image, ImageDraw

ROOT = pathlib.Path(__file__).resolve().parents[1]
_s = importlib.util.spec_from_file_location("video_s4_for_thumb", ROOT / "scripts" / "video_scenes.py")
S = importlib.util.module_from_spec(_s)
_s.loader.exec_module(S)
V, text = S.V, S.text
PAPER, INK, SUB, BLUE, RED, BAR0, TRACK = S.PAPER, S.INK, S.SUB, S.BLUE, S.RED, S.BAR0, S.TRACK
TW, TH = 1280, 720
CLIP = "s318_L2v2_c/trial_00.mp4"
T_DROP, T_DONE = 42.1, 86.0                         # 緑が手から落ちている所・3 個とも箱に入った後（映像の時刻 s）
DROP_AT = (621, 374)                                # 落ちている緑の立方体の中心（発表用カメラ 1280×720 の画素）
HEAD = ["落としても、", "やり直して", "最後まで片付ける"]


def shot(t, box):
    """映像のフレームを box=(x0, y0, x1, y1) で切り出す。"""
    return S.still(CLIP, t).crop(box)


def ring(d, c, r, col, ox=0, oy=0, k=1.0):
    """落ちた立方体を囲む輪（赤＝失敗）。c は元の画素、(ox, oy) と k は貼った位置と倍率。"""
    x, y = ox + c[0] * k, oy + c[1] * k
    d.ellipse((x - r, y - r, x + r, y + r), outline=col, width=7)


def label(d, x, y, s, size, col):
    """フレームの左上に、白地の四角に色の文字で札を付ける。"""
    w = d.textlength(s, font=V.font(size, True))
    d.rectangle((x, y, x + w + 32, y + size + 24), fill=PAPER)
    text(d, (x + 16, y + 10), s, size, col, True)


def paste(im, d, img, x, y, w, h, col):
    im.paste(img.resize((w, h), Image.LANCZOS), (x, y))
    d.rectangle((x - 6, y - 6, x + w + 5, y + h + 5), outline=col, width=6)


def case_a():
    im = Image.new("RGB", (TW, TH), PAPER)
    d = ImageDraw.Draw(im)
    box = (250, 130, 1010, 480)                     # 腕の先・落ちる緑・箱が入る横長の範囲
    x, w = 620, 620
    h = int(w * (box[3] - box[1]) / (box[2] - box[0]))
    y1, y2 = 40, TH - 40 - h
    k = w / (box[2] - box[0])
    paste(im, d, shot(T_DROP, box), x, y1, w, h, RED)
    ring(d, (DROP_AT[0] - box[0], DROP_AT[1] - box[1]), 34, RED, x, y1, k)
    label(d, x, y1, "落とした", 36, RED)
    paste(im, d, shot(T_DONE, box), x, y2, w, h, BLUE)
    label(d, x, y2, "やり直して 3 個とも箱へ", 36, BLUE)
    y = 70
    for s in HEAD:
        text(d, (52, y), s, 76, INK, True, maxw=540)
        y += 104
    d.rectangle((52, y + 12, 560, y + 16), fill=INK)
    text(d, (52, y + 44), "VLA ＋ LLM のロボット", 40, INK, True, maxw=520)
    text(d, (52, y + 100), "シミュレーション・Panda", 32, SUB, False, maxw=520)
    return im


def case_b():
    im = Image.new("RGB", (TW, TH), PAPER)
    d = ImageDraw.Draw(im)
    text(d, (52, 36), "落としても、やり直して", 84, INK, True, maxw=TW - 104)
    text(d, (52, 140), "最後まで片付ける", 84, BLUE, True, maxw=TW - 104)
    box = (300, 120, 940, 500)
    w, gap = 576, 24
    h = int(w * (box[3] - box[1]) / (box[2] - box[0]))
    y = TH - 36 - h
    x1, x2 = 52, 52 + w + gap
    k = w / (box[2] - box[0])
    paste(im, d, shot(T_DROP, box), x1, y, w, h, RED)
    ring(d, (DROP_AT[0] - box[0], DROP_AT[1] - box[1]), 32, RED, x1, y, k)
    label(d, x1, y, "落とした", 34, RED)
    paste(im, d, shot(T_DONE, box), x2, y, w, h, BLUE)
    label(d, x2, y, "3 個とも箱へ", 34, BLUE)
    return im


def case_c(tag="VLA ＋ LLM"):
    vals, _ = S.load_spec()
    n, a, b = vals["clc_n"], vals["clc_l2v2"], vals["clc_n0"]
    im = Image.new("RGB", (TW, TH), PAPER)
    d = ImageDraw.Draw(im)
    box = (330, 40, 930, 520)
    w = 600
    h = int(w * (box[3] - box[1]) / (box[2] - box[0]))
    x, y = TW - 40 - w, (TH - h) // 2
    paste(im, d, shot(T_DONE, box), x, y, w, h, BLUE)
    label(d, x, y, "3 個とも箱へ", 36, BLUE)
    yy = 40
    for s in HEAD:
        text(d, (52, yy), s, 70, INK, True, maxw=540)
        yy += 92
    d.rectangle((52, yy + 6, 560, yy + 10), fill=INK)
    text(d, (52, yy + 26), tag, 42, INK, True, maxw=508)
    yy += 110
    text(d, (52, yy), f"3 個とも片付いた数（{n} 回中）", 30, INK, True, maxw=540)
    bw = 380
    for name, v, col in (("回復あり", a, BLUE), ("回復なし", b, BAR0)):
        yy += 56
        text(d, (52, yy), name, 30, col if col != BAR0 else INK, True)
        by = yy + 42
        d.rectangle((52, by, 52 + bw, by + 44), fill=TRACK)
        d.rectangle((52, by, 52 + int(bw * v / n), by + 44), fill=col)
        text(d, (52 + bw + 16, by - 2), str(v), 44, col if col != BAR0 else INK, True)
        yy += 60
    return im


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:                                    # noqa: BLE001
        pass
    for name, f in (("A", case_a), ("B", case_b), ("C", case_c),
                    ("C_rl", lambda: case_c("模倣学習 ＋ 強化学習 ＋ LLM"))):   # 技術の言葉だけ替えた版
        p = S.BUILD / f"サムネ_{name}.png"
        f().save(p, optimize=True)
        print(p, f"{p.stat().st_size / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
