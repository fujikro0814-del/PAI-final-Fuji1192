"""README の冒頭の GIF（掴み損ね → その場で掴み直して箱へ）を、動画の冒頭と同じ試行の記録から作る。

    .venv\\Scripts\\python.exe scripts\\readme_gif.py <試行の mp4> [--out docs/figures/regrasp.gif]

動画の冒頭（video_final.py の open の場面）と同じ切り出し・同じ部分: 5〜15 s は等倍、15〜23 s は 2 倍速（右上に「×2」）。
"""
import argparse
import pathlib

import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = pathlib.Path(__file__).resolve().parents[1]
CROP = (40, 600, 120, 1116)                  # 行 40〜600、列 120〜1116（video_v5.yaml の open の crop）
SEGMENTS = [(5.0, 15.0, 1.0), (15.0, 23.0, 2.0)]
FPS_OUT, WIDTH = 8, 480
FONT = r"C:\Windows\Fonts\YuGothB.ttc"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mp4")
    ap.add_argument("--out", default=str(ROOT / "docs" / "figures" / "regrasp.gif"))
    a = ap.parse_args(argv)
    cap = cv2.VideoCapture(a.mp4)
    font = ImageFont.truetype(FONT, 22)
    frames = []
    for t0, t1, speed in SEGMENTS:
        t = t0
        while t < t1:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, f = cap.read()
            if not ok:
                break
            y0, y1, x0, x1 = CROP
            im = Image.fromarray(cv2.cvtColor(f[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
            im = im.resize((WIDTH, round(im.height * WIDTH / im.width)), Image.LANCZOS)
            if speed != 1.0:
                d = ImageDraw.Draw(im)
                d.rectangle((WIDTH - 52, 6, WIDTH - 6, 34), fill=(255, 255, 255))
                d.text((WIDTH - 48, 7), f"×{speed:g}", font=font, fill=(20, 20, 20))
            frames.append(im.convert("P", palette=Image.ADAPTIVE, colors=128))
            t += speed / FPS_OUT
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / FPS_OUT), loop=0, optimize=True)
    print(out, len(frames), "こま", f"{out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
