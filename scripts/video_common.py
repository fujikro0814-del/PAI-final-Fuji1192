"""前の版の動画（約 2 分、1920×1080、30 fps、H.264）を組み立てる。

scripts/video_scenes.py が、文字・映像の読み出し・左右比較などの部品として読む。

    .venv\\Scripts\\python.exe scripts\\video_common.py build     # paper/build/動画_PAI最終課題_<アカウント名>.mp4
    .venv\\Scripts\\python.exe scripts\\video_common.py check     # 画面の文字に使わない語（開発中の記号を含む）がないか・長さ・場面の再現
    .venv\\Scripts\\python.exe scripts\\video_common.py stills    # 冒頭の静止画を outputs/demo_v2/_review/ に書き出す（--times 0,4,…）

動画は簡易な説明にとどめ、詳しい説明は説明資料に任せる。開発中の記号は画面に出さない。同時に出す文字は 3 つまで
（見出し 1 行・左右の短いラベル・映像の下の一言か数字）と角の小さな札だけ。
場面の並びと長さは configs/demo/video_s3.yaml（説明資料の「動画 m:ss〜m:ss」も同じ表から計算する）。
映像は scripts/63_demo_v2.py で同じ設定で回し直したもの（outputs/demo_v2）で、元の記録と同じ経過のもの（meta.json の
same_course）だけを使う。字幕の数字は説明資料の値（paper/build/values.json、scripts/paper_pdf.py build）から入れる。
"""
import argparse
import json
import re
import subprocess

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from recovla.common import config

CFG = config.load_v2()
ROOT = config.ROOT
OUT = config.path(CFG["paths"]["outputs"])
DEMO = OUT / "demo_v2"
EVAL = OUT / "v2eval" / "V3DEMO"
BUILD = ROOT / "paper" / "build"
VIDEO = ROOT / "configs" / "demo" / "video_s3.yaml"
from recovla.common.account import ACCOUNT  # noqa: E402  値は account.py だけに書く
OUT_NAME = f"動画_PAI最終課題_{ACCOUNT}.mp4"
W, H, FPS = 1920, 1080, 30
FONT_R = r"C:\Windows\Fonts\BIZ-UDGothicR.ttc"
FONT_B = r"C:\Windows\Fonts\BIZ-UDGothicB.ttc"
BG, INK, MUTED = (246, 247, 244), (29, 33, 38), (91, 99, 109)
C_A, C_B, C_C = (45, 95, 139), (31, 107, 92), (149, 86, 25)
WITH, WITHOUT = "復帰デモありで学習したモデル", "復帰デモなしで学習したモデル"        # 前の版の説明資料と同じ語
RERUN_NOTE = "同じ設定で回し直した映像（テスト用の評価と同じ経過のもの）"
COLOR = {"red": "赤", "green": "緑", "blue": "青"}
TEXTS = []                                     # 画面に出した文字（check で語を調べる）


def font(size, bold=False):
    return ImageFont.truetype(FONT_B if bold else FONT_R, size)


def text(d, xy, s, size=32, fill=INK, bold=False, anchor="la", maxw=W - 80):
    """文字を書く。幅が maxw を超えるときは収まるまで字を小さくする（はみ出しを防ぐ）。"""
    TEXTS.append(s)
    while size > 14 and d.textlength(s, font=font(size, bold)) > maxw:
        size -= 1
    d.text(xy, s, font=font(size, bold), fill=fill, anchor=anchor)


def spec() -> dict:
    s = yaml.safe_load(VIDEO.read_text(encoding="utf-8"))
    return {sc["id"]: sc for sc in s["scenes"]}, s["scenes"]


class Clip:
    """outputs/demo_v2/<場面>/trial_NN.mp4 を、単調に進む時刻で読む。same_course が偽の場面は使わない。"""

    def __init__(self, ref):
        import av
        clip, trial = ref.split("/")
        k = int(trial.replace("trial_", "").replace(".mp4", ""))
        meta_p = DEMO / clip / "meta.json"
        if meta_p.is_file():
            rows = [r for r in json.loads(meta_p.read_text(encoding="utf-8"))["rows"] if r["trial"] == k]
            if not rows or not rows[0]["same_course"]:
                raise SystemExit(f"{ref} は元の記録と同じ経過ではないので使わない")
        self.n = len(np.load(DEMO / clip / f"trial_{k:02d}_frames.npz")["t"])
        self.c = av.open(str(DEMO / clip / f"trial_{k:02d}.mp4"))
        self.it = self.c.decode(video=0)
        self.i, self.img = -1, None
        self.dur = self.n / FPS

    def at(self, t):
        want = min(max(0, int(round(t * FPS))), self.n - 1)
        while self.i < want:
            try:
                self.img = next(self.it).to_ndarray(format="rgb24")
            except StopIteration:
                break
            self.i += 1
        return self.img


def header(d, title, speed=None, note=None):
    d.rectangle((0, 0, W, 86), fill=(29, 33, 38))
    text(d, (40, 22), title, 40, (255, 255, 255), True)
    if speed:
        text(d, (W - 40, 26), f"{speed:.1f} 倍速" if abs(speed - 1) > 0.05 else "実時間", 36, (255, 214, 102), True, "ra")
    if note:
        text(d, (40, H - 44), note, 22, MUTED)


MAX_SPEED = 2.0   # 課題の規則: 倍速は 2 倍まで


def _speed(sc, clip_dur):
    t0 = float(sc.get("t0", 0.0))
    t1 = float(sc.get("t1", clip_dur))
    sp = (t1 - t0) / float(sc["dur_s"])
    if sp > MAX_SPEED + 1e-9:
        raise SystemExit(f"場面 {sc.get('id')} の倍速 {sp:.2f} が {MAX_SPEED:g} 倍を超える（dur_s を延ばす）")
    return t0, sp


# ------------------------------------------------------------------ 場面
def card(sc, title, lines):
    def f(j):
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)
        if title:
            text(d, (W // 2, 330), title, 64, INK, True, "ma")
        y = 470 if title else 300
        for s, size, col in lines:
            text(d, (W // 2, y), s, size, col, False, "ma")
            y += size + 26
        return im
    return int(sc["dur_s"] * FPS), f


def single(sc, title, label, caption):
    clip = Clip(sc["clip"])
    t0, sp = _speed(sc, clip.dur)

    def f(j):
        t = t0 + j * sp / FPS
        im = Image.new("RGB", (W, H), BG)
        im.paste(Image.fromarray(clip.at(t)).resize((1440, 810)), (0, 86))
        d = ImageDraw.Draw(im)
        header(d, title, sp, RERUN_NOTE)
        text(d, (1470, 120), label, 28, INK, True)
        y = 200
        for s in caption:
            text(d, (1470, y), s, 26, MUTED)
            y += 40
        return im
    return int(sc["dur_s"] * FPS), f


CROP = (120, 600, 250, 1103)                   # 左右比較で拡大する範囲（縦 y0:y1、横 x0:x1、16:9。腕の先・立方体・箱が入る）
PW, PH = 930, 523                              # 左右比較の 1 枚の大きさ（間に 20 px あける）


def _events(evs, t):
    for e in evs or []:
        if e["t0"] <= t < e["t1"]:
            return e
    return None


def pair(sc, lines=None, band=None):
    """同じ配置の 2 本（左 立て直しデータあり・右 なし）を拡大して並べる。
    lines: 上の見出しの文（1〜2 行、太字）。yaml に headline があれば、それを 1 行だけ大きく書く（冒頭）。
    yaml の corner: 右上の小さな札（後ろに「・実時間」か「・N 倍速」を足す）。labels: 左右のラベル（既定は長い名前）。
    footnote: false なら回し直しの注記を出さない。
    events_with / events_without（yaml）: 各映像の下に出す一言（映像の時刻 s）。done_with: 左が成功した時刻（緑の枠）。
    band: (説明, [(文字, 色), ...]) 下に出す数字の欄。band_from（yaml、映像の時刻 s）から出し、出ている間は映像の下の一言を消す
    （同時に出す文字を 3 つまでにするため）。"""
    cl = [Clip(sc["clip_with"]), Clip(sc["clip_without"])]
    t0, sp = _speed(sc, max(c.dur for c in cl))
    y0, y1, x0, x1 = CROP
    headline = sc.get("headline")
    if headline:
        lines = [headline]
    top_h = 116 if headline else 150
    labels = sc.get("labels") or [WITH, WITHOUT]
    lab_size = 40 if sc.get("labels") else 34
    py = top_h + (lab_size + 40) + (50 if headline else 26)      # 映像の上端（冒頭は縦の中央に寄せる）
    xs = (20, W - 20 - PW)
    sub = sc.get("sub")
    spd = f"{sp:.1f} 倍速" if abs(sp - 1) > 0.05 else "実時間"
    corner = f"{sc['corner']}・{spd}" if sc.get("corner") else spd
    footnote = sc.get("footnote", True)

    def f(j):
        t = t0 + j * sp / FPS
        im = Image.new("RGB", (W, H), BG)
        for i, c in enumerate(cl):
            im.paste(Image.fromarray(c.at(t)[y0:y1, x0:x1]).resize((PW, PH)), (xs[i], py))
        d = ImageDraw.Draw(im)
        d.rectangle((0, 0, W, top_h), fill=(29, 33, 38))
        if headline:
            text(d, (36, (top_h - 56) // 2), headline, 56, (255, 255, 255), True, maxw=W - 520)
            text(d, (W - 30, (top_h - 28) // 2), corner, 28, (255, 214, 102), False, "ra", maxw=440)
        else:
            text(d, (W - 30, 20), corner, 32, (255, 214, 102), True, "ra")
            y = 18
            for k, s in enumerate(lines):
                text(d, (36, y), s, 44 if len(lines) > 1 else 50, (255, 255, 255) if k == 0 else (255, 214, 102), True,
                     maxw=W - 260)
                y += 62
        if sub:
            text(d, (36, top_h + 14), sub, 27, MUTED)
        for i, (lab, col) in enumerate(zip(labels, (C_B, INK))):
            text(d, (xs[i] + 4, py - lab_size - 12), lab, lab_size, col, True, maxw=PW)
        done = sc.get("done_with")
        if done is not None and t >= done:       # 左が成功した: 緑の枠
            d.rectangle((xs[0] - 6, py - 6, xs[0] + PW + 5, py + PH + 5), outline=C_B, width=8)
        show_band = band and t >= float(sc.get("band_from", 0.0))
        if not show_band:
            for i, key in enumerate(("events_with", "events_without")):
                e = _events(sc.get(key), t)
                if e:
                    col = {"ok": C_B, "ng": C_C}.get(e.get("kind"), INK)
                    text(d, (xs[i] + PW // 2, py + PH + 22), e["text"], 44 if headline else 38, col, True, "ma", maxw=PW)
        if show_band:
            by = py + PH + 40
            d.rounded_rectangle((20, by, W - 20, by + 180), 14, fill=(255, 255, 255), outline=C_B, width=3)
            text(d, (W // 2, by + 14), band[0], 44, INK, True, "ma")       # 数字の意味の行
            parts = band[1]
            fb = font(72, True)
            wsum = sum(d.textlength(s, font=fb) for s, _ in parts)
            x = (W - wsum) / 2
            for s, col in parts:
                text(d, (x, by + 84), s, 72, col, True)
                x += d.textlength(s, font=fb)
        if footnote:
            text(d, (W - 30, H - 34), RERUN_NOTE, 20, MUTED, False, "ra")
        return im
    return int(sc["dur_s"] * FPS), f


def task_scene(sc, title):
    clip = Clip(sc["clip"])
    run = json.loads((EVAL / sc["clip"].split("/")[0] / "run_0000.json").read_text(encoding="utf-8"))
    plan = [COLOR[c] for c in run["plan"]["steps"]]
    t0, sp = _speed(sc, clip.dur)

    def f(j):
        t = t0 + j * sp / FPS
        im = Image.new("RGB", (W, H), BG)
        im.paste(Image.fromarray(clip.at(t)).resize((1440, 810)), (0, 86))
        d = ImageDraw.Draw(im)
        header(d, title, sp, RERUN_NOTE)
        text(d, (1470, 110), WITH, 24, INK, True)
        text(d, (1470, 160), f"指示「{run['text']}」", 30, INK, True)
        text(d, (1470, 210), "LLM による分解:", 24, MUTED)
        text(d, (1470, 246), " → ".join(plan), 34, C_A, True)
        y = 320
        for s in run["steps"]:
            tj = s["attempts"][-1]["t_judge"]
            done = s["judged_complete"] and tj is not None and tj <= t
            active = s["attempts"][0]["t_begin"] <= t and not done
            mark = "✓" if done else ("▶" if active else "・")
            text(d, (1470, y), f"{mark} {COLOR[s['color']]}を箱へ", 32, C_B if done else (INK if active else MUTED), True)
            y += 50
            if len(s["attempts"]) > 1 and s["attempts"][1]["t_begin"] <= t:
                a1 = s["attempts"][1]
                quick = a1["t_judge"] is not None and a1["t_judge"] - a1["t_begin"] < 3.0   # 戻った時点で既に箱の中
                text(d, (1510, y), "待機位置へ戻ってから完了を確認" if quick else "待機位置へ戻して再試行", 24, C_C)
                y += 38
        for e in sc.get("events", []):
            if e["t0"] <= t < e["t1"]:
                text(d, (1470, y + 20), e["text"], 30, C_C, True, maxw=W - 1470 - 20)
        return im
    return int(sc["dur_s"] * FPS), f


FIG_PICK = {                                    # 説明資料の HTML から図の SVG を選ぶ（要点の棒グラフなど、ほかの SVG を拾わない）
    "fig1": r"<figure class=\"sys\">\s*(<svg.*?</svg>)",
    "rec": r"(<svg[^>]*aria-label=\"復帰デモの作り方.*?</svg>)",
}


def fig_image(name):
    """説明資料の図 1 つだけの HTML を Edge で画像にする（説明資料と同じ SVG・値）。name は FIG_PICK の鍵。"""
    import importlib.util
    s = importlib.util.spec_from_file_location("paper", ROOT / "scripts" / "paper_pdf.py")
    pm = importlib.util.module_from_spec(s)
    s.loader.exec_module(pm)
    v = json.loads((BUILD / "values.json").read_text(encoding="utf-8"))
    body = pm.render(v)
    style = re.search(r"<style>.*?</style>", body, re.S).group(0)
    m = re.search(FIG_PICK[name], body, re.S)
    if not m:
        raise SystemExit(f"説明資料の HTML に図 {name} が見つからない（{FIG_PICK[name]}）")
    svg = re.sub(r"（\d+(\.\d+)? 節）", "", m.group(1))          # 説明資料の節番号は動画に出さない
    page = BUILD / f"{name}_only.html"
    page.write_text(f"<!doctype html><meta charset='utf-8'>{style}<body style='margin:0;background:#fff'>"
                    f"<div style='width:1800px;padding:20px 60px'>{svg}</div></body>", encoding="utf-8")
    png = BUILD / f"{name}_video.png"
    subprocess.run([str(pm.EDGE), "--headless", "--disable-gpu", "--hide-scrollbars", f"--screenshot={png}",
                    "--window-size=1920,2000", page.as_uri()], check=True, capture_output=True, timeout=120)
    from PIL import ImageChops
    im = Image.open(png).convert("RGB")
    box = ImageChops.difference(im, Image.new("RGB", im.size, "white")).getbbox()   # 図の下の余白を落とす
    return im.crop((0, 0, im.width, min(im.height, box[3] + 20)))


def image_scene(sc, img, title, lines):
    im0 = img.copy()
    im0.thumbnail((1500, H - 100 - 44 * len(lines) - 60))      # 説明文が図に重ならない大きさ
    dur = float(sc["dur_s"])

    top = 96 + max(0, (H - 96 - 40 - (im0.height + 20 + 44 * len(lines))) // 2)   # 図と説明文をまとめて縦の中央に

    def f(j):
        im = Image.new("RGB", (W, H), BG)
        im.paste(im0, ((W - im0.width) // 2, top))
        d = ImageDraw.Draw(im)
        header(d, title)
        k = min(len(lines), 1 + j // int(dur * FPS / max(1, len(lines))))
        y = top + im0.height + 20
        for s in lines[:k]:
            text(d, (W // 2, y), s, 30, INK, False, "ma")
            y += 44
        return im
    return int(dur * FPS), f


def scenes(v):
    S, order = spec()
    figs = {}

    def fig(name):                               # 図は要る場面があるときだけ描く
        if name not in figs:
            figs[name] = fig_image(name)
        return figs[name]

    # 冒頭の数字は 1 つだけ: わざと掴み損ねさせた同じ配置・同じ失敗の組で、掴み直して成功した回数（配置の数として書く）。
    # どの評価かの詳しい条件は主な結果の場面と説明資料に書く
    band = (f"{v['e3a_pairs']} 通りの配置で同じ失敗をさせ、掴み直して成功した回数",
            [("復帰デモあり ", C_B), (f"{v['e3a_xk']} 回", C_B), ("　／　", MUTED), ("なし ", INK), (f"{v['e3a_yk']} 回", INK)])
    make = {
        # 冒頭: 方策が自分で掴み損ね、自分で掴み直す（左）と、掴み直さない（右）。見出し 1 行（yaml の headline）と数字 1 つ
        "contrast": lambda sc: pair(sc, band=band),
        "title": lambda sc: card(sc, "失敗からの復帰を学習するロボットアーム", [
            ("シミュレーション上のロボットアームが、日本語の指示で立方体を箱へ片付ける", 42, INK),
            ("実機も、人の手によるデモ集めも使わない。学習データはすべて自動で作る", 42, INK)]),
        # 課題の難しさを 1 行で
        "problem": lambda sc: card(sc, "課題", [
            ("成功デモだけで学ぶと失敗の後を知らず、掴み損ねても掴み直さない（冒頭の右）", 38, INK),
            ("難しい点: 動きを書いたプログラムではなく、学習した方策が、動きながら自分で掴み直す", 38, C_A),
            ("一方、実機で失敗と回復のデモを人の手で集めるのは手間がかかる", 34, MUTED)]),
        "recipe": lambda sc: image_scene(sc, fig("rec"), "方法: 復帰デモを自動で作る", [
            "エキスパート（シミュレーションの正解を使えるプログラム）が、デモを作る途中で意図的に失敗する",
            "失敗した後から、回復して箱に入れるまでを記録し、成功デモと一緒に学習する（人手なし）"]),
        "contrast_p1": lambda sc: pair(sc, ["把持を意図的に失敗させた評価から 1 回",
                                            "左は自分で掴み直して成功し、右は回復しない"]),
        "method": lambda sc: image_scene(sc, fig("fig1"), "仕組み", [
            "カメラ画像と関節の状態だけで動く（シミュレーションの正解の情報は使わない）",
            "LLM が指示を色の順番に分け、VLA が 1 色ずつ箱へ運ぶ"]),
        "task_intro": lambda sc: card(sc, "実演の条件", [
            ("指示は「全部片付けて」の 1 文だけ。LLM が片付ける順番を決める（例: 赤 → 緑 → 青）", 34, INK),
            ("学習したのは「指示した 1 色を箱へ入れる」動作だけ。箱に入ったとカメラで確かめたら次の色へ", 34, INK),
            (f"1 個につき {v['step_timeout']} 秒以内。終わらなければ待機位置に戻って {v['retry']} 回だけやり直す", 34, INK),
            (f"学習にも調整にも使っていない {v['e7_n']} 配置のうち、3 個とも片付いたのは {v['e7_k']} 配置。次の映像はその 1 つ", 30, MUTED)]),
        "task": lambda sc: task_scene(sc, "実演: 日本語の指示から 3 個を順に片付ける"),
        "numbers": lambda sc: card(sc, "主な結果（学習にも調整にも使っていない配置で測定）", [
            (f"■ 把持を意図的に失敗させたとき（{v['e3a_pairs']} 通りの配置で、両方に同じ失敗）", 30, MUTED),
            (f"掴み直して成功: 復帰デモあり {v['e3a_xk']} 回、なし {v['e3a_yk']} 回", 42, INK),
            ("", 6, BG),
            (f"■ 意図的に失敗させない、1 個を片付ける試行（{v['time_limit']} 秒以内）", 30, MUTED),
            (f"成功 {v['nat_n']} 回中 {v['e1_k']} 回（{v['e1_pct']}）", 42, INK),
            ("", 6, BG),
            ("■ 3 個を片付ける実演（「全部片付けて」の 1 文）", 30, MUTED),
            (f"3 個とも片付いたのは {v['e7_n']} 配置中 {v['e7_k']} 配置", 42, INK),
            ("", 6, BG),
            ("詳しい数字と条件は説明資料に記載", 26, MUTED)]),
        "limits": lambda sc: card(sc, "限界と今後", [
            ("方策はシミュレーションのみで、実機では動かしていない", 34, INK),
            ("落下や置き損ねからは、ほとんど回復できない", 34, INK),
            ("", 10, BG),
            ("動画の映像はすべて、評価と同じ設定で回し直し、評価と同じ経過になったもの（冒頭を含む）", 26, MUTED)]),
    }
    return [make[sc["id"]](sc) for sc in order]


def cmd_build(a) -> None:
    import av
    v = json.loads((BUILD / "values.json").read_text(encoding="utf-8"))
    sc = scenes(v)
    total = sum(n for n, _ in sc)
    out = BUILD / "video.mp4"
    box = av.open(str(out), mode="w")
    st = box.add_stream("libx264", rate=FPS)
    st.width, st.height, st.pix_fmt = W, H, "yuv420p"
    st.options = {"crf": "22", "preset": "medium", "threads": "4"}     # ほかの処理と並行できるよう 4 スレッドに抑える
    for n, f in sc:
        for j in range(n):
            fr = av.VideoFrame.from_image(f(j))
            for p in st.encode(fr):
                box.mux(p)
    for p in st.encode():
        box.mux(p)
    box.close()
    (BUILD / OUT_NAME).write_bytes(out.read_bytes())
    uniq = sorted(set(TEXTS))
    (BUILD / "video_texts.json").write_text(json.dumps({"seconds": total / FPS, "texts": uniq}, ensure_ascii=False, indent=1),
                                            encoding="utf-8")
    print(BUILD / OUT_NAME, f"{total / FPS:.1f} s", f"{out.stat().st_size / 1e6:.1f} MB")


def cmd_check(a) -> None:
    import importlib.util
    s = importlib.util.spec_from_file_location("paper", ROOT / "scripts" / "paper_pdf.py")
    pm = importlib.util.module_from_spec(s)
    s.loader.exec_module(pm)
    info = json.loads((BUILD / "video_texts.json").read_text(encoding="utf-8"))
    joined = "\n".join(info["texts"])
    hits = [(p, m.group(0)) for p in pm.FORBIDDEN for m in re.finditer(p, joined)]
    problems = [f"使わない語 {h[1]!r}（{h[0]}）" for h in hits]
    if not 60 <= info["seconds"] <= 180:   # 課題の規則: 1〜3 分
        problems.append(f"長さ {info['seconds']:.0f} s（1〜3 分の外）")
    S, order = spec()
    for sc in order:
        if "t1" in sc and _speed(sc, float(sc["t1"]))[1] > MAX_SPEED + 1e-9:
            problems.append(f"場面 {sc['id']} の倍速 {_speed(sc, float(sc['t1']))[1]:.2f} が {MAX_SPEED:g} 倍を超える")
    want = sum(float(sc["dur_s"]) for sc in order)
    if abs(info["seconds"] - want) > 0.5:
        problems.append(f"長さ {info['seconds']:.1f} s が場面の表の合計 {want:.0f} s と違う（説明資料の時刻がずれる）")
    size = (BUILD / OUT_NAME).stat().st_size
    res = {"seconds": round(info["seconds"], 1), "mb": round(size / 1e6, 1), "texts": len(info["texts"]), "problems": problems}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    raise SystemExit(1 if problems else 0)


def cmd_stills(a) -> None:
    """組み上げた動画から指定の時刻の静止画を書き出す（確認用）。"""
    import av
    out = OUT / "demo_v2" / "_review"
    out.mkdir(parents=True, exist_ok=True)
    want = sorted(float(x) for x in a.times.split(","))
    c = av.open(str(BUILD / OUT_NAME))
    k = 0
    for i, fr in enumerate(c.decode(video=0)):
        while k < len(want) and i >= int(round(want[k] * FPS)):
            p = out / f"video_{want[k]:05.1f}s.png"
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
