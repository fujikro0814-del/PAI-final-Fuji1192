"""動画（1920×1080・30 fps・H.264、3 分以内）を組み立てる。VLA の掴み直しと LLM の手順の組み直しを、仕組み全体として見せる。

scripts/video_final.py・thumbnail.py が、場面の描き方（比較・図・横棒など）の部品として読む。

    .venv\\Scripts\\python.exe scripts\\video_scenes.py build     # paper/build/動画_PAI最終課題_<アカウント名>.mp4
    .venv\\Scripts\\python.exe scripts\\video_scenes.py check     # 画面の文字に使わない語・長さ・倍速・場面の経過
    .venv\\Scripts\\python.exe scripts\\video_scenes.py stills --times 0,5,10

映像はすべて、実機の Panda で測った応答に合わせたシミュレーション。場面の並び・長さ・数字は configs/demo/video_s4.yaml、
映像は outputs/demo_v4（scripts/api/demo_ladder.py で同じ設定で回し直したもの）。回し直しは元の記録と同じ経過になるとは
限らないので、画面に「同じ設定で再実行した試行」と書き、場面の説明と経過が合うかは check が expect で確かめる。
文字・映像の読み込み・倍速の部品は scripts/video_common.py を使う。見た目は白地に黒の文字、色は意味のあるところだけ
（青＝立て直しの仕組みあり・うまくいった、赤＝失敗）、左揃え、数字は棒。同時に出す文字は 3 つまで、倍速は 2 倍まで・画面に書く。
"""
import argparse
import importlib.util
import json
import pathlib
import sys

import yaml
from PIL import Image, ImageDraw

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_s = importlib.util.spec_from_file_location("video_s3_parts", ROOT / "scripts" / "video_common.py")
V = importlib.util.module_from_spec(_s)
_s.loader.exec_module(V)
V.DEMO = ROOT / "outputs" / "demo_v4"                 # 部品（Clip）が読む置き場所を差し替える
V.EVAL = ROOT / "outputs" / "v2eval" / "V4DEMO"
SPEC = ROOT / "configs" / "demo" / "video_s4.yaml"
from recovla.common.account import ACCOUNT  # noqa: E402  値は src/recovla/common/account.py だけに書く
OUT_NAME = f"動画_PAI最終課題_{ACCOUNT}.mp4"
BUILD = V.BUILD
W, H, FPS = V.W, V.H, V.FPS
PAPER, INK, SUB = (255, 255, 255), (17, 17, 17), (70, 70, 70)
BLUE, RED, TRACK, BAR0 = (0, 87, 217), (200, 16, 16), (232, 232, 232), (130, 130, 130)
M = 64                                             # 左右の余白
text = V.text
COLOR = V.COLOR
TITLE = ["失敗しても、やり直して", "作業を続けるロボット"]
STAGES = [("落下・タイムアウト", "LLM が状況を報告して再計画"),                      # 題の場面の 2 段の対応（動画で見せる順）
          ("把持失敗", "VLA がその場で再把持（回復デモで学習）")]
CL2_NAME = "スクリプト方策"                       # 学習を使わない方策（点群で物体の位置を推定し、把持・搬送・配置を状態機械で切り替える）。
                                                  # 低レベル制御（手先の速度 → 逆運動学 → 制限）は VLA と共通で、違うのは手先の動きを決める層だけ


def load_spec():
    s = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    return s["values"], s["scenes"]


def seg_time(segs, j):
    """segments の倍速で、動画のフレーム j → (映像の時刻, その倍速)。"""
    tv = j / FPS
    for sg in segs:
        d = (float(sg["t1"]) - float(sg["t0"])) / float(sg["speed"])
        if tv < d or sg is segs[-1]:
            return float(sg["t0"]) + min(tv, d) * float(sg["speed"]), float(sg["speed"])
        tv -= d
    return float(segs[-1]["t1"]), float(segs[-1]["speed"])


def seg_dur(segs) -> float:
    return sum((float(s["t1"]) - float(s["t0"])) / float(s["speed"]) for s in segs)


def timing(sc, clips):
    """場面の時刻の進め方: segments（切れ目ごとの倍速）か、t0〜t1 を一定の倍速で。フレーム j → (映像の時刻, 倍速)。"""
    segs = sc.get("segments")
    if segs:
        if abs(seg_dur(segs) - float(sc["dur_s"])) > 0.05:
            raise SystemExit(f"場面 {sc['id']} の区間の長さの合計 {seg_dur(segs):.2f} s が dur_s {sc['dur_s']} と違う")
        if max(float(s["speed"]) for s in segs) > V.MAX_SPEED + 1e-9:
            raise SystemExit(f"場面 {sc['id']} の倍速が 2 倍を超える")
        return lambda j: seg_time(segs, j)
    t0, sp = V._speed(sc, max(c.dur for c in clips))
    return lambda j: (t0 + j * sp / FPS, sp)


class RerunClip(V.Clip):
    """回し直しの映像。元の記録と同じ経過でなくてよい（same_course を見ない）。経過の照らしは check の expect で行う。"""

    def __init__(self, ref):
        import av
        import numpy as np
        clip, trial = ref.split("/")
        k = int(trial.replace("trial_", "").replace(".mp4", ""))
        self.n = len(np.load(V.DEMO / clip / f"trial_{k:02d}_frames.npz")["t"])
        self.c = av.open(str(V.DEMO / clip / f"trial_{k:02d}.mp4"))
        self.it = self.c.decode(video=0)
        self.i, self.img = -1, None
        self.dur = self.n / FPS


V.Clip = RerunClip
V.RERUN_NOTE = RERUN_NOTE = "映像は評価と同じ設定で再実行した試行"
PW, PH = 880, 495                                  # 左右比較の 1 枚の大きさ（16:9、間 32 px）
XS = (M, W - M - PW)


# ------------------------------------------------------------------ 共通の部品
def header(d, title, speed=None):
    """上の見出し（左揃えの黒い太字と下の線）と、右上の倍速の札（黒地に白）。"""
    text(d, (M, 40), title, 50, INK, True, maxw=W - 2 * M - 260)
    d.rectangle((M, 122, W - M, 125), fill=INK)
    if speed:
        s = f"{speed:.1f} 倍速" if abs(speed - 1) > 0.05 else "実時間"
        tw = d.textlength(s, font=V.font(34, True))
        d.rectangle((W - M - tw - 40, 34, W - M, 100), fill=INK)
        text(d, (W - M - 20, 46), s, 34, PAPER, True, "ra")


def note(d):
    text(d, (M, H - 58), RERUN_NOTE, 26, SUB)


def frame(d, x, y, w, h, col):
    d.rectangle((x - 8, y - 8, x + w + 7, y + h + 7), outline=col, width=8)


def bars(d, x, y, w, rows, total, size=40, gap=150, num=56):
    """横棒（数 / total）。rows: [(名前, 数, 色)]。名前を上に、棒と数を下に。棒の後ろの薄い灰色が total の長さ。"""
    bw = w - int(num * 2.4)
    for k, row in enumerate(rows):
        name, val, col = row[:3]
        shown = row[3] if len(row) > 3 else str(val)     # 4 つ目があれば、棒の先に書く文字（割合など）
        yy = y + k * gap
        text(d, (x, yy), name, size, col if col != BAR0 else INK, True, maxw=w)
        by = yy + size + 14
        bh = int(num * 1.1)
        d.rectangle((x, by, x + bw, by + bh), fill=TRACK)
        d.rectangle((x, by, x + int(bw * val / total), by + bh), fill=col)
        text(d, (x + int(bw * val / total) + 16, by + (bh - num) // 2 - 4), shown, num, col if col != BAR0 else INK, True)


def still(ref, t, crop=None):
    c = V.Clip(ref)
    img = c.at(t)
    if crop:
        y0, y1, x0, x1 = crop
        img = img[y0:y1, x0:x1]
    return Image.fromarray(img)


def records(ref):
    """回し直しの記録の並び（outputs/v2eval/V4DEMO/<場面>/run_NNNN.json、古い場面は trial_N.json）。"""
    name = ref.split("/")[0]
    return sorted((V.EVAL / name).glob("run_[0-9][0-9][0-9][0-9].json")) or sorted((V.EVAL / name).glob("trial_[0-9]*.json"))


def record(ref) -> dict:
    k = int(ref.split("/")[1].replace("trial_", "").replace(".mp4", ""))
    recs = records(ref)
    return json.loads(recs[k].read_text(encoding="utf-8")) if len(recs) > k else {}


def wrap(d, s, size, maxw, bold=False):
    """字の大きさを変えずに、幅 maxw で折り返す（句読点・閉じ括弧を行頭に置かない）。"""
    fo = V.font(size, bold)
    lines = []
    for part in s.split("\n"):                       # \n は手で決めた改行
        cur = ""
        for ch in part:
            if cur and d.textlength(cur + ch, font=fo) > maxw and ch not in "、。）」":
                k = max(cur.rfind("、"), cur.rfind("。"))
                if k >= 0 and len(cur) - k <= 8:     # 近くに句読点があれば、その後ろで折る
                    lines.append(cur[:k + 1])
                    cur = cur[k + 1:]
                else:
                    lines.append(cur)
                    cur = ""
            cur += ch
        if cur:
            lines.append(cur)
    return lines


def report_box(d, r, x0, y0, x1):
    """LLM が返した報告（日本語）を、黒い線の四角に折り返して（3 行まで）。"""
    ls = wrap(d, str(r["report"]), 30, x1 - x0 - 56)[:3]
    d.rectangle((x0, y0, x1, y0 + 78 + 44 * len(ls)), fill=PAPER, outline=INK, width=3)
    text(d, (x0 + 28, y0 + 20), "LLM の状況報告", 30, INK, True)
    for k, s in enumerate(ls):
        text(d, (x0 + 28, y0 + 70 + 44 * k), s, 30, INK, False, maxw=x1 - x0 - 56)


# ------------------------------------------------------------------ 場面
def compare(sc):
    """同じ配置の 2 本を左右に並べる（左 = 立て直しの仕組みあり・青、右 = なし・黒）。映像の下に経過の一言。
    yaml: crop（発表用カメラ 1280×720 のうち映す範囲）、segments か t0/t1、labels、events_with/without、done_with（左の成功の時刻）、
    count: true なら箱の中の数（記録の truth_success_t）、左の LLM の報告（report_s 秒）、右が中止した時刻に赤の枠。"""
    cl = [V.Clip(sc["clip_with"]), V.Clip(sc["clip_without"])]
    runs = [record(sc[k]) for k in ("clip_with", "clip_without")]
    at = timing(sc, cl)
    y0, y1, x0, x1 = sc.get("crop") or V.CROP
    reps = [r for r in (runs[0].get("replans") or []) if r.get("report")]
    stop_t = (runs[1].get("stopped") or {}).get("t")
    box_t = [runs[i].get("truth_success_t") or {} for i in range(2)]
    py = 236
    plan_line = None                                       # 指示と LLM の最初の計画
    if sc.get("plan_until") is not None:
        plans = [(r.get("plan") or {}).get("llm_steps") for r in runs]
        if plans[0] != plans[1]:
            raise SystemExit(f"場面 {sc['id']} の左右で LLM の最初の計画が違う {plans}")
        plan_line = "指示「全部片付けて」　→　LLM の計画: " + " → ".join(COLOR.get(c, c) for c in plans[0])

    def n_in(i, t):
        return sum(1 for v in box_t[i].values() if v is not None and float(v) <= t)

    def f(j):
        t, sp = at(j)
        im = Image.new("RGB", (W, H), PAPER)
        for i, c in enumerate(cl):
            im.paste(Image.fromarray(c.at(t)[y0:y1, x0:x1]).resize((PW, PH)), (XS[i], py))
        d = ImageDraw.Draw(im)
        header(d, sc["headline"], sp)
        for i, col in enumerate((BLUE, INK)):
            text(d, (XS[i], py - 66), sc["labels"][i], 40, col, True, maxw=PW - 220)
            if sc.get("count"):
                n = n_in(i, t)
                text(d, (XS[i] + PW, py - 60), f"箱の中 {n}/3", 34, BLUE if n == 3 else INK, True, "ra")
        done = sc.get("done_with")
        if (sc.get("count") and n_in(0, t) == 3) or (done is not None and t >= float(done)):
            frame(d, XS[0], py, PW, PH, BLUE)
        if sc.get("count") and stop_t is not None and t >= float(stop_t):
            frame(d, XS[1], py, PW, PH, RED)
        if plan_line and t < float(sc["plan_until"]):        # 最初は計画を映像の下に（経過の一言はまだ出ない時間）
            text(d, (M, py + PH + 36), plan_line, 44, INK, True, maxw=W - 2 * M)
            note(d)
            return im
        shown = False
        for r in reps:
            td = float(r.get("t_decided") or 0.0)
            if td <= t < td + float(sc.get("report_s", 14.0)):
                report_box(d, r, XS[0], py + PH + 30, XS[0] + PW)
                shown = True
        for i, key in enumerate(("events_with", "events_without")):
            if i == 0 and shown:
                continue
            e = V._events(sc.get(key), t)
            if e:
                col = {"ok": BLUE, "ng": RED}.get(e.get("kind"), INK)
                text(d, (XS[i], py + PH + 36), e["text"], 44, col, True, maxw=PW)
        note(d)
        return im
    return int(round(float(sc["dur_s"]) * FPS)), f


def instr_card(sc):
    """指示を変えた実演: 同じ配置で指示だけ変えた 3 本を横に並べ、それぞれ箱に入る所を流す。
    yaml の panels: [{clip, text（指示）, plan（計画の説明）, t0, t1}]。評価とは別の実演（判定に使っていない）。"""
    pans = sc["panels"]
    cl = [V.Clip(p["clip"]) for p in pans]
    recs = [record(p["clip"]) for p in pans]
    dur = float(sc["dur_s"])
    sps = [(float(p["t1"]) - float(p["t0"])) / dur for p in pans]
    if max(sps) > V.MAX_SPEED + 1e-9:
        raise SystemExit(f"場面 {sc['id']} の倍速 {max(sps):.2f} が 2 倍を超える")
    y0, y1, x0, x1 = sc.get("crop") or V.CROP
    gap = 24
    pw = (W - 2 * M - 2 * gap) // 3
    ph = pw * (y1 - y0) // (x1 - x0)
    py = 290
    box_t = [r.get("truth_success_t") or {} for r in recs]

    def f(j):
        im = Image.new("RGB", (W, H), PAPER)
        for i, c in enumerate(cl):
            t = float(pans[i]["t0"]) + j * sps[i] / FPS
            im.paste(Image.fromarray(c.at(t)[y0:y1, x0:x1]).resize((pw, ph)), (M + i * (pw + gap), py))
        d = ImageDraw.Draw(im)
        header(d, sc["headline"], max(sps))
        for i, p in enumerate(pans):
            x = M + i * (pw + gap)
            t = float(p["t0"]) + j * sps[i] / FPS
            text(d, (x, 165), f"「{p['text']}」", 36, INK, True, maxw=pw)
            n = sum(1 for v in box_t[i].values() if v is not None and float(v) <= t)
            text(d, (x + pw, 228), f"箱の中 {n}", 30, INK, True, "ra")
            for k, ln in enumerate(wrap(d, p["plan"], 32, pw)):
                text(d, (x, py + ph + 28 + k * 48), ln, 32, BLUE if k == 0 else INK, k == 0, maxw=pw)
        text(d, (M, H - 58), "評価とは別の実演（同じ配置で指示だけ変えた）。映像は箱に入る所を 2 倍速で", 26, SUB)
        return im
    return int(round(dur * FPS)), f


def side(sc, title, sub, rows, total, last=None, bold_last=True):
    """1 本の映像を大きく左に、右の欄に横棒の比べ。sub: 棒の上の説明、last: 棒の下の一言。"""
    clip = V.Clip(sc["clip"])
    at = timing(sc, [clip])
    vw, vh = 1152, 648
    cx = M + vw + 56
    cw = W - M - cx

    def f(j):
        t, sp = at(j)
        im = Image.new("RGB", (W, H), PAPER)
        im.paste(Image.fromarray(clip.at(t)).resize((vw, vh)), (M, 180))
        d = ImageDraw.Draw(im)
        header(d, title, sp)
        y = 180
        for s in sub:
            for ln in wrap(d, s, 34, cw):
                text(d, (cx, y), ln, 34, INK, False, maxw=cw)
                y += 50
            y += 22
        bars(d, cx, y + 20, cw, rows, total, size=36, gap=160, num=56)
        if last:
            ls = wrap(d, last, 34, cw, bold_last)
            for k, ln in enumerate(ls):
                text(d, (cx, 180 + vh - 50 * (len(ls) - k)), ln, 34, INK if bold_last else SUB, bold_last, maxw=cw)
        note(d)
        return im
    return int(round(float(sc["dur_s"]) * FPS)), f


def title_card(sc, img):
    """題（左揃え）と 2 段の対応表、右に 3 個とも入った場面の静止画。冒頭と終わりで同じもの。"""
    im0 = img.resize((900, 506))

    def f(j):
        im = Image.new("RGB", (W, H), PAPER)
        im.paste(im0, (W - M - 900, 250))
        d = ImageDraw.Draw(im)
        text(d, (M + 16, 250), TITLE[0], 68, INK, True, maxw=860)
        text(d, (M + 16, 340), TITLE[1], 68, INK, True, maxw=860)
        y = 500
        for a, b in STAGES:
            text(d, (M + 16, y), a, 36, INK, True, maxw=840)
            text(d, (M + 56, y + 52), f"→ {b}", 36, BLUE, True, maxw=800)
            y += 130
        text(d, (W - M - 900, 250 + 506 + 20), "指示「全部片付けて」で、立方体を箱へ（シミュレーション）", 28, SUB, False, maxw=900)
        return im
    return int(round(float(sc["dur_s"]) * FPS)), f


def list_card(sc, title, items, foot=None, numbered=False):
    def f(j):
        im = Image.new("RGB", (W, H), PAPER)
        d = ImageDraw.Draw(im)
        header(d, title)
        y = 260
        for k, s in enumerate(items):
            if numbered:
                text(d, (M, y), f"{k + 1}", 48, BLUE, True)
            text(d, (M + (80 if numbered else 0), y), s, 48, INK, False, maxw=W - 2 * M - 80)
            y += 130
        if foot:
            text(d, (M, y + 40), foot, 34, SUB, False)
        return im
    return int(round(float(sc["dur_s"]) * FPS)), f


def numbers_card(sc, vals):
    """上: 3 個とも箱に入った数（事前に決めた比較）。下: 評価中に起きた掴み損ねから方策が自分で立て直した割合
    （事後の集計、scripts/api/natural_recovery.py）。"""
    rows = [("回復なし（成功デモのみ・再計画なし）", vals["clc_n0"], BAR0), ("回復あり（回復デモ＋再計画）", vals["clc_l2v2"], BLUE)]
    k0, n0, k1, n1 = vals["nr_n0_k"], vals["nr_n0_n"], vals["nr_u0_k"], vals["nr_u0_n"]
    rows2 = [("成功デモのみで学習", round(1000 * k0 / n0), BAR0, f"{100 * k0 / n0:.0f}%（{k0}/{n0}）"),
             ("回復デモも学習", round(1000 * k1 / n1), BLUE, f"{100 * k1 / n1:.0f}%（{k1}/{n1}）")]

    def f(j):
        im = Image.new("RGB", (W, H), PAPER)
        d = ImageDraw.Draw(im)
        header(d, "結果: 回復あり・なしの比較（同じ配置の組）")
        text(d, (M, 165), f"学習・調整に使っていない {vals['clc_n']} 配置で、「全部片付けて」の 3 個とも収納できた数", 34, INK, False)
        bars(d, M, 245, W - 2 * M, rows, vals["clc_n"], size=34, gap=140, num=52)
        text(d, (M, 600), "同じ本番で起きた把持失敗のうち、方策が自分で掴み直して箱に入れた割合（事後の集計）", 34, INK, False)
        bars(d, M, 680, W - 2 * M, rows2, 1000, size=34, gap=140, num=52)
        return im
    return int(round(float(sc["dur_s"]) * FPS)), f


def qa_card(sc, vals):
    """締め: 確かめた 3 つの問いと答え。"""
    qa = [("① 自動生成した学習データでも、VLA は機能するか",
           f"機能する（3 個とも収納 {vals['clc_l2v2']}/{vals['clc_n']}）。ただし同じ作業ではスクリプト方策（{vals['clc_cl2']}）に及ばない"),
          ("② 回復を学習データに入れると、本番で立て直せるか",
           f"立て直せる（本番の把持失敗から自力で {100 * vals['nr_n0_k'] / vals['nr_n0_n']:.0f}% → "
           f"{100 * vals['nr_u0_k'] / vals['nr_u0_n']:.0f}%）"),
          ("③ 上位の判断に LLM を入れると、成功率は上がるか",
           f"立方体では上がる（再計画あり・なし 103 → 153、全体 {vals['clc_n0']} → {vals['clc_l2v2']}）")]

    def f(j):
        im = Image.new("RGB", (W, H), PAPER)
        d = ImageDraw.Draw(im)
        header(d, "確かめた問いと答え")
        y = 190
        for q, a in qa:
            text(d, (M, y), q, 40, INK, True, maxw=W - 2 * M)
            text(d, (M + 56, y + 62), "→ " + a, 36, BLUE, False, maxw=W - 2 * M - 56)
            y += 220
        return im
    return int(round(float(sc["dur_s"]) * FPS)), f


def recipe_card(sc):
    """図 2（立て直しデータの作り方）を同じ見た目で描く。4 つの箱と矢印、その下に記録する範囲を示す時間の線。"""
    boxes = [("成功デモの生成", ["エキスパートが", "通常どおり把持し", "箱へ搬送"]),
             ("意図的な失敗", ["把持失敗: 位置か高さを", "ずらして閉じる", "置き損ね: 机の上で放す"]),   # 学習データの立て直しは
             ("失敗の確定", ["物体の状態で判定", "（持ち上がらない、", "机の上で静止）"]),               # 掴み損ね 60・置き損ね 30
             ("回復して収納", ["エキスパートが", "再把持・再搬送して", "箱に入れるまで"])]
    bw, gap, by, bh = 410, 50, 220, 300
    lines = ["エキスパート: シミュレーションの真の状態を使うスクリプト方策。デモ生成の途中で意図的に失敗させる",
             "失敗の確定以降を回復デモとして記録。成功デモ 240 本＋回復デモ 90 本で学習（人手の遠隔操作なし）"]

    def f(j):
        im = Image.new("RGB", (W, H), PAPER)
        d = ImageDraw.Draw(im)
        header(d, "回復デモの自動生成: エキスパートに意図的に失敗させる")
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
            d.rectangle((x, ty + 30, x + 18, ty + 33), fill=BAR0)
        text(d, (M, ty - 8), "記録しない", 32, SUB, False)
        d.rectangle((rec_x, ty, W - M, ty + 70), fill=BLUE)
        text(d, (rec_x + 20, ty + 16), "回復デモとして記録", 36, PAPER, True, maxw=W - M - rec_x - 40)
        d.rectangle((M, ty + 100, W - M - 24, ty + 103), fill=INK)
        d.polygon([(W - M - 24, ty + 90), (W - M, ty + 101), (W - M - 24, ty + 112)], fill=INK)
        text(d, (W - M, ty + 120), "時間", 30, SUB, False, "ra")
        y = ty + 230
        for s in lines:
            text(d, (M, y), s, 40, INK, False)
            y += 66
        return im
    return int(round(float(sc["dur_s"]) * FPS)), f


def loop_card(sc):
    """立て直しの仕組みの制御ループの図。右が速いループ（0.1 秒ごと、VLA が掴み直す）、
    左が手順のループ（落とした・2 回試して時間切れのとき、LLM が報告して手順を組み直す）。青は立て直しのための部分。"""
    def box(d, x0, y0, x1, y1, head, sub=None, fill=PAPER, edge=INK):
        d.rectangle((x0, y0, x1, y1), fill=fill, outline=edge, width=3)
        col = PAPER if fill == BLUE else INK
        cy = (y0 + y1) // 2 - (24 if sub else 0)
        text(d, ((x0 + x1) // 2, cy - 18), head, 34, col, True, "ma", maxw=x1 - x0 - 24)
        if sub:
            text(d, ((x0 + x1) // 2, cy + 28), sub, 26, col, False, "ma", maxw=x1 - x0 - 24)

    def arrow(d, pts, col=INK):
        for a, b in zip(pts, pts[1:]):
            d.line([a, b], fill=col, width=5)
        (xa, ya), (xb, yb) = pts[-2], pts[-1]
        ux, uy = (xb > xa) - (xb < xa), (yb > ya) - (yb < ya)
        px, py = -uy, ux
        d.polygon([(xb, yb), (xb - 22 * ux + 12 * px, yb - 22 * uy + 12 * py), (xb - 22 * ux - 12 * px, yb - 22 * uy - 12 * py)],
                  fill=col)

    def f(j):
        im = Image.new("RGB", (W, H), PAPER)
        d = ImageDraw.Draw(im)
        header(d, "階層的な回復: 方策ループとタスクループ")
        # 左: タスク計画の側
        box(d, M, 170, 664, 250, "指示「全部片付けて」")
        box(d, M, 310, 664, 410, "LLM（タスク分解）", "指示 → サブタスクの列")
        box(d, M, 470, 664, 570, "タスク計画", "赤 → 緑 → 青")
        box(d, M, 800, 664, 920, "LLM（再計画）", "状況報告と計画の修正", fill=BLUE, edge=BLUE)
        arrow(d, [(364, 250), (364, 310)])
        arrow(d, [(364, 410), (364, 470)])
        arrow(d, [(364, 800), (364, 570)], BLUE)
        text(d, (390, 640), "修正した計画", 28, BLUE, True)
        # 右: 方策ループ
        box(d, 820, 470, 1130, 570, "VLA", "回復デモで学習", edge=BLUE)
        box(d, 1210, 470, 1520, 570, "逆運動学", "低レベル制御")
        box(d, 1600, 470, W - M, 570, "ロボット")
        box(d, 1210, 640, 1520, 730, "観測", "画像・関節角")
        arrow(d, [(664, 520), (820, 520)])
        text(d, (742, 470), "サブタスク", 26, INK, False, "md")
        arrow(d, [(1130, 520), (1210, 520)])
        arrow(d, [(1520, 520), (1600, 520)])
        arrow(d, [(1728, 570), (1728, 685), (1520, 685)])
        arrow(d, [(1210, 685), (975, 685), (975, 570)])
        text(d, (820, 330), "方策ループ（10 Hz）", 32, BLUE, True)
        text(d, (820, 380), "把持失敗 → VLA がその場で再把持", 32, BLUE, True)
        # 下: 失敗検知と完了判定 → LLM
        box(d, 1210, 800, W - M, 920, "失敗検知・完了判定", "完了なら次のサブタスクへ")
        arrow(d, [(1365, 730), (1365, 800)])
        arrow(d, [(1210, 860), (664, 860)], BLUE)
        text(d, (937, 800), "落下・2 回タイムアウト", 28, RED, True, "ma", maxw=520)
        text(d, (M, 960), "タスクループ: 落下・タイムアウト → LLM が再計画", 32, BLUE, True)
        return im
    return int(round(float(sc["dur_s"]) * FPS)), f


def diagram_card(sc):
    """詳しいブロック線図（scripts/report_figs.py の fig_detail）を、部分ごとに拡大して字幕で説明する。
    yaml の keys: [{part: all|A|B|CD|E|F, hold: 秒, caption: [1 行目, 2 行目]}]。部分の範囲は report_figs の DETAIL_PARTS。
    次の部分へは move_s 秒でなめらかに移る（中心は線形、倍率は対数で補間）。"""
    import math
    s65 = importlib.util.spec_from_file_location("report_figs", ROOT / "scripts" / "report_figs.py")
    rf = importlib.util.module_from_spec(s65)
    s65.loader.exec_module(rf)
    fig = rf.fig_detail()
    V.TEXTS.extend(rf.TEXTS)                                 # 図の中の文字も使わない語の点検に入れる
    pad = 2000
    big = Image.new("RGB", (fig.width + 2 * pad, fig.height + 2 * pad), PAPER)
    big.paste(fig, (pad, pad))
    fw, fh = W, 880                                          # 図を映す範囲（下は字幕）
    keys = sc["keys"]
    move = float(sc.get("move_s", 0.75))
    total = sum(float(k["hold"]) for k in keys) + move * (len(keys) - 1)
    if abs(total - float(sc["dur_s"])) > 0.05:
        raise SystemExit(f"場面 {sc['id']} の長さの合計 {total:.2f} s が dur_s {sc['dur_s']} と違う")

    def view(part):
        x0, y0, x1, y1 = rf.DETAIL_PARTS[part]
        s = min(fw / (x1 - x0), fh / (y1 - y0)) * 0.97
        return (x0 + x1) / 2, (y0 + y1) / 2, s

    views = [view(k["part"]) for k in keys]
    cache = {}

    def render(cx, cy, s):
        k = (round(cx, 1), round(cy, 1), round(s, 4))
        if k not in cache:
            hw, hh = fw / s / 2, fh / s / 2
            box = (int(pad + cx - hw), int(pad + cy - hh), int(pad + cx + hw), int(pad + cy + hh))
            cache.clear() if len(cache) > 4 else None
            cache[k] = big.crop(box).resize((fw, fh), Image.LANCZOS)
        return cache[k]

    spans = []                                               # (始め, 終わり, 止めるか動かすか, i)
    t = 0.0
    for i, k in enumerate(keys):
        spans.append((t, t + float(k["hold"]), "hold", i))
        t += float(k["hold"])
        if i < len(keys) - 1:
            spans.append((t, t + move, "move", i))
            t += move

    def f(j):
        tt = j / FPS
        sp = next((x for x in spans if x[0] <= tt < x[1]), spans[-1])
        i = sp[3]
        if sp[2] == "hold":
            cx, cy, s = views[i]
            cap = keys[i]["caption"]
        else:
            u = (tt - sp[0]) / move
            u = u * u * (3 - 2 * u)                          # なめらかに動き出して止まる
            (ax, ay, as_), (bx, by, bs) = views[i], views[i + 1]
            cx, cy = ax + (bx - ax) * u, ay + (by - ay) * u
            s = math.exp(math.log(as_) + (math.log(bs) - math.log(as_)) * u)
            cap = keys[i + 1]["caption"]
        im = Image.new("RGB", (W, H), PAPER)
        im.paste(render(cx, cy, s), (0, 0))
        d = ImageDraw.Draw(im)
        d.rectangle((0, fh, W, fh + 3), fill=INK)
        for n, line in enumerate(cap or []):
            if line:
                text(d, (M, fh + 34 + n * 70), line, 42, INK, n == 0, maxw=W - 2 * M)
        return im
    return int(round(float(sc["dur_s"]) * FPS)), f


def scenes(v):
    vals, order = load_spec()
    S = {sc["id"]: sc for sc in order}
    sy = S["system"]
    cover = still(sy["clip_with"], float(sy.get("cover_t", 86.0)), sy.get("crop"))

    make = {
        "title": lambda sc: title_card(sc, cover),
        "system": compare,
        "contrast": compare,
        "recipe": recipe_card,
        "loop": loop_card,
        "diagram": diagram_card,
        "numbers": lambda sc: numbers_card(sc, vals),
        "cl2": lambda sc: side(sc, "比較: VLA の代わりにスクリプト方策で実行",
                               ["点群で物体の位置を推定し、\n把持・搬送・配置を\n状態機械で切り替える（学習なし）",
                                f"3 個とも収納できた数（{vals['clc_n']} 配置中）"],
                               [(CL2_NAME, vals["clc_cl2"], INK), ("VLA（回復あり）", vals["clc_l2v2"], BLUE)],
                               vals["clc_n"], "低レベル制御\n（逆運動学など）は共通", bold_last=False),
        "cup": lambda sc: side(sc, "別タスク: 「カップを皿に置いて」",
                               ["カップのデモも同じ方法で\n自動生成し、再学習",
                                f"皿に置けた数（{vals['cup_n']} 配置中）"],
                               [("再学習前（立方体のデモのみ）", vals["cup_rreal"], BAR0), ("再学習後", vals["cup_rcp"], BLUE)],
                               vals["cup_n"]),
        "real2sim": lambda sc: list_card(sc, "Real2Sim: 実機の計測値をシミュレーションに反映", [
            "実機の Panda で腕のサーボの応答を測り、シミュレーションに反映した",
            "手先の遅延: 98 ms → 6 ms（実機との差 2 ms 以内）",          # 合わせ込みに使わなかった 32 本で確かめた差（configs/real_v1.yaml）
            "グリッパが閉じるまでの時間と、指の幅のセンサ（6.5 Hz、遅延 0.08 s）も反映"]),
        "limits": lambda sc: list_card(sc, "限界", [
            "方策の評価はシミュレーションのみ（実機での評価なし）",
            "落下の検知は一部のみ。転倒した物体を起こす動作はない",
            "2 つの回復機構の個別の寄与は分離していない（全体として評価）",
            "物体を色で識別するため、同じ色の物体が複数あると扱えない"], numbered=True),
        "closing": lambda sc: title_card(sc, cover),
        "qa": lambda sc: qa_card(sc, vals),
        "instr": instr_card,
    }
    return [make[sc["id"]](sc) for sc in order], order


def cmd_build(a) -> None:
    import av
    vals, _ = load_spec()
    sc, order = scenes(vals)
    total = sum(n for n, _ in sc)
    out = BUILD / "video_s4.mp4"
    box = av.open(str(out), mode="w")
    st = box.add_stream("libx264", rate=FPS)
    st.width, st.height, st.pix_fmt = W, H, "yuv420p"
    st.options = {"crf": "22", "preset": "medium", "threads": "4"}
    for n, f in sc:
        for j in range(n):
            for p in st.encode(av.VideoFrame.from_image(f(j))):
                box.mux(p)
    for p in st.encode():
        box.mux(p)
    box.close()
    (BUILD / OUT_NAME).write_bytes(out.read_bytes())
    (BUILD / "video_s4_texts.json").write_text(json.dumps({"seconds": total / FPS, "texts": sorted(set(V.TEXTS))},
                                                          ensure_ascii=False, indent=1), encoding="utf-8")
    print(BUILD / OUT_NAME, f"{total / FPS:.1f} s", f"{out.stat().st_size / 1e6:.1f} MB")


def cmd_check(a) -> None:
    import re
    s = importlib.util.spec_from_file_location("paper", ROOT / "scripts" / "paper_pdf.py")
    pm = importlib.util.module_from_spec(s)
    s.loader.exec_module(pm)
    info = json.loads((BUILD / "video_s4_texts.json").read_text(encoding="utf-8"))
    joined = "\n".join(info["texts"])
    problems = [f"使わない語 {m.group(0)!r}（{p}）" for p in pm.FORBIDDEN for m in re.finditer(p, joined)]
    if not 60 <= info["seconds"] <= 180:
        problems.append(f"長さ {info['seconds']:.0f} s（1〜3 分の外）")
    _, order = load_spec()
    want = sum(float(sc["dur_s"]) for sc in order)
    if abs(info["seconds"] - want) > 0.5:
        problems.append(f"長さ {info['seconds']:.1f} s が場面の表の合計 {want:.1f} s と違う")
    for sc in order:
        if "t1" in sc and (float(sc["t1"]) - float(sc.get("t0", 0))) / float(sc["dur_s"]) > V.MAX_SPEED + 1e-9:
            problems.append(f"場面 {sc['id']} の倍速が 2 倍を超える")
        for p in sc.get("panels") or []:                  # 指示を変えた実演: 評価の記録はないので、指示の色が全部箱に入ったかだけ確かめる
            m = record(p["clip"])
            if not m.get("all_planned_in_box"):
                problems.append(f"場面 {sc['id']} の {p['clip']} で、指示の色が全部は箱に入っていない")
            if m.get("text") != p["text"]:
                problems.append(f"場面 {sc['id']} の {p['clip']} の指示 {m.get('text')!r} が字幕 {p['text']!r} と違う")
        for key in ("clip", "clip_with", "clip_without"):
            if sc.get(key):
                name = sc[key].split("/")[0]
                if not (V.DEMO / name / "meta.json").is_file():
                    problems.append(f"場面 {sc['id']} の {sc[key]} は元の記録（判定済み）と照らしていない（compare）")
                want = sc.get("expect" + key[4:]) or {}
                recs = sorted((V.EVAL / name).glob("run_[0-9][0-9][0-9][0-9].json")) or sorted((V.EVAL / name).glob("trial_[0-9]*.json"))
                k = int(sc[key].split("/")[1].replace("trial_", "").replace(".mp4", ""))
                if want and len(recs) > k:
                    m = json.loads(recs[k].read_text(encoding="utf-8"))
                    ok = (all(bool(x) for x in m["final_in_place"].values()) if m.get("final_in_place")
                          else bool(m.get("all_three_in_box", m.get("success"))))
                    got = {"success": ok, "replan": bool(m.get("replans")),
                           "stopped": bool(m.get("stopped")), "dropped": bool(m.get("early_events"))}
                    bad = {k: (v, got.get(k)) for k, v in want.items() if got.get(k) != v}
                    if bad:
                        problems.append(f"場面 {sc['id']} の {sc[key]} の回し直しの経過が場面の説明と違う {bad}")
    print(json.dumps({"seconds": round(info["seconds"], 1), "texts": len(info["texts"]), "problems": problems},
                     ensure_ascii=False, indent=1))
    raise SystemExit(1 if problems else 0)


def cmd_stills(a) -> None:
    import av
    out = ROOT / "outputs" / "demo_v4" / "_review"
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
