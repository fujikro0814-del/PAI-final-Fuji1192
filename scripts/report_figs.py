"""説明資料の図（システム構成・立て直しデータの作り方・詳しいブロック線図）を描く。

scripts/video_scenes.py が、ブロック線図（fig_detail・DETAIL_PARTS）を動画の部品として読む。

    .venv\\Scripts\\python.exe scripts\\report_figs.py      # paper/build/fig1_system.png・fig2_recovery_demo.png

図 1: システム構成（方策ループとタスクループ）。記号の定義は説明資料の図の説明文に書く。
図 2: 立て直しデータの自動生成。学習データは成功 240・掴み損ねからの立て直し 60・置き損ねからの立て直し 30
（docs/real/regen_models.md）。見た目は動画と同じ（白地に黒、青＝立て直しのための部分、赤＝失敗、角の立った四角）。
"""
import pathlib
import re

from PIL import Image, ImageDraw, ImageFont

ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD = ROOT / "paper" / "build"
PAPER, INK, SUB = (255, 255, 255), (17, 17, 17), (70, 70, 70)
BLUE, RED, BAR0 = (0, 87, 217), (200, 16, 16), (130, 130, 130)
F_JA = (r"C:\Windows\Fonts\BIZ-UDGothicR.ttc", r"C:\Windows\Fonts\BIZ-UDGothicB.ttc")
F_IT, F_RM = r"C:\Windows\Fonts\cambriai.ttf", r"C:\Windows\Fonts\cambria.ttc"
_cache = {}


def font(path, size):
    k = (path, size)
    if k not in _cache:
        _cache[k] = ImageFont.truetype(path, size)
    return _cache[k]


def _runs(s):
    """数式まじりの文字列を (文字, 種類, 立体か) に分ける。_{…} は下付き、^{…} は上付き。1 文字のラテン文字・ギリシャ文字は
    変数として斜体、2 文字以上続くラテン文字（LLM・VLA・done・top など）は立体、数字・記号はローマン、日本語はゴシック。"""
    out, i = [], 0
    while i < len(s):
        if s[i] in "_^" and i + 1 < len(s) and s[i + 1] == "{":
            j = s.index("}", i)
            for ch in s[i + 2:j]:
                out.append([ch, "sub" if s[i] == "_" else "sup", False])
            i = j + 1
            continue
        out.append([s[i], "base", False])
        i += 1
    k = 0
    depth = 0
    while k < len(out):                                   # 同じ種類の中で続くラテン文字の長さ
        ch = out[k][0]
        depth += {"[": 1, "]": -1}.get(ch, 0) if out[k][1] == "base" else 0
        if re.match(r"[A-Za-z]", ch):
            j = k
            while j < len(out) and re.match(r"[A-Za-z]", out[j][0]) and out[j][1] == out[k][1]:
                j += 1
            prev = out[k - 1][0] if k > 0 else ""
            prev2 = out[k - 2][0] if k > 1 else ""
            nxt = out[j][0] if j < len(out) else ""
            nxt2 = out[j + 1][0] if j + 1 < len(out) else ""
            unit = (prev.isdigit() or prev == "/" or (prev == " " and prev2.isdigit()) or nxt.isdigit() or depth > 0
                    or (nxt == " " and nxt2 and ord(nxt2) > 0x3000) or (nxt and ord(nxt) > 0x3000 and nxt not in "∈（）、。"))
            for m in range(k, j):
                out[m][2] = j - k >= 2 or unit             # 2 文字以上・単位・型番・層の名前は立体
            k = j
        else:
            k += 1
    return out


def _font_for(ch, size, bold, upright=False):
    if re.match(r"[A-Za-zα-ωΑ-Ω]", ch):
        return font(F_RM if upright else F_IT, size)
    if re.match(r"[0-9()=,.:|·∈↦~…{}′+\-−→←\[\]ℝ]", ch) or ch == " ":
        return font(F_RM, size)
    return font(F_JA[1 if bold else 0], size)


def _layout(d, s, size, bold):
    """各文字の (文字, フォント, x のずれ, y のずれ) と全体の幅。上付きの直後の下付きは、同じ位置に重ねる（I^{top}_{t}）。"""
    items, x = [], 0.0
    runs = _runs(s)
    i = 0
    while i < len(runs):
        ch, kind, up = runs[i]
        if kind == "base":
            fo = _font_for(ch, size, bold, up)
            items.append((ch, fo, x, 0))
            x += d.textlength(ch, font=fo)
            i += 1
            continue
        groups = []                                       # 続く上付き・下付きのかたまり
        while i < len(runs) and runs[i][1] != "base":
            k0 = runs[i][1]
            g = []
            while i < len(runs) and runs[i][1] == k0:
                g.append(runs[i])
                i += 1
            groups.append((k0, g))
        x0, wmax = x, 0.0
        for k0, g in groups:
            xx = x0
            for ch, _, up2 in g:
                fo = _font_for(ch, int(size * 0.68), bold, up2)
                items.append((ch, fo, xx, int(size * 0.42) if k0 == "sub" else -int(size * 0.12)))
                xx += d.textlength(ch, font=fo)
            wmax = max(wmax, xx - x0)
        x = x0 + wmax
    return items, x


def mwidth(d, s, size, bold=False):
    return _layout(d, s, size, bold)[1]


TEXTS = []                                                  # 図に書いた文字（使わない語の点検に使う）


def mtext(d, xy, s, size, fill=INK, bold=False, anchor="l"):
    """数式まじりの文字列を書く（anchor: l 左揃え・m 中央揃え。y は行の上端）。"""
    TEXTS.append(s)
    items, w = _layout(d, s, size, bold)
    x, y = xy
    if anchor == "m":
        x -= w / 2
    for ch, fo, dx, dy in items:
        d.text((x + dx, y + dy), ch, font=fo, fill=fill)


def box(d, x0, y0, x1, y1, head, sub=None, fill=PAPER, edge=INK, size=40):
    d.rectangle((x0, y0, x1, y1), fill=fill, outline=edge, width=4)
    col = PAPER if fill == BLUE else INK
    cx = (x0 + x1) // 2
    if sub:
        mtext(d, (cx, (y0 + y1) // 2 - size - 4), head, size, col, True, "m")
        mtext(d, (cx, (y0 + y1) // 2 + 10), sub, int(size * 0.9), col, False, "m")
    else:
        mtext(d, (cx, (y0 + y1) // 2 - size // 2 - 4), head, size, col, True, "m")


def arrow(d, pts, col=INK, w=6):
    for a, b in zip(pts, pts[1:]):
        d.line([a, b], fill=col, width=w)
    (xa, ya), (xb, yb) = pts[-2], pts[-1]
    ux, uy = (xb > xa) - (xb < xa), (yb > ya) - (yb < ya)
    px, py = -uy, ux
    d.polygon([(xb, yb), (xb - 28 * ux + 15 * px, yb - 28 * uy + 15 * py), (xb - 28 * ux - 15 * px, yb - 28 * uy - 15 * py)],
              fill=col)


def fig1():
    W, H = 2400, 1330
    im = Image.new("RGB", (W, H), PAPER)
    d = ImageDraw.Draw(im)
    L0, L1 = 60, 860                                         # 左の列（タスクループ）
    box(d, L0, 40, L1, 150, "指示 ℓ「全部片付けて」")
    box(d, L0, 230, L1, 380, "タスク分解 LLM", "(ℓ, z_{0}) ↦ G")
    box(d, L0, 460, L1, 610, "サブタスク列", "G = (g_{1}, …, g_{K}),  g_{k} ∈ {赤, 緑, 青}")
    box(d, L0, 1000, L1, 1150, "再計画 LLM", "(ℓ, G, h, z_{t}, e) ↦ (G′, r)", fill=BLUE, edge=BLUE)
    arrow(d, [(460, 150), (460, 230)])
    arrow(d, [(460, 380), (460, 460)])
    arrow(d, [(460, 1000), (460, 610)], BLUE)
    mtext(d, (490, 780), "修正した計画 G′", 36, BLUE, True)
    mtext(d, (L0, 1175), "状況報告 r（日本語）を人に返す", 34, BLUE, True)
    # 右: 方策ループ
    box(d, 1020, 460, 1500, 610, "VLA  π_{θ}", "a_{t:t+H} ~ π_{θ}(· | o_{t}, g_{k})", edge=BLUE)
    box(d, 1600, 460, 1950, 610, "逆運動学・制限", "a_{t} ↦ q^{cmd}")
    box(d, 2050, 460, W - 60, 610, "Panda", "シミュレーション", size=34)
    box(d, 1560, 690, 2140, 820, "観測  o_{t} = (I^{top}_{t}, I^{wr}_{t}, s_{t})", size=40)
    arrow(d, [(L1, 535), (1020, 535)])
    mtext(d, (940, 470), "g_{k}", 40, INK, False, "m")
    arrow(d, [(1500, 535), (1600, 535)])
    arrow(d, [(1950, 535), (2050, 535)])
    arrow(d, [(2195, 610), (2195, 755), (2140, 755)])
    arrow(d, [(1560, 755), (1260, 755), (1260, 610)])
    mtext(d, (1020, 300), "方策ループ（10 Hz、チャンク長 H = 50）", 40, BLUE, True)
    mtext(d, (1020, 360), "把持失敗 → π_{θ} がその場で再把持（回復デモで学習）", 36, BLUE, False)
    # 下: 失敗の検知・完了判定 → LLM の組み直し
    box(d, 1560, 1000, W - 60, 1150, "失敗検知・完了判定", "e_{t} ∈ {done, drop, timeout}")
    arrow(d, [(1850, 820), (1850, 1000)])
    mtext(d, (1880, 880), "画像・指の幅・関節角", 32, SUB, False)
    arrow(d, [(1560, 1075), (L1, 1075)], BLUE)
    mtext(d, (1210, 1010), "e ∈ {drop, timeout}", 40, RED, False, "m")
    mtext(d, (1210, 1095), "done なら k ← k+1", 34, SUB, False, "m")
    mtext(d, (1020, 1230), "タスクループ（イベント駆動）: 落下・タイムアウト → LLM が再計画", 40, BLUE, True)
    return im


def fig2():
    W, H = 2400, 760
    im = Image.new("RGB", (W, H), PAPER)
    d = ImageDraw.Draw(im)
    boxes = [("成功デモの生成", ["エキスパートが", "通常どおり把持し", "箱へ搬送"]),
             ("意図的な失敗", ["把持失敗: 位置か高さを", "ずらして閉じる", "置き損ね: 机の上で放す"]),
             ("失敗の確定", ["物体の状態で判定", "（持ち上がらない、", "机の上で静止）"]),
             ("回復して収納", ["エキスパートが", "再把持・再搬送して", "箱に入れるまで"])]
    M, gap = 60, 70
    bw = (W - 2 * M - 3 * gap) // 4
    by, bh = 40, 330
    for k, (head, body) in enumerate(boxes):
        x = M + k * (bw + gap)
        last = k == len(boxes) - 1
        d.rectangle((x, by, x + bw, by + bh), fill=BLUE if last else PAPER, outline=BLUE if last else INK, width=4)
        col = PAPER if last else INK
        mtext(d, (x + 30, by + 30), head, 46, col, True)
        for i, s in enumerate(body):
            mtext(d, (x + 30, by + 130 + i * 58), s, 38, col)
        if not last:
            ay = by + bh // 2
            arrow(d, [(x + bw + 8, ay), (x + bw + gap - 8, ay)])
    rec_x = M + 3 * (bw + gap)
    ty = by + bh + 70
    for x in range(M, rec_x - 20, 44):
        d.rectangle((x, ty + 36, x + 22, ty + 40), fill=BAR0)
    mtext(d, (M, ty - 4), "記録しない", 38, SUB)
    d.rectangle((rec_x, ty, W - M, ty + 84), fill=BLUE)
    mtext(d, (rec_x + 26, ty + 18), "回復デモとして記録", 44, PAPER, True)
    arrow(d, [(M, ty + 125), (W - M, ty + 125)])
    mtext(d, (W - M - 90, ty + 150), "時間", 36, SUB)
    mtext(d, (M, ty + 220), "学習データ: 成功デモ 240 本 ＋ 回復デモ 90 本（把持失敗 60・置き損ね 30）＝ 330 本。すべて自動生成",
          40, INK)
    return im


# ------------------------------------------------------------------ 詳しいブロック線図
# 決まり: 信号は各ブロックの左辺から入り右辺から出る。帰還は下側を通って左へ戻る。値はコードと設定で確かめたもの
# （docs/report/details.md の 1〜3 節）。動画では A〜F を順に拡大して字幕で説明する（DETAIL_PARTS の範囲）。
LIGHT_BLUE, ENV = (232, 241, 253), (238, 238, 238)
DW, DH = 3200, 1990
DETAIL_PARTS = {"all": (0, 0, DW, DH), "A": (0, 20, 980, 1000), "B": (940, 100, 1600, 1000),
                "CD": (1560, 60, 2620, 1000), "E": (2580, 100, 3200, 1000), "F": (170, 1150, 3195, 1750),
                "F1": (170, 1150, 1900, 1750), "F2": (1500, 1150, 3195, 1750)}


def dblock(d, x0, y0, x1, y1, title, lines=(), kind="plain", ts=28, ls=23):
    """ブロック: kind は plain（設計した処理・借りたモデル）・learned（学習した要素）・recovery（立て直しのための要素）・env（環境）。"""
    fill = {"learned": LIGHT_BLUE, "env": ENV}.get(kind, PAPER)
    edge = BLUE if kind in ("learned", "recovery") else INK
    d.rectangle((x0, y0, x1, y1), fill=fill, outline=edge, width=4 if kind == "recovery" else 3)
    mtext(d, (x0 + 14, y0 + 10), title, ts, INK, True)
    for i, s in enumerate(lines):
        mtext(d, (x0 + 14, y0 + 16 + ts + i * (ls + 9)), s, ls, SUB if kind != "learned" else INK)


def frame(d, x0, y0, x1, y1, title, dashed=False):
    if dashed:
        for x in range(x0, x1, 28):
            d.line([(x, y0), (min(x + 16, x1), y0)], fill=SUB, width=3)
            d.line([(x, y1), (min(x + 16, x1), y1)], fill=SUB, width=3)
        for y in range(y0, y1, 28):
            d.line([(x0, y), (x0, min(y + 16, y1))], fill=SUB, width=3)
            d.line([(x1, y), (x1, min(y + 16, y1))], fill=SUB, width=3)
    else:
        d.rectangle((x0, y0, x1, y1), outline=(150, 150, 150), width=3)
    mtext(d, (x0 + 14, y0 + 12), title, 32, INK, True)


def wire(d, pts, label=None, at=None, col=INK, size=24, dashed=False):
    """直角に折れる線と矢印。label は at（x, y）に書く。"""
    if dashed:
        for a, b in zip(pts, pts[1:]):
            n = max(1, int(max(abs(b[0] - a[0]), abs(b[1] - a[1])) // 24))
            for i in range(0, n, 2):
                p = (a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n)
                q = (a[0] + (b[0] - a[0]) * min(n, i + 1) / n, a[1] + (b[1] - a[1]) * min(n, i + 1) / n)
                d.line([p, q], fill=col, width=4)
        (xa, ya), (xb, yb) = pts[-2], pts[-1]
        ux, uy = (xb > xa) - (xb < xa), (yb > ya) - (yb < ya)
        px, py = -uy, ux
        d.polygon([(xb, yb), (xb - 22 * ux + 11 * px, yb - 22 * uy + 11 * py), (xb - 22 * ux - 11 * px, yb - 22 * uy - 11 * py)],
                  fill=col)
    else:
        for a, b in zip(pts, pts[1:]):
            d.line([a, b], fill=col, width=4)
        (xa, ya), (xb, yb) = pts[-2], pts[-1]
        ux, uy = (xb > xa) - (xb < xa), (yb > ya) - (yb < ya)
        px, py = -uy, ux
        d.polygon([(xb, yb), (xb - 22 * ux + 11 * px, yb - 22 * uy + 11 * py), (xb - 22 * ux - 11 * px, yb - 22 * uy - 11 * py)],
                  fill=col)
    if label:
        mtext(d, at, label, size, col)


def pill(d, x0, y0, x1, y1, s):
    d.rounded_rectangle((x0, y0, x1, y1), radius=(y1 - y0) // 2, outline=INK, width=3, fill=PAPER)
    mtext(d, ((x0 + x1) // 2, (y0 + y1) // 2 - 16), s, 28, INK, True, "m")


def fig_detail():
    im = Image.new("RGB", (DW, DH), PAPER)
    d = ImageDraw.Draw(im)
    mtext(d, (20, 14), "立て直しの仕組み（ブロック線図）", 30, INK, True)
    pill(d, 20, 235, 170, 295, "指示 ℓ")
    pill(d, 2900, 40, 3180, 100, "状況報告 r（人へ）")

    # A タスク層
    frame(d, 190, 130, 930, 990, "A タスク層（LLM・実行器、イベント駆動）")
    dblock(d, 230, 200, 890, 330, "タスク分解 LLM（Claude Haiku 5.5）", ["(ℓ, z_{0}) ↦ G、構造化出力"])
    dblock(d, 230, 360, 890, 465, "計画の検査（規則）", ["机にない色・箱の中の色・重複を弾く", "弾いたら人に聞き返す"], ls=22)
    dblock(d, 230, 500, 890, 690, "実行器", ["サブタスク g_{k} を 1 つずつ VLA へ（done で k ← k+1）",
                                            "30 s でタイムアウト → 待機位置に戻り 1 回だけ再試行",
                                            "2 回目のタイムアウト → e = timeout", "（組み直しの LLM を使わないときは、ここで停止）"])
    dblock(d, 230, 720, 890, 830, "組み直しの LLM（Claude Sonnet 5.5）", ["(ℓ, G, h, z_{t}, e) ↦ (d, r)",
                                                                      "d（次の動き）: 次の色・順の入れ替え・飛ばす・終了・停止"],
           kind="recovery", ls=22)
    dblock(d, 230, 855, 890, 965, "次の動きの検査（規則）", ["箱の中の色を弾く・同じ色の手順は 2 回まで",
                                                                  "組み直しは 1 試行に 2 回まで・弾いたら停止して人に知らせる"], kind="recovery", ls=22)
    wire(d, [(170, 265), (230, 265)])
    wire(d, [(560, 330), (560, 360)])
    wire(d, [(560, 465), (560, 500)], "G = (g_{1}, …, g_{K})", (580, 467))
    wire(d, [(560, 690), (560, 720)], "e = timeout", (580, 692), RED)
    wire(d, [(560, 830), (560, 855)])
    wire(d, [(230, 910), (205, 910), (205, 600), (230, 600)], "G′", (212, 760), BLUE)

    # A → B（サブタスク）、A → C（待機位置）、A → 人（報告）
    wire(d, [(890, 560), (975, 560), (975, 320), (1030, 320)], "g_{k}", (982, 520))
    wire(d, [(890, 650), (945, 650), (945, 112), (1950, 112), (1950, 180)], "待機位置・ホームへ", (1200, 76), size=22)
    wire(d, [(890, 790), (960, 790), (960, 70), (2900, 70)], None, None, BLUE)

    # B 方策層
    frame(d, 990, 130, 1530, 990, "B 方策層（VLA）")
    dblock(d, 1030, 200, 1490, 440, "VLA  π_{θ}（SmolVLA）", ["VLM（SmolVLM2-500M）: 固定", "行動エキスパート・状態の射影: 学習",
                                                             "フローマッチング: ノイズから", "10 ステップで行動の列を生成",
                                                             "a_{t:t+49} ∈ ℝ^{50×7}（1 行 0.1 s）"], kind="learned")
    dblock(d, 1030, 480, 1490, 630, "非同期実行", ["0.6 s ごとに次の 50 行（5 s 分）を推論", "その間も前の行を実行",
                                              "観測からの経過時間に合う行から使う"], ls=22)
    dblock(d, 1030, 660, 1490, 800, "グリッパの開閉の確定", ["閉じている間の「開く」は", "0.5 s 続いたら開く", "閉じるのはすぐ"], ls=22)
    mtext(d, (1040, 825), "g_{k} は指示文 \"put the {c} cube in the box\" で入る", 21, SUB)
    mtext(d, (1040, 860), "推論時間は測った分布から与える（中央値 0.32 s）", 21, SUB)
    wire(d, [(1260, 440), (1260, 480)])
    wire(d, [(1260, 630), (1260, 660)])

    # C 低レベル制御
    frame(d, 1590, 130, 2030, 990, "C 低レベル制御")
    dblock(d, 1630, 180, 1990, 260, "待機位置・ホームへの移動", ["実行器の指令で（VLA の代わりに）"], ls=21)
    dblock(d, 1630, 300, 1990, 400, "目標 x^{cmd} の積分（500 Hz）", ["作業空間で制限"], ls=22)
    dblock(d, 1630, 425, 1990, 520, "目標への追従", ["2 次・臨界減衰、ω = 20 rad/s"], ls=22)
    dblock(d, 1630, 540, 1990, 660, "微分逆運動学", ["減衰つき最小二乗（λ = 0.05）", "零空間の姿勢項"], ls=22)
    dblock(d, 1630, 690, 1990, 790, "関節の制限（1 kHz）", ["位置・速度・加速度・躍度"], ls=22)
    dblock(d, 1630, 830, 1990, 930, "グリッパの指令", ["開閉 → 指の幅の指令"], ls=22)
    wire(d, [(1810, 260), (1810, 300)])
    wire(d, [(1810, 400), (1810, 425)])
    wire(d, [(1810, 520), (1810, 540)])
    wire(d, [(1810, 660), (1810, 690)])
    wire(d, [(1490, 550), (1555, 550), (1555, 350), (1630, 350)], "Δx_{t}", (1560, 300), size=24)
    wire(d, [(1490, 730), (1570, 730), (1570, 880), (1630, 880)], "g_{t} ∈ {±1}", (1420, 905), size=22)

    # D プラント
    frame(d, 2090, 130, 2600, 990, "D プラント（Panda、MuJoCo）")
    d.rectangle((2120, 180, 2575, 545), outline=(150, 150, 150), width=2)
    mtext(d, (2130, 186), "センサ", 22, SUB, True)
    dblock(d, 2140, 220, 2560, 315, "俯瞰 RGB-D", ["D455 相当・縦の画角 45°", "遅延 33〜100 ms"], kind="env", ts=24, ls=20)
    dblock(d, 2140, 322, 2560, 395, "手先 RGB-D", ["D405 相当・縦の画角 58°"], kind="env", ts=24, ls=20)
    dblock(d, 2140, 402, 2560, 450, "関節角 q", [], kind="env", ts=24)
    dblock(d, 2140, 458, 2560, 535, "指の幅 w ▲", ["6.5 Hz・遅延 0.08 s"], kind="env", ts=24, ls=20)
    dblock(d, 2120, 600, 2330, 780, "腕のサーボ ▲", ["実機の応答で同定", "比例ゲイン・速度 FF", "遅延 8 ms・重力補償"], kind="env",
           ts=26, ls=20)
    dblock(d, 2120, 820, 2330, 950, "ハンド ▲", ["0.15 s で動き出す", "0.08 m/s"], kind="env", ts=26, ls=20)
    dblock(d, 2370, 600, 2575, 950, "接触・物体", ["物理エンジン", "立方体 3・箱"], kind="env", ts=26, ls=20)
    wire(d, [(1990, 740), (2120, 740)], "q^{cmd} [rad]", (1996, 700), size=20)
    wire(d, [(1990, 885), (2120, 885)], "w^{cmd}", (2030, 843), size=22)
    wire(d, [(2330, 690), (2370, 690)])
    wire(d, [(2330, 885), (2370, 885)])
    wire(d, [(2470, 600), (2470, 545)])

    # D → E（センサの信号の束）
    d.line([(2620, 230), (2620, 960)], fill=INK, width=6)
    for y in (267, 358, 426, 496):
        d.line([(2560, y), (2620, y)], fill=INK, width=4)
    mtext(d, (2600, 965), "I^{top}, I^{wr}, 深度, q, w", 21, SUB)

    # E 知覚・判定
    frame(d, 2660, 130, 3130, 990, "E 知覚・判定")
    eb = [(195, 335, "物体の推定", ["画像と深度 → 色の分割", "→ 物体・箱の位置", "計算 0.22〜0.62 s（実測から）"], "plain"),
          (350, 455, "目標位置 c_{t}", ["指示の色の重心", "→ 机の高さの平面へ投影"], "plain"),
          (470, 575, "状態 s_{t} ∈ ℝ^{19}", ["手先の位置・姿勢・指の幅・関節角", "c_{t}・配置位置 p_{t}（箱の空きセル）"], "plain"),
          (590, 665, "場面の要約 z_{t}", ["机の上・箱の中・把持中の色"], "plain"),
          (680, 755, "完了判定", ["箱の中の画素・手先の画像・深度"], "plain"),
          (790, 950, "失敗検知（落下）", ["把持中に指の幅が縮む・開く", "→ 画像で 1.0 s 確かめる"], "recovery")]
    for y0, y1, t, ls_, k in eb:
        dblock(d, 2700, y0, 3100, y1, t, ls_, kind=k, ts=24, ls=20)
        wire(d, [(2620, (y0 + y1) // 2), (2700, (y0 + y1) // 2)])

    # 帰還（下側を通って左へ）。札は各線の下に
    wire(d, [(3100, 522), (3140, 522), (3140, 1030), (1005, 1030), (1005, 400), (1030, 400)],
         "o_{t} = (I^{top}_{t}, I^{wr}_{t}, s_{t})（観測、B へ）", (1700, 1034), size=22)
    wire(d, [(3100, 627), (3157, 627), (3157, 1070), (175, 1070), (175, 790), (230, 790)],
         "z_{t}（場面の要約）・e = drop（落下、A の組み直しへ）", (1700, 1074), BLUE, size=22)
    wire(d, [(3100, 870), (3157, 870)], None, None, BLUE)
    wire(d, [(175, 300), (230, 300)], None, None, BLUE)
    d.line([(175, 1070), (175, 300)], fill=BLUE, width=4)
    wire(d, [(3100, 717), (3175, 717), (3175, 1110), (160, 1110), (160, 520), (230, 520)],
         "e = done（完了、A の実行器へ）", (1700, 1114), size=22)

    # F 学習（オフライン）
    frame(d, 190, 1160, 3180, 1740, "F 学習（オフライン）: 立て直しのデータの自動生成と、VLA 自身の失敗からの学び直し", dashed=True)
    mtext(d, (230, 1214), "(1) わざと起こした失敗から", 22, SUB)
    fb = [(230, 560, "場面のサンプル", ["学習用の乱数シード", "（評価とは別）"], "env"),
          (610, 1000, "生成プログラム", ["シミュレーションの正確な", "位置を使う（学習なし）"], "plain"),
          (1050, 1520, "意図的な失敗", ["把持失敗: 把持点を 2.5〜3.5 cm ずらす", "または閉じる高さを 2.5〜5 cm 上げる",
                                         "置き損ね: 机の上で放す"], "recovery"),
          (1570, 1880, "失敗の確定", ["物体の状態で判定", "（持ち上がらない・机で静止）"], "recovery"),
          (1930, 2290, "立て直しの記録", ["確定以降、生成プログラムが", "掴み直して箱へ"], "recovery"),
          (2340, 2680, "データ 330 本", ["成功 240・立て直し 90", "10 fps・50,659 フレーム"], "plain"),
          (2730, 3140, "追加学習（1 回目）", ["行動エキスパート・状態の射影", "20,000 ステップ・約 2 時間 20 分", "RTX 4080 SUPER ×1"], "learned")]
    for i, (x0, x1, t, ls_, k) in enumerate(fb):
        dblock(d, x0, 1260, x1, 1430, t, ls_, kind=k, ts=26, ls=21)
        if i:
            wire(d, [(fb[i - 1][1], 1345), (x0, 1345)])
    wire(d, [(805, 1260), (805, 1225), (2510, 1225), (2510, 1260)], "成功 240 本", (1600, 1188), size=22)

    # 2 段目: 1 回目の VLA を学習用の初期状態で動かし、失敗した直後から生成プログラムに替えて記録する
    wire(d, [(2935, 1430), (2935, 1450), (395, 1450), (395, 1495)])
    mtext(d, (425, 1458), "(2) VLA 自身の失敗から", 22, SUB)
    fb2 = [(230, 560, "VLA で試す", ["1 回目の VLA を", "学習用の初期状態で動かす"], "plain"),
           (610, 1000, "VLA の失敗を検知", ["掴み損ね・落下・", "置く直前に外した", "（シミュの状態で判定）"], "recovery"),
           (1050, 1520, "生成プログラムに交代", ["失敗の直後から", "箱に入れるまでを記録"], "recovery"),
           (1570, 2290, "足すデータ 340 本", ["交代後の立て直し 200", "途中で物をずらす外乱 80", "箱に 2 個入りの場面の成功 60"], "plain"),
           (2340, 3140, "学び直し（使う重み）", ["330 本＋340 本＝670 本・99,402 フレーム", "配置位置を入力に加える・40,000 ステップ・約 4 時間 45 分",
                                       "RTX 4080 SUPER ×1"], "learned")]
    for i, (x0, x1, t, ls_, k) in enumerate(fb2):
        dblock(d, x0, 1495, x1, 1655, t, ls_, kind=k, ts=26, ls=21)
        if i:
            wire(d, [(fb2[i - 1][1], 1575), (x0, 1575)])
    wire(d, [(2740, 1655), (2740, 1680), (1545, 1680), (1545, 1150), (1290, 1150), (1290, 990)], None, None, BLUE, dashed=True)
    mtext(d, (1580, 1688), "学び直した重み θ を π_{θ} へ（B 方策層）", 24, BLUE, True)

    # Real2Sim と凡例
    dblock(d, 230, 1780, 900, 1890, "実機の Panda で計測（2026-10-11）", ["サーボの応答・グリッパの時間・指の幅のセンサ"],
           ts=24, ls=20)
    dblock(d, 980, 1780, 1700, 1890, "シミュレーションのパラメータ（▲）", ["データの生成（F）と評価（D）の両方で使う"],
           kind="env", ts=24, ls=20)
    wire(d, [(900, 1835), (980, 1835)])
    lx, ly = 1800, 1758
    sw = [("learned", "学習した要素"), ("recovery", "立て直しのための要素"), ("env", "シミュレーションの環境"), ("plain", "設計した処理・借りたモデル")]
    for i, (k, s) in enumerate(sw):
        x = lx + (i % 2) * 690
        y = ly + (i // 2) * 46
        fill = {"learned": LIGHT_BLUE, "env": ENV}.get(k, PAPER)
        d.rectangle((x, y + 4, x + 56, y + 36), fill=fill, outline=BLUE if k in ("learned", "recovery") else INK,
                    width=4 if k == "recovery" else 3)
        mtext(d, (x + 70, y + 6), s, 24, INK)
    mtext(d, (lx, ly + 98), "▲: 実機の計測値を反映（Real2Sim）　赤: 失敗のイベント　青の線: 立て直しのための信号", 22, SUB)
    mtext(d, (lx, ly + 127), "ℓ: 指示　G: サブタスクの列　G′: 組み直した列　g_{k}: k 番目のサブタスク（色）　o_{t}: 観測　s_{t}: 状態", 22, SUB)
    mtext(d, (lx, ly + 156), "c_{t}: 目標位置　p_{t}: 配置位置　a_{t} = (Δx_{t}, g_{t}): 行動（0.1 s あたりの手先の変位 [m]、グリッパの開閉）", 22, SUB)
    mtext(d, (lx, ly + 185), "z_{t}: 場面の要約　h: 各サブタスクの結果　e: イベント　r: 状況報告", 22, SUB)
    return im


def main():
    BUILD.mkdir(parents=True, exist_ok=True)
    for name, f in (("fig1_system.png", fig1), ("fig2_recovery_demo.png", fig2), ("fig1_system_detail.png", fig_detail)):
        f().save(BUILD / name)
        print(BUILD / name)
    import json
    (BUILD / "report_figs_texts.json").write_text(json.dumps(sorted(set(TEXTS)), ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
