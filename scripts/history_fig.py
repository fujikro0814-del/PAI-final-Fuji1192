"""説明資料の図「作品の変遷」（採用した改善を上から開発の順に、試して採らなかったことを右に）を描く。

    python scripts\\history_fig.py      # paper/build/fig_history.png

左の列が採用した改善（上から順に矢印でつなぐ）、右の列はその改善で試して採らなかったこと。黒い細い線の四角に黒の文字、
結果の行だけ色（青＝うまくいった、赤＝うまくいかなかった・採らなかった、灰＝評価中）。
数は各段の記録から写した（paper/build/values.json、docs/real/regen_result.md、configs/demo/video_s4.yaml、
自分の失敗・自分の成功から学び直した評価の判定 outputs/api/r1_judge_R1E.json・r2_judge_R2E.json）。
"""
import pathlib
import re

from PIL import Image, ImageDraw, ImageFont

ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD = ROOT / "paper" / "build"
PAPER, INK = (255, 255, 255), (17, 17, 17)
BLUE, RED, GRAY = (0, 87, 217), (200, 16, 16), (110, 110, 110)
F_R, F_B = r"C:\Windows\Fonts\BIZ-UDGothicR.ttc", r"C:\Windows\Fonts\BIZ-UDGothicB.ttc"
W, PAD, SZ = 2000, 40, 34                            # SZ: A4 の幅 17 cm で約 8.5 pt
MX0, MX1 = PAD, 1250                                  # 採用した改善の列
SX0, SX1 = 1350, W - PAD                              # 採らなかったことの列
IN, ARROW = 28, 70                                    # 枠の内側の余白・矢印の長さ
COL = {"ok": BLUE, "ng": RED, "wait": GRAY, "base": INK}

# 採用した改善: (結果の種類, 見出し, 中身, 結果, 採らなかったこと [(試したこと, 結果)])
# 言葉は説明資料の本文にそろえる。数には何の評価かを添える（/99 は 99 試行で立方体 1 個、/200 は 200 配置で 3 個とも）。
# 出どころ: docs/results/results_s3.md、docs/real/regen_result.md、docs/real/real2sim_log.md（接触の設定）
STEPS = [
    ("ng", "SmolVLA を成功データで追加学習", "生成プログラムが作った成功データで学習（入力は画像と指示文）",
     "指示と違う色の立方体に向かう", []),
    ("ok", "目標の位置を VLA に渡す", "LLM が指示を色の順に分け、カメラ画像の色検出で求めた目標の位置を VLA の状態に加える",
     "指示した色へ向かう（99/99）", []),
    ("base", "立て直しデータを自動生成して追加",
     "生成プログラムにわざと失敗させ、失敗が確定した後だけを記録。比べるために成功データだけのモデルも学習",
     "以降の改善は、すべてこのデータの上に重ねた", []),
    ("ng", "センサの出力だけで動作",
     "立方体の位置は画像から推定。センサの雑音・遅延と、腕の指令の速さ・加速度の制限を再現し、シミュレーションの真の状態は使わない",
     "1 個を箱へ 12/99。雑音・遅延を入れて作り直したデータで学び直しても 23/99", []),
    ("ok", "原因を直したデータで学び直す",
     "掴む直前に、手先が立方体から水平にずれたまま下り始めていた（約 21 mm）。生成プログラムの最後の接近を遅くしたデータで学び直した",
     "1 個を箱へ 85/99。同じ配置・同じわざとの失敗の 27 組で、掴み直して成功したのは立て直しデータあり 13・なし 0",
     [("安全フィルタ（障害物に近づくと停止）", "成功が 84% → 64% に下がり不採用"),
      ("VLA が止まった・ずれたまま閉じようとした場面で、生成プログラムに交代したデータを追加", "事前に決めた条件を満たさず不採用（78 → 78/99）"),
      ("運搬中の落下からの立て直しを VLA に学ばせる", "わざと落とした後の状態が実際の落下の後と合わず、データの段階で中止")]),
    ("ok", "シミュレーションを実機に合わせる",
     "実機の Panda で測った値（腕のサーボ・グリッパ・指の幅のセンサ）を反映し、データを作り直して学び直した",
     "合わせたシミュレーションで 1 個を箱へ 73/99（合わせる前のデータで学んだモデルは 2/99）",
     [("掴みの接触の設定を標準的なものに変える", "斜めに掴んだときの実機の動きと合わず、元に戻した")]),
    ("ok", "落下・時間切れは LLM が手順を組み直す",
     "掴み損ねは VLA がその場で掴み直し、落下・2 回の時間切れは LLM が状況を報告して手順を組み直す",
     "以前の 200 配置で 3 個とも箱へ 143（立て直しデータも LLM の組み直しもないモデルは 71）", []),
    ("ok", "自分の失敗から学び直す",
     "学習用の配置で VLA を動かし、失敗した直後に生成プログラムに交代した立て直しを学習データに追加（外乱・2 個入りのデータも追加）",
     "表 1 と同じ 200 配置で、自力の立て直し率 42.7% → 83.1%（LLM なし）、3 個とも箱へ 144 → 189（LLM あり）",
     [("自分が成功した試行も学習データに追加", "別の新しい 200 配置（LLM なし）で 145 → 142、差は確かめられず（p = 0.81）")]),
]
SIDE_HEAD = "試して採らなかったこと"
MAIN_HEAD = "採用した改善（上から開発の順）"

_fonts = {}


def font(size, bold=False):
    k = (size, bold)
    if k not in _fonts:
        _fonts[k] = ImageFont.truetype(F_B if bold else F_R, size)
    return _fonts[k]


def wrap(d, s, size, maxw, bold=False):
    """幅 maxw で折り返す（句読点・閉じ括弧を行頭に置かない、数・英字のひと続きを折らない、最後の行を 1〜3 字だけにしない）。"""
    fo, lines, cur = font(size, bold), [], ""
    for ch in re.findall(r"[A-Za-z0-9/.%]+(?: (?:mm|cm|ms|s)(?![A-Za-z]))?|.", s):     # 数と単位（21 mm）も折らない
        if cur and d.textlength(cur + ch, font=fo) > maxw and ch not in "、。）」・":
            lines.append(cur)
            cur = ""
        cur += ch
    if lines and len(cur) <= 3:
        k = max(1, 4 - len(cur))
        lines[-1], cur = lines[-1][:-k].rstrip(), (lines[-1][-k:] + cur).lstrip()
    return lines + ([cur] if cur else [])


def block(d, x, y, s, maxw, col=INK, bold=False, size=SZ, draw=True):
    """折り返して書き、下端の y を返す。"""
    for ln in wrap(d, s, size, maxw, bold):
        if draw:
            d.text((x, y), ln, font=font(size, bold), fill=col)
        y += int(size * 1.45)
    return y


def main_box(d, y, st, draw=True):
    """採用した改善の枠: 見出し（太字）・中身・結果（色の太字）。下端の y を返す。"""
    kind, head, body, res, _ = st
    x, w = MX0 + IN, MX1 - MX0 - 2 * IN
    yy = block(d, x, y + IN, head, w, INK, True, 36, draw)
    if body:
        yy = block(d, x, yy + 4, body, w, INK, False, SZ, draw)
    yy = block(d, x, yy + 8, res, w, COL[kind], True, SZ, draw)
    y1 = yy + IN - 10
    if draw:
        d.rectangle((MX0, y, MX1, y1), outline=INK, width=2)
    return y1


def side_box(d, y, item, draw=True):
    tried, res = item
    x, w = SX0 + IN, SX1 - SX0 - 2 * IN
    yy = block(d, x, y + 20, tried, w, INK, False, 30, draw)
    yy = block(d, x, yy + 4, res, w, RED, True, 30, draw)
    y1 = yy + 10
    if draw:
        d.rectangle((SX0, y, SX1, y1), outline=INK, width=2)
    return y1


def arrow(d, x, y0, y1, w=3, head=14):
    d.line((x, y0, x, y1 - head), fill=INK, width=w)
    d.polygon([(x - head * 0.7, y1 - head - 2), (x + head * 0.7, y1 - head - 2), (x, y1)], fill=INK)


def harrow(d, x0, x1, y, head=12):
    d.line((x0, y, x1 - head, y), fill=INK, width=2)
    d.polygon([(x1 - head - 2, y - head * 0.6), (x1 - head - 2, y + head * 0.6), (x1, y)], fill=INK)


def layout(d, draw):
    if draw:
        d.text((MX0, 8), MAIN_HEAD, font=font(30, True), fill=INK)
        d.text((SX0, 8), SIDE_HEAD, font=font(30, True), fill=INK)
    y = 60
    for k, st in enumerate(STEPS):
        y_main = main_box(d, y, st, False)
        ys, tops = y, []
        for it in st[4]:
            tops.append(ys)
            ys = side_box(d, ys, it, False) + 16
        y_end = max(y_main, ys - 16)
        if draw:
            main_box(d, y, st)
            if st[4]:                                # 採用した改善の枠の右から、採らなかったことの枠それぞれへ
                xm, ym = (MX1 + SX0) // 2, y + 44
                d.line((MX1, ym, xm, ym), fill=INK, width=2)
                d.line((xm, ym, xm, tops[-1] + 44), fill=INK, width=2)
            for it, t in zip(st[4], tops):
                side_box(d, t, it)
                harrow(d, (MX1 + SX0) // 2, SX0, t + 44)
            if k < len(STEPS) - 1:
                arrow(d, MX0 + 120, y_main, y_end + ARROW)
        y = y_end + ARROW
    return y - ARROW + 12


def main() -> int:
    probe = ImageDraw.Draw(Image.new("RGB", (W, 10)))
    im = Image.new("RGB", (W, layout(probe, False)), PAPER)
    layout(ImageDraw.Draw(im), True)
    p = BUILD / "fig_history.png"
    im.save(p, optimize=True)
    print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
