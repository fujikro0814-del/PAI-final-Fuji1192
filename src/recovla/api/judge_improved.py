"""完了判定の見落としを減らした判定（手順書 docs/api/r1_protocol.md 5 節の 3）。LLM の組み直しありの条件（Lf・Ld）で
api/executor_improved.install が JudgeV2 をこれに替える。ほかの条件の判定は変えない。

見落としの多くは、真値では箱の中なのに俯瞰の画素（box_pixels ≥ box_min_pixels）だけが足りないもの。
立方体が箱の奥の壁ぎわにある、または手前に別の立方体がある（俯瞰のカメラから隠れる）と、俯瞰で見える目標の色の画素が
減る。このとき手首の画像の画素・割合（待機位置から箱を上から見る）と深度の条件は満たしている。
そこで、手首の画像の条件（画素と割合）と深度の条件（上面の高さ）がそろっているときだけ、俯瞰の画素の閾値を
OVERHEAD_RELAX 倍にする。ほかの条件（開・待機位置・静止・hold_s）は同じ。重なり・縁は深度の条件で落ちる。
記録: last に overhead_relaxed（緩めた閾値で通ったか）。
"""
from recovla.runtime.judge import JudgeV2

OVERHEAD_RELAX = 0.5


class JudgeR1(JudgeV2):
    def conditions(self, sensor, wm, hand_pos, speed) -> dict:
        out = super().conditions(sensor, wm, hand_pos, speed)
        p = self.p
        wrist = out["wrist_in_pixels"] >= p["wrist_min_pixels"] and out["wrist_fraction"] >= p["wrist_min_fraction"]
        relaxed = bool(not out["in_box"] and wrist and out["depth_ok"]
                       and out["box_pixels"] >= OVERHEAD_RELAX * float(p["box_min_pixels"]))
        out["overhead_relaxed"] = relaxed
        if relaxed:
            out["in_box"] = True
        return out


def install(judge) -> "JudgeR1":
    judge.__class__ = JudgeR1
    return judge
