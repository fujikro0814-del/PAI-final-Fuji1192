"""色の判定と、目標位置の入力（俯瞰画像の指示色の位置）の numpy だけの部品。

学習の変換（recovla.perception.color が再び公開する）と実行系が同じ関数を使う。違うのは較正の値だけで、
変換は描画に使った本当のカメラ、実行系は SetupInfo の外部パラメータ（実行系が持つ値）と机の面を使う。
実行系がシミュレーションの真値に触れないように、runtime の中に置く。

画素の分類: 色 c の画素 = そのチャンネルが min_value 以上、かつ ほかの 2 チャンネルの大きい方より min_margin 以上大きい。
"""
import dataclasses

import numpy as np

CHANNEL = {"red": 0, "green": 1, "blue": 2}


@dataclasses.dataclass(frozen=True)
class Calibration:
    pos: np.ndarray          # (3,) カメラの位置（世界）
    rot: np.ndarray          # (3, 3) 列: カメラの x（右）・y（上）・z（後ろ向き）
    f: float                 # 焦点距離 [画素]
    width: int
    height: int

    def to_json(self) -> dict:
        return {"pos": [float(v) for v in self.pos], "rot": [[float(v) for v in r] for r in self.rot],
                "f": float(self.f), "width": self.width, "height": self.height}


@dataclasses.dataclass(frozen=True)
class Thresholds:
    min_value: int           # その色のチャンネルの下限（0〜255）
    min_margin: int          # ほかの 2 チャンネルの大きい方との差の下限
    min_pixels: int          # これ以上の画素があれば「見えている」

    @classmethod
    def from_dict(cls, c: dict) -> "Thresholds":
        if c is None:
            raise ValueError("planner.color_detect is not set (scripts/color_fit.py で学習データの種から決める)")
        return cls(int(c["min_value"]), int(c["min_margin"]), int(c["min_pixels"]))

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


def calibration_from_setup(cam) -> Calibration:
    """SetupInfo の CameraSetup（俯瞰、方策の RGB の切り出し）から。"""
    I = cam.rgb_crop
    return Calibration(np.array(cam.extrinsic.t, float), np.array(cam.extrinsic.R, float), float(I.fy), I.width, I.height)


def pixel_to_plane(calib: Calibration, u: float, v: float, z: float) -> np.ndarray:
    """画素 (u, v)（生の画像）を通る光線と、高さ z の水平面の交点 (x, y)。"""
    d_cam = np.array([(u - calib.width / 2.0) / calib.f, -(v - calib.height / 2.0) / calib.f, -1.0])
    d = calib.rot @ d_cam
    if abs(d[2]) < 1e-12:
        raise ValueError("ray parallel to the plane")
    t = (z - calib.pos[2]) / d[2]
    p = calib.pos + t * d
    return p[:2].copy()


def color_mask(img: np.ndarray, color: str, thr: Thresholds) -> np.ndarray:
    """(H, W, 3) uint8 → (H, W) bool。"""
    a = np.asarray(img).astype(np.int16)
    ch = CHANNEL[color]
    others = np.max(np.delete(a, ch, axis=2), axis=2)
    return (a[:, :, ch] >= thr.min_value) & (a[:, :, ch] - others >= thr.min_margin)


def detect(img: np.ndarray, color: str, thr: Thresholds) -> dict:
    """画素の数と重心（画素の中心の座標、生の画像）。"""
    m = color_mask(img, color, thr)
    n = int(m.sum())
    if n == 0:
        return {"pixels": 0, "u": None, "v": None, "visible": False}
    rows, cols = np.nonzero(m)
    return {"pixels": n, "u": float(cols.mean() + 0.5), "v": float(rows.mean() + 0.5),
            "visible": n >= thr.min_pixels}


class TargetCue:
    """推論（またはデータの 1 フレーム）ごとに update する。見えないときは最後に見えた値を保つ。
    plane_z: 立方体の中心の高さ（机の面＋一辺の半分）。fallback: 一度も見えていないときの (x, y)（作業域の中心）。"""

    def __init__(self, calib: Calibration, thr: Thresholds, plane_z: float, fallback):
        self.calib = calib
        self.thr = thr
        self.plane_z = float(plane_z)
        self.fallback = np.asarray(fallback, float)
        self.reset()

    def reset(self) -> None:
        self.last = None
        self.color = None

    def update(self, raw_overhead: np.ndarray, color: str) -> np.ndarray:
        """→ (3,) [x, y, 見えているか（1 / 0）] float32。色が変わったら（指示の切り替え）最後の値を捨てる。"""
        if color not in CHANNEL:
            raise ValueError(f"color {color!r} not in {tuple(CHANNEL)}")
        if color != self.color:
            self.last, self.color = None, color
        det = detect(raw_overhead, color, self.thr)
        if det["visible"]:
            self.last = pixel_to_plane(self.calib, det["u"], det["v"], self.plane_z)
            return np.array([self.last[0], self.last[1], 1.0], dtype=np.float32)
        xy = self.last if self.last is not None else self.fallback
        return np.array([xy[0], xy[1], 0.0], dtype=np.float32)


def color_of_instruction(task: str) -> str:
    """指示文（convert.instruction の書式）から色を取り出す。LLM の手順の色と同じもの。"""
    words = [w for w in str(task).replace(",", " ").split() if w in CHANNEL]
    if len(set(words)) != 1:
        raise ValueError(f"instruction {task!r}: expected exactly one color word")
    return words[0]
