"""俯瞰の画像から、箱の中と机の上にある立方体の色を判定する。

    reg = Regions(calib, frames.box_outer_half(model))   # calib = perception.color.overhead_calibration(model)
    c = reg.counts(raw_overhead)               # {"box": {色: 画素数}, "table": {色: 画素数}}
    reg.box_colors(raw_overhead), reg.table_colors(raw_overhead)

- 画素の色の分類は perception.color（閾値 planner.color_detect は学習用の配置の画像だけで決めた）
- 箱の領域: 箱の内寸の四隅を、箱の中の立方体の上面の高さ（箱の底＋立方体の一辺）でカメラへ投影した四角形
- 机の領域: 箱の外寸（壁の上端の高さ）に余白 1 cm を足して投影した四角形の外
- 「ある」とみなす画素数の閾値は planner.presence（学習用の配置の画像だけで決める。scripts/planner_tools.py fit）
"""
import cv2
import numpy as np

from recovla.common import config
from recovla.common.seeds import COLORS
from recovla.perception import color as C
from recovla.sim import frames

_CFG = config.load()


def world_to_pixel(calib: C.Calibration, p) -> tuple:
    """frames.project と同じ透視投影（静的な俯瞰カメラの較正の値から）。"""
    rel = calib.rot.T @ (np.asarray(p, float) - calib.pos)
    return (calib.width / 2.0 + calib.f * rel[0] / -rel[2], calib.height / 2.0 - calib.f * rel[1] / -rel[2])


def _poly_mask(calib, center_xy, half, z) -> np.ndarray:
    pts = [world_to_pixel(calib, (center_xy[0] + sx * half, center_xy[1] + sy * half, z))
           for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    m = np.zeros((calib.height, calib.width), np.uint8)
    cv2.fillPoly(m, [np.round(np.array(pts) * 16).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=4)
    return m.astype(bool)


def wrist_box_mask(model, data, width: int, height: int):
    """手首のカメラの画像での箱の領域（内寸の四隅を箱の中の立方体の上面の高さで投影）。カメラの位置はロボットの関節から
    決まる（data の順運動学。立方体の真値は使わない）。カメラの後ろに回る点があれば None。"""
    b = np.asarray(_CFG["scene"]["box"]["pos"][:2], float)
    h, z = frames.BOX_INNER_HALF, frames.BOX_FLOOR_Z + frames.CUBE_SIZE
    pts = [frames.project(model, data, "wrist", (b[0] + sx * h, b[1] + sy * h, z), width, height)
           for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    if any(p is None for p in pts):
        return None
    m = np.zeros((height, width), np.uint8)
    cv2.fillPoly(m, [np.round(np.array(pts) * 16).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=4)
    return m.astype(bool)


def wrist_counts(img, mask, color: str, thr: C.Thresholds) -> tuple:
    """手首の画像で、その色の画素のうち箱の領域の内側の数と、全体の数。"""
    m = C.color_mask(img, color, thr)
    return (int((m & mask).sum()) if mask is not None else 0), int(m.sum())


class Regions:
    def __init__(self, calib: C.Calibration, box_outer_half: float, box_xy=None, cfg: dict = None):
        cfg = cfg or _CFG
        box = np.asarray(box_xy if box_xy is not None else cfg["scene"]["box"]["pos"][:2], float)
        self.thr = C.Thresholds.from_config(cfg)
        self.box_mask = _poly_mask(calib, box, frames.BOX_INNER_HALF, frames.BOX_FLOOR_Z + frames.CUBE_SIZE)
        self.table_mask = ~_poly_mask(calib, box, float(box_outer_half) + 0.01, frames.BOX_WALL_TOP_Z)   # 余白 1 cm
        p = cfg["planner"].get("presence") or {}
        self.box_min = p.get("box_min_pixels")
        self.table_min = p.get("table_min_pixels")
        self.wrist_min = p.get("wrist_min_pixels")
        self.wrist_frac = p.get("wrist_min_fraction")

    def counts(self, img) -> dict:
        out = {"box": {}, "table": {}}
        for c in COLORS:
            m = C.color_mask(img, c, self.thr)
            out["box"][c] = int((m & self.box_mask).sum())
            out["table"][c] = int((m & self.table_mask).sum())
        return out

    def box_colors(self, img) -> list:
        if self.box_min is None:
            raise ValueError("planner.presence が決まっていない（planner_tools.py fit）")
        c = self.counts(img)["box"]
        return [k for k in COLORS if c[k] >= self.box_min]

    def table_colors(self, img) -> list:
        if self.table_min is None:
            raise ValueError("planner.presence が決まっていない（planner_tools.py fit）")
        c = self.counts(img)["table"]
        return [k for k in COLORS if c[k] >= self.table_min]
