"""色の判定と、目標位置の入力（学習データの変換用の入口）。

    calib = overhead_calibration(model)                 # 俯瞰カメラの内部・外部（静的なカメラ。モデルから読む）
    thr = Thresholds.from_config                      # configs の planner.color_detect（学習用の配置だけで決めた値）
    det = detect(raw_overhead, "red", thr)              # 画素の数・重心（生の画像の座標）
    cue = TargetCue(calib, thr); cue.reset
    x, y, visible = cue.update(raw_overhead, "red")     # 机の面（立方体の中心の高さ）へ投影した位置と、見えているかの旗

目標位置の入力は俯瞰画像（方策の画像の反転の前）からだけ計算し、シミュレータの真値は使わない。見えないとき（画素が min_pixels
未満）は最後に見えた値を保ち、旗を 0 にする。一度も見えていなければ配置の範囲（scene.region）の中心を返す（旗 0）。
判定の関数は実行系と同じ recovla.runtime.cue。ここでは較正にモデルのカメラ、机の面に既知の高さを使う。
"""
import mujoco
import numpy as np

from recovla.common import config
from recovla.runtime import cue as _cue
from recovla.runtime.cue import CHANNEL, Calibration, color_mask, color_of_instruction, detect, pixel_to_plane  # noqa: F401
from recovla.sim import frames

_CFG = config.load()
OVERHEAD = "overhead"


class Thresholds(_cue.Thresholds):
    @classmethod
    def from_config(cls, cfg: dict = None) -> "Thresholds":
        c = _cue.Thresholds.from_dict((cfg or _CFG)["planner"]["color_detect"])
        return cls(c.min_value, c.min_margin, c.min_pixels)


def overhead_calibration(model, size: int = None) -> Calibration:
    """俯瞰カメラの較正の値。frames.project と同じ透視投影。"""
    size = int(size or _CFG["sim"]["image_size"])
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    cam = model.camera(OVERHEAD).id
    f = 0.5 * size / np.tan(np.radians(model.cam_fovy[cam]) / 2.0)
    return Calibration(d.cam_xpos[cam].copy(), d.cam_xmat[cam].reshape(3, 3).copy(), float(f), size, size)


class TargetCue(_cue.TargetCue):
    """机の面は既知の高さ（frames.CUBE_REST_Z）、一度も見えていなければ scene.region の中心。"""

    def __init__(self, calib: Calibration, thr, cfg: dict = None):
        r = (cfg or _CFG)["scene"]["region"]
        super().__init__(calib, thr, frames.CUBE_REST_Z, [np.mean(r["x"]), np.mean(r["y"])])
