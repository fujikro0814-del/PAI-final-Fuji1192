"""評価時の入口: カメラの描画と腕の状態から、学習した方策に渡す観測を作る。

画像は学習データの変換と同じ vla_image_spec.policy_images を通すので、学習と評価で変換が食い違わない。
画像の仕様は学習の出力フォルダ（<output_dir>/conversion.json。学習の起動役が学習データの
meta/conversion.json を写す）と照らす。写しがない、版 1 のデータ、変換の表が違う、のどれかなら方策を動かす前に止める。

observation.state は今のフレームの腕の状態（ee_pos・ee_quat・fingers・joints）から vla_state.policy_state_frame で作る
（学習データと同じ定義）。出力は lerobot-train が前処理に渡す 1 項目と同じ形: 画像は float32 C x H x W [0, 1]、
observation.state は float32 (15,)（目標位置の入力を足すと 17・18）、task は文字列（バッチの次元なし）。
"""
import json
import pathlib

import numpy as np

from recovla.data import vla_image_spec as spec
from recovla.data import vla_state

COPIED_CONVERSION = "conversion.json"     # <output_dir>/ に学習の起動役が写す


def training_output_dir(checkpoint_dir) -> pathlib.Path:
    """<output_dir> 自身か .../checkpoints/<step>/pretrained_model から <output_dir> を返す。"""
    p = pathlib.Path(checkpoint_dir).resolve()
    if p.name == "pretrained_model" and p.parent.parent.name == "checkpoints":
        return p.parent.parent.parent
    return p


def load_training_spec(checkpoint_dir) -> dict:
    """学習の出力フォルダに写した conversion.json を読んで確かめる。"""
    out = training_output_dir(checkpoint_dir)
    path = out / COPIED_CONVERSION
    if not path.is_file():
        raise spec.ImageSpecError(
            f"{path} not found: the training launcher (start_training.bat) copies the training dataset's "
            f"meta/conversion.json there; without it the image transforms used in training are unknown")
    conversion = json.loads(path.read_text(encoding="utf-8"))
    spec.check_record(conversion, str(path))
    return conversion


def observation_images(raw_by_view: dict) -> dict:
    """{視点: H x W x 3 uint8} → {特徴の名前: float32 C x H x W [0, 1]}。"""
    out = {}
    for key, img in spec.policy_images(raw_by_view).items():
        if img.dtype != np.uint8:
            raise spec.ImageSpecError(f"{key}: expected uint8 renders, got {img.dtype}")
        out[key] = np.ascontiguousarray(img.transpose(2, 0, 1)).astype(np.float32) / 255.0
    return out


class PolicyObservationBuilder:
    """評価する checkpoint ごとに 1 度作る。仕様の検査はコンストラクタで行う（方策を呼ぶ前）。

    目標位置の入力: 学習の conversion.json に "target_cue" があれば、俯瞰画像から変換と同じ部品・
    同じ閾値で目標位置の入力（vla_state.CUE_NAMES）を計算して observation.state に足す。試行の始めに reset を呼ぶ
    （目標が隠れている間は最後に見えた位置を保つため）。"""

    def __init__(self, checkpoint_dir):
        self.conversion = load_training_spec(checkpoint_dir)
        self.checkpoint_dir = pathlib.Path(checkpoint_dir)
        self.cue = None
        self.cue_keep_flag = True
        rec = self.conversion.get("target_cue")
        if rec is not None:
            from recovla.data.convert import cue_tracker
            self.cue = cue_tracker()
            self.cue_keep_flag = "cue_visible" in rec["names"]          # 旗を入力から外した学習なら 17 次元
            if (rec["thresholds"] != self.cue.thr.to_json()
                    or rec["names"] != vla_state.cue_names(self.cue_keep_flag)):
                raise spec.ImageSpecError(f"target_cue of the training data {rec['thresholds']} {rec['names']} != "
                                          f"runtime {self.cue.thr.to_json()}")
        self.cue_offset = None                    # 目標位置の入力に足すずれ [m]（雑音の試験用。既定はなし）
        self.last_cue = None

    def reset(self) -> None:
        if self.cue is not None:
            self.cue.reset()
        self.last_cue = None

    def build(self, raw_by_view: dict, proprio: dict, task: str, cue_color: str = None) -> dict:
        """raw_by_view: {視点: 描画}、proprio: シミュレーションから読んだ 1 フレームの
        {ee_pos (3,), ee_quat (4,), fingers (2,), joints (7,)}。
        cue_color: 目標位置の入力を計算する色（既定は task の色の語。指示文と別の色にする試験で使う）。"""
        state = vla_state.policy_state_frame(proprio)
        if self.cue is not None:
            from recovla.perception.color import color_of_instruction
            self.last_cue = self.cue.update(raw_by_view["overhead"], cue_color or color_of_instruction(task))
            if self.cue_offset is not None:       # 目標位置の入力の (x, y) にだけ足す（部品の中に保った値は変えない）
                self.last_cue = self.last_cue.copy()
                self.last_cue[:2] += np.asarray(self.cue_offset, np.float32)
            state = vla_state.with_cue(state, self.last_cue, keep_flag=self.cue_keep_flag)
        return {**observation_images(raw_by_view), "observation.state": state, "task": str(task)}
