"""センサのモデルつきのデータ生成の世界: 学習データを、実行時と同じセンサ・作動のモデルで記録する。

生成プログラムは真の状態で動き、指令は実行系と同じ口（DrivenRig: IK・制限層・関節の位置の指令・ハンドの move/grasp）を通る。
記録は実行時の方策が見るものと同じにする:
  画像      センサのモデルの、そのとき届いている最新のフレーム（俯瞰・手首の切り出し。雑音・遅延・照明を含む）
  状態      測った関節角（雑音つき）、その順運動学の手先、測った開き幅の半分ずつの指
  行動      x_des = 実行系の x_cmd、グリッパ = 開閉の指令
  目標位置の入力  変換（data/convert.py）が、エピソードに記録した較正の値（episode_meta）で計算する
センサの試行ごとの値（較正誤差など）は、配置のシードから作ったシードで引く（同じ配置なら同じ設置）。
"""
import numpy as np

from recovla.common import config
from recovla.harness.driven import DrivenRig
from recovla.harness.sensors import SensorSuite
from recovla.harness.world import WorldRig


class SensedDrivenRig(DrivenRig):
    def __init__(self, cfg: dict = None):
        cfg = cfg or config.load_v2()
        super().__init__(render=False, cfg=cfg)
        self.suite = SensorSuite(self.model, cfg)
        self._sf = None
        self.setup_believed = None

    def close(self) -> None:
        self.suite.close()
        super().close()

    def sensor_seed(self, layout) -> int:
        return int(layout.seed) * 10 + 7

    def reset(self, layout) -> None:
        super().reset(layout)
        self.setup_believed = self.suite.start_trial(self.sensor_seed(layout), self.data, self.setup_info)
        self.suite.prime(self.data)
        self._sf = None

    def physics_step(self, on_step=None) -> None:
        self.controller.desired_pos = self.motion.x_cmd              # 記録（writer.add_step）が読む参照を実行系の値に
        WorldRig.physics_step(self, None)
        self.suite.on_physics_step(self.data)
        if on_step is not None:
            on_step(self)

    def render(self, data=None) -> list:
        sf = self.suite.sense(self.data, self.hand, self._last_cmd, True)
        self._sf = sf
        return [sf.cameras["overhead"].rgb.copy(), sf.cameras["wrist"].rgb.copy()]

    def postprocess_frame(self, f: dict) -> dict:
        sf = self._sf if self._sf is not None else self.suite.sense(self.data, self.hand, self._last_cmd, False)
        self._sf = None
        q = np.asarray(sf.joints.q, float)
        pos, quat = self.motion.hand_pose(q)
        if quat[0] < 0.0:
            quat = -quat
        w = float(sf.gripper.width)
        f.update({"ee_pos": pos, "ee_quat": quat, "joints": q.copy(), "joint_vel": np.asarray(sf.joints.dq, float).copy(),
                  "fingers": np.array([0.5 * w, 0.5 * w]), "x_des": self.motion.x_cmd, "gripper_closed": bool(self.closed),
                  "sensor_t": float(sf.t)})
        return f

    def episode_meta(self) -> dict:
        cam = self.setup_believed.cameras["overhead"]
        su = self.setup_believed
        out = int(self.cfg["sensor"]["cameras"]["overhead"]["color"]["out"])
        return {"generation": "v2", "sensor_model_version": int(self.cfg["sensor_model_version"]),
                "cameras": {"names": list(self.cameras), "width": out, "height": out},
                "sensor_seed": int(self.suite.seed),
                "cue_calibration": {"pos": [float(v) for v in cam.extrinsic.t], "rot": [[float(v) for v in r] for r in cam.extrinsic.R],
                                    "f": float(cam.rgb_crop.fy), "width": cam.rgb_crop.width, "height": cam.rgb_crop.height,
                                    "plane_z": float(su.table_z + 0.5 * su.cube_size),
                                    "fallback": [float(v) for v in su.cue_fallback_xy]}}
