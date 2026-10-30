"""センサの口。センサごとに「実機にある」「シミュレーションにモデルがある」の印を持ち、失敗の検知・技能が
宣言した needs（使うセンサの名前）がそろうかを決める。

    reg = standard_registry("sim")              # "sim"（モデルがあれば使える）か "real"（実機にあれば使える）
    reg.available(("gripper", "joints"))        # 真なら使える
    reg.table                                 # 一覧（実機にある／モデルあり）
    s = Senses.from_sensor_frame(frame, kin)    # runtime/types.SensorFrame をそのまま包む（コピーしない）
    s.get("gripper").width、s.time("gripper")、s.hand、s.tip

標準の名前: joints・gripper（幅・is_grasped）・camera/overhead・camera/wrist。足せる名前: ext_wrench（Panda の外力の推定
O_F_ext_hat_K・tau_ext_hat_filtered）・tactile。ext_wrench・tactile のモデルは無い（印だけ）。
kin（順運動学。測った関節角 → (手先の位置, 指先の中心)）を渡すと、hand・tip を初めて読んだときに 1 回だけ計算する。
"""
import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class SenseInfo:
    name: str
    real: bool                    # 実機にある
    sim_model: bool               # シミュレーションにモデルがある
    rate_hz: float
    latency_s: float
    note: str = ""


# 頻度・遅延は configs/sensor_v1.yaml の値（実機の Panda で測った指の幅のセンサの値は configs/v3_real.yaml）。
# 使った実機の Panda にはカメラが無い（付ければ real を真に）
STANDARD = (
    SenseInfo("joints", True, True, 1000.0, 0.0, "関節の角度・速度・最後の指令 q_d（libfranka の RobotState）"),
    SenseInfo("gripper", True, True, 10.0, 0.0, "開き幅（2 本の和）と is_grasped。v3_real は 6.5 Hz・遅れ 0.08 s"),
    SenseInfo("camera/overhead", False, True, 30.0, 0.066, "俯瞰の RGB-D（遅れは 33〜100 ms の一様）。今の実機には無い"),
    SenseInfo("camera/wrist", False, True, 30.0, 0.066, "手首の RGB-D（同上）。今の実機には無い"),
)
EXTRA = (
    SenseInfo("ext_wrench", True, False, 1000.0, 0.0, "外力の推定 O_F_ext_hat_K・tau_ext_hat_filtered。模型は無い（印だけ）"),
    SenseInfo("tactile", False, False, 0.0, 0.0, "触覚。実機にも模型にも無い（名前だけ）"),
)
TARGETS = ("sim", "real")


class SenseRegistry:
    def __init__(self, target: str = "sim"):
        if target not in TARGETS:
            raise ValueError(f"target {target!r} は {TARGETS} のどれか")
        self.target = target
        self._infos = {}

    def register(self, info: SenseInfo) -> None:
        if info.name in self._infos:
            raise ValueError(f"感覚 {info.name} は登録済み")
        self._infos[info.name] = info

    def get(self, name: str) -> SenseInfo:
        return self._infos[name]

    @property
    def names(self) -> tuple:
        return tuple(self._infos)

    def usable(self, name: str) -> bool:
        i = self._infos.get(name)
        if i is None:
            return False
        return i.real if self.target == "real" else i.sim_model

    def missing(self, names) -> list:
        return [n for n in names if not self.usable(n)]

    def available(self, names) -> bool:
        return not self.missing(names)

    def table(self) -> list:
        return [dict(dataclasses.asdict(i), usable=self.usable(i.name)) for i in self._infos.values()]

    def format_table(self) -> str:
        rows = [f"  {'感覚':16s} 実機  模型  頻度[Hz]  遅れ[s]  {self.target}で使える"]
        for r in self.table():
            rows.append(f"  {r['name']:16s} {'有' if r['real'] else '無':4s}  {'有' if r['sim_model'] else '無':4s}  "
                        f"{r['rate_hz']:8.1f}  {r['latency_s']:7.3f}  {'○' if r['usable'] else '×'}")
        return "\n".join(rows)


def standard_registry(target: str = "sim", extra: bool = True) -> SenseRegistry:
    reg = SenseRegistry(target)
    for i in STANDARD + (EXTRA if extra else ()):
        reg.register(i)
    return reg


def fingertip(pos, quat, offset: float) -> np.ndarray:
    """手先の位置と四元数 (w, x, y, z) から指先の中心（runner.PolicyRuntime._tip と同じ式）。"""
    w, x, y, z = quat
    zaxis = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
    return np.asarray(pos, float) + float(offset) * zaxis


class Senses:
    """名前 → 最新の値と時刻。SensorFrame の中身をそのまま指す（コピーしない）。"""

    def __init__(self, t: float, values: dict, kin=None, frame=None):
        self.t = float(t)
        self._v = dict(values)                  # {名前: (値, 時刻)}
        self._kin = kin
        self._fk = None
        self.frame = frame

    @classmethod
    def from_sensor_frame(cls, frame, kin=None) -> "Senses":
        v = {"joints": (frame.joints, frame.joints.t), "gripper": (frame.gripper, frame.gripper.t)}
        for name, cam in (frame.cameras or {}).items():
            v[f"camera/{name}"] = (cam, cam.t_capture)
        return cls(frame.t, v, kin=kin, frame=frame)

    @property
    def names(self) -> tuple:
        return tuple(self._v)

    def has(self, name: str) -> bool:
        return name in self._v

    def get(self, name: str):
        return self._v[name][0]

    def time(self, name: str) -> float:
        return self._v[name][1]

    def cameras(self) -> dict:
        """{カメラの名前: CameraFrame}（知覚の update に渡す形）。"""
        return {n.split("/", 1)[1]: v for n, (v, _) in self._v.items() if n.startswith("camera/")}

    def _fk_once(self):
        if self._fk is None:
            if self._kin is None or "joints" not in self._v:
                return None
            hand, tip = self._kin(self.get("joints").q)
            self._fk = (np.asarray(hand, float), np.asarray(tip, float))
        return self._fk

    def hand(self):
        """測った関節角の順運動学の手先（hand の原点）。kin が無ければ None。"""
        fk = self._fk_once()
        return None if fk is None else fk[0]

    def tip(self):
        """同じく指先の中心。"""
        fk = self._fk_once()
        return None if fk is None else fk[1]
