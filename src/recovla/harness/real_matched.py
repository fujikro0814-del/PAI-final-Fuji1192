"""実機の Panda で測った応答に合わせたシミュレーションの世界 v3_real（configs/v3_real.yaml、docs/real/v3_real.md）。

    cfg = load_config("v3_real", generation=True)       # 生成: v2 の層 + expert_v3 + 重ね
    rig = V3RealGenRig(cfg)                             # データ生成の世界（SensedDrivenRig を継ぐ）
    world, suite = make_eval_world(load_config("v3_real", generation=False))   # 評価の世界とセンサ

入り・切り（configs/v3_real.yaml の v3_real.switches。どれも独立に切れる）:
- arm_servo:    harness/arm_real.py の ArmServo（real_v1.arm_servo）を入れる
- hand_timing:  harness/hand_real.py の RealTimingHand（real_v1.gripper）に替える
- width_sensor: 幅のセンサを RealGripperSuite（6.5 Hz で保つ・遅延 delay_s・実機の雑音）にする。
                生成の記録の状態（SensedDrivenRig.postprocess_frame）と実行系（SimRobotIO.sense）は同じセンサを読む
- expert_width: 生成プログラムが読む指の幅・速さを、真の値からセンサの値に替える（width_sensor が要る）
- closed_fix:   最後の開閉の指令から closed_fix.delay_s の間、生成プログラムには指が動いていると見せる
                （ハンドが動き始める前の遅延の間に、指が止まったと判断しないため）
- b_slip:       失敗の意図的な失敗 B を浅く掴む型にする（expert/inject_real.py。データ生成の側だけ）
- contact_std:  掴みの接触を MuJoCo の標準的な設定にする（指の当て板の condim 4・楕円の摩擦の錐・impratio 10）
既定の世界・ハンド・センサ（world.py・hand.py・sensors.py・gen_sensed.py）は変えず、ここで差し替える。
"""
import collections
import copy
import dataclasses

import mujoco
import numpy as np

from recovla.common import config, seeds
from recovla.harness.gen_sensed import SensedDrivenRig
from recovla.harness.sensors import SensorSuite
from recovla.runtime.types import GripperState

WORLDS = ("v3_real",)
SWITCHES = ("arm_servo", "hand_timing", "width_sensor", "expert_width", "closed_fix", "b_slip", "contact_std")
FINGER_BODIES = ("left_finger", "right_finger")
CONES = {"pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL, "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC}


@dataclasses.dataclass(frozen=True)
class Switches:
    arm_servo: bool = False
    hand_timing: bool = False
    width_sensor: bool = False
    expert_width: bool = False
    closed_fix: bool = False
    b_slip: bool = False
    contact_std: bool = False

    @classmethod
    def from_config(cls, cfg: dict) -> "Switches":
        sw = ((cfg or {}).get("v3_real") or {}).get("switches") or {}
        bad = set(sw) - set(SWITCHES)
        if bad:
            raise ValueError(f"v3_real.switches に知らない名前 {sorted(bad)}")
        return cls(**{k: bool(v) for k, v in sw.items()})


def _v2_layers() -> list:
    return [n for n in config.V2_LAYERS if (config.CONFIG_DIR / f"{n}.yaml").is_file()]


def load_config(world: str = "v3_real", generation: bool = True, switches: dict = None) -> dict:
    """世界の設定。generation なら生成（v2 の層 + expert_v3）、でなければ評価（v2 の層）の上に重ねる。
    switches で入り・切りを上書きできる（例 {"b_slip": True}）。"""
    if world not in WORLDS:
        raise ValueError(f"world {world!r} は {WORLDS} のどれか")
    base = config.load(*_v2_layers(), *(("expert_v3",) if generation else ()))
    return apply_overlay(base, world, switches)


def apply_overlay(cfg: dict, world: str = "v3_real", switches: dict = None) -> dict:
    over = copy.deepcopy(config._load_file(world)["v3_real"])
    real = config._load_file("real_v1")
    sw = dict(over.get("switches") or {})
    for k, v in (switches or {}).items():
        if k not in SWITCHES:
            raise ValueError(f"switch {k!r} は {SWITCHES} のどれか")
        sw[k] = bool(v)
    over["switches"] = sw
    over["world"] = world
    out = copy.deepcopy(cfg)
    out["real_v1"] = copy.deepcopy(real["real_v1"])
    out["v3_real"] = over
    if sw.get("expert_width") and not sw.get("width_sensor"):
        raise ValueError("expert_width は width_sensor が要る（台本が読むセンサの値がない）")
    if sw.get("width_sensor"):
        g = dict(out["sensor"]["gripper"])
        rg = real["sensor"]["gripper"]
        g.update(rate_hz=float(rg["rate_hz"]), width_sigma=float(rg["width_sigma"]), width_unit=float(rg["width_unit"]),
                 delay_s=float(over["width_sensor"]["delay_s"]))
        out["sensor"]["gripper"] = g
    return out


# ------------------------------------------------------------------------------ 幅のセンサ
class RealGripperSuite(SensorSuite):
    """SensorSuite の幅だけを替える: rate_hz で値を取り、次まで保つ。取る値は delay_s 前の幅と is_grasped（実機は
    約 0.08 s 古い値が届く）。値は物理の 1 ステップごと（on_physics_step）に、取る時刻ちょうどで決める（読む側の頻度によらない）。
    雑音は別の乱数列（sensor_frame の 98）。カメラ・関節は SensorSuite のまま。"""

    def __init__(self, model, cfg: dict, hand_of):
        super().__init__(model, cfg)
        g = self.sc["gripper"]
        self.delay_s = float(g.get("delay_s", 0.0))
        self._hand_of = hand_of                            # 今のハンドを返す関数（ハンドは差し替わることがある）
        self._fq = np.array([model.joint(n).qposadr[0] for n in ("finger_joint1", "finger_joint2")])
        self._dt = float(model.opt.timestep)
        self.samples = []                                  # (届いた時刻, 届いた幅, その幅を取った時刻)

    def start_trial(self, seed: int, data, base_setup):
        out = super().start_trial(seed, data, base_setup)
        self.grip_rng = np.random.default_rng(seeds.seed_sequence(seed, "sensor_frame", 98))
        self._hist = collections.deque()
        self.samples = []
        self.trial_values["gripper_v3_real"] = {"rate_hz": 1.0 / self.grip_period, "delay_s": self.delay_s,
                                                "phase_s": float(self.grip_next)}
        return out

    def _record(self, data) -> None:
        h = self._hand_of()
        self._hist.append((float(data.time), float(data.qpos[self._fq].sum()), bool(h.is_grasped())))
        keep = float(data.time) - self.delay_s - 4 * self._dt
        while len(self._hist) > 1 and self._hist[1][0] <= keep:
            self._hist.popleft()

    def _sample(self, t: float) -> GripperState:
        tq = t - self.delay_s
        src = self._hist[0]
        for e in self._hist:
            if e[0] <= tq + 1e-9:
                src = e
            else:
                break
        g = self.sc["gripper"]
        w = src[1] + float(self.grip_rng.normal(0.0, float(g["width_sigma"])))
        w = round(w / float(g["width_unit"])) * float(g["width_unit"])
        self.samples.append((t, float(w), src[0]))
        return GripperState(t, float(w), src[2])

    def prime(self, data) -> None:
        super().prime(data)
        self._hist = collections.deque()
        self._record(data)
        self.grip_state = self._sample(float(data.time))   # 時刻 0 の値（場面は止まっている）
        self._grip_due = False

    def on_physics_step(self, data) -> None:
        super().on_physics_step(data)                      # カメラと、取る時刻の印（_grip_due）
        self._record(data)
        if getattr(self, "_grip_due", False):
            self.grip_state = self._sample(float(data.time))
            self._grip_due = False

    def width_reading(self) -> tuple:
        """届いている最新の幅 [m] と、直前の 2 つの値から出した幅の速さ [m/s]（生成プログラムの判断に使う）。"""
        if len(self.samples) < 2:
            return float(self.grip_state.width), 0.0
        (t1, w1, _), (t0, w0, _) = self.samples[-1], self.samples[-2]
        return float(w1), (w1 - w0) / max(t1 - t0, 1e-9)


# ------------------------------------------------------------------------------ 腕とハンド
def install_actuation(rig, cfg: dict) -> dict:
    """腕のサーボとハンドの遅延を、WorldRig（とその子）に入れる。rig.reset の前に呼ぶ。入れたものを返す。"""
    from recovla.harness import arm_real, hand_real
    sw = Switches.from_config(cfg)
    info = {"arm_servo": None, "hand_timing": None}
    if sw.hand_timing:
        timing = hand_real.RealTiming.from_config(cfg["real_v1"]["gripper"])
        hand_real.install(rig, timing)
        info["hand_timing"] = dataclasses.asdict(timing)
    if sw.arm_servo:
        servo = arm_real.ArmServo.from_config(cfg["real_v1"]["arm_servo"])
        rig._arm_servo_saved = arm_real.install(rig, servo)
        info["arm_servo"] = dataclasses.asdict(servo)
    rig.v3_real_actuation = info
    return info


def pad_geoms(model) -> list:
    """指の当て板の衝突の geom（left_finger・right_finger の体の、接触する geom 全部）。"""
    return [g for g in range(model.ngeom) if model.body(model.geom_bodyid[g]).name in FINGER_BODIES
            and (model.geom_contype[g] or model.geom_conaffinity[g])]


def install_contact(rig, cfg: dict) -> dict:
    """contact_std: 当て板の condim と、摩擦の錐・impratio をモデルに入れる。摩擦係数・solref・質量・形は変えない。"""
    info = None
    if Switches.from_config(cfg).contact_std:
        c = cfg["v3_real"]["contact_std"]
        m = rig.model
        gs = pad_geoms(m)
        m.geom_condim[gs] = int(c["pad_condim"])
        m.opt.cone = CONES[str(c["cone"])]
        m.opt.impratio = float(c["impratio"])
        info = {"pad_geoms": len(gs), "pad_condim": int(c["pad_condim"]), "cone": str(c["cone"]),
                "impratio": float(c["impratio"])}
    rig.v3_real_contact = info
    return info


def make_suite(world, cfg: dict):
    if Switches.from_config(cfg).width_sensor:
        return RealGripperSuite(world.model, cfg, lambda: world.hand)
    return SensorSuite(world.model, cfg)


def make_eval_world(cfg: dict, gravcomp=None):
    """評価の世界（WorldRig）とセンサのモデルに、v3_real の重ねを入れる。"""
    from recovla.harness.world import WorldRig
    world = WorldRig(render=False, cfg=cfg, gravcomp=gravcomp)
    install_actuation(world, cfg)
    install_contact(world, cfg)
    return world, make_suite(world, cfg)


def world_info(cfg: dict) -> dict:
    """記録に残す世界の要約。"""
    v = cfg.get("v3_real") or {}
    return {"world": v.get("world"), "switches": dict(v.get("switches") or {}),
            "gripper_sensor": dict(cfg["sensor"]["gripper"]), "closed_fix_delay_s": (v.get("closed_fix") or {}).get("delay_s"),
            "b_slip": dict(v.get("b_slip") or {}), "contact_std": dict(v.get("contact_std") or {})}


# ------------------------------------------------------------------------------ 生成の世界
class V3RealGenRig(SensedDrivenRig):
    """データ生成の世界（SensedDrivenRig）に v3_real の重ねを入れたもの。"""

    def __init__(self, cfg: dict = None):
        cfg = cfg or load_config("v3_real", generation=True)
        super().__init__(cfg)
        self.switches = Switches.from_config(cfg)
        if self.switches.width_sensor:
            self.suite.close()
            self.suite = RealGripperSuite(self.model, cfg, lambda: self.hand)
        install_actuation(self, cfg)
        install_contact(self, cfg)
        self.closed_delay_s = float(cfg["v3_real"]["closed_fix"]["delay_s"])
        self._t_grip_cmd = -np.inf

    def reset(self, layout) -> None:
        # 腕を最初の指令へ置き直してから（重力の補償あり）、センサの試行を始める（時刻 0 のフレームに置き直した姿勢を写す）
        from recovla.harness import arm_real
        from recovla.harness.driven import DrivenRig
        DrivenRig.reset(self, layout)
        if self.switches.arm_servo:
            saved = self._arm_servo_saved
            if saved.get("fifo") is not None:
                saved["fifo"].clear()                          # 指令の遅延の列を試行ごとに空にする
            if saved.get("gravity_comp"):
                arm_real.place_arm(self, self.motion.limiter.q)
        self.setup_believed = self.suite.start_trial(self.sensor_seed(layout), self.data, self.setup_info)
        self.suite.prime(self.data)
        self._sf = None
        self._t_grip_cmd = -np.inf

    def pad_read(self, vel=None, press: bool = False, on_step=None) -> None:
        if press:
            self._t_grip_cmd = float(self.data.time)
        super().pad_read(vel, press, on_step)

    def truth(self, target: str, data=None):
        """生成プログラム・意図的な失敗が読む値（data なし）だけ、指の幅・速さをセンサの側に替える。記録のフレーム（scratch）は真の値のまま。"""
        tr = super().truth(target, data)
        if data is not None:
            return tr
        fingers, fvel = tr.fingers, tr.finger_vel
        if self.switches.expert_width:
            w, v = self.suite.width_reading()
            fingers, fvel = np.array([0.5 * w, 0.5 * w]), np.array([0.5 * v, 0.5 * v])
        if self.switches.closed_fix and float(self.data.time) < self._t_grip_cmd + self.closed_delay_s - 1e-9:
            s = 0.5 * self.gripper_speed                       # 指令の速さで動いているとみなす（静止の時間を数え直す）
            fvel = np.array([s, s])
        if fingers is tr.fingers and fvel is tr.finger_vel:
            return tr
        return dataclasses.replace(tr, fingers=fingers, finger_vel=fvel)

    def episode_meta(self) -> dict:
        out = super().episode_meta()
        out["world"] = world_info(self.cfg)
        out["world"]["actuation"] = self.v3_real_actuation
        out["world"]["contact"] = self.v3_real_contact
        return out
