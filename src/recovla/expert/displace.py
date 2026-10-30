"""外乱 D: 生成プログラムが近づく途中（閉じる前）に、目標の立方体を机の上でずらして回す。

    dp = sample_params(rng)
    dis = Displacer(dp, rig, color, cfg)
    dis.install                                # rig.pad_read と rig.physics_step を包む
    ... generate.run_attempt ...                 # 生成プログラムは真の状態を読むので、動いた目標を追う
    dis.uninstall

- 発動: 開いていて、指先が目標より CLEAR_M 以上高く、水平の距離が trigger_r_m 未満になったとき。
  先に descend に入ったらその時点で発動する。閉じる段階まで発動しなければ "missed"
- 動かし方: 机の上で 3〜8 cm ずらし、向きを ±20° 回す。slide_s かけて滑らかに運ぶ（物理の 1 ステップごとに位置と速さを書く）。
  行き先は配置の範囲の中で、他の立方体から scene.min_center_dist 以上、通り道も PATH_CLEAR_M 以上離す。
  満たす行き先がなければ "no_target"
- 記録は試みの最初から全部（立て直しデータではないので inject.KINDS に入れない）
- データ生成の側でだけ使うので、真の状態と場面の寸法を使う（実行時には使わない）
"""
import dataclasses

import numpy as np

from recovla.common import seeds
from recovla.common.seeds import COLORS
from recovla.expert import script as S
from recovla.sim import frames

KIND = "D"
D_KEY = 4                          # 乱数列: seed_sequence(シード, "inject", 色, 作り直し, D_KEY)
DIST_M = (0.03, 0.08)
DYAW_DEG = 20.0
TRIGGER_R_M = (0.03, 0.12)
SLIDE_S = (0.3, 0.6)
CLEAR_M = 0.08                     # 指先と目標の中心の高さの差の下限（手が高いうちだけ動かす）
PATH_CLEAR_M = 0.06                # 通り道と他の立方体の中心の距離の下限
MAX_TRIES = 200


@dataclasses.dataclass(frozen=True)
class DisplaceParams:
    trigger_r_m: float
    slide_s: float
    u: tuple                       # 行き先を選ぶ一様乱数の列（距離・向き・回り）

    def to_json(self) -> dict:
        return {"kind": KIND, "trigger_r_m": self.trigger_r_m, "slide_s": self.slide_s, "n_u": len(self.u)}


def rng_for(layout_seed: int, color: str, retry: int) -> np.random.Generator:
    return seeds.generator(seeds.seed_sequence(layout_seed, "inject", COLORS.index(color), retry, D_KEY))


def sample_params(rng: np.random.Generator) -> DisplaceParams:
    r = float(rng.uniform(*TRIGGER_R_M))
    s = float(rng.uniform(*SLIDE_S))
    u = tuple(float(v) for v in rng.random(3 * MAX_TRIES))
    return DisplaceParams(round(r, 6), round(s, 6), u)


def seg_point_dist(p, a, b) -> float:
    p, a, b = (np.asarray(v, float)[:2] for v in (p, a, b))
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / max(float(np.dot(ab, ab)), 1e-18), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def choose_goal(p: DisplaceParams, start_xy, start_yaw: float, others_xy, cfg: dict) -> dict:
    """行き先（xy・向き）を u の列から順に試して選ぶ。others_xy は他の机上の立方体の中心。"""
    sc = cfg["scene"]
    rx, ry = sc["region"]["x"], sc["region"]["y"]
    yaw_max = np.radians(float(sc["yaw_range_deg"]))
    min_d = float(sc["min_center_dist"])
    p0 = np.asarray(start_xy, float)[:2]
    for i in range(MAX_TRIES):
        ud, ua, uy = p.u[3 * i:3 * i + 3]
        d = DIST_M[0] + ud * (DIST_M[1] - DIST_M[0])
        a = 2.0 * np.pi * ua
        dyaw = np.radians(DYAW_DEG) * (2.0 * uy - 1.0)
        p1 = p0 + d * np.array([np.cos(a), np.sin(a)])
        yaw1 = float(start_yaw) + dyaw
        if not (rx[0] <= p1[0] <= rx[1] and ry[0] <= p1[1] <= ry[1]):
            continue
        if abs(yaw1) > yaw_max:
            continue
        if any(np.hypot(*(p1 - np.asarray(o, float)[:2])) < min_d for o in others_xy):
            continue
        if any(seg_point_dist(o, p0, p1) < PATH_CLEAR_M for o in others_xy):
            continue
        return {"ok": True, "xy": [float(v) for v in p1], "yaw": yaw1, "dist_m": float(d), "dyaw_deg": float(np.degrees(dyaw)),
                "tries": i + 1}
    return {"ok": False, "tries": MAX_TRIES}


def smooth(s: float) -> tuple:
    """0〜1 の滑らかな進み（3 次の補間）。戻り値は（値, 微分）。"""
    s = float(np.clip(s, 0.0, 1.0))
    return 3 * s * s - 2 * s ** 3, 6 * s - 6 * s * s


class Displacer:
    def __init__(self, params: DisplaceParams, rig, color: str, cfg: dict):
        self.p, self.rig, self.color, self.cfg = params, rig, color, cfg
        self.ti = COLORS.index(color)
        self.pp = S.PhaseParams.from_config(cfg)
        self.status = "armed"          # armed → sliding → done | missed | no_target
        self.info = {}
        self._saved = {}

    # -------------------------------------------------------------- 包む・外す
    def install(self) -> None:
        r = self.rig
        for name in ("pad_read", "physics_step"):
            self._saved[name] = r.__dict__.get(name)
        orig_pad, orig_phys = r.pad_read, r.physics_step

        def pad_read(vel=None, press: bool = False, on_step=None):
            self.check(r.truth(self.color))
            return orig_pad(vel, press, on_step)

        def physics_step(on_step=None):
            self.before_physics()
            return orig_phys(on_step)
        r.pad_read, r.physics_step = pad_read, physics_step

    def uninstall(self) -> None:
        r = self.rig
        for name, v in self._saved.items():
            if v is None:
                r.__dict__.pop(name, None)
            else:
                setattr(r, name, v)
        self._saved = {}

    # -------------------------------------------------------------- 発動
    def check(self, tr: S.Truth) -> None:
        if self.status != "armed":
            return
        ph = S.phase_of(tr, 0.0, self.pp)
        if tr.gripper_closed or ph in (S.Phase.close, S.Phase.lift, S.Phase.carry, S.Phase.release):
            self.status = "missed"
            self.info["missed_phase"] = ph.name
            return
        high = float(tr.fingertip[2] - tr.target_pos[2]) >= CLEAR_M
        near = float(np.hypot(*(tr.fingertip[:2] - tr.target_pos[:2]))) < self.p.trigger_r_m
        if not high:
            if ph == S.Phase.descend:
                self.status = "missed"
                self.info["missed_phase"] = "descend_low"
            return
        if near or ph == S.Phase.descend:
            self.fire(tr, "near" if near else "descend")

    def fire(self, tr: S.Truth, why: str) -> None:
        others = [tr.cube_pos[i] for i in range(len(COLORS))
                  if i != self.ti and not frames.in_box(tr.cube_pos[i], tr.box)]
        yaw0 = float(frames.quat_yaw(tr.cube_quat[self.ti]))
        g = choose_goal(self.p, tr.target_pos, yaw0, others, self.cfg)
        self.info.update(t_fire=round(tr.t, 3), why=why, start_xy=[float(v) for v in tr.target_pos[:2]],
                         start_yaw_deg=round(float(np.degrees(yaw0)), 3),
                         fingertip_at_fire=[float(v) for v in tr.fingertip], goal=g)
        if not g["ok"]:
            self.status = "no_target"
            return
        self.status = "sliding"
        self._p0 = np.asarray(tr.target_pos, float).copy()
        self._p1 = np.array([g["xy"][0], g["xy"][1], self._p0[2]])
        self._yaw0, self._yaw1 = yaw0, float(g["yaw"])
        self._t0 = float(self.rig.data.time)

    # -------------------------------------------------------------- 運ぶ（物理の 1 ステップごと）
    def before_physics(self) -> None:
        if self.status != "sliding":
            return
        from recovla.sim import scene
        d = self.rig.data
        qa, va = scene.cube_qpos_adr(self.rig.model, self.color)
        T = float(self.p.slide_s)
        s = (float(d.time) - self._t0) / T
        f, df = smooth(s)
        pos = self._p0 + (self._p1 - self._p0) * f
        yaw = self._yaw0 + (self._yaw1 - self._yaw0) * f
        d.qpos[qa:qa + 3] = pos
        d.qpos[qa + 3:qa + 7] = frames.yaw_quat(yaw)
        if s >= 1.0:
            d.qvel[va:va + 6] = 0.0
            self.status = "done"
            self.info["t_done"] = round(float(d.time), 3)
            self.info["end_xy"] = [float(v) for v in pos[:2]]
            self.info["end_yaw_deg"] = round(float(np.degrees(yaw)), 3)
            return
        d.qvel[va:va + 3] = (self._p1 - self._p0) * df / T
        d.qvel[va + 3:va + 6] = [0.0, 0.0, (self._yaw1 - self._yaw0) * df / T]

    def summary(self) -> dict:
        return {"params": self.p.to_json(), "status": self.status, **self.info}
