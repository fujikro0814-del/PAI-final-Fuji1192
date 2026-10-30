"""失敗の意図的な失敗 B（運ぶ途中の落下）の「浅く掴んで滑らせる」型（configs/v3_real.yaml の v3_real.b_slip）。既定は切。

通常の B は運ぶ途中で指を開く。この型は開かない: 立方体の中心から当て板の幅の向き（手の x ＝ 世界の x）に
ずらした点で掴み（当て板と立方体の重なり overlap_m）、あとは生成プログラムのまま持ち上げて運ぶ。持ち上げから運ぶへの
変わり目で立方体が指から抜けて机に落ち、静止したら確定する（閉じたまま）。記録はその後から（生成プログラムが開いて立て直す）。
重なりの範囲は、閉じたまま立方体が抜けた例（重なり 3.1・3.9 mm）に合わせた（docs/real/real2sim_log.md）。

| 結果 | 意味 |
|---|---|
| confirmed | 持ち上げた後に手から抜け、机の上で静止し、着地の検査（inject.landing_check）を通った |
| not_effective / no_slip | 抜けずに放す位置まで運んだ |
| not_effective / grasp_miss | 持ち上がらなかった（掴み損ね） |
| not_effective / timeout | timeout_s を超えた |
| landing_invalid | 抜けたが着地が検査を通らない、または静止しない |

意図的な失敗の値は B の u（inject.sample_params の意図的な失敗の乱数列）から決める: 向き = u < 0.5 なら +x、それ以外は −x、
重なり = overlap_m の下限 + (2u mod 1) × 幅。generate.run_attempt は inject.Injector を呼ぶので、
expert/generate_real.py の生成の間だけ inject.Injector を make_injector に差し替える。
"""
import dataclasses

import mujoco
import numpy as np

from recovla.common.seeds import COLORS
from recovla.expert import inject
from recovla.expert import script as S
from recovla.sim import frames

ORIG_INJECTOR = inject.Injector


@dataclasses.dataclass(frozen=True)
class SlipParams(inject.InjectParams):
    variant: str = None
    overlap_m: float = None       # 当て板と立方体の重なり（狙い）
    side: float = None            # ずらす向き（世界の x の符号）
    shift_m: float = None         # 立方体の中心から指先の中心までのずれ（狙い）


def pad_half_width(model) -> float:
    """指の当て板の、手の x の向きの半分の幅（指の箱形の当たりの外端。panda.xml では 0.0085 m）。"""
    out = 0.0
    for g in range(model.ngeom):
        body = model.body(model.geom_bodyid[g]).name
        if body in ("left_finger", "right_finger") and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX and model.geom_contype[g]:
            out = max(out, abs(float(model.geom_pos[g][0])) + float(model.geom_size[g][0]))
    return out


def slip_params(base: inject.InjectParams, cfg: dict, model) -> SlipParams:
    bs = cfg["v3_real"]["b_slip"]
    lo, hi = (float(v) for v in bs["overlap_m"])
    u = float(base.b_u)
    side = 1.0 if u < 0.5 else -1.0
    overlap = lo + ((2.0 * u) % 1.0) * (hi - lo)
    shift = frames.CUBE_HALF + pad_half_width(model) - overlap
    return SlipParams("B", b_u=u, variant="shallow_slip", overlap_m=round(overlap, 6), side=side, shift_m=round(shift, 6))


def make_injector(params, expert, rig, cfg: dict = None):
    """inject.Injector の代わり。v3_real の b_slip が入りで種類が B のときだけ SlipInjector、ほかは元の Injector。"""
    from recovla.harness.real_matched import Switches
    if params.kind == "B" and Switches.from_config(cfg or {}).b_slip:
        return SlipInjector(params, expert, rig, cfg)
    return ORIG_INJECTOR(params, expert, rig, cfg)


class SlipInjector(ORIG_INJECTOR):
    def __init__(self, params, expert, rig, cfg: dict):
        super().__init__(slip_params(params, cfg, rig.model), expert, rig, cfg)
        self.timeout_s = float(cfg["v3_real"]["b_slip"]["timeout_s"])
        self.yaw_max_deg = float(cfg["v3_real"]["b_slip"]["yaw_max_deg"])
        self._lifted_t = None
        self._last_phase = None

    def modify(self, cmd: S.Command, truth: S.Truth) -> S.Command:
        if self.handed:
            return cmd
        if self.finished:
            return self._cmd(phase=cmd.phase)                 # 確定した後、記録を始めるまで静止
        if self.status == "armed":
            if cmd.phase not in (S.Phase.descend, S.Phase.close):
                return cmd
            self.status, self.t_fire = "active", round(truth.t, 3)
            self._t_arm = truth.t
            yaw = float(np.degrees(frames.quat_yaw(truth.cube_quat[truth.target])))
            self.info["yaw_rel_at_fire_deg"] = round((yaw + 45.0) % 90.0 - 45.0, 2)
            if abs(self.info["yaw_rel_at_fire_deg"]) > self.yaw_max_deg:
                self.status, self.reason = "not_effective", "yaw_out_of_range"   # 斜めの立方体は端で掴めない（押し出す）
                return self._cmd(phase=cmd.phase)
            self.stage = "move"
            self._goal_xy = truth.target_pos[:2] + inject.ORTHO_AXIS * float(self.p.side) * float(self.p.shift_m)
            self.info["target_z_at_fire"] = float(truth.target_pos[2])
        if truth.t - self._t_arm > self.timeout_s and self.stage != "dropped":
            self.status, self.reason = "not_effective", "timeout"
            return self._cmd(phase=cmd.phase)
        if self.stage == "move":
            return self._move(cmd, truth)
        if self.stage == "held":
            return self._held(cmd, truth)
        return self._dropped(truth)

    def _move(self, cmd, truth):
        """ずらした把持点の真上へ行き、把持の高さへ下ろして閉じる（A の横ずらしと同じ動き）。"""
        ex = self.ex
        x = np.asarray(truth.x_cmd, float)
        off = self._goal_xy - truth.fingertip[:2]
        aim = x[:2] + off
        if np.hypot(*off) > ex.pp.above_tol:
            goal = ex._rise_first(x, np.array([aim[0], aim[1], ex.approach_z]), ex.approach_z)
            return self._cmd(ex._vel_to(x, goal), False, S.Phase.approach)
        goal = np.array([aim[0], aim[1], ex.pp.grasp_z])
        if np.hypot(*off) <= ex.move_tol and abs(x[2] - ex.pp.grasp_z) <= ex.move_tol and ex._hand_still(truth):
            self.stage = "held"
            rel = truth.target_pos - truth.fingertip
            self.info.update(fingertip_at_close=[float(v) for v in truth.fingertip], cube_rel_at_close=[float(v) for v in rel],
                             overlap_at_close_m=float(frames.CUBE_HALF + pad_half_width(self.rig.model) - abs(rel[0])),
                             t_close=round(truth.t, 3))
            return self._press_once(S.Phase.close)
        return self._cmd(ex._vel_to(x, goal, aim), False, S.Phase.descend)

    def _held(self, cmd, truth):
        """生成プログラムのまま持ち上げて運ぶ。持ち上げた後に手から抜けたら止まる。"""
        pp = self.ex.pp
        hold, up = S.holding(truth, pp), S.lifted(truth, pp)
        if hold and up and self._lifted_t is None:
            self._lifted_t = truth.t
            rel = truth.target_pos - truth.fingertip
            self.info["overlap_at_lift_m"] = float(frames.CUBE_HALF + pad_half_width(self.rig.model) - abs(rel[0]))
        if self._lifted_t is not None and not hold:
            self.stage = "dropped"
            self._t_open = truth.t
            self.info.update(slip_t=round(truth.t, 3), t_lifted=round(self._lifted_t, 3),
                             phase_before_slip=self._last_phase, slip_point=[float(v) for v in truth.target_pos],
                             hand_at_slip=[float(v) for v in truth.hand_pos])
            return self._cmd()
        if cmd.phase in (S.Phase.release, S.Phase.settle, S.Phase.retreat, S.Phase.done) and self._lifted_t is not None:
            self.status, self.reason = "not_effective", "no_slip"
            return self._cmd(phase=cmd.phase)
        if cmd.phase == S.Phase.reopen and self._lifted_t is None:
            self.status, self.reason = "not_effective", "grasp_miss"
            return self._cmd(phase=cmd.phase)
        self._last_phase = cmd.phase.name
        return cmd

    def _dropped(self, truth):
        """閉じたまま静止し、立方体が机の上で静止したら確定する（B の開いた後と同じ検査。開いていなくてよい）。"""
        self._update_rest(truth)
        lc = self.ic["landing"]
        if truth.t - self._t_open > float(lc["timeout_s"]):
            self.status, self.reason = "landing_invalid", "no_rest"
            self.info.update(self.landing_check(truth))
            return self._cmd()
        apart = np.linalg.norm(truth.target_pos - truth.fingertip) > self.ex.pp.drop_dist
        on_table = truth.target_pos[2] - frames.CUBE_REST_Z < self.ex.pp.grasped_lift_min
        if apart and on_table and self._rest_s >= float(lc["rest_hold_s"]) - 1e-9:
            self.info.update(finger_gap_m=float(np.sum(truth.fingers)), rest_s=self._rest_s,
                             gripper_closed_at_confirm=bool(truth.gripper_closed))
            self._confirm(truth)
        return self._cmd()


def start_condition(data: dict, meta: dict, cfg: dict) -> dict:
    """浅く掴んで滑らせた B の、保存した最初のフレームの条件（inject.start_condition の B は開いていることを見るので別に置く）。
    閉じたまま、指の間に何もなく（幅 < min_grip_gap）、立方体が指から離れて机の上で静止している。"""
    ph, lc = cfg["expert"]["phase"], cfg["inject"]["landing"]
    t = COLORS.index(meta["target"])
    info = meta["inject"]["info"]
    cube = np.asarray(data["cube_pos"][0, t], float)
    step0 = int(meta["record_start_step"])
    split_every = int(cfg["sim"]["record_every"]) * int(cfg["sim"]["stride"])
    checks = {"on_10hz_boundary": step0 % split_every == 0 and int(data["step"][0]) == step0,
              "closed": bool(data["gripper_closed"][0]),
              "nothing_between": float(np.sum(data["fingers"][0])) < float(ph["min_grip_gap_m"]),
              "not_between": float(np.linalg.norm(cube - np.asarray(data["fingertip"][0], float))) > float(ph["drop_dist_m"]),
              "on_table": float(cube[2] - frames.CUBE_REST_Z) < float(ph["grasped_lift_min_m"]),
              "at_rest": float(np.linalg.norm(data["cube_linvel"][0, t])) < float(lc["rest_speed"]),
              "rested_long": float(info.get("rest_s", 0.0)) >= float(lc["rest_hold_s"]) - 1e-9}
    return {"ok": all(checks.values()), "checks": checks}
