"""立て直しデータのための失敗の意図的な失敗。数値は configs/default.yaml の inject。

    ip = sample_params(kind, seeds.inject_rng(layout_seed, color, retry), layout, color)
    inj = Injector(ip, expert, rig)
    cmd = inj.modify(expert.act(truth), truth)      # 毎回の入力の読み取りで、生成プログラムの指令を上書きする
    inj.status                                      # armed → active → confirmed / not_effective / landing_invalid / natural_failure

意図的な失敗の中身（パラメータは試みの開始時に意図的な失敗の乱数列から全部引く）:

| 種類 | 発動 | 意図的な失敗 | 記録を始める時点（物理の状態で決める） |
|---|---|---|---|
| A 掴み損ね | descend・close に入った | 把持点を、指の閉じる向き（世界の y）と直交する水平方向（x）に 2.5〜3.5 cm ずらす、または閉じる高さを 1.5〜2.5 cm 上げる（半々）→ 閉じる → 3〜5 cm 持ち上げる | 持ち上げ終わって手が止まった時点で、指が閉じ切り（指の速さが close_settle_s 続けて止まっている）、手先が 3 cm 以上上がり、立方体の上がりが 1 cm 未満 |
| B 運ぶ途中の落下 | 掴んで持ち上げ、carry に入った | 最初に carry に入った時点の箱の中心までの水平距離を r0 とし、距離が 0.15 + u·(r0 − 0.15) を切った時点で開く（経路の進み具合について一様） | 指の開きが立方体の幅＋余裕を超え、立方体が指の間になく、机の上で静止（速さ 1 cm/s 未満が 0.3 s） |
| C 置き損ね | 箱の上で開く指令を出した | 開かずに、運ぶ高さへ上がって、ずらし先（scene.region の一様。他の立方体・壁から 3 cm 以上）の真上へ運び、放す高さへ下ろして開く | B と同じ |

発動の後は、確定するまで生成プログラムの指令を使わない（A は意図的な失敗の動作、B・C は開いたまま静止）。確定した後も、
記録を始める（呼ぶ側が handover を呼ぶ）まで静止を続ける。記録を始めた後は生成プログラムがそのまま立て直す
（reopen → approach → …。立て直し専用の分岐は作らない）。

確定の時点で着地（A は目標の立方体の状態）を検査する: 傾き 10° 以下、箱の外（外寸の上にない）、他の立方体・
箱の壁との隙間 3 cm 以上（水平面の足跡どうしの距離＝frames.polygon_distance。mj_geomDistance は箱どうしで最大
19 mm 誤るので使わない）、作業空間の水平の範囲の中、俯瞰カメラに写る。
外れたら landing_invalid（保存しない）。
"""
import dataclasses

import numpy as np

from recovla.common import config
from recovla.common.seeds import COLORS
from recovla.expert import script as S
from recovla.sim import frames

_CFG = config.load()
KINDS = ("A", "B", "C")
# 指は世界の y の向きに閉じる（hand の y 軸 = 世界の −y。ヨーは固定）。直交する水平方向は世界の x
CLOSE_AXIS = np.array([0.0, 1.0])
ORTHO_AXIS = np.array([1.0, 0.0])


@dataclasses.dataclass(frozen=True)
class InjectParams:
    kind: str
    a_mode: str = None            # "lateral" | "raise"
    a_offset_m: float = None      # lateral: ORTHO_AXIS の向きの符号つきのずれ。raise: 閉じる高さの上げ幅
    a_lift_m: float = None        # 持ち上げる量（手先）
    b_u: float = None             # B: 経路の進み具合
    c_drop_xy: tuple = None       # C: ずらし先（立方体の中心の水平位置）
    c_tries: int = None

    def to_json(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in dataclasses.asdict(self).items()
                if v is not None}


def sample_params(kind: str, rng: np.random.Generator, layout, target: str, cfg: dict = None) -> InjectParams:
    """意図的な失敗の乱数列から、種類によらず同じ順で引く（A の 4 つ → B の u → C のずらし先）。"""
    if kind not in KINDS:
        raise ValueError(f"inject kind {kind!r} not in {KINDS}")
    ic = (cfg or _CFG)["inject"]
    a = ic["A"]
    mode = "raise" if rng.random() < float(a["raise_ratio"]) else "lateral"
    lateral = float(rng.uniform(*a["lateral_offset_m"])) * (1.0 if rng.random() < 0.5 else -1.0)
    raise_m = float(rng.uniform(*a["raise_close_m"]))
    lift = float(rng.uniform(*a["lift_m"]))
    u = float(rng.random())
    drop, tries = _sample_drop(rng, layout, target, cfg or _CFG) if kind == "C" else (None, None)
    if kind == "A":
        return InjectParams("A", mode, lateral if mode == "lateral" else raise_m, lift)
    if kind == "B":
        return InjectParams("B", b_u=u)
    return InjectParams("C", c_drop_xy=drop, c_tries=tries)


def _sample_drop(rng, layout, target: str, cfg: dict):
    """C のずらし先: scene.region の一様。他の机上の立方体（配置のまま）と箱の外寸から、立方体の外接円どうしで
    min_clearance_m 以上離れる（保守側。実際の隙間は着地の後に測り直す）。"""
    sc, c = cfg["scene"], cfg["inject"]["C"]
    rx, ry = sc["region"]["x"], sc["region"]["y"]
    half_diag = frames.CUBE_HALF * np.sqrt(2.0)
    clear = float(c["min_clearance_m"])
    box = np.asarray(sc["box"]["pos"], float)
    outer = _box_outer_half(cfg)
    others = [np.asarray(v[:2], float) for col, v in layout.cubes.items() if col != target]
    for i in range(1, int(c["drop_max_tries"]) + 1):
        p = np.array([rng.uniform(*rx), rng.uniform(*ry)])
        if any(np.hypot(*(p - o)) < 2.0 * half_diag + clear for o in others):
            continue
        gap = np.maximum(np.abs(p - box[:2]) - outer, 0.0)       # 箱の外寸までの距離（軸ごと）
        if np.hypot(*gap) < half_diag + clear:
            continue
        return (float(p[0]), float(p[1])), i
    raise RuntimeError(f"layout {layout.seed}: no drop point for C after {c['drop_max_tries']} draws")


def _box_outer_half(cfg: dict = None) -> float:
    """箱の外寸の半分（壁の外面まで）。壁の厚みは設定にないので、場面の XML から一度だけ読む。"""
    global _OUTER_HALF
    if _OUTER_HALF is None:
        from recovla.sim import scene
        _OUTER_HALF = frames.box_outer_half(scene.build_model("3cube"))
    return _OUTER_HALF


_OUTER_HALF = None


def start_condition(data: dict, meta: dict, cfg: dict = None) -> dict:
    """保存したエピソードの最初のフレームが「記録を始める時点」の条件を満たすか（データの点検用）。
    data は data.npz の配列、meta は meta.json。フレームに無い量（指の静止の持続、確定の時点の手の高さ）は meta の
    inject.info から読む。"""
    cfg = cfg or _CFG
    ph, ic = cfg["expert"]["phase"], cfg["inject"]
    kind = meta["kind"]
    t = COLORS.index(meta["target"])
    info = meta["inject"]["info"]
    cube = np.asarray(data["cube_pos"][0, t], float)
    cube_rise = float(cube[2] - frames.CUBE_REST_Z)
    gap = float(np.sum(data["fingers"][0]))
    step0 = int(meta["record_start_step"])
    split_every = int(cfg["sim"]["record_every"]) * int(cfg["sim"]["stride"])
    checks = {"on_10hz_boundary": step0 % split_every == 0 and int(data["step"][0]) == step0}
    if kind == "A":
        hand_rise = float(data["ee_pos"][0, 2]) - float(info["hand_z_at_close"])
        checks.update(closed=bool(data["gripper_closed"][0]), nothing_between=gap < float(ph["min_grip_gap_m"]),
                      fingers_rested=float(info.get("finger_rest_s", 0.0)) >= float(cfg["expert"]["close_settle_s"]) - 1e-9,
                      hand_rise=hand_rise >= float(ic["A"]["confirm"]["hand_rise_min_m"]),
                      cube_low=cube_rise < float(ic["A"]["confirm"]["cube_rise_max_m"]))
    else:
        lc = ic["landing"]
        speed = float(np.linalg.norm(data["cube_linvel"][0, t]))
        apart = float(np.linalg.norm(cube - np.asarray(data["fingertip"][0], float)))
        checks.update(open=not bool(data["gripper_closed"][0]),
                      fingers_apart=gap > frames.CUBE_SIZE + float(lc["finger_gap_margin_m"]),
                      not_between=apart > float(ph["drop_dist_m"]),
                      on_table=cube_rise < float(ph["grasped_lift_min_m"]),
                      at_rest=speed < float(lc["rest_speed"]),
                      rested_long=float(info.get("rest_s", 0.0)) >= float(lc["rest_hold_s"]) - 1e-9)
    return {"ok": all(checks.values()), "checks": checks}


# ------------------------------------------------------------------------------------ 意図的な失敗

class Injector:
    def __init__(self, params: InjectParams, expert: S.Expert, rig, cfg: dict = None):
        cfg = cfg or _CFG
        self.p = params
        self.ex = expert
        self.rig = rig
        self.ic = cfg["inject"]
        self.cfg = cfg
        self.status = "armed"          # armed → active → confirmed | not_effective | landing_invalid | natural_failure
        self.stage = None              # 発動の後の段取り（種類ごと）
        self.handed = False
        self.t_fire = None
        self.t_confirm = None
        self.info = {}                 # 確定の時点の物理の量（記録と start_condition の検査に使う）
        self.reason = None
        self._rest_s = 0.0
        self._press_prev = False

    # -------------------------------------------------------------- 補助
    def _cmd(self, vel=None, press=False, phase=None) -> S.Command:
        return S.Command(np.zeros(3) if vel is None else np.asarray(vel, float), bool(press),
                         phase if phase is not None else S.Phase.settle)

    def _press_once(self, phase) -> S.Command:
        """ボタンの押し始めで開閉が切り替わる（制御器は立ち上がりを見る）ので、1 回だけ押す。"""
        return self._cmd(None, True, phase)

    def _update_rest(self, truth: S.Truth) -> None:
        lc = self.ic["landing"]
        self._rest_s = self._rest_s + self.ex.dt if truth.target_speed < float(lc["rest_speed"]) else 0.0

    def handover(self) -> None:
        """記録を始めた。以後は生成プログラムの指令をそのまま通す。"""
        self.handed = True

    @property
    def finished(self) -> bool:
        return self.status in ("confirmed", "not_effective", "landing_invalid", "natural_failure")

    # -------------------------------------------------------------- 指令の上書き
    def modify(self, cmd: S.Command, truth: S.Truth) -> S.Command:
        if self.handed:
            return cmd
        if self.status in ("confirmed", "not_effective", "landing_invalid", "natural_failure"):
            return self._cmd(phase=cmd.phase)            # 確定した後、記録を始めるまで静止
        if self.status == "armed":
            fired = {"A": self._arm_a, "B": self._arm_b, "C": self._arm_c}[self.p.kind](cmd, truth)
            if fired is None:
                return cmd
            self.status, self.t_fire = "active", round(truth.t, 3)
            return fired
        return {"A": self._run_a, "B": self._run_dropped, "C": self._run_c}[self.p.kind](cmd, truth)

    # ---------------------------------------------------------------- A
    def _arm_a(self, cmd, truth):
        if cmd.phase not in (S.Phase.descend, S.Phase.close):
            return None
        g = self.ex.grasp_point(truth)
        close_z = self.ex.pp.grasp_z
        if self.p.a_mode == "lateral":
            g = g + ORTHO_AXIS * float(self.p.a_offset_m)
        else:
            close_z += float(self.p.a_offset_m)
        self._a_goal_xy, self._a_close_z = g, close_z
        self._t_arm = truth.t
        self.stage = "move"
        self.info["target_z_at_fire"] = float(truth.target_pos[2])
        return self._run_a(cmd, truth)

    def _run_a(self, cmd, truth):
        ex, a = self.ex, self.ic["A"]
        x = np.asarray(truth.x_cmd, float)
        if truth.t - self._t_arm > float(a["timeout_s"]):
            self.status, self.reason = "not_effective", "timeout"
            return self._cmd(phase=cmd.phase)
        if self.stage == "move":
            off = self._a_goal_xy - truth.fingertip[:2]
            aim = x[:2] + off
            if np.hypot(*off) > ex.pp.above_tol:            # ずらした把持点の真上にまだいない: 接近の高さで向かう
                goal = ex._rise_first(x, np.array([aim[0], aim[1], ex.approach_z]), ex.approach_z)
                return self._cmd(ex._vel_to(x, goal), False, S.Phase.approach)
            goal = np.array([aim[0], aim[1], self._a_close_z])
            if np.hypot(*off) <= ex.move_tol and abs(x[2] - self._a_close_z) <= ex.move_tol and ex._hand_still(truth):
                self.stage = "closing"
                self._hand_z0 = float(truth.hand_pos[2])
                self.info["hand_z_at_close"] = self._hand_z0
                self.info["fingertip_at_close"] = [float(v) for v in truth.fingertip]
                return self._press_once(S.Phase.close)
            return self._cmd(ex._vel_to(x, goal), False, S.Phase.descend)
        if self.stage == "closing":
            if truth.gripper_closed and ex.clocks.finger_rest_s >= ex.close_settle_s - 1e-9:
                self.stage = "lift"
                rise = max(float(self.p.a_lift_m), float(a["confirm"]["hand_rise_min_m"]) + float(a["lift_rise_margin_m"]))
                self._hand_goal_z = self._hand_z0 + rise
                self.info["lift_goal_rise_m"] = rise
            return self._cmd(phase=S.Phase.lift)
        # lift: 手先（hand の原点）の高さで狙う（手先と x_des の定常的なずれを打ち消す）
        err = self._hand_goal_z - float(truth.hand_pos[2])
        goal = np.array([x[0], x[1], x[2] + err])
        if abs(err) <= float(a["lift_done_tol_m"]) and ex._hand_still(truth):
            c = a["confirm"]
            hand_rise = float(truth.hand_pos[2]) - self._hand_z0
            cube_rise = float(truth.target_pos[2]) - frames.CUBE_REST_Z
            closed_rest = truth.gripper_closed and ex.clocks.finger_rest_s >= ex.close_settle_s - 1e-9
            self.info.update(hand_rise_m=hand_rise, cube_rise_m=cube_rise, fingers=[float(v) for v in truth.fingers],
                             finger_rest_s=round(ex.clocks.finger_rest_s, 3),
                             finger_speed=float(np.sum(np.abs(truth.finger_vel))),
                             gripper_closed=bool(truth.gripper_closed))
            if cube_rise < float(c["cube_rise_max_m"]) and truth.gripper_closed and not closed_rest:
                return self._cmd(phase=S.Phase.lift)      # 持ち上げの揺れで指の速さが戻った: 止まるまで待つ（上限は timeout_s）
            if closed_rest and hand_rise >= float(c["hand_rise_min_m"]) and cube_rise < float(c["cube_rise_max_m"]):
                self._confirm(truth)
            else:
                self.status = "not_effective"
                self.reason = "cube_followed" if cube_rise >= float(c["cube_rise_max_m"]) else "not_closed_or_low"
            return self._cmd(phase=S.Phase.lift)
        return self._cmd(self.ex._vel_to(x, goal), False, S.Phase.lift)

    # ---------------------------------------------------------------- B
    def _arm_b(self, cmd, truth):
        if cmd.phase != S.Phase.carry or not (S.holding(truth, self.ex.pp) and S.lifted(truth, self.ex.pp)):
            return None
        d = float(np.hypot(*(truth.target_pos[:2] - truth.box[:2])))
        if not hasattr(self, "_r0"):
            self._r0 = d
            dmin = float(self.ic["B"]["min_dist_from_box_m"])
            self._b_threshold = dmin + float(self.p.b_u) * (d - dmin)
            self.info.update(r0_m=d, threshold_m=self._b_threshold)
            if d < dmin:
                self.status, self.reason = "natural_failure", "r0_below_min"
                return None
        if d >= self._b_threshold:
            return None
        self.stage = "dropped"
        self._t_open = truth.t
        self.info["drop_point"] = [float(v) for v in truth.target_pos]
        return self._press_once(S.Phase.settle)

    def _run_dropped(self, cmd, truth):
        """B・C の開いた後: 開いたまま静止し、立方体が机の上で静止したら確定する。"""
        self._update_rest(truth)
        lc = self.ic["landing"]
        if truth.t - self._t_open > float(lc["timeout_s"]):
            self.status, self.reason = "landing_invalid", "no_rest"
            self.info.update(self.landing_check(truth))
            return self._cmd()
        gap = float(np.sum(truth.fingers))
        apart = np.linalg.norm(truth.target_pos - truth.fingertip) > self.ex.pp.drop_dist
        on_table = truth.target_pos[2] - frames.CUBE_REST_Z < self.ex.pp.grasped_lift_min
        if (not truth.gripper_closed and gap > frames.CUBE_SIZE + float(lc["finger_gap_margin_m"]) and apart
                and on_table and self._rest_s >= float(lc["rest_hold_s"]) - 1e-9):
            self.info.update(finger_gap_m=gap, rest_s=self._rest_s)
            self._confirm(truth)
        return self._cmd()

    # ---------------------------------------------------------------- C
    def _arm_c(self, cmd, truth):
        if not (cmd.phase == S.Phase.release and cmd.press):
            return None
        self.stage = "rise"
        return self._run_c(cmd, truth)

    def _run_c(self, cmd, truth):
        ex = self.ex
        if self.stage == "dropped":
            return self._run_dropped(cmd, truth)
        if not S.holding(truth, ex.pp):
            self.status, self.reason = "natural_failure", "slipped_while_moving"
            return self._cmd()
        x = np.asarray(truth.x_cmd, float)
        drop = np.asarray(self.p.c_drop_xy, float)
        off = drop - truth.target_pos[:2]
        aim = x[:2] + off
        if self.stage == "rise":                          # 箱の上から、運ぶ高さへ真上に上がる
            if abs(x[2] - ex.carry_z) <= ex.move_tol:
                self.stage = "move"
            else:
                return self._cmd(ex._vel_to(x, np.array([x[0], x[1], ex.carry_z])), False, S.Phase.carry)
        if self.stage == "move":
            if np.hypot(*off) > ex.pp.above_tol:
                return self._cmd(ex._vel_to(x, np.array([aim[0], aim[1], ex.carry_z])), False, S.Phase.carry)
            self.stage = "lower"
        goal = np.array([aim[0], aim[1], ex.pp.release_z])
        if np.hypot(*off) <= ex.move_tol and abs(x[2] - ex.pp.release_z) <= ex.move_tol and ex._hand_still(truth):
            self.stage = "dropped"
            self._t_open = truth.t
            self.info["drop_point"] = [float(v) for v in truth.target_pos]
            return self._press_once(S.Phase.release)
        return self._cmd(ex._vel_to(x, goal), False, S.Phase.release)

    # -------------------------------------------------------------- 確定と着地の検査
    def _confirm(self, truth) -> None:
        self.t_confirm = round(truth.t, 3)
        land = self.landing_check(truth)
        self.info.update(land)
        if land["landing_ok"]:
            self.status = "confirmed"
        else:
            self.status, self.reason = "landing_invalid", ",".join(land["landing_fail"])

    def landing_check(self, truth: S.Truth) -> dict:
        """目標の立方体の状態の検査。評価で失敗を起こす側（recovla.eval.induce）も同じ関数を使う。"""
        return landing_check(self.rig, truth, self.cfg)


def landing_check(rig, truth: S.Truth, cfg: dict = None) -> dict:
    """着地の検査: 傾き 10° 以下、箱の外、他の立方体・壁との隙間 3 cm 以上、作業空間と俯瞰カメラの視野の中。"""
    cfg = cfg or _CFG
    lc = cfg["inject"]["landing"]
    m, d = rig.model, rig.data
    t = truth.target
    pos = truth.target_pos
    tilt = frames.tilt_deg(truth.cube_quat[t])
    mine = frames.cube_footprint(pos, truth.cube_quat[t])
    dist = {}
    for i, c in enumerate(COLORS):
        if i != t:
            dist[f"cube_{c}"] = frames.polygon_distance(mine, frames.cube_footprint(truth.cube_pos[i], truth.cube_quat[i]))
    for w, fp in frames.wall_footprints(m).items():
        dist[w] = frames.polygon_distance(mine, fp)
    outer = _box_outer_half(cfg)
    over_box = bool(np.all(np.abs(pos[:2] - truth.box[:2]) < outer + frames.CUBE_HALF))
    ws = cfg["sim"]["workspace"]
    in_ws = bool(ws["x"][0] <= pos[0] <= ws["x"][1] and ws["y"][0] <= pos[1] <= ws["y"][1])
    size = int(cfg["sim"]["image_size"])
    margin = float(lc["view_margin_px"])
    uv = frames.project(m, d, "overhead", pos, size, size)
    in_view = uv is not None and margin <= uv[0] <= size - margin and margin <= uv[1] <= size - margin
    fail = []
    if tilt > float(lc["max_tilt_deg"]):
        fail.append("tilt")
    if over_box:
        fail.append("over_box")
    if min(dist.values()) < float(lc["min_clearance_m"]):
        fail.append("clearance")
    if not in_ws:
        fail.append("workspace")
    if not in_view:
        fail.append("view")
    yaw = np.degrees(frames.quat_yaw(truth.cube_quat[t]))
    rel = (yaw + 45.0) % 90.0 - 45.0                        # 指（ヨー 0）に対する、90° の対称性を除いた角
    return {"landing_ok": not fail, "landing_fail": fail, "tilt_deg": round(tilt, 3),
            "clearance_m": {k: round(v, 5) for k, v in dist.items()}, "over_box": over_box,
            "in_workspace": in_ws, "overhead_uv": None if uv is None else [round(uv[0], 1), round(uv[1], 1)],
            "yaw_rel_deg": round(float(rel), 2), "target_pos": [round(float(v), 5) for v in pos]}
