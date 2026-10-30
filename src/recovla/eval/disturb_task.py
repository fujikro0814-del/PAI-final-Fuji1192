"""評価の外乱（docs/api/r1_protocol.md の外乱の評価）: 「全部片付けて」の試行で、目標の立方体を持ち上げてから 0.5〜2 s 後に、
横から押して手から落とす。評価の枠の側（真値を使ってよい。実行系・方策には知らせない）。既定は切（scripts/api/ladder.py の
--disturb push のときだけ）。

    d = PushDropDisturber(seed)
    d.pre_physics_step(world, target_color)     # 物理の 1 ステップごと（mj_step の前）
    d.record                                  # 試行の json の "disturb"

決まり（真値）:
- 持ち上げた: その時の手順の目標の立方体の中心が、試行の始めの高さより LIFT_M 以上高く、指先の中心から HOLD_M 以内
- 押す時刻: 持ち上げた時刻 ＋ 一様(delay_s)。シードごとの乱数。その時刻にまだ持っていれば押す。持っていなければ
  （その前に置いた・落とした）、次の持ち上げを待つ（押すのは 1 試行に max_pushes 回まで）
- 押し方は eval/induce_slip.PullOutInducer（引き抜き）と同じ: 物理の 1 ステップごとに、指に対する立方体の位置と速さの
  「押す向きの成分」を speed で離れていく道に合わせる（ほかの向き・回転・重力・指との接触は物理のまま）。指先の中心から
  押す向きに exit_m 離れたら手を離す（max_s で打ち切り）。向きは指の溝の向き（hand の x 軸を水平にしたもの。指の閉じる
  向きと直交）の ±に、下向きを down_deg 混ぜたもの。力で押す形は、静止摩擦が切れた瞬間に立方体が飛ぶので使わない
- 乱数: SeedSequence(seed, spawn_key=(SPAWN_KEY,))（common/seeds の列とは別の番号。ほかの列の値は変わらない）
記録: 持ち上げの列（時刻・押す予定の時刻・押したか・押さなかった理由）、押した時刻・向き・速さ・抜けた時刻・理由・力積の見積もり、
  押した後 2 s の立方体の行き先（指先からの距離・高さ）。
"""
import numpy as np

LIFT_M = 0.015          # 持ち上げたとみなす高さ [m]
HOLD_M = 0.04           # 指先の中心からこの中なら持っている [m]
SPAWN_KEY = 9101        # 乱数の列の番号（common/seeds.STREAM_ID と重ならない）
PARAMS = {"delay_s": (0.5, 2.0), "speed_m_s": (0.2, 0.5), "down_deg": (0.0, 60.0), "exit_m": 0.045, "max_s": 0.4,
          "max_pushes": 1, "after_s": 2.0}
COLORS = ("red", "green", "blue")


class PushDropDisturber:
    def __init__(self, seed: int, params: dict = None):
        self.seed = int(seed)
        self.p = dict(PARAMS, **(params or {}))
        self.rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(self.seed, spawn_key=(SPAWN_KEY,))))
        self.z0 = None
        self.lifts, self.pushes = [], []
        self._wait = None          # 持ち上げの後の待ち {"color", "t_lift", "t_push"}
        self._on = None            # 押している途中の状態
        self._after = None         # 押した後の見届け

    # ------------------------------------------------------------------ 真値の読み
    @staticmethod
    def _tip(world):
        from recovla.sim import frames
        return frames.fingertip_center(world.data, world.hand_id)

    def _held(self, world, i: int, tip) -> bool:
        p = world.data.xpos[world.cube_ids[i]]
        return bool(p[2] - self.z0[i] >= LIFT_M and np.linalg.norm(p - tip) <= HOLD_M)

    # ------------------------------------------------------------------ 物理の 1 ステップごと
    def pre_physics_step(self, world, target) -> None:
        d = world.data
        if self.z0 is None:
            self.z0 = np.array([float(d.xpos[b][2]) for b in world.cube_ids])
        t = float(d.time)
        tip = self._tip(world)
        if self._on is not None:
            self._push_step(world, t, tip)
            return
        if self._after is not None:
            self._watch_after(world, t, tip)
        if len(self.pushes) >= int(self.p["max_pushes"]) or target not in COLORS:
            return
        i = COLORS.index(target)
        held = self._held(world, i, tip)
        w = self._wait
        if w is not None and w["color"] != target:
            w["result"], self._wait, w = "target_changed", None, None
        if w is None:
            if held:
                dl = float(self.rng.uniform(*self.p["delay_s"]))
                self._wait = {"color": target, "t_lift": round(t, 3), "delay_s": round(dl, 3), "t_push": round(t + dl, 3),
                              "result": None}
                self.lifts.append(self._wait)
            return
        if not held:
            w["result"], self._wait = "released_before_push", None
            return
        if t >= w["t_push"] - 1e-9:
            w["result"], self._wait = "pushed", None
            self._start(world, i, t, tip)

    def _start(self, world, i: int, t: float, tip) -> None:
        d = world.data
        R = d.xmat[world.hand_id].reshape(3, 3)
        x = np.array([R[0, 0], R[1, 0], 0.0])
        x /= max(np.linalg.norm(x), 1e-9)
        side = 1.0 if self.rng.random() < 0.5 else -1.0
        down = float(self.rng.uniform(*self.p["down_deg"]))
        speed = float(self.rng.uniform(*self.p["speed_m_s"]))
        th = np.radians(down)
        u = np.cos(th) * side * x + np.sin(th) * np.array([0.0, 0.0, -1.0])
        u /= np.linalg.norm(u)
        b = world.cube_ids[i]
        self._on = {"i": i, "dir": u, "speed": speed, "t0": t, "p0": float(np.dot(d.xpos[b] - tip, u)), "tip_prev": None,
                    "imp": 0.0}
        self.pushes.append({"color": COLORS[i], "t_start": round(t, 4), "side": side, "down_deg": round(down, 2),
                            "speed_m_s": round(speed, 3), "dir": [round(float(v), 4) for v in u],
                            "mass_kg": round(float(world.model.body_mass[b]), 4), "t_end": None, "end_reason": None,
                            "impulse_ns": None, "after": None})

    def _push_step(self, world, t: float, tip) -> None:
        from recovla.sim import scene
        d, m, s = world.data, world.model, self._on
        i = s["i"]
        b, va = world.cube_ids[i], int(world.cube_vadr[i])
        dt = float(m.opt.timestep)
        v_tip = np.zeros(3) if s["tip_prev"] is None else (tip - s["tip_prev"]) / dt
        s["tip_prev"] = tip.copy()
        u = s["dir"]
        p = float(np.dot(d.xpos[b] - tip, u))
        el = t - s["t0"]
        if p > self.p["exit_m"] or el > self.p["max_s"]:
            rec = self.pushes[-1]
            rec.update(t_end=round(t, 4), end_reason="exit" if p > self.p["exit_m"] else "max_s", impulse_ns=round(s["imp"], 4))
            self._on = None
            self._after = {"rec": rec, "t_end": t}
            return
        qa = scene.cube_qpos_adr(m, COLORS[i])[0]
        dp = (s["p0"] + s["speed"] * el) - p              # 外の手: 指に対する立方体の位置を、押す向きに speed で動かす
        d.qpos[qa:qa + 3] = d.qpos[qa:qa + 3] + dp * u
        v = d.qvel[va:va + 3]
        dv = (float(np.dot(v_tip, u)) + s["speed"]) - float(np.dot(v, u))
        d.qvel[va:va + 3] = v + dv * u
        s["imp"] += float(m.body_mass[b]) * dv

    def _watch_after(self, world, t: float, tip) -> None:
        a = self._after
        if t - a["t_end"] < self.p["after_s"] - 1e-9:
            return
        i = COLORS.index(a["rec"]["color"])
        p = world.data.xpos[world.cube_ids[i]]
        a["rec"]["after"] = {"t": round(t, 3), "dist_tip_m": round(float(np.linalg.norm(p - tip)), 4),
                             "z_above_start_m": round(float(p[2] - self.z0[i]), 4),
                             "dropped": bool(np.linalg.norm(p - tip) > HOLD_M)}
        self._after = None

    def record(self) -> dict:
        if self._on is not None:
            self.pushes[-1].update(end_reason="trial_end", impulse_ns=round(self._on["imp"], 4))
            self._on = None
        return {"kind": "push_drop", "seed": self.seed, "params": {k: list(v) if isinstance(v, tuple) else v for k, v in self.p.items()},
                "spawn_key": SPAWN_KEY, "lifts": list(self.lifts), "pushes": list(self.pushes),
                "pushed": bool(self.pushes), "dropped": any((x.get("after") or {}).get("dropped") for x in self.pushes)}


def attach(world, disturber, target_of) -> None:
    """world.physics_step を包んで、mj_step の前に disturber.pre_physics_step(world, target_of) を呼ぶ（この世界の物だけ）。"""
    if not hasattr(world, "_physics_step_orig"):             # 同じ世界に 2 回付けても重ねない
        world._physics_step_orig = world.physics_step
    orig = world._physics_step_orig

    def physics_step(on_step=None):
        disturber.pre_physics_step(world, target_of())
        orig(on_step)
    world.physics_step = physics_step
