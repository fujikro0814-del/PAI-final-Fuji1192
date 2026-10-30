"""方策の非同期実行（シミュレーションの時刻の上で、推論の間も世界は一度も止まらない）。

    rt = PolicyRuntime(io, setup, policy, mode="naive", s=10)
    rt.start(task, seed)            # 試行・サブタスクの始め（腕は保持から始まる）
    rt.tick                       # 物理の 1 ステップ（500 Hz）ごとに 1 回。腕の指令（1 kHz 相当で 2 つ）とハンドの指令を io に出す
    rt.reset_chunks               # サブタスクの切り替え・やり直し（行動列を持ち越さない）

- 推論: 行動の区切り（0.1 s）に観測を組み、io.compute で計算する。結果は「始めた時刻＋計算の時間」から使える
  （計算の時間は評価の枠が実測の分布から試行のシードで引く。前処理を含む）。使える前に使うと Pending が止める
- naive・rtc: 前の推論を始めてから s 行ごとに次の推論を始め、その間は前の行動列を実行し続ける。新しい行動列は、使えるように
  なった後の最初の区切りから、観測から数えた行の位置で使う。行動列がない間（最初・切り替えの直後）は腕をその場で保持し、
  そのとき届いた行動列は 0 行目から使う（保持していた間、腕は観測の時点から動いていないため）
- sync（止まって推論する）: 推論の間は腕をその場で保持し（世界は進む）、使えるようになった区切りから s 行を実行して、また保持して推論する
- rtc: 方策に渡す inference_delay は実行系が見積もった値（直前の推論で実際にかかった行の数。最初は configs の d）。
  実際に使う行の位置は、実際に届いた時刻で決まる（未来の計算時間は実行系には分からないため）
- グリッパ: 行動の 7 番目が正なら閉（grasp）、そうでなければ開（move）。変わったときだけ口に出す
- 落下の反射（reflex。既定は None で切）: ハンドの新しい読みごとに recovla.runtime.reflex で検出し、発火したら古い行動列と
  計算中の推論を捨て、参照を止めて指を開き、待ちの後に新しい観測から推論し直す。切なら動きも記録も反射のないときと同じ
"""
import math

import numpy as np

from recovla.policy.schedule import normalize_left_over, noise_generator
from recovla.runtime.motion import Motion

ACTION_DT = 0.1
STEPS_PER_ACTION = 50


def enable_rtc_for(policy, horizon: int, schedule: str, max_guidance_weight: float) -> dict:
    """RTC を入れる（policy.runner.enable_rtc と同じ設定）。"""
    from recovla.policy.runner import enable_rtc
    return enable_rtc(policy.policy, horizon, schedule, max_guidance_weight)


def disable_rtc_for(policy) -> None:
    from recovla.policy.runner import disable_rtc
    disable_rtc(policy.policy)


class PolicyRuntime:
    def __init__(self, io, setup, policy, mode: str = "naive", s: int = 10, d_init: int = 4, rtc_horizon: int = 40,
                 motion: Motion = None, limiter_enabled: bool = True, margin: float = 0.99,
                 perception=None, safety=None, checks: dict = None, tip_offset: float = 0.1034, gripper_gate: dict = None,
                 reflex=None):
        if mode not in ("sync", "naive", "rtc"):
            raise ValueError(mode)
        self.io, self.setup, self.policy = io, setup, policy
        self.mode, self.s, self.d_init, self.E = mode, int(s), int(d_init), int(rtc_horizon)
        self.motion = motion or Motion(setup, limiter_enabled=limiter_enabled, margin=margin)
        self.action_filter = None          # 評価の道具（意図的な失敗）が行動を上書きする口。None なら方策のまま
        self.injecting = False             # 評価の道具が上書きしている間は真（安全フィルタを切る）
        self.external = False              # 上位層が速さを直接出している間は真（方策を止め、安全フィルタを切る）
        self.perception = perception       # recovla.runtime.perception.Perception（None なら知覚なし）
        self.safety = safety               # recovla.runtime.safety.PerceptionSafetyFilter（None ならフィルタなし）
        self.checks = dict(checks or {})   # 知覚の失敗の止まり方の閾値（configs の runtime_v2.checks）
        self.tip_offset = float(tip_offset)
        self.gripper_gate = gripper_gate if gripper_gate and gripper_gate.get("enabled") else None
        self.t_grip_switch = -1e9
        self.n_open_req = 0                # 閉じている間に「開」が続いた手の数（gripper_gate の open_steps。無ければ使わない）
        self.n_open_held = 0               # open_steps で開かずに保った手の数（記録だけ）
        self.reflex = reflex               # recovla.runtime.reflex.GraspLossReflex（None なら切）

    # -------------------------------------------------------------------- 試行
    def start(self, task: str, seed: int) -> None:
        self.task, self.seed = task, int(seed)
        sensor = self.io.sense(cameras=False)
        self.motion.reset(sensor.joints)
        self.policy.start_trial(seed)
        from recovla.runtime.cue import color_of_instruction
        self.color = color_of_instruction(task)
        self.ready = self.perception is None          # 知覚があるなら、起動時の確かめ（机の面・箱）の後に始める
        self.boxf = []
        self.pend_per = None
        self.wm = None
        self.stop_reason = None
        self.t_task0 = None
        self.log_per = []
        if self.safety is not None:
            self.safety.start_trial(self.color)
        self.n_tick = 0
        self.n_open_req = 0
        self.k = -1
        self.closed = False
        self.t_grip_switch = -1e9
        self.i = 0                          # 推論の3 個の連続タスク番号（雑音の列）
        self.d_est = self.d_init
        self.log_inf, self.log_act = [], []
        self.reset_chunks()
        if self.reflex is not None:
            self.reflex.start_trial()
            self.rx_intent = False          # 方策が最後に出した開閉（評価の道具が上書きする前）
            self.rx_grip_t = None           # 最後に読んだハンドの読みの時刻

    def set_task(self, task: str, seed: int) -> None:
        """サブタスクの切り替え・やり直し（知覚の起動時の確かめと推定は保つ）。行動列は持ち越さない。雑音の列は seed から。"""
        from recovla.runtime.cue import color_of_instruction
        self.task, self.seed = task, int(seed)
        self.color = color_of_instruction(task)
        self.policy.start_trial(seed)
        self.i = 0
        self.t_task0 = None
        self.stop_reason = None if self.stop_reason == "target_not_found" else self.stop_reason
        if self.safety is not None:
            wm = self.wm
            self.safety.start_trial(self.color)
            if wm is not None:
                self.safety.set_world(wm)
        self.reset_chunks()

    def reset_chunks(self) -> None:
        self.active = None                  # {"i", "post", "raw", "k0", "o0", "rows_done"}
        self.pending = None                 # {"i", "k_obs", "t_obs", "fut", "rtc"}
        self.next_infer_k = None
        self.policy.reset_cue()
        self.motion.set_velocity(np.zeros(3))

    # --------------------------------------------------------------------- 物理の 1 ステップ
    def tick(self) -> None:
        if self.n_tick % STEPS_PER_ACTION == 0:
            self.k += 1
            self._action_boundary()
        sensor = self.io.sense(cameras=False)
        if self.reflex is not None:
            self._reflex_sample(sensor)
        if self.perception is not None:
            self.perception.record_joints(sensor.joints)
        if self.safety is not None and self.n_tick % 10 == 0:          # 入力の読み取り（50 Hz）ごと
            self.safety.begin_read(sensor.joints.q, sensor.gripper.width, self.motion.x_cmd)
            self.motion.set_command_filter(self.safety.filter if self.safety.on() else None)
        self.n_tick += 1
        for q in self.motion.step(sensor.joints):
            self.io.command_joints(q)

    # ------------------------------------------------------------------ reflex（落下の反射。既定は切）
    def _reflex_sample(self, sensor) -> None:
        """ハンドの新しい読みごとに検出する（読みは 10 Hz。行動の区切りを待たない）。発火したらその場で古い行動列を捨て、
        手先の参照を止めて指を開く（実機の口: 直交座標の参照と、ハンドの move）。"""
        g = sensor.gripper
        if self.external or g is None or g.t == self.rx_grip_t:
            return
        self.rx_grip_t = g.t
        rx = self.reflex
        if rx.overriding:
            rx.observe(self.io.now(), self.k, g.width, self.rx_intent)          # 待ちの間は開き幅を控えるだけ
            return
        hand = self.motion.hand_pose(sensor.joints.q)[0]
        if rx.observe(self.io.now(), self.k, g.width, self.rx_intent, hand=hand, x_cmd=self.motion.x_cmd, sample_t=g.t):
            rx.cur["discarded"] = {"active": None if self.active is None else int(self.active["i"]),
                                   "pending": None if self.pending is None else int(self.pending["i"])}
            self.active = None
            self.pending = None
            self.next_infer_k = None
            if self.safety is not None:
                self.safety.gate = False
            a = rx.hold_action(self.motion.x_cmd)
            self._apply(a)
            # 行動の記録は「実際に出した指令」にする。同じ手で区切りの行動を出した直後に発火したなら、その記録（捨てた方策の
            # 行動）を止めの指令に置き換え、区切りの途中で発火したなら、発火の時刻の行として足す（保持の扱い。行動列の番号なし）
            t = self.io.now()
            row = (self.k, t, None, True, a.copy())
            if self.log_act and self.log_act[-1][0] == self.k and abs(self.log_act[-1][1] - t) < 1e-9:
                self.log_act[-1] = row
                rx.cur["log"] = "replaced_boundary_row"
            else:
                self.log_act.append(row)
                rx.cur["log"] = "appended_row"
            rx.cur["a_sent"] = [round(float(v), 6) for v in a]

    # ------------------------------------------------------------------ 知覚
    def _tip(self, q) -> np.ndarray:
        pos, quat = self.motion.hand_pose(q)
        w, x, y, z = quat
        zaxis = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
        return pos + self.tip_offset * zaxis

    def _perception(self, t: float) -> None:
        per = self.perception
        if self.pend_per is not None and self.pend_per.ready(t):
            out = self.pend_per.result(t)
            self.pend_per = None
            if out["kind"] == "startup":
                self._after_startup(out)
            else:
                self.wm = out["wm"]
                if self.safety is not None:
                    self.safety.set_world(self.wm)
                self.log_per.append({"t": t, "t_obs": out["t_obs"],
                                     "cubes": {c: {"pos": e.pos.tolist(), "status": e.status, "in_box": e.in_box,
                                                   "source": e.source} for c, e in self.wm.cubes.items()},
                                     "box": {"xy": np.asarray(self.wm.box["xy"]).tolist(), "yaw": self.wm.box["yaw"],
                                             "ok": self.wm.box["ok"]}})
                self._check_target(t)
        if self.pend_per is not None or self.stop_reason:
            return
        sensor = self.io.sense(cameras=True)
        if not {"overhead", "wrist"} <= set(sensor.cameras):
            return
        if not self.ready:
            self.boxf.append((sensor.cameras["overhead"], sensor.gripper.width))
            if len(self.boxf) < int(self.checks.get("box_frames", 5)):
                return
            frames, self.boxf = self.boxf, []

            def startup():
                chk = per.check_table(frames[0][0], frames[0][1])
                box = per.init_box(frames)
                return {"kind": "startup", "table": chk, "box": box}
            self.pend_per = self.io.compute("perception_startup", startup)    # 起動時の 1 回（机と箱の当てはめ）は別の遅延
            return
        tip = self._tip(sensor.joints.q)
        t_obs = t

        def work():
            return {"kind": "update", "wm": per.update(sensor.cameras, sensor.gripper, sensor.t, tip), "t_obs": t_obs}
        # 計算の時間の種類は知覚が決める（推定を足した知覚は重いので別の遅延。推定なしの知覚は "perception"）
        self.pend_per = self.io.compute(getattr(per, "compute_kind", "perception"), work)

    def _after_startup(self, out) -> None:
        c = self.checks
        tab, box = out["table"], out["box"]
        self.startup = {"table": tab, "box": {k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v)
                                               for k, v in box.items()}}
        if abs(tab["dz_m"]) > float(c.get("table_dz_m", 1e9)) or tab["angle_deg"] > float(c.get("table_angle_deg", 1e9)):
            self.stop_reason = "table_mismatch"
        elif not box.get("ok"):
            self.stop_reason = "box_not_found"
        elif box.get("dev_m", 0.0) > float(c.get("box_dev_m", 1e9)) or box.get("dev_yaw_deg", 0.0) > float(c.get("box_dev_deg", 1e9)):
            self.stop_reason = "box_moved"
        self.ready = self.stop_reason is None
        if tab.get("corrected") and hasattr(self.policy, "set_cue_pose"):
            R, t = self.perception.camera_pose("overhead", self.io.now(), 0.08)
            self.policy.set_cue_pose(R, t)

    def _check_target(self, t: float) -> None:
        if self.t_task0 is None:
            self.t_task0 = t
        e = self.wm.cubes.get(self.color)
        missing = e is None or e.status == "lost"
        if missing and t - self.t_task0 > float(self.checks.get("target_missing_s", 1e9)) and self.active is None:
            self.stop_reason = "target_not_found"

    def _action_boundary(self) -> None:
        k, t = self.k, self.io.now()
        if self.perception is not None:
            self._perception(t)
            if not self.ready or self.stop_reason:           # 起動時の確かめの間・知覚の失敗: 腕をその場で保持する
                self.motion.set_velocity(np.zeros(3))
                self.log_act.append((k, t, None, True, np.array([0, 0, 0, 0, 0, 0, 1.0 if self.closed else -1.0])))
                return
        if self.external:                                     # 上位層が腕を動かしている間（待機位置へ戻す動き）: 方策は止める
            if self.safety is not None:
                self.safety.gate = False
            return
        # 1) 届いた推論を行動列にする
        if self.pending is not None and self.pending["fut"].ready(t):
            p = self.pending
            post = p["fut"].result(t)
            lag = k - p["k_obs"]
            # 観測から今までに腕が前の行動列を実行していたなら、その分だけ先の行から使う。保持していた（行動列がなかった）なら、
            # 腕は観測の時点から動いていないので 0 行目から使う（最初の推論・切り替えの直後・同期）
            o0 = lag if (self.mode != "sync" and p["was_moving"]) else 0
            self.active = {"i": p["i"], "post": post, "raw": p["raw"](), "k0": k, "o0": o0, "rows_done": 0}
            self.d_est = max(1, lag)
            self.log_inf[-1].update({"k_act": k, "t_act": t, "offset": o0})
            self.pending = None
            rx = self.reflex
            if rx is not None and rx.overriding and not rx.accept_chunk(t, k, p["i"], self.log_inf[-1].get("cue")):
                self.active = None                            # 反射の待ちの後の行動列の条件を満たさない: 捨てて推論し直す
        # 2) 推論を始めるか
        start = False
        if self.reflex is not None and self.reflex.blocking(t):
            pass                                              # 反射の待ちの間は推論を始めない
        elif self.pending is None:
            if self.mode == "sync":
                start = self.active is None or self.active["rows_done"] >= self.s
                if start:
                    self.active = None
            else:
                start = self.active is None or (self.next_infer_k is not None and k >= self.next_infer_k)
        if start:
            self._start_inference(k, t)
        # 3) この区切りの行動
        a = None
        if self.active is not None:
            row = self.active["o0"] + (k - self.active["k0"])
            if row < len(self.active["post"]) and not (self.mode == "sync" and self.active["rows_done"] >= self.s):
                a = np.array(self.active["post"][row], float)
                self.active["rows_done"] += 1
        held = a is None
        if held:
            a = np.zeros(7)
            a[6] = 1.0 if self.closed else -1.0
        if self.reflex is not None:
            if self.reflex.overriding:                         # 反射の止め方（参照を止めて保ち、指を開く）
                a, held = self.reflex.hold_action(self.motion.x_cmd), True
            else:
                self.rx_intent = bool(a[6] > 0.0)              # 方策の開閉（評価の道具が上書きする前）
        if self.action_filter is not None:
            a = np.asarray(self.action_filter(k, a), float)
        if self.safety is not None:                          # 方策が指令を出している間だけ（保持・評価の道具の上書き中は切る）
            self.safety.gate = (not held) and not self.injecting
        self._apply(a)
        self.log_act.append((k, t, None if held else self.active["i"], held, a.copy()))

    def _start_inference(self, k: int, t: float) -> None:
        sensor = self.io.sense(cameras=True)
        if not {"overhead", "wrist"} <= set(sensor.cameras):
            return                          # まだフレームが届いていない（試行の始め）。保持して次の区切りでやり直す
        i = self.i
        self.i += 1
        rtc = None
        if self.mode == "rtc" and self.active is not None:
            j = self.active["o0"] + (k - self.active["k0"])
            left = self.active["raw"][min(j, len(self.active["raw"])):]
            rtc = {"inference_delay": int(min(self.d_est, self.s - 1)),
                   "prev_chunk_left_over": normalize_left_over(left, self.E)}
        pol = self.policy
        holder = {}

        def work():
            obs = pol.observe(sensor, self.task)
            out = pol.infer(obs, noise_generator(self.seed, i), rtc)
            holder["raw"] = pol.last_raw.copy()
            holder["cue"] = None if pol.last_cue is None else np.asarray(pol.last_cue).copy()
            holder["state"] = np.asarray(obs["observation.state"], float).copy()       # 記録だけ
            return out

        fut = self.io.compute("policy_rtc" if self.mode == "rtc" else "policy", work)
        self.pending = {"i": i, "k_obs": k, "t_obs": t, "fut": fut, "raw": lambda: holder["raw"],
                        "was_moving": self.active is not None}
        self.next_infer_k = k + self.s
        self.log_inf.append({"i": i, "k_obs": k, "t_obs": t, "t_ready": fut.t_ready, "latency_s": fut.latency,
                             "wall_s": fut.wall_s, "rtc_delay": None if rtc is None else rtc["inference_delay"],
                             "img_t": {n: c.t_capture for n, c in sensor.cameras.items()},
                             "cue": holder.get("cue"), "state": holder.get("state")})

    def _apply(self, a) -> None:
        self.motion.set_velocity(a[:3] / ACTION_DT)
        g = self.gripper_gate
        if g is None:
            want = bool(a[6] > 0.0)
        else:
            want = self.closed
            asks_open = self.closed and a[6] < float(g["open"])
            self.n_open_req = self.n_open_req + 1 if asks_open else 0
            if self.io.now() - self.t_grip_switch >= float(g["min_hold_s"]) - 1e-9:
                if not self.closed and a[6] > float(g["close"]):
                    want = True
                elif asks_open:
                    # open_steps: 「開」が続けてこの手数そろったときだけ開く（運ぶ途中の短い開の跳ねを通さない。無ければ 1）
                    if self.n_open_req >= int(g.get("open_steps", 1)):
                        want = False
                    else:
                        self.n_open_held += 1
        if want != self.closed:
            self.closed = want
            self.t_grip_switch = self.io.now()
            su = self.setup
            if want:
                eps = 0.005
                self.io.gripper_grasp(su.cube_size, su.gripper_speed, su.grasp_force, eps, eps)
            else:
                self.io.gripper_move(0.08, su.gripper_speed)

    def trace(self) -> dict:
        out = self._trace()
        if self.reflex is not None:                           # 反射が入っているときだけ足す
            out["reflex"] = self.reflex.summary()
        if self.gripper_gate is not None and "open_steps" in self.gripper_gate:   # 絞りが入っているときだけ足す
            out["gripper_gate"] = {"open_steps": int(self.gripper_gate["open_steps"]), "open_held_steps": self.n_open_held}
        return out

    def _trace(self) -> dict:
        return {"inference": self.log_inf, "actions": [
            {"k": k, "t": t, "chunk": c, "held": h, "a": a.tolist()} for k, t, c, h, a in self.log_act],
            "perception": self.log_per, "startup": getattr(self, "startup", None), "stop_reason": self.stop_reason,
            "safety": None if self.safety is None else self.safety.summary(),
            "motion": {**self.motion.lag_stats, "cart_clipped": self.motion.n_cart_clipped,
                       "xcmd_leashed": self.motion.n_xcmd_leashed, "xcmd_leash_m": self.motion.xcmd_leash_m,
                       "cart_unresolved": self.motion.n_cart_unresolved, "pos_clipped": self.motion.limiter.n_pos_clipped,
                       "joint_clipped": self.motion.limiter.n_clipped}}
