"""3 個を続けて片付けるタスクで、手順の切り替えのときの腕の戻し先を比べる診断と、その集計。

条件（ARMS。名前は識別子のまま）:
  E0  今の実行器のまま（同じ配置で 2 回回し、回ごとのばらつきを測る）
  EH  2 番目以降の手順の前に、学習データの始めの姿勢（home）の関節角へ関節空間で戻す
  ES  同じく、学習データの待機位置の始めの姿勢へ戻す
  EO  指示文で色の順（緑 → 赤 → 青）を決める
  EX  手順の切り替えで実行系を作り直す

実行系の側（シミュレーションの真値は持たない）:
    GoalMotion        runtime.motion.Motion の子。関節の目標があれば、関節空間で目標へ動かす指令を制限層（limiter）と
                      手先の上限（Motion._cartesian_limit）に通して出す。目標がなければ Motion.step のまま
    DiagTaskRuntime   runtime.executor.TaskRuntime の子。前の手順を終えた後、次の手順を始める前に
                      EH・ES: 指を開く → 低ければ真上に rise_z まで → 関節空間で戻し先へ → hold_s 保つ、を挟む
                      EX: 実行系（PolicyRuntime の起動時の確かめ・知覚の推定・行動列・方策の試行の状態）を作り直す
    install が、作った TaskRuntime と Motion の型をこの子に差し替える（__class__ を替える。持っている値はそのまま）。
集計（記録だけを読む）:
    attempt_first_closes(meta, z)   各手順・各試みの最初の閉じ
    summarize_condition(dir)        条件 1 つ（run_NNNN.json・npz）の集計
    gate_T_inputs(summaries)        事前に決めた判定に使う値（E0 の基準・要る差・条件ごとの値）を並べる。判定はしない

事前に決めた項目:
  - 戻し先の関節角: EH は home [0, -0.684, 0, -2.907, 0, 2.216, 0.785]。ES は学習データの成功データ
    （start=retreat の 78 本）の 0 フレーム目の関節角の平均（手先 z 0.314）。train_pose_means で計算し直せる。
  - 戻す時: 2 番目以降の手順の 1 回目の試みを始める前（前の手順の完了の判定の後）。やり直しの前の戻す動きは
    E0 と同じ（待機位置へ）。1 番目の手順の前には戻さない（試行は home から始まる）。
  - 動かし方: GOAL_PARAMS。関節の速さ = gain × 残り（各関節の最大が vmax を超えれば全体を縮める＝関節空間の直線）を、
    制限層（libfranka の limitRate、上限の 0.99 倍）と手先の上限（1.7 m/s・13 m/s²・6500 m/s³ の 0.95 倍）に通す。
    着いた＝指令の関節角の残りの最大 <= 1e-3 rad かつ速さの最大 <= 5e-3 rad/s。着いたら meas_settle_s 待ち、測った関節角の差
    （目標 − 測った値）だけ指令の目標をずらしてもう一度動かす（meas_corrections 回）。その後、関節の目標を保ったまま hold_s 保ち、
    手順を始める時に直交座標の動きへ戻す（先に戻すと IK の定常のずれで手先が動く）。
    10 s で着かなければ、その場から手順を始め、arrived=false を記録する。
  - EO の指示文: EO_TEXT（緑 → 赤 → 青の順）。手順に分ける LLM はキャッシュに無ければ 1 回だけ呼ぶ。
"""
import json
import pathlib

import numpy as np

from recovla.common.seeds import COLORS
from recovla.runtime.executor import TaskRuntime, step_seed
from recovla.runtime.motion import SUBSTEPS, Motion
from recovla.sim import control, frames
from recovla.sim.device import pad_state

# ---------------------------------------------------------------- 戻し先の関節角 [rad]（事前に固定）
HOME_Q = (0.0, -0.684, 0.0, -2.907, 0.0, 2.216, 0.785)
STANDBY_START_Q = (0.162272, -0.394371, 0.27625, -2.802853, 0.159527, 2.408117, 1.081562)
STANDBY_END_Q = (0.166045, -0.359234, 0.280733, -2.804154, 0.155684, 2.432485, 1.092411)
POSE_SOURCE = ("outputs/manifests/R1v3_20261005-133737.json の kind n（通常デモ）: start=home 162 本・start=retreat 78 本の 0 こま目、"
               "全 240 本の最後のこまの joints の平均（担当 B が 10/08 に読み取りで計算。train_pose_means で計算し直せる）。"
               "HOME_Q は目標書 8-3・8-5 の値そのもの（学習の home の平均 [0.000032, -0.684448, ..., 0.785001] と 5e-4 rad 以内）")
TRAIN_EE_Z = {"home": 0.355284, "standby_start": 0.314438, "standby_end": 0.303591}   # 学習データの手先 z の平均 [m]

DEFAULT_TEXT = "全部片付けて"
EO_TEXT = "緑、赤、青の順に全部片付けて"
EO_ORDER = ["green", "red", "blue"]

GOAL_PARAMS = {"vmax_rad_s": 0.5, "gain_per_s": 2.0, "tol_cmd_rad": 1e-3, "tol_vel_rad_s": 5e-3, "hold_s": 0.5,
               "timeout_s": 10.0, "open_wait_max_s": 1.0, "meas_corrections": 1, "meas_settle_s": 0.25}
# meas_corrections: 指令が着いて meas_settle_s 待った後、測った関節角の差（目標 − 測った値）だけ指令の目標をずらして、もう一度動かす回数。
# 位置のサーボの定常のずれ（試しの実行で 0.0053 rad、手先 z 3 mm）を、実行系が測る関節角だけで直す（真値は使わない）

ARMS = {
    "E0": {"ja": "今の実行器（同じ 40 種で 2 回。条件名 E0_run1・E0_run2）", "goal": None, "text": DEFAULT_TEXT},
    "EH": {"ja": "学習の home の関節角へ、関節空間で制限層を通して戻す", "goal": "home", "goal_q": HOME_Q, "text": DEFAULT_TEXT},
    "ES": {"ja": "学習の待機位置の始めの姿勢（z 0.314）へ、関節空間で制限層を通して戻す", "goal": "standby_start",
           "goal_q": STANDBY_START_Q, "text": DEFAULT_TEXT},
    "EO": {"ja": "緑 → 赤 → 青の順（指示文を変え、段階 4 の計画役で 1 回呼んでキャッシュ）", "goal": None, "text": EO_TEXT,
           "expected_order": EO_ORDER},
    "EX": {"ja": "条件つき（8-4 の行 1・行 3 の矛盾の枝のときだけ）。手順の切り替えで実行系を作り直す", "goal": None,
           "rebuild": True, "text": DEFAULT_TEXT, "conditional": True},
}


def arm_record(arm: str) -> dict:
    """試行の meta["diag"] と run.json に残す条件の定義。"""
    a = dict(ARMS[arm])
    if a.get("goal_q") is not None:
        a["goal_q"] = list(a["goal_q"])
        a["goal_params"] = dict(GOAL_PARAMS)
    return {"diag": "D-E7", "arm": arm, **a}


def train_pose_means(manifest_path, root=None) -> dict:
    """学習データの成功データの、開始（home・retreat）の 0 フレーム目と最後のフレームの関節角・手先の平均（読むだけ。定数の照合用）。"""
    root = pathlib.Path(root) if root else pathlib.Path(manifest_path).resolve().parents[2]
    man = json.loads(pathlib.Path(manifest_path).read_text(encoding="utf-8"))
    st, sx, en, ex = {"home": [], "retreat": []}, {"home": [], "retreat": []}, [], []
    for e in man["entries"]:
        p = root / e["run"] / e["key"]
        m = json.loads((p / "meta.json").read_text(encoding="utf-8"))
        if m["kind"] != "n":
            continue
        z = np.load(p / "data.npz")
        st[m["layout"]["start"]].append(z["joints"][0])
        sx[m["layout"]["start"]].append(z["ee_pos"][0])
        en.append(z["joints"][-1])
        ex.append(z["ee_pos"][-1])
    out = {}
    for k in st:
        out[f"start_{k}"] = {"n": len(st[k]), "joints": np.mean(st[k], 0).tolist(), "ee": np.mean(sx[k], 0).tolist()}
    out["end"] = {"n": len(en), "joints": np.mean(en, 0).tolist(), "ee": np.mean(ex, 0).tolist()}
    return out


# ---------------------------------------------------------------- 実行系の側: 関節空間の戻す動き
class GoalMotion(Motion):
    """Motion の子（install が __class__ を差し替える。__init__ は呼ばないので、足す値は getattr で既定を持つ）。"""

    def set_joint_goal(self, q, vmax: float, gain: float) -> None:
        self._jgoal = (np.asarray(q, float).copy(), float(vmax), float(gain))
        self.set_velocity(np.zeros(3))

    def joint_goal_active(self) -> bool:
        return getattr(self, "_jgoal", None) is not None

    def joint_goal_error(self) -> tuple:
        """(指令の関節角の残りの最大 [rad], 指令の速さの最大 [rad/s])。"""
        g = self._jgoal[0]
        return float(np.max(np.abs(g - self.limiter.q))), float(np.max(np.abs(self.limiter.v)))

    def clear_joint_goal(self) -> None:
        """関節の目標を外し、直交座標の追従器・IK・x_cmd を今の指令の姿勢に揃え直す（Motion.reset と同じ手順。制限層の速さ・加速度は
        引き継ぐ＝指令が跳ばない）。"""
        self._jgoal = None
        c, integ = self.controller, self.integrator
        q = self.limiter.q.copy()
        self._load_state(q)
        c.q_des = q.copy()
        c.gripper_closed = False
        c.sync_target_to_hand()
        integ.reset(c.target_pos)
        c.device_ref_pos = integ.x_cmd.copy()
        c.target_ref_pos = c.target_pos.copy()
        c.device_ref_quat = control._IDENTITY_QUAT.copy()
        c.target_ref_quat = c.target_quat.copy()
        c.prev_clutch = True
        self.vel = np.zeros(3)
        self.pad.state = pad_state(vel=self.vel)
        integ.refresh()

    def step(self, joints) -> list:
        jg = getattr(self, "_jgoal", None)
        if jg is None:
            return Motion.step(self, joints)
        goal, vmax, gain = jg
        v_des = gain * (goal - self.limiter.q)
        s = float(np.max(np.abs(v_des))) / vmax
        if s > 1.0:
            v_des = v_des / s                                       # 関節空間の直線（向きを保って縮める）
        out = []
        for _ in range(SUBSTEPS):                                   # Motion.step と同じ: 500 Hz の 1 ステップで 1 kHz の 2 刻み
            v = self.limiter.propose(v_des)
            if self.limiter.enabled:
                v = self._cartesian_limit(v)
            q = self.limiter.commit(v)
            self._x_hist.append(self._fk_hand(q))
            del self._x_hist[:-4]
            out.append(q)
        self.integrator.x_cmd = self._fk_hand(out[-1])              # 記録（x_des）と切り替えの後の出発点を指令の手先に合わせる
        self.controller.q_des = out[-1].copy()
        return out


class DiagTaskRuntime(TaskRuntime):
    """TaskRuntime の子（__class__ を差し替えてから diag_setup を呼ぶ）。手順の切り替えに戻す動き（EH・ES）か作り直し（EX）を挟む。"""

    def diag_setup(self, arm: str, goal_q=None, params: dict = None, rebuild: bool = False, new_perception=None) -> None:
        self.diag_arm = arm
        self.goal_q = None if goal_q is None else np.asarray(goal_q, float)
        self.gp = dict(GOAL_PARAMS, **(params or {}))
        self.rebuild = bool(rebuild)
        self.new_perception = new_perception                         # EX: 新しい Perception を作る関数（実行系の側の物だけ）
        self.goal_state = None

    def _diag_on(self) -> bool:
        return self.goal_q is not None or self.rebuild

    def _step_done(self, t: float, ok: bool) -> None:
        if not (ok and self._diag_on() and self.j + 1 < len(self.plan["steps"])):
            TaskRuntime._step_done(self, t, ok)
            return
        rec = self.steps[-1]                                         # TaskRuntime._step_done と同じ記録
        rec["attempts"][-1]["t_judge"] = self.done_t
        rec.update(judged_complete=True, t_judge=self.done_t, t_end=t)
        if self.goal_q is not None:
            self._begin_goal(t, self.j + 1)
        else:
            self._begin_rebuild(t, self.j + 1)

    def _control(self, t: float) -> None:
        if self.phase == "goal":
            self._goal_tick(t)
        elif self.phase == "rebuild":
            self._rebuild_tick(t)
        else:
            TaskRuntime._control(self, t)

    # ---------------------------------------------------------- EH・ES
    def _begin_goal(self, t: float, nxt: int) -> None:
        prt = self.prt
        prt.external = True
        prt.motion.set_velocity(np.zeros(3))
        opened = False
        if prt.closed:
            self.io.gripper_move(0.08, self.setup.gripper_speed)
            prt.closed = False
            opened = True
        self.goal_state = {"next": nxt, "t_begin": t, "stage": "open", "opened": opened, "rest": 0.0, "last_w": None,
                           "t_last": t, "rose": False, "t_joint": None, "t_arrive": None, "arrived": False}
        self.phase = "goal"

    def _goal_tick(self, t: float) -> None:
        prt, gs, gp = self.prt, self.goal_state, self.gp
        mo = prt.motion
        sensor = self.io.sense(cameras=False)
        if gs["stage"] == "open":                                   # 開き幅が止まるのを待つ（_return_tick と同じ。上限 1 s）
            w = sensor.gripper.width
            dt = t - gs["t_last"]
            if gs["last_w"] is not None and dt > 0:
                gs["rest"] = gs["rest"] + dt if abs(w - gs["last_w"]) / dt < 0.005 and w >= self.open_m else 0.0
            gs["last_w"], gs["t_last"] = w, t
            if not gs["opened"] or gs["rest"] >= 0.3 - 1e-9 or t - gs["t_begin"] >= gp["open_wait_max_s"] - 1e-9:
                if mo.x_cmd[2] < self.rm["rise_z"] - self.rm["z_tol"]:
                    gs["stage"], gs["rose"] = "rise", True
                else:
                    gs["stage"] = "joint"
                    gs["t_joint"] = t
                    mo.set_joint_goal(self.goal_q, gp["vmax_rad_s"], gp["gain_per_s"])
            return
        if gs["stage"] == "rise":                                   # 真上に rise_z まで（やり直しの前の戻す動きと同じ速さの決まり）
            z = mo.x_cmd[2]
            if abs(z - self.rm["rise_z"]) <= self.rm["z_tol"] or t - gs["t_begin"] >= gp["timeout_s"] - 1e-9:
                mo.set_velocity(np.zeros(3))
                gs["stage"] = "joint"
                gs["t_joint"] = t
                mo.set_joint_goal(self.goal_q, gp["vmax_rad_s"], gp["gain_per_s"])
            else:
                vz = float(np.clip(self.rm["gain"] * (self.rm["rise_z"] - z), -self.rm["z_max"], self.rm["z_max"]))
                mo.set_velocity(np.array([0.0, 0.0, vz]))
            return
        if gs["stage"] == "joint":
            e, v = mo.joint_goal_error()
            timeout = t - gs["t_begin"] >= gp["timeout_s"] - 1e-9
            if e <= gp["tol_cmd_rad"] and v <= gp["tol_vel_rad_s"]:
                if gs.setdefault("n_corr", 0) < int(gp["meas_corrections"]) and not timeout:
                    gs["stage"], gs["t_settle"] = "settle_meas", t   # 関節の目標は保ったまま（その場で保持）
                    return
                gs["arrived"], gs["t_arrive"] = True, t
            elif not timeout:
                return
            gs["err_cmd_rad"] = e
            gs["stage"], gs["t_hold"] = "hold", t                    # 関節の目標は保ったまま保持する（直交座標の IK に戻すと、
            return                                                  # IK の定常のずれで手先が数 mm 動く。試しの実行で +7 mm）
        if gs["stage"] == "settle_meas":
            if t - gs["t_settle"] < gp["meas_settle_s"] - 1e-9:
                return
            q = np.asarray(sensor.joints.q, float)
            cmd = gs.get("goal_cmd", self.goal_q) + (self.goal_q - q)
            gs.setdefault("corr", []).append({"t": t, "err_meas_rad": float(np.max(np.abs(self.goal_q - q)))})
            gs["goal_cmd"], gs["n_corr"] = cmd, gs["n_corr"] + 1
            mo.set_joint_goal(cmd, gp["vmax_rad_s"], gp["gain_per_s"])
            gs["stage"] = "joint"
            return
        if gs["stage"] == "hold" and t - gs["t_hold"] >= gp["hold_s"] - 1e-9:
            q = np.asarray(sensor.joints.q, float)
            self.returns.append({"step": gs["next"], "attempt": 0, "kind": "goal", "goal": self.diag_arm,
                                 "t_begin": gs["t_begin"], "t_joint": gs["t_joint"], "t_arrive": gs["t_arrive"], "t_end": t,
                                 "arrived": gs["arrived"], "opened": gs["opened"], "rose": gs["rose"], "judged": False,
                                 "err_cmd_rad": gs.get("err_cmd_rad"), "corrections": gs.get("corr", []),
                                 "err_meas_rad": float(np.max(np.abs(self.goal_q - q))),
                                 "hand_meas": self._hand(q).tolist(), "q_meas": q.tolist()})
            self.goal_state = None
            mo.clear_joint_goal()                                    # 手順を始める時に直交座標の動きへ戻す（E0 の手順の始めと同じ）
            self._begin_step(t, gs["next"], 0)

    # ---------------------------------------------------------- EX（実行系の作り直し）
    def _begin_rebuild(self, t: float, nxt: int) -> None:
        """まず方策を止めて腕を止める（制限層の速さ・加速度が 0 に近づくまで。PolicyRuntime.start は制限層の速さ・加速度を 0 に
        戻すので、動いている間に作り直すと躍度が上限を超える。試しの実行で上限の 1.195 倍）。止まったら _rebuild_now。"""
        prt = self.prt
        prt.external = True
        prt.motion.set_velocity(np.zeros(3))
        if prt.closed:                                               # start は閉の印を消すだけなので、先に指を開く
            self.io.gripper_move(0.08, self.setup.gripper_speed)
            prt.closed = False
        self.goal_state = {"next": nxt, "t_begin": t, "stage": "brake", "t_rebuild": None}
        self.phase = "rebuild"

    def _rebuild_now(self, t: float, nxt: int) -> None:
        """実行系を作り直す: 新しい Perception（判定も同じものを見る）、PolicyRuntime.start（起動時の確かめ・推定・行動列・方策の試行の
        状態・動きの出発点をやり直す）。記録（推論・行動・知覚の列）と行動の区切りの数え方は続ける。"""
        prt = self.prt
        keep = (prt.log_inf, prt.log_act, prt.log_per, prt.k, prt.n_tick)
        lim = prt.motion.limiter
        lim_va = (lim.v.copy(), lim.a.copy())                      # 制限層の速さ・加速度は引き継ぐ（指令を跳ばさない）
        if self.new_perception is not None and prt.perception is not None:
            per = self.new_perception(prt.perception)
            prt.perception = per
            self.judge.per = per
        prt.start(self.instruction(self.plan["steps"][nxt]), step_seed(self.seed, nxt, 0))
        prt.motion.limiter.v, prt.motion.limiter.a = lim_va
        prt.log_inf, prt.log_act, prt.log_per, prt.k, prt.n_tick = keep
        prt.external = True
        self.goal_state.update(stage="startup", t_rebuild=t)

    def _rebuild_tick(self, t: float) -> None:
        prt, gs = self.prt, self.goal_state
        if gs["stage"] == "brake":
            lim = prt.motion.limiter
            still = float(np.max(np.abs(lim.v))) <= 1e-3 and float(np.max(np.abs(lim.a))) <= 0.05
            if (still and t - gs["t_begin"] >= 0.3 - 1e-9) or t - gs["t_begin"] >= 2.0 - 1e-9:
                gs["still"] = still
                self._rebuild_now(t, gs["next"])
            return
        if prt.stop_reason:
            self._stop(t, f"作り直した実行系の知覚の失敗（{prt.stop_reason}）で {gs['next'] + 1} 番目の手順を始められませんでした")
            return
        if not (prt.ready and prt.wm is not None):
            if t - gs["t_begin"] >= self.gp["timeout_s"] - 1e-9:
                self._stop(t, "作り直した実行系の起動時の確かめが終わりませんでした")
            return
        self.returns.append({"step": gs["next"], "attempt": 0, "kind": "rebuild", "t_begin": gs["t_begin"], "t_arrive": t,
                             "t_rebuild": gs["t_rebuild"], "still_at_rebuild": gs.get("still"),
                             "t_end": t, "arrived": True, "opened": False, "judged": False,
                             "startup": getattr(prt, "startup", None)})
        self.goal_state = None
        self._begin_step(t, gs["next"], 0)


def install(ex, arm: str, perception_cls=None):
    """作った TaskRuntime を、条件に合わせて差し替える（E0・EO は何もしない）。返り値は ex。"""
    a = ARMS[arm]
    if a.get("goal_q") is None and not a.get("rebuild"):
        return ex
    ex.__class__ = DiagTaskRuntime
    newp = None
    if a.get("rebuild") and perception_cls is not None:
        def newp(old):
            return perception_cls(old.setup, old.p, old.thr)
    ex.diag_setup(arm, goal_q=a.get("goal_q"), rebuild=bool(a.get("rebuild")), new_perception=newp)
    if a.get("goal_q") is not None:
        ex.prt.motion.__class__ = GoalMotion
    return ex


# ---------------------------------------------------------------- 集計（記録だけを読む。判定はしない）
LIFT_M = 0.02          # 最初の閉じから次に開くまでに目標が CUBE_REST_Z より 2 cm 上がった
PLUS_Y_CM = 1.0        # 最初の閉じの時点の（指先 − 目標）の y が +1 cm を超えた


def first_close(z, ci: int, f_from: int, f_to: int) -> dict:
    """フレーム [f_from, f_to) の中の最初の閉じ（gripper_closed の 0→1）。無ければ None。
    lift: 閉じたフレームから次に開く（gripper_closed=0）フレームの手前までに、目標の z − CUBE_REST_Z > 0.02 のフレームがある。
    dx_cm・dy_cm: 閉じたフレームでの（指先 − 目標）の xy [cm]。"""
    gc = np.asarray(z["gripper_closed"]).astype(int)
    rises = np.nonzero(np.diff(gc) == 1)[0] + 1
    fs = [int(f) for f in rises if f_from <= f < f_to]
    if not fs:
        return None
    f = fs[0]
    nxt = np.nonzero((np.arange(len(gc)) > f) & (gc == 0))[0]
    f2 = int(nxt[0]) if len(nxt) else len(gc)
    cube = np.asarray(z["cube_pos"])
    d = (np.asarray(z["fingertip"])[f, :2] - cube[f, ci, :2]) * 100.0
    return {"frame": f, "t": float(z["sim_time"][f]), "lift": bool(np.any(cube[f:f2, ci, 2] - frames.CUBE_REST_Z > LIFT_M)),
            "dx_cm": float(d[0]), "dy_cm": float(d[1]), "plus_y": bool(d[1] > PLUS_Y_CM),
            "cube_xy": cube[f, ci, :2].tolist()}


def attempt_first_closes(meta: dict, z) -> list:
    """手順・試みごとの行。窓は [その試みの t_begin, 次の試みの t_begin か手順の t_end か記録の終わり)。"""
    t = np.asarray(z["sim_time"])
    rows = []
    for s in meta.get("steps") or []:
        ci = COLORS.index(s["color"])
        atts = s["attempts"]
        for k, a in enumerate(atts):
            t0 = float(a["t_begin"])
            t1 = atts[k + 1]["t_begin"] if k + 1 < len(atts) else (s.get("t_end") if s.get("t_end") is not None else float(t[-1]) + 1.0)
            f0, f1 = int(np.searchsorted(t, t0)), int(np.searchsorted(t, float(t1)))
            fc = first_close(z, ci, f0, f1)
            rows.append({"step": s["step"], "color": s["color"], "attempt": a["attempt"], "t_begin": t0,
                         "first_close": fc, "fc_t_rel": None if fc is None else round(fc["t"] - t0, 3),
                         "judged": a.get("t_judge") is not None})
    return rows


def _frac(k: int, n: int) -> dict:
    return {"k": int(k), "n": int(n), "rate": None if n == 0 else round(k / n, 4)}


def _step_metrics(trials: list, step: int, attempt: int = 0) -> dict:
    """手順 step・試み attempt の、1 回目の閉じの持ち上がり（分母は手順が始まった試行）と
    +y のずれ（分母は最初の閉じが起きた試行）。参考に持ち上がりの分母を閉じた試行にした値も出す。"""
    rows = [r for tr in trials for r in tr["attempts"] if r["step"] == step and r["attempt"] == attempt]
    fc = [r["first_close"] for r in rows if r["first_close"] is not None]
    dy = [f["dy_cm"] for f in fc]
    return {"n_started": len(rows), "n_close": len(fc),
            "first_close_lift": _frac(sum(f["lift"] for f in fc), len(rows)),
            "first_close_lift_of_closed": _frac(sum(f["lift"] for f in fc), len(fc)),
            "plus_y_shift": _frac(sum(f["plus_y"] for f in fc), len(fc)),
            "dy_cm_median": None if not dy else round(float(np.median(dy)), 3),
            "colors": sorted({r["color"] for r in rows})}


def summarize_condition(d) -> dict:
    """条件のフォルダ 1 つ（task の記録）を集計する。読むだけ。"""
    d = pathlib.Path(d)
    trials = []
    for p in sorted(d.glob("run_[0-9][0-9][0-9][0-9].json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        z = np.load(p.with_suffix(".npz"))
        steps = m.get("steps") or []
        goal = [r for r in (m.get("returns") or []) if r.get("kind") in ("goal", "rebuild")]
        trials.append({"run": m.get("run"), "seed": m.get("seed"), "plan": (m.get("plan") or {}).get("steps"),
                       "plan_model": (m.get("plan") or {}).get("model"), "from_cache": (m.get("plan") or {}).get("from_cache"),
                       "all_three": bool(m.get("all_three_in_box")), "timed_out": bool(m.get("timed_out")),
                       "t_end": m.get("t_end"), "stopped": m.get("stopped") is not None,
                       "judge_all": bool(steps) and len(steps) == len((m.get("plan") or {}).get("steps") or []) and
                       all(s.get("judged_complete") for s in steps) and m.get("stopped") is None,
                       "n_steps_started": len(steps), "attempts": attempt_first_closes(m, z), "goal_returns": goal,
                       "diag": m.get("diag"), "api_retry": (m.get("diag") or {}).get("api_retry")})
    n = len(trials)
    gr = [g for t in trials for g in t["goal_returns"]]
    out = {"dir": str(d), "n": n,
           "step2": _step_metrics(trials, 1), "step1": _step_metrics(trials, 0), "step3": _step_metrics(trials, 2),
           "later_att0": {"n_close": 0},
           "all_three_true": _frac(sum(t["all_three"] for t in trials), n),
           "success_by_judge": _frac(sum(t["judge_all"] for t in trials), n),
           "timed_out": sum(t["timed_out"] for t in trials),
           "t_end_max": None if not trials else max(float(t["t_end"]) for t in trials),
           "plans": sorted({json.dumps(t["plan"]) for t in trials}),
           "plan_models": sorted({str(t["plan_model"]) for t in trials}),
           "api_retries": sum(1 for t in trials if t["api_retry"]),
           "goal_returns": {"n": len(gr), "arrived": sum(bool(g.get("arrived")) for g in gr),
                            "err_meas_rad_max": None if not gr or gr[0].get("err_meas_rad") is None
                            else round(max(float(g["err_meas_rad"]) for g in gr), 5)},
           "per_seed": {str(t["seed"]): {"all_three": t["all_three"],
                                          "step2_lift": next((r["first_close"]["lift"] for r in t["attempts"]
                                                              if r["step"] == 1 and r["attempt"] == 0 and r["first_close"]), None)
                                          if any(r["step"] == 1 and r["attempt"] == 0 for r in t["attempts"]) else None}
                        for t in trials}}
    later = [r for t in trials for r in t["attempts"] if r["step"] >= 1 and r["attempt"] == 0]
    fc = [r["first_close"] for r in later if r["first_close"] is not None]
    out["later_att0"] = {"n_started": len(later), "n_close": len(fc), "first_close_lift_of_closed": _frac(sum(f["lift"] for f in fc), len(fc)),
                         "plus_y_shift": _frac(sum(f["plus_y"] for f in fc), len(fc))}
    out["trials"] = trials
    return out


def _paired(a: dict, b: dict, key: str) -> list:
    return [(a["per_seed"][s][key], b["per_seed"][s][key]) for s in a["per_seed"] if s in b["per_seed"]]


def gate_T_inputs(summ: dict) -> dict:
    """事前に決めた判定に使う値を並べる（判定はしない）。summ = {条件名: summarize_condition}。
    条件名は E0_run1・E0_run2・EH・ES・EO（無いものは飛ばす）。"""
    out = {"source": {k: v["dir"] for k, v in summ.items()}, "arms": {}}
    for k, v in summ.items():
        out["arms"][k] = {"first_close_lift": v["step2"]["first_close_lift"], "plus_y_shift": v["step2"]["plus_y_shift"],
                          "all_three_true": v["all_three_true"], "timed_out": v["timed_out"],
                          "step1_first_close_lift": v["step1"]["first_close_lift"], "step1_colors": v["step1"]["colors"]}
    e1, e2 = summ.get("E0_run1"), summ.get("E0_run2")
    if e1 and e2:
        l1, l2 = e1["step2"]["first_close_lift"]["rate"], e2["step2"]["first_close_lift"]["rate"]
        d = None if l1 is None or l2 is None else abs(l1 - l2)
        out["E0_ref_lift"] = None if d is None else max(l1, l2)
        out["d_E0"] = None if d is None else round(d, 4)
        out["required_margin"] = None if d is None else round(max(0.25, d + 0.10), 4)
        out["E0_all_three_max"] = max(e1["all_three_true"]["k"], e2["all_three_true"]["k"])
        pairs = _paired(e1, e2, "all_three")
        n = len(pairs)
        disc = sum(a != b for a, b in pairs)
        x = np.array(pairs, float) if pairs else np.zeros((0, 2))
        phi = None
        if n and x[:, 0].std() > 0 and x[:, 1].std() > 0:
            phi = round(float(np.corrcoef(x[:, 0], x[:, 1])[0, 1]), 4)
        out["K_e7"] = {"all_three_discordance": _frac(disc, n), "rho_hat_phi": phi, "d_E0": out["d_E0"],
                       "note": "同じ種の対で数える。rho_hat は成功（3 個とも・真値）の二値の phi 係数"}
    if "EO" in summ:
        s1 = summ["EO"]["step1"]
        out["EO_green_first"] = {"first_close_lift": s1["first_close_lift"], "colors": s1["colors"],
                                 "plans": summ["EO"]["plans"],
                                 "note": "T.b1 の材料: EO で 1 番目の手順（緑のはず。colors で確かめる）の 1 回目の閉じの持ち上がり"}
    out["note"] = "判定はしない（s4_gates.json の T.continue・T.b1・T.b2 の当てはめは二重集計役が行う）"
    return out
