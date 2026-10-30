"""技能の口。VLA と手で書いたスクリプト方策を、実行器を書き換えずに差し替えられるようにする。

実行器（runtime/executor.TaskRuntime など）が使う口は PolicyRuntime と同じ: start(task, seed)・set_task(task, seed)・tick・
external・motion・wm。ほかに closed・ready・stop_reason・reset_chunks・trace・tip_offset と、getattr で読む startup。

  VLASkill     PolicyRuntime そのもの（名前を付けるだけ）
  ScriptSkill  手で書いたスクリプト方策（段階の決まりは api/skills_script）。知覚の推定・測った関節角の順運動学・ハンドの幅・
               自分の指令だけで動き、毎回段階を決め直すので、落とせばひとりでに掴み直しに行く。PolicyRuntime を継いで知覚・腕の動き
               （runtime/motion と制限層 runtime/limiter）・ハンドの口をそのまま使い、方策の推論の代わりに ctrl_every 手ごと
               （既定 10 ステップ = 20 ms）に指令を出す。知覚は引数 finder で選ぶ: "estimated"（既定。api/estimated の推定、
               wm.geometry を読む）か "color"（比べ用。既知の値の ColorWorldModelFinder）。
ScriptSkill はシミュレーションの真値を読まない（recovla.expert・sim・harness・eval を import しない。テストで検査）。
"""
import dataclasses

import numpy as np

from recovla.api import perception as AP
from recovla.api import relations as REL
from recovla.api import senses as AS
from recovla.api import task_spec as TS
from recovla.api.skills_script import Obs, Phase, ScriptBrain
from recovla.runtime.runner import ACTION_DT, STEPS_PER_ACTION, PolicyRuntime

EXECUTOR_IO = ("start", "set_task", "tick", "external", "motion", "wm")
EXECUTOR_ALSO = ("closed", "ready", "stop_reason", "reset_chunks", "trace", "tip_offset")   # startup は getattr で読む（任意）


@dataclasses.dataclass(frozen=True)
class SkillInfo:
    name: str
    needs: tuple
    note: str = ""


VLASkill = PolicyRuntime
VLA_INFO = SkillInfo("vla", ("joints", "gripper", "camera/overhead", "camera/wrist"),
                     "今の PolicyRuntime（方策の推論は io.compute、知覚・安全フィルタ・反射は 96 の make のまま）")
SCRIPT_INFO = SkillInfo("script", ("joints", "gripper", "camera/overhead", "camera/wrist"),
                        "古典の台本（知覚の推定・順運動学・幅だけ）。段階の決まりは api/skills_script")


class _NoPolicy:
    """方策なし（PolicyRuntime の start・set_task・reset_chunks が呼ぶ口だけ）。"""
    cue = None

    def start_trial(self, seed) -> None:
        pass

    def reset_cue(self) -> None:
        pass


class ScriptSkill(PolicyRuntime):
    info = SCRIPT_INFO
    needs = SCRIPT_INFO.needs

    def __init__(self, io, setup, rules, spec, instruction_fmt: str, perception=None, checks: dict = None,
                 tip_offset: float = 0.1034, motion=None, ctrl_every: int = 10, grasp_offset=(0.0, 0.0), slot_order=None,
                 finder: str = "estimated"):
        super().__init__(io, setup, _NoPolicy(), mode="naive", s=10, motion=motion, perception=perception, safety=None,
                         checks=checks, tip_offset=tip_offset, gripper_gate=None, reflex=None)
        self.rules, self.spec, self.fmt = rules, spec, instruction_fmt
        self.ctrl_every = int(ctrl_every)
        if finder not in AP.FINDER_KINDS:
            raise ValueError(f"知覚の口 {finder!r} は {AP.FINDER_KINDS} のどれか")
        if finder == "estimated" and perception is not None and not hasattr(perception, "geom"):
            raise ValueError("finder='estimated' は GeometryPerception（api/estimated）の知覚が要る")
        self.finder_kind = finder
        self.finder = AP.ColorWorldModelFinder.from_spec(spec, setup=setup) if finder == "color" else None
        cams = getattr(setup, "cameras", None) or {}
        fixed = [c for c in cams.values() if getattr(c, "mount", None) == "world"]      # 固定のカメラ（較正の値）
        view = np.asarray(fixed[0].extrinsic.t, float)[:2] if fixed else None
        self.brain = ScriptBrain(rules, dt=self.ctrl_every * float(self.motion.dt), grasp_offset=grasp_offset,
                                 slot_order=slot_order, view_from=view)
        self.step = None
        self.last_phase, self.k_logged = None, None
        self.log_script, self.grasp_log = [], []
        self._obs = None
        self.grasp_eps_frac = 0.125                    # 掴んだ判定で許す幅の割合（推定した物の幅に掛ける）

    # -------------------------------------------------------------------- trial
    def _begin(self, task: str) -> None:
        self.step = TS.step_of_instruction(self.spec, task, self.fmt)
        self.brain.reset()
        self.last_phase = None
        self.k_logged = None
        self.log_script.append({"t": round(float(self.io.now()), 4), "kind": "task", "obj": self.step.obj,
                                "place": self.step.place})

    def start(self, task: str, seed: int) -> None:
        super().start(task, seed)                       # 作り直しでも呼ばれる（記録は続ける）
        self._begin(task)

    def set_task(self, task: str, seed: int) -> None:
        super().set_task(task, seed)
        self._begin(task)

    def _check_target(self, t: float) -> None:
        """狙いの物を見失っても止めない（search の段階で探す。PolicyRuntime は target_not_found で止まる）。"""
        if self.t_task0 is None:
            self.t_task0 = t

    # --------------------------------------------------------------------- tick
    def _action_boundary(self) -> None:
        """行動の区切り（0.1 s）: 知覚だけ PolicyRuntime と同じに回す。方策の推論はしない。"""
        k, t = self.k, self.io.now()
        if self.perception is not None:
            self._perception(t)
            if not self.ready or self.stop_reason:
                self.motion.set_velocity(np.zeros(3))
                self.log_act.append((k, t, None, True, np.array([0, 0, 0, 0, 0, 0, 1.0 if self.closed else -1.0])))

    def tick(self) -> None:
        if self.n_tick % STEPS_PER_ACTION == 0:
            self.k += 1
            self._action_boundary()
        sensor = self.io.sense(cameras=False)
        if self.perception is not None:
            self.perception.record_joints(sensor.joints)
        if (self.n_tick % self.ctrl_every == 0 and self.ready and not self.stop_reason and not self.external
                and self.step is not None):
            self._script_step(sensor)
        self.n_tick += 1
        for q in self.motion.step(sensor.joints):
            self.io.command_joints(q)

    def estimate(self):
        """知覚の推定（WorldEstimate か None）。estimated は知覚の更新で一緒に作った wm.geometry、color は wm を写したもの。"""
        if self.finder_kind == "estimated":
            return getattr(self.wm, "geometry", None)
        return self.finder.estimate(self.wm)

    def observation(self, sensor) -> Obs:
        """スクリプト方策の 1 回の入力（知覚の推定・順運動学・幅・自分の指令）。"""
        hand, quat = self.motion.hand_pose(sensor.joints.q)
        tip = AS.fingertip(hand, quat, self.tip_offset)
        est = self.estimate()
        objs = {} if est is None else est.objects
        st = self.step
        su = self.setup
        return Obs(t=float(self.io.now()), target=objs.get(st.obj), others={n: o for n, o in objs.items() if n != st.obj},
                   place=None if est is None else est.places.get(st.place), place_name=st.place, closed=bool(self.closed),
                   width=float(sensor.gripper.width), width_t=float(sensor.gripper.t), hand=np.asarray(hand, float),
                   tip=tip, x_cmd=self.motion.x_cmd, table_z=float(su.table_z),
                   est_t=None if est is None else est.t)

    def _apply(self, a) -> None:
        """PolicyRuntime._apply と同じ口（腕の参照の速さ・ハンドの grasp / move）。grasp に渡す幅は、既知の cube_size の代わりに
        狙いの物の推定の短い幅、許す幅は grasp_eps_frac × 幅 ＋ unc_k × σ（推定が無ければ PolicyRuntime と同じ）。"""
        self.motion.set_velocity(a[:3] / ACTION_DT)
        want = bool(a[6] > 0.0)
        if want == self.closed:
            return
        o = None if self._obs is None else self._obs.target
        w = None if o is None else REL.width_min(o)
        if not want or w is None:
            super()._apply(a)
            return
        self.closed = want
        self.t_grip_switch = self.io.now()
        su = self.setup
        eps = self.grasp_eps_frac * w + self.rules.unc_k * REL.sigma(o)
        self.grasp_log.append({"t": round(float(self.io.now()), 4), "width": round(w, 5), "eps": round(eps, 5)})
        self.io.gripper_grasp(w, su.gripper_speed, su.grasp_force, eps, eps)

    def _script_step(self, sensor) -> None:
        obs = self.observation(sensor)
        self._obs = obs
        cmd = self.brain.step(obs)
        a = np.zeros(7)
        a[:3] = cmd.vel * ACTION_DT
        a[6] = 1.0 if cmd.closed else -1.0
        self._apply(a)
        if cmd.phase != self.last_phase:
            self.log_script.append({"t": round(obs.t, 4), "k": int(self.k), "phase": Phase(cmd.phase).name,
                                    "obj": self.step.obj, "target_status": None if obs.target is None else obs.target.status,
                                    "width": round(obs.width, 5), "closed": bool(cmd.closed)})
            self.last_phase = cmd.phase
        if self.k != self.k_logged:                       # 行動の記録は区切りごとに 1 行（方策と同じ 10 Hz の行の数）
            self.log_act.append((self.k, obs.t, None, False, a.copy()))
            self.k_logged = self.k

    def trace(self) -> dict:
        out = super().trace()
        out["script"] = {"skill": self.info.name, "finder": self.finder_kind, "rules": self.rules.to_dict(),
                         "ctrl_every": self.ctrl_every, "grasp": list(self.grasp_log),
                         "log": list(self.log_script)}
        return out
