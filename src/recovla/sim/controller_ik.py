"""微分逆運動学（減衰つき最小二乗、DLS）の手先位置の制御器。"""
import numpy as np
import mujoco

from recovla.sim.device import DeviceInput, DeviceState


class TeleopControllerIK:
    """
    微分逆運動学（DLS）の制御器。入力（クラッチ）→ 追従器 → IK → 関節の位置のサーボ。

    - 追従器: クラッチが書く desired_pos は参照で、実際の目標 target_pos は臨界減衰の 2 次系
          a = w^2 (desired - target) - 2 w v
      で追う（制御の周期で半陰的 Euler）。速さは max_target_vel [m/s] で切る。段状の目標が硬い関節のサーボ
      （kp 2000〜4500）を振動させるのを防ぐ。tracker_omega=0 なら 1 ステップごとの移動量の制限（max_target_step）
    - 目標は作業空間の中に切る（切ったときに目標が跳ばないよう、追従器と移動量の制限を通す）
    - q_des の巻き上がり防止: q_des は測った関節角から q_des_leash [rad] より先へ行かない
    - 零空間の姿勢の項（q_neutral へ。null_gain=0 なら切）

    直立の特異姿勢（q がすべて 0）から始めないこと。作る前に開始姿勢（control.HOME_QPOS）を書く。
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        scale: float = 2.0,
        workspace_x=(0.25, 0.70),
        workspace_y=(-0.35, 0.35),
        workspace_z=(-0.06, 0.60),
        ik_damping: float = 0.05,
        ik_gain: float = 0.5,           # 1 ステップあたりのタスクの利得
        max_joint_step: float = 0.01,   # 制御の 1 ステップあたり [rad]
        max_target_step: float = 0.004, # 制御の 1 ステップあたり [m]（tracker_omega=0 のとき）
        tracker_omega: float = 20.0,    # 目標の追従器のシード範囲域 [rad/s]
        max_target_vel: float = 0.8,    # 目標の速さの上限 [m/s]
        ctrl_dt: float = 0.0,           # 制御の周期 [s]。0 ならモデルの timestep
        q_des_leash: float = 0.10,      # q_des が qpos より先へ行ける量の上限 [rad]
        null_gain: float = 0.0,         # 零空間の姿勢の利得（0 で切）
        max_dx_norm: float = 0.10,      # タスクの誤差の飽和
        pos_weight: float = 1.0,
        rot_weight: float = 0.0,        # 向きの追従（既定は切）
        lock_orientation: bool = False, # True: クラッチは目標を回さない
                                        # （rot_weight>0 なら開始時の手の向きを保つ）
        end_effector_body: str = "hand",
    ):
        self.model = model
        self.data = data
        self.scale = scale

        self.workspace_x = workspace_x
        self.workspace_y = workspace_y
        self.workspace_z = workspace_z

        self.ik_damping = ik_damping
        self.ik_gain = ik_gain
        self.max_joint_step = max_joint_step
        self.max_target_step = max_target_step
        self.tracker_omega = tracker_omega
        self.max_target_vel = max_target_vel
        self.ctrl_dt = ctrl_dt if ctrl_dt > 0.0 else float(model.opt.timestep)
        self.q_des_leash = q_des_leash
        self.null_gain = null_gain
        self.max_dx_norm = max_dx_norm
        self.pos_weight = pos_weight
        self.rot_weight = rot_weight
        self.lock_orientation = lock_orientation

        if model.nmocap < 1:
            raise RuntimeError("Scene must contain at least one mocap body (visual target).")
        self.mocap_id = 0

        self.hand_body_id = model.body(end_effector_body).id

        # ---- グリッパの作動器（名前で探す） ----
        self.gripper_indices = []
        for i in range(model.nu):
            name = model.actuator(i).name
            if name is not None:
                lname = name.lower()
                if "gripper" in lname or "finger" in lname:
                    self.gripper_indices.append(i)

        # ---- 腕の関節の作動器: (act_idx, joint_id, qpos_adr, dof_adr) ----
        self.arm = []
        for i in range(model.nu):
            if i in self.gripper_indices:
                continue
            if model.actuator_trntype[i] != 0:  # mjTRN_JOINT
                continue
            joint_id = model.actuator_trnid[i][0]
            self.arm.append((i, joint_id,
                             model.jnt_qposadr[joint_id],
                             model.jnt_dofadr[joint_id]))
        if len(self.arm) == 0:
            raise RuntimeError("No arm joint actuators found.")

        self.dof_indices = np.array([a[3] for a in self.arm], dtype=int)
        self.n_arm = len(self.arm)
        self.q_min = np.array([model.jnt_range[a[1]][0] for a in self.arm])
        self.q_max = np.array([model.jnt_range[a[1]][1] for a in self.arm])

        mujoco.mj_forward(model, data)

        self.q_des = np.array([data.qpos[a[2]] for a in self.arm])
        self.q_neutral = self.q_des.copy()

        # 目標は今の手の姿勢から始め、作業空間の中に切る
        self.target_pos = self._clamp_workspace(
            data.xpos[self.hand_body_id].copy())
        self.target_quat = data.xquat[self.hand_body_id].copy()

        # 追従器の状態: クラッチが desired_pos を書き、追従器が毎手 target_pos をそこへ寄せる
        # （クラッチを離した後も動かし、滑らかに止める）
        self.desired_pos = self.target_pos.copy()
        self.target_vel = np.zeros(3)

        # クラッチの基準
        self.device_ref_pos = None
        self.target_ref_pos = None
        self.device_ref_quat = None
        self.target_ref_quat = None

        self.gripper_closed = True
        self.prev_grip = False
        self.prev_clutch = False

        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))
        self._log_counter = 0

        # 直前の IK のタスクのヤコビアン（6×N）。_solve_ik_step が書く（可操作性の診断用）
        self.last_jac = None

    # ------------------------------------------------------------------ 補助

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def sync_target_to_hand(self) -> None:
        """目標（mocap）を今の手の姿勢に置き直し、追従器とクラッチの状態を消す。

        data.qpos を外から書き換えて mj_forward した直後に呼ぶ。目標が古いままだと、次の手で目標と手の差が大きくなり
        IK が跳ねる。target_vel とクラッチの基準を 0 にして、静止した状態から始め直す。
        """
        self.target_pos = self._clamp_workspace(
            self.data.xpos[self.hand_body_id].copy())
        self.target_quat = self.data.xquat[self.hand_body_id].copy()
        self.desired_pos = self.target_pos.copy()
        self.target_vel = np.zeros(3)
        self.device_ref_pos = None
        self.target_ref_pos = None
        self.device_ref_quat = None
        self.target_ref_quat = None
        # 次にクラッチをつないだとき、新しい基準を取らせる
        self.prev_clutch = False
        self.data.mocap_pos[self.mocap_id] = self.target_pos
        self.data.mocap_quat[self.mocap_id] = self.target_quat

    @property
    def last_sigma_min(self) -> float:
        """直前のタスクのヤコビアンの最小特異値（最初の IK の前は 0.0）。読むたびに last_jac から計算する。"""
        if self.last_jac is None:
            return 0.0
        return float(np.min(np.linalg.svd(self.last_jac, compute_uv=False)))

    def set_target_quat(self, quat) -> None:
        """向きの目標を直接書く（正規化だけ。制御則・クラッチの基準・位置の目標には触れない）。"""
        q = np.array(quat, dtype=float)
        mujoco.mju_normalize4(q)
        self.target_quat = q
        self.data.mocap_quat[self.mocap_id] = self.target_quat

    def _clamp_workspace(self, pos: np.ndarray) -> np.ndarray:
        pos[0] = np.clip(pos[0], self.workspace_x[0], self.workspace_x[1])
        pos[1] = np.clip(pos[1], self.workspace_y[0], self.workspace_y[1])
        pos[2] = np.clip(pos[2], self.workspace_z[0], self.workspace_z[1])
        return pos

    # -------------------------------------------------------------- グリッパ

    def _update_gripper(self, grip_button: bool, grip_analog=None) -> None:
        """グリッパの指令。

        grip_analog が None（既定）: grip_button の押し始めで開閉を切り替える。
        grip_analog が数: 連続の指令 ctrl = hi + (lo - hi) * clip(grip_analog, 0, 1)（0 = 開、1 = 閉）。
        どちらでも prev_grip は更新する（呼び方を切り替えても古い押し始めが効かない）。
        """
        if grip_analog is not None:
            g = float(np.clip(grip_analog, 0.0, 1.0))
            for idx in self.gripper_indices:
                lo = self.model.actuator_ctrlrange[idx][0]
                hi = self.model.actuator_ctrlrange[idx][1]
                self.data.ctrl[idx] = hi + (lo - hi) * g
            self.prev_grip = grip_button
            return

        if grip_button and not self.prev_grip:
            self.gripper_closed = not self.gripper_closed
            for idx in self.gripper_indices:
                lo = self.model.actuator_ctrlrange[idx][0]
                hi = self.model.actuator_ctrlrange[idx][1]
                self.data.ctrl[idx] = lo if self.gripper_closed else hi
        self.prev_grip = grip_button

    # --------------------------------------------------------------- クラッチ

    def _update_clutch(self, clutch_button, device_pos, device_quat) -> None:
        if clutch_button and not self.prev_clutch:
            self.device_ref_pos = device_pos.copy()
            self.target_ref_pos = self.target_pos.copy()
            self.device_ref_quat = device_quat.copy()
            self.target_ref_quat = self.target_quat.copy()

        if clutch_button and self.device_ref_pos is not None:
            delta_pos = device_pos - self.device_ref_pos
            desired = self.target_ref_pos + self.scale * delta_pos
            # これは追従器の参照。実際の目標は _track_target で滑らかに動かす
            self.desired_pos = self._clamp_workspace(desired)

            if self.rot_weight > 0.0 and not self.lock_orientation:
                inv_ref = np.zeros(4)
                mujoco.mju_negQuat(inv_ref, self.device_ref_quat)
                dquat = np.zeros(4)
                mujoco.mju_mulQuat(dquat, device_quat, inv_ref)
                new_quat = np.zeros(4)
                mujoco.mju_mulQuat(new_quat, dquat, self.target_ref_quat)
                mujoco.mju_normalize4(new_quat)
                self.target_quat = new_quat

        self.prev_clutch = clutch_button

        self._track_target()

        self.data.mocap_pos[self.mocap_id] = self.target_pos
        self.data.mocap_quat[self.mocap_id] = self.target_quat

    # -------------------------------------------------- 目標の追従器

    def _track_target(self) -> None:
        """target_pos を desired_pos へ制御の 1 ステップぶん進める。

        tracker_omega > 0: 臨界減衰の 2 次系（半陰的 Euler）、速さは max_target_vel で切る。
        クラッチを離している間も動かし、途中で止まらずに滑らかに落ち着かせる。
        tracker_omega <= 0: 1 ステップごとの移動量を max_target_step で切る。
        """
        if self.tracker_omega <= 0.0:
            step = self.desired_pos - self.target_pos
            n = np.linalg.norm(step)
            if n > self.max_target_step:
                step *= self.max_target_step / n
            self.target_pos = self.target_pos + step
            return

        dt = self.ctrl_dt
        w = self.tracker_omega
        acc = w * w * (self.desired_pos - self.target_pos) \
            - 2.0 * w * self.target_vel
        self.target_vel = self.target_vel + acc * dt

        speed = np.linalg.norm(self.target_vel)
        if speed > self.max_target_vel:
            self.target_vel *= self.max_target_vel / speed

        new_pos = self.target_pos + self.target_vel * dt
        clamped = self._clamp_workspace(new_pos.copy())
        # 作業空間の境界へ向かう速さを 0 にする（境界に押し付けた後、離れるときに跳ねないように）
        hit = clamped != new_pos
        if np.any(hit):
            self.target_vel[hit] = 0.0
        self.target_pos = clamped

    # ------------------------------------------------------------------- IK

    def _solve_ik_step(self) -> None:
        model, data = self.model, self.data

        hand_pos = data.xpos[self.hand_body_id]
        hand_quat = data.xquat[self.hand_body_id]

        pos_err = self.target_pos - hand_pos

        if self.rot_weight > 0.0:
            inv_hand = np.zeros(4)
            mujoco.mju_negQuat(inv_hand, hand_quat)
            err_quat = np.zeros(4)
            mujoco.mju_mulQuat(err_quat, self.target_quat, inv_hand)
            if err_quat[0] < 0.0:  # 四元数の二重被覆: 短い回り方を選ぶ
                err_quat = -err_quat
            rot_err = np.zeros(3)
            mujoco.mju_quat2Vel(rot_err, err_quat, 1.0)
        else:
            rot_err = np.zeros(3)

        dx = np.concatenate([self.pos_weight * pos_err,
                             self.rot_weight * rot_err])
        dx_norm = np.linalg.norm(dx)
        if dx_norm > self.max_dx_norm:
            dx *= self.max_dx_norm / dx_norm

        mujoco.mj_jacBody(model, data, self._jacp, self._jacr, self.hand_body_id)
        J = np.vstack([self._jacp[:, self.dof_indices],
                       self._jacr[:, self.dof_indices]])
        self.last_jac = J

        lam2 = self.ik_damping ** 2
        A = J @ J.T + lam2 * np.eye(6)
        J_pinv = J.T @ np.linalg.solve(A, np.eye(6))
        dq = J_pinv @ (self.ik_gain * dx)

        # 零空間の姿勢の項（null_gain > 0 のとき）
        if self.null_gain > 0.0:
            q_now = np.array([data.qpos[a[2]] for a in self.arm])
            dq_null = self.null_gain * (self.q_neutral - q_now)
            N = np.eye(self.n_arm) - J_pinv @ J
            dq = dq + N @ dq_null

        dq = np.clip(dq, -self.max_joint_step, self.max_joint_step)
        self.q_des = np.clip(self.q_des + dq, self.q_min, self.q_max)

        # 巻き上がり防止: 測った関節角のまわり ±q_des_leash に切る
        q_now = np.array([data.qpos[a[2]] for a in self.arm])
        self.q_des = np.clip(self.q_des,
                             q_now - self.q_des_leash,
                             q_now + self.q_des_leash)

        for k, (act_idx, _, _, _) in enumerate(self.arm):
            data.ctrl[act_idx] = self.q_des[k]

        # ---- 診断の出力（呼ぶ側が標準出力を捨てる） ----
        self._log_counter += 1
        if self._log_counter % 25 == 0:
            print(
                f"t={data.time:7.3f} "
                f"|pos_err|={np.linalg.norm(pos_err):.4f} "
                f"|dq|={np.linalg.norm(dq):.4f} "
                f"tgt=({self.target_pos[0]:.3f},{self.target_pos[1]:.3f},{self.target_pos[2]:.3f}) "
                f"hand=({hand_pos[0]:.3f},{hand_pos[1]:.3f},{hand_pos[2]:.3f})"
            )

    # ---------------------------------------------------------------- 1 ステップの更新

    def update(self, device: DeviceInput) -> DeviceState:
        state: DeviceState = device.read()
        # grip_analog の欄を持たない入力もあるので getattr で読む
        self._update_gripper(state.button_grip,
                              getattr(state, "grip_analog", None))
        self._update_clutch(state.button_clutch, state.pos, state.quat)
        self._solve_ik_step()
        return state