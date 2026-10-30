"""実行系の方策（学習した SmolVLA のチェックポイント）と、SensorFrame からの観測。

    pol = SensorPolicy(checkpoint, setup, fk)       # fk(q) -> (手先の位置, 四元数)。ロボットのモデルの順運動学
    pol.start_trial(seed)
    obs = pol.observe(sensor, task)                 # 観測（画像 2 枚・状態・指示）。前処理の時間も推論の時間に入る
    chunk = pol.infer(obs, i, rtc_kwargs)           # 後処理の後の行動列 (H, 7)

観測の作り方は学習と同じ関数（recovla.data.vla_observation・vla_state）で、入力だけをセンサの値にする:
  画像: SensorFrame の俯瞰・手首の RGB（センサのモデルの切り出し・雑音・遅延を通ったもの）
  状態: 手先の位置・向きは**測った**関節角の順運動学、指の 2 値は**測った**開き幅の半分ずつ、関節は測った角度
  目標位置の入力: 俯瞰の RGB と、実行系が持つ外部パラメータ・机の面（SetupInfo）
雑音の乱数（推論 i ごと）は policy.schedule.noise_generator から引く。
"""
import pathlib

import numpy as np

from recovla.common import config
from recovla.data import vla_image_spec as spec
from recovla.data import vla_observation, vla_state
from recovla.runtime import cue as C

_CFG = config.load()


class SensorPolicy:
    def __init__(self, checkpoint, setup, fk, device: str = "cuda"):
        self.checkpoint = pathlib.Path(checkpoint).resolve()
        self.conversion = vla_observation.load_training_spec(self.checkpoint)      # 画像の変換の照合
        self.setup = setup
        self.fk = fk
        import torch
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.processor import PolicyProcessorPipeline
        from lerobot.processor.converters import (batch_to_transition, policy_action_to_transition,
                                                  transition_to_batch, transition_to_policy_action)
        self.torch = torch
        self.device = device
        rt = _CFG["runtime"]
        torch.backends.cuda.matmul.allow_tf32 = bool(rt["matmul_tf32"])
        torch.backends.cudnn.allow_tf32 = bool(rt["cudnn_tf32"])
        self.policy = SmolVLAPolicy.from_pretrained(str(self.checkpoint)).to(device).eval()
        self.pre = PolicyProcessorPipeline.from_pretrained(
            str(self.checkpoint), config_filename="policy_preprocessor.json",
            overrides={"device_processor": {"device": device}},
            to_transition=batch_to_transition, to_output=transition_to_batch)
        self.post = PolicyProcessorPipeline.from_pretrained(
            str(self.checkpoint), config_filename="policy_postprocessor.json",
            to_transition=policy_action_to_transition, to_output=transition_to_policy_action)
        self.rename = {}
        for step in self.pre.steps:
            if hasattr(step, "rename_map"):
                self.rename.update(step.rename_map)
        self.cue = None
        self.cue_keep_flag = True
        rec = self.conversion.get("target_cue")
        self.cue_threshold_note = None
        if rec is not None:
            # 色の閾値は実行系の値（カメラのモデルの画像で学習用の配置から決めた runtime_v2.color_detect）。
            # 学習データの目標位置の入力を計算した閾値と違うときは、止めずに記録する
            thr = C.Thresholds.from_dict(config.color_detect(config.load_v2()))
            if rec["thresholds"] != thr.to_json():
                self.cue_threshold_note = {"training": rec["thresholds"], "runtime": thr.to_json()}
            self.cue_keep_flag = "cue_visible" in rec["names"]
            if rec["names"] != vla_state.cue_names(self.cue_keep_flag):
                raise spec.ImageSpecError(f"target_cue names {rec['names']}")
            self.cue = C.TargetCue(C.calibration_from_setup(setup.cameras["overhead"]), thr,
                                   setup.table_z + 0.5 * setup.cube_size, setup.cue_fallback_xy)
        self.last_cue = None
        # 置く位置の入力（r2_protocol.md 6.4 節）: 学習の conversion.json に place_cue があるときだけ、目標位置の入力の後ろに 2 つ足す。
        # 値は上位層が手順・やり直しの始めに set_place で渡す（api/place_cue.free_cell）。渡されていなければ place_fallback
        rec_p = self.conversion.get("place_cue")
        if rec_p is not None and rec_p["names"] != vla_state.PLACE_NAMES:
            raise spec.ImageSpecError(f"place_cue names {rec_p['names']}")
        self.place_cue = rec_p is not None
        self.place_xy = None
        self.place_fallback = np.asarray(setup.box_nominal_xy, float)[:2].copy()
        self.input_check = None

    @property
    def chunk_size(self) -> int:
        return int(self.policy.config.chunk_size)

    def start_trial(self, seed: int) -> None:
        self.policy.reset()
        self.reset_cue()
        self.input_check = None

    def set_cue_pose(self, R, t) -> None:
        """実行系が持つ俯瞰の外部パラメータが直ったとき（知覚の起動時の机の面での補正）、目標位置の入力の投影も直す。"""
        if self.cue is not None:
            c = self.cue.calib
            self.cue.calib = C.Calibration(np.asarray(t, float), np.asarray(R, float), c.f, c.width, c.height)

    def reset_cue(self) -> None:
        if self.cue is not None:
            self.cue.reset()
        self.last_cue = None

    def set_place(self, xy) -> None:
        """置く位置（世界の x, y）。None なら place_fallback を使う。置く位置の入力の無い重みでは何もしない。"""
        self.place_xy = None if xy is None else np.asarray(xy, np.float32)[:2].copy()

    def sync(self) -> None:
        if str(self.device).startswith("cuda"):
            self.torch.cuda.synchronize()

    # ------------------------------------------------------------------ 観測
    def observe(self, sensor, task: str, cue_color: str = None) -> dict:
        views = {"overhead": sensor.cameras["overhead"].rgb, "wrist": sensor.cameras["wrist"].rgb}
        pos, quat = self.fk(sensor.joints.q)
        w = float(sensor.gripper.width)
        proprio = {"ee_pos": pos, "ee_quat": quat, "fingers": np.array([0.5 * w, 0.5 * w]), "joints": np.asarray(sensor.joints.q)}
        state = vla_state.policy_state_frame(proprio)
        if self.cue is not None:
            self.last_cue = self.cue.update(views["overhead"], cue_color or C.color_of_instruction(task))
            state = vla_state.with_cue(state, self.last_cue, keep_flag=self.cue_keep_flag)
        if self.place_cue:
            state = vla_state.with_place(state, self.place_fallback if self.place_xy is None else self.place_xy)
        return {**vla_observation.observation_images(views), "observation.state": state, "task": str(task)}

    def _check_inputs(self, obs, batch) -> dict:
        t = self.torch
        out, slots = {}, []
        for view, key in spec.IMAGE_KEYS.items():
            slot = self.rename.get(key, key)
            same = bool(slot in batch and t.equal(batch[slot].detach().cpu().reshape(-1), t.from_numpy(obs[key]).reshape(-1)))
            if not same:
                raise RuntimeError(f"{view} image does not reach policy slot {slot}")
            out[view] = {"slot": slot, "equals_view": same}
            slots.append(slot)
        if t.equal(batch[slots[0]], batch[slots[1]]):
            raise RuntimeError(f"camera slots {slots} hold identical images")
        return out

    def infer(self, obs: dict, noise_generator, rtc_kwargs: dict = None, noise=None) -> np.ndarray:
        """観測 → 後処理の後の行動列 (H, 7)。rtc_kwargs は RTC の inference_delay・prev_chunk_left_over（正規化の空間）。
        noise を渡したら乱数を引かずにそれを使う（雑音の空間で行動を選ぶ方策を試すとき。形は (1, H, max_action_dim)）。"""
        t = self.torch
        batch = {k: t.from_numpy(v) for k, v in obs.items() if isinstance(v, np.ndarray)}
        batch["task"] = obs["task"]
        batch = self.pre(batch)
        if self.input_check is None:
            self.input_check = self._check_inputs(obs, batch)
        cfgp = self.policy.config
        if noise is None:
            noise = t.randn((1, cfgp.chunk_size, cfgp.max_action_dim), generator=noise_generator).to(self.device)
        else:
            noise = t.as_tensor(noise, dtype=t.float32).reshape(1, cfgp.chunk_size, cfgp.max_action_dim).to(self.device)
        kw = {}
        if rtc_kwargs:
            kw = {"inference_delay": int(rtc_kwargs["inference_delay"]),
                  "prev_chunk_left_over": t.as_tensor(rtc_kwargs["prev_chunk_left_over"], dtype=t.float32)[None].to(self.device)}
        with t.no_grad():
            chunk = self.policy.predict_action_chunk(batch, noise=noise, **kw)
        self.last_raw = chunk[0].detach().cpu().numpy()
        post = self.post(chunk).detach().cpu().numpy().reshape(-1, chunk.shape[-1])[:, :7].astype(np.float64)
        return post
