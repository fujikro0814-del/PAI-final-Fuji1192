"""方策の実行器（docs/interfaces/runner.md）。時間の流れ（schedule.Schedule）を方策に繋ぐ。

    run = SceneRunner(ScenePolicy(ckpt), RuntimeConfig("rtc", 10, 2), rtc={"guidance_horizon": 10, "schedule": "EXP",
                                                                           "max_guidance_weight": 10.0})
    run.start_trial(seed)
    a = run(k, frame, raw_by_view, task)        # 10 fps の k ステップ目の行動（7）。scene_trial.run_trial の act にそのまま渡せる
    run.reset                                 # 指示の切り替え・やり直しの直後（前の行動列を持ち越さない）
    run.trace                                 # 行動列の記録（trial_record.md の chunk_* と inference[]）

- sync: k が s の倍数のときに推論し、推論の間は物理を止める。naive・rtc: 推論は直前の推論の開始から s 手ごとに k の観測で
  始め、結果は k+d から有効（その d 手は前の行動列を実行）。rtc は LeRobot の RTC（predict_action_chunk に inference_delay=d と、
  後処理の前の空間の前の行動列の残りを execution_horizon の長さに揃えたもの）。最初の行動列と reset の直後は持ち越さない
- この模擬では、推論の実時間の間も物理は止まっている（遅延は d 手として模擬する）。実時間は torch.cuda.synchronize 込みで
  推論ごとに測って記録する（前処理と推論に分けて）
- 雑音: 推論 i の雑音は schedule.noise_generator(試行のシード, i) から (1, H, max_action_dim) を CPU で引く。
  推論の回数が違う条件でも i 回目どうしは同じ雑音になり、naive の d=0 は sync と一致する
- 評価器の TF32 は PolicyActions が設定から固定する
"""
import time

import numpy as np

from recovla.common import config
from recovla.policy.schedule import RuntimeConfig, Schedule, noise_generator

_CFG = config.load()


def runtime_from_config(mode: str = None, exec_interval: int = None, delay_steps: int = None,
                        cfg: dict = None) -> RuntimeConfig:
    rt = (cfg or _CFG)["runtime"]
    mode = mode or rt["mode"]
    d = rt["delay_steps"] if delay_steps is None else delay_steps
    return RuntimeConfig(mode, int(exec_interval or rt["exec_interval"]), None if d is None else int(d),
                         execution_horizon=int(rt["rtc_guidance_horizon"]))


def _base_policy(policy):
    """PEFT で包んだ方策なら中の SmolVLAPolicy。"""
    return policy.get_base_model() if hasattr(policy, "get_base_model") else policy


def enable_rtc(policy, execution_horizon: int, schedule: str, max_guidance_weight: float) -> dict:
    """rtc_config を入れて init_rtc_processor。明示した値を返す（記録用）。"""
    from lerobot.configs import RTCAttentionSchedule
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    base = _base_policy(policy)
    sched = RTCAttentionSchedule[str(schedule).upper()]
    base.config.rtc_config = RTCConfig(enabled=True, prefix_attention_schedule=sched,
                                       max_guidance_weight=float(max_guidance_weight),
                                       execution_horizon=int(execution_horizon))
    base.init_rtc_processor()
    return {"enabled": True, "prefix_attention_schedule": sched.name, "max_guidance_weight": float(max_guidance_weight),
            "execution_horizon": int(execution_horizon)}


def disable_rtc(policy) -> None:
    base = _base_policy(policy)
    base.config.rtc_config = None
    base.init_rtc_processor()
    if getattr(base, "model", None) is not None:
        base.model.rtc_processor = None


class SceneRunner:
    def __init__(self, pol, rt: RuntimeConfig, rtc: dict = None):
        self.pol = pol
        self.rt = rt
        self.torch = pol.torch
        self.rtc = None
        if rt.mode == "rtc":
            r = rtc or {}
            self.rtc = enable_rtc(pol.policy, rt.execution_horizon, r.get("schedule", _CFG["runtime"]["rtc_schedule"]),
                                  r.get("max_guidance_weight", _CFG["runtime"]["rtc_max_guidance_weight"]))
        else:
            disable_rtc(pol.policy)
        self.seed = None
        self.sched = None

    # ------------------------------------------------------------------ 試行
    def start_trial(self, seed: int) -> None:
        self.pol.start_trial(seed)                      # policy.reset・目標位置の入力の保った値を捨てる・入力照合の準備
        self.seed = int(seed)
        self.sched = Schedule(self.rt)
        self.post = {}
        self.exec_log = []                              # k ごとの (行動列の番号 i, 添字)
        self.chunk_rows = []                            # 推論ごと: k_obs・k_valid・x_des（観測の時点）・行動列（後処理の後）
        self.cue_rows = []                              # 推論ごと: [k_obs, 目標位置の入力の x, y, 旗]（記録だけ）

    def reset(self) -> None:
        self.sched.reset()
        self.pol.builder.reset()

    def __call__(self, k, frame, raw_by_view, task):
        t = self.torch
        dec = self.sched.decide(int(k))
        if dec.infer:
            self.pol._sync()
            t0 = time.perf_counter()
            batch = self.pol.prepare(self.pol.builder.build(raw_by_view, frame, task))
            self.pol._sync()
            t1 = time.perf_counter()
            cue = self.pol.builder.last_cue          # 方策に渡した目標位置の入力（記録だけ）
            if cue is not None:
                self.cue_rows.append([float(k), float(cue[0]), float(cue[1]), float(cue[2])])
            i = len(self.sched.log) - 1
            cfgp = _base_policy(self.pol.policy).config
            noise = t.randn((1, cfgp.chunk_size, cfgp.max_action_dim),
                            generator=noise_generator(self.seed, i)).to(self.pol.device)
            kw = {}
            if self.rt.mode == "rtc" and dec.left_over is not None:
                kw = {"inference_delay": int(dec.inference_delay),
                      "prev_chunk_left_over": t.as_tensor(dec.left_over, dtype=t.float32)[None].to(self.pol.device)}
            with t.no_grad():
                chunk = self.pol.policy.predict_action_chunk(batch, noise=noise, **kw)
            self.pol._sync()
            t2 = time.perf_counter()
            raw_chunk = chunk[0].detach().cpu().numpy()                    # 後処理の前（正規化された空間）
            i_got = self.sched.deliver(raw_chunk)
            post = self.pol.post(chunk).detach().cpu().numpy().reshape(-1, chunk.shape[-1])[:, :7].astype(np.float64)
            self.post[i_got] = post
            entry = self.sched.log[-1]
            entry["wall_s"] = t2 - t0
            entry["wall_breakdown_s"] = {"preprocess": t1 - t0, "policy": t2 - t1}
            entry["block"] = bool(dec.block)
            entry["inference_delay"] = int(dec.inference_delay)
            self.chunk_rows.append({"i": i_got, "k_obs": int(k), "k_valid": int(entry["k_valid"]),
                                    "offset": int(entry["offset"]), "x_des": np.asarray(frame["x_des"], float).copy(),
                                    "post": post})
            self.pol.timing.append((t2 - t0, True))
        else:
            self.pol.timing.append((0.0, False))
        self.exec_log.append(dec.execute)
        return self.sched.action(self.post)[:7]

    # ------------------------------------------------------------------ 記録
    def trace(self) -> dict:
        """trial_record.md の chunk_*（行動の番号 k ごとと、行動列ごと）と inference[]。"""
        H = self.rt.chunk_size
        xdes = np.zeros((len(self.chunk_rows), H, 3))
        grip = np.zeros((len(self.chunk_rows), H))
        for m, r in enumerate(self.chunk_rows):
            xdes[m] = r["x_des"] + np.cumsum(r["post"][:, :3], axis=0)       # 行動列の頭（観測の時点）からの予測経路
            grip[m] = r["post"][:, 6]
        inference = [{"i": e["i"], "k_obs": e["k_obs"], "k_valid": e["k_valid"], "offset": e["offset"],
                      "left_over_len": e["left_over_len"], "reset": e["reset"], "block": e.get("block"),
                      "inference_delay": e.get("inference_delay"), "wall_s": e["wall_s"],
                      "wall_breakdown_s": e["wall_breakdown_s"]} for e in self.sched.log]
        return {"exec": list(self.exec_log), "chunk_k_valid": np.array([r["k_valid"] for r in self.chunk_rows], np.int32),
                "chunk_k_obs": np.array([r["k_obs"] for r in self.chunk_rows], np.int32),
                "chunk_xdes_pred": xdes, "chunk_grip_pred": grip, "inference": inference,
                "cue_at_inference": np.array(self.cue_rows, dtype=np.float64).reshape(-1, 4)}

    def runtime_record(self) -> dict:
        rt = _CFG["runtime"]
        return {"mode": self.rt.mode, "exec_interval": self.rt.exec_interval, "delay_steps": self.rt.delay,
                "rtc_guidance_horizon": self.rt.execution_horizon if self.rt.mode == "rtc" else None,
                "rtc_schedule": (self.rtc or {}).get("prefix_attention_schedule"),
                "rtc_max_guidance_weight": (self.rtc or {}).get("max_guidance_weight"),
                "safety_filter": bool(_CFG["safety_filter"]["enabled"]), "tf32": self.pol.config.get("tf32"),
                "num_steps": int(_base_policy(self.pol.policy).config.num_steps)}     # flow matching の積分の刻み数
