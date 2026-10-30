"""3 色の場面の方策（学習した SmolVLA のチェックポイント）。立方体 1 個の評価器の PolicyActions を継ぎ、指示を呼び出しごとに渡す。

    pol = ScenePolicy(checkpoint)
    pol.start_trial(seed)                        # 雑音の乱数列を試行のシードから作る（seeds.torch_seed(noise の列)）
    a = pol(k, frame, raw_by_view, task)         # 同期の実行: select_action（50 ステップの行動列を使い切るまで待ち行列）
    chunk = pol.chunk_for(raw_by_view, frame, task, generator)   # 1 つの観測から行動列を 1 つ（後処理の後、(50, 7)）

観測は recovla.data.vla_observation（学習の変換と同じ定義）だけで作る。入力照合（2 視点が正しい位置に入っているか）は
試行ごとに最初の推論で行う。
"""
import time

import numpy as np

from recovla.common import seeds
from recovla.data import vla_image_spec as spec
from recovla.eval.closed_loop import PolicyActions


class ScenePolicy(PolicyActions):
    def start_trial(self, seed: int) -> None:
        super().start_trial(seed)
        self.generator = self.torch.Generator().manual_seed(seeds.torch_seed(seeds.seed_sequence(seed, "noise")))

    def __call__(self, k, frame, raw_by_view, task):
        obs = self.builder.build(raw_by_view, frame, task)
        self.policy_frames.append(np.hstack([
            np.round(obs[spec.IMAGE_KEYS[v]].transpose(1, 2, 0) * 255.0).astype(np.uint8)
            for v in ("overhead", "wrist")]))
        batch = self.prepare(obs)
        noise = self.noise()
        inferred = self._queue_empty()
        self._sync()
        t0 = time.perf_counter()
        with self.torch.no_grad():
            action = self.policy.select_action(batch, noise=noise)
        self._sync()
        self.timing.append((time.perf_counter() - t0, inferred))
        action = self.post(action)
        return action.detach().cpu().numpy().reshape(-1)[:7].astype(np.float64)

    def chunk_for(self, raw_by_view, frame, task, generator, cue_color=None) -> np.ndarray:
        """1 つの観測から行動列を 1 つ（後処理の後の (chunk_size, 7)）。雑音は渡された生成器から引く。
        cue_color: 目標位置の入力を計算する色（既定は指示の色。指示の差し替え試験で目標位置の入力だけを変えるとき）。
        行動列ごとに独立に見るので、目標位置の入力の「最後に見えた値」は毎回捨てる。"""
        self.builder.reset()
        batch = self.prepare(self.builder.build(raw_by_view, frame, task, cue_color))
        noise = self.torch.randn((1, self.policy.config.chunk_size, self.policy.config.max_action_dim),
                                 generator=generator).to(self.device)
        with self.torch.no_grad():
            chunk = self.policy.predict_action_chunk(batch, noise=noise)
        return self.post(chunk).detach().cpu().numpy().reshape(-1, chunk.shape[-1])[:, :7].astype(np.float64)
