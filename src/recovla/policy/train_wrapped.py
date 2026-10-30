"""lerobot-train を包む。LeRobot のファイルは書き換えず、モジュールの名前を差し替えるだけ。

    python -m recovla.policy.train_wrapped [--first-frames N --first-weight W] [--reference]
                                           [--cue-aug-prob P --cue-aug-max-m M] [--fast-query] -- <lerobot-train の引数>

1. 重みつきの並べ方（--first-frames・--first-weight）: lerobot-train が作る EpisodeAwareSampler を、各エピソードの
   最初の N フレームを W 倍の重みで引く WeightedEpisodeAwareSampler に差し替える。1 周の長さは同じ（全フレーム数）で、
   (シード, 周回) から決まる乱数で復元抽出する。引いたフレームの数を数え、保存のたびに sampler_stats.json に書く
2. 学習時の出力の記録（--reference）: 保存のたびに、学習中の（メモリ上の）方策に、データセットの決まったフレームと
   決まった雑音を与えた行動列（predict_action_chunk、後処理の前）をチェックポイントの pretrained_model/recovla_reference.pt に
   書く。PEFT のときは、アダプタを切った出力も書く（対照）。推論の経路で読み込んだ方策の出力がこれと一致すれば、
   保存と読み込みで LoRA の差分が失われていない
3. 学習時の目標位置の入力のずらし（--cue-aug-prob・--cue-aug-max-m）: CueAugment を参照。前処理の後・更新の前に
   足す（update_policy を包む）。保存のたびに cue_aug_stats.json に、足した数・大きさと向きの度数を書く
4. 行動列の読み出しの高速化（--fast-query）: LeRobot の DatasetReader._query_hf_dataset は
   hf_dataset[key][50 個] で行全体（画像を含む）を読んで変換するので、1 試料に 160 ms かかる。画像でない
   列を、読み出しの最初に一度だけ同じ変換（torch.tensor）で表にしておき、そこから引く。値は元の経路と同じ
   （scripts/91_fast_query_check.py で 1,000 試料を比べる）
"""
import argparse
import json
import pathlib
import sys

import numpy as np

from lerobot.datasets.sampler import EpisodeAwareSampler

REFERENCE_FILE = "recovla_reference.pt"
SAMPLER_STATS = "sampler_stats.json"
CUE_AUG_STATS = "cue_aug_stats.json"
CUE_AUG_KEY = 54               # ずらしの乱数列: SeedSequence([学習のシード, 54])


class CueAugment:
    """学習時の目標位置の入力のずらし。試料ごとに確率 prob で、目標位置の入力の (x, y) に大きさ U(0, max_m)・向き
    U(0, 2π) のずれを足す（残りはそのまま）。前処理（正規化）の後に足すので、ずれは次元の標準偏差で割って足す。
    教師の行動は変えない。評価と推論では使わない。乱数は学習のシードから作る 1 本の列（主プロセスだけで引く）。"""

    last = None

    def __init__(self, seed: int, prob: float, max_m: float, idx: tuple, std: tuple):
        self.rng = np.random.default_rng(np.random.SeedSequence([int(seed), CUE_AUG_KEY]))
        self.seed, self.prob, self.max_m = int(seed), float(prob), float(max_m)
        self.idx, self.std = tuple(int(i) for i in idx), tuple(float(s) for s in std)
        self.n = 0
        self.applied = 0
        self.mag_hist = np.zeros(20, np.int64)          # 0〜max_m を 20 に分けた度数（足したものだけ）
        self.ang_hist = np.zeros(12, np.int64)          # 向きを 30° ごと
        self.mag_sum = 0.0
        CueAugment.last = self

    def draw(self, n: int):
        """(n, 2) のずれ [m] と、足したかの (n,) 真偽。"""
        on = self.rng.random(n) < self.prob
        mag = self.rng.uniform(0.0, self.max_m, n)
        ang = self.rng.uniform(0.0, 2 * np.pi, n)
        off = np.stack([np.cos(ang), np.sin(ang)], axis=1) * mag[:, None] * on[:, None]
        self.n += n
        self.applied += int(on.sum())
        self.mag_hist += np.histogram(mag[on], bins=20, range=(0.0, self.max_m))[0]
        self.ang_hist += np.histogram(ang[on], bins=12, range=(0.0, 2 * np.pi))[0]
        self.mag_sum += float(mag[on].sum())
        return off, on

    def apply(self, batch: dict) -> dict:
        import torch
        st = batch["observation.state"]
        off, _ = self.draw(st.shape[0])
        add = torch.zeros_like(st)
        for j, (i, s) in enumerate(zip(self.idx, self.std)):
            add[..., i] = torch.as_tensor(off[:, j] / s, dtype=st.dtype, device=st.device).reshape(
                (st.shape[0],) + (1,) * (st.ndim - 2))
        batch["observation.state"] = st + add
        return batch

    def stats(self) -> dict:
        return {"seed": self.seed, "key": CUE_AUG_KEY, "prob": self.prob, "max_m": self.max_m,
                "state_index": list(self.idx), "state_std": list(self.std), "samples": self.n,
                "applied": self.applied, "applied_share": (self.applied / self.n) if self.n else None,
                "magnitude_mean_m": (self.mag_sum / self.applied) if self.applied else None,
                "magnitude_hist_20": self.mag_hist.tolist(), "angle_hist_12": self.ang_hist.tolist()}


def cue_augment_for(lerobot_args: list, prob: float, max_m: float) -> CueAugment:
    """lerobot-train の引数から学習のシードとデータセットを読み、目標位置の入力の次元と標準偏差を取る。"""
    args = dict(a[2:].split("=", 1) for a in lerobot_args if a.startswith("--") and "=" in a)
    root = pathlib.Path(args["dataset.root"])
    conv = json.loads((root / "meta" / "conversion.json").read_text(encoding="utf-8"))
    stats = json.loads((root / "meta" / "stats.json").read_text(encoding="utf-8"))
    names = conv["state"]
    if "cue_x" not in names:
        raise SystemExit(f"{root}: the dataset has no target cue (state {names})")
    idx = (names.index("cue_x"), names.index("cue_y"))
    std = tuple(float(stats["observation.state"]["std"][i]) for i in idx)
    return CueAugment(int(args.get("seed", 1000)), prob, max_m, idx, std)
REFERENCE_NOISE_SEED = 20260926
REFERENCE_ITEMS = 3                  # データセットの最初の 3 エピソードの最初のフレーム


class WeightedEpisodeAwareSampler(EpisodeAwareSampler):
    """最初の first_frames フレーム（エピソードの中の位置 < first_frames）を weight 倍の重みで引く。"""

    last = None                      # 保存のときに統計を読むため、最後に作ったものを覚えておく

    def __init__(self, *args, first_frames: int = 20, weight: float = 5.0, drop_n_first_frames: int = 0, **kw):
        super().__init__(*args, drop_n_first_frames=drop_n_first_frames, **kw)
        lengths = np.diff(np.concatenate([[0], self._cum_lengths]))
        pos = np.concatenate([np.arange(n) + drop_n_first_frames for n in lengths])   # エピソードの中の位置
        self.is_first = pos < int(first_frames)
        self.weights = np.where(self.is_first, float(weight), 1.0)
        self.first_frames, self.weight = int(first_frames), float(weight)
        self.drawn = 0
        self.drawn_first = 0
        WeightedEpisodeAwareSampler.last = self

    @property
    def expected_first_share(self) -> float:
        w = self.weights
        return float(w[self.is_first].sum() / w.sum())

    def _iter_epoch(self, epoch: int, start: int):
        import torch
        draws = torch.multinomial(torch.as_tensor(self.weights, dtype=torch.float64), self._num_frames,
                                  replacement=True, generator=self._epoch_generator(epoch))
        for k in range(start, self._num_frames):
            p = int(draws[k])
            self.drawn += 1
            self.drawn_first += int(self.is_first[p])
            yield self._frame_index(p)

    def stats(self) -> dict:
        return {"first_frames": self.first_frames, "weight": self.weight, "num_frames": int(self._num_frames),
                "first_frames_total": int(self.is_first.sum()), "expected_first_share": self.expected_first_share,
                "uniform_first_share": float(self.is_first.mean()), "drawn": self.drawn,
                "drawn_first": self.drawn_first,
                "drawn_first_share": (self.drawn_first / self.drawn) if self.drawn else None}


_ORIGINAL_QUERY = None
IMAGE_DTYPES = ("image", "video")


def fast_query_hf_dataset(self, query_indices: dict) -> dict:
    """DatasetReader._query_hf_dataset の置き換え（モジュールの説明の 4）。画像でない列（行動・状態など）だけを
    表にして引く。画像の列（観測の 1 フレーム）は元の経路のまま。列の表は hf_dataset ごとに作り直す。"""
    import torch
    cache = self.__dict__.get("_recovla_columns")
    if cache is None or cache[0] is not self.hf_dataset:
        cache = (self.hf_dataset, {})
        self._recovla_columns = cache
    result, rest = {}, {}
    for key, q_idx in query_indices.items():
        if key in self._meta.video_keys:
            continue
        if self._meta.features.get(key, {}).get("dtype") in IMAGE_DTYPES:
            rest[key] = q_idx
            continue
        rel = q_idx if self._absolute_to_relative_idx is None else [self._absolute_to_relative_idx[i] for i in q_idx]
        table = cache[1].get(key)
        if table is None:
            table = torch.stack([torch.tensor(x) for x in self.hf_dataset.with_format(None).select_columns([key])[key][:]])
            cache[1][key] = table
        result[key] = table[torch.as_tensor(rel, dtype=torch.long)]
    if rest:
        result.update(_ORIGINAL_QUERY(self, rest))
    return result


def install_fast_query() -> None:
    global _ORIGINAL_QUERY
    from lerobot.datasets.dataset_reader import DatasetReader
    if _ORIGINAL_QUERY is None:
        _ORIGINAL_QUERY = DatasetReader._query_hf_dataset
    DatasetReader._query_hf_dataset = fast_query_hf_dataset


def fast_query_worker_init(worker_id: int) -> None:
    """DataLoader の worker_init_fn。名前で pickle されるので、読み手のプロセスでこのモジュールが読み込まれて差し替わる。"""
    install_fast_query()


def write_reference(pretrained_dir: pathlib.Path, cfg, policy, preprocessor) -> dict:
    """学習中の方策の出力をチェックポイントに書く（モジュールの説明の 2）。"""
    import torch
    from lerobot.datasets.factory import make_dataset
    ds = make_dataset(cfg)
    ep_from = [int(v) for v in ds.meta.episodes["dataset_from_index"]][:REFERENCE_ITEMS]
    items = [ds[i] for i in ep_from]
    batch = {}
    for k in items[0]:
        v = [it[k] for it in items]
        batch[k] = torch.stack(v) if isinstance(v[0], torch.Tensor) else v
    batch = preprocessor(batch)
    pcfg = policy.config
    gen = torch.Generator().manual_seed(REFERENCE_NOISE_SEED)
    noise = torch.randn((len(items), pcfg.chunk_size, pcfg.max_action_dim), generator=gen)
    was_training = policy.training
    policy.eval()
    out = {"dataset_indices": ep_from, "tasks": [it["task"] for it in items], "noise": noise}
    with torch.no_grad():
        out["chunk"] = policy.predict_action_chunk(batch, noise=noise.to(batch["observation.state"].device)).cpu()
        if hasattr(policy, "disable_adapter"):
            with policy.disable_adapter():
                out["chunk_adapter_disabled"] = policy.predict_action_chunk(
                    batch, noise=noise.to(batch["observation.state"].device)).cpu()
    if was_training:
        policy.train()
    torch.save(out, pretrained_dir / REFERENCE_FILE)
    return {"items": len(items), "has_disabled": "chunk_adapter_disabled" in out}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        raise SystemExit("usage: python -m recovla.policy.train_wrapped [options] -- <lerobot-train args>")
    cut = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--first-frames", type=int, default=None)
    ap.add_argument("--first-weight", type=float, default=None)
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--cue-aug-prob", type=float, default=None)
    ap.add_argument("--cue-aug-max-m", type=float, default=None)
    ap.add_argument("--fast-query", action="store_true")
    a = ap.parse_args(argv[:cut])
    import lerobot.scripts.lerobot_train as LT

    if a.fast_query:
        install_fast_query()            # 主プロセス（num_workers = 0 のとき）
        original_worker_kwargs = LT._dataloader_worker_kwargs

        def worker_kwargs(cfg):         # データの読み手（Windows では spawn で、差し替えを引き継がない）にも入れる
            kw = original_worker_kwargs(cfg)
            if cfg.num_workers > 0:
                kw["worker_init_fn"] = fast_query_worker_init
            return kw
        LT._dataloader_worker_kwargs = worker_kwargs

    if a.cue_aug_prob is not None:
        aug = cue_augment_for(argv[cut + 1:], a.cue_aug_prob, a.cue_aug_max_m)
        original_update = LT.update_policy

        def update_policy(train_metrics, policy, batch, *args, **kw):
            return original_update(train_metrics, policy, aug.apply(batch), *args, **kw)
        LT.update_policy = update_policy

    if a.first_frames is not None:
        ff, fw = a.first_frames, float(a.first_weight)

        def make_sampler(*args, **kw):
            return WeightedEpisodeAwareSampler(*args, first_frames=ff, weight=fw, **kw)
        LT.EpisodeAwareSampler = make_sampler

    original_save = LT.save_checkpoint

    def save_checkpoint(checkpoint_dir, step, cfg, policy, *args, **kw):
        original_save(checkpoint_dir, step, cfg, policy, *args, **kw)
        pretrained = pathlib.Path(checkpoint_dir) / "pretrained_model"
        s = WeightedEpisodeAwareSampler.last
        if s is not None:
            (pathlib.Path(checkpoint_dir) / SAMPLER_STATS).write_text(
                json.dumps({"step": int(step), **s.stats()}, indent=2), encoding="utf-8")
        if CueAugment.last is not None:
            (pathlib.Path(checkpoint_dir) / CUE_AUG_STATS).write_text(
                json.dumps({"step": int(step), **CueAugment.last.stats()}, indent=2), encoding="utf-8")
        if a.reference:
            pre = kw.get("preprocessor", args[2] if len(args) > 2 else None)
            info = write_reference(pretrained, cfg, policy, pre)
            print(f"recovla: wrote {pretrained / REFERENCE_FILE} {info}", flush=True)
    LT.save_checkpoint = save_checkpoint

    sys.argv = ["lerobot-train"] + argv[cut + 1:]
    LT.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
