"""名前つきの独立した乱数列。

1 つの試行の乱数のシードから、用途ごとに別の乱数列を作る。番号（STREAM_ID）は固定し、変えない。

    stream(seed, "layout")                    配置     SeedSequence(seed, spawn_key=(0,))
    stream(seed, "induce")                    失敗の意図的な失敗     (1,)
    torch_seed(seed_sequence(seed, "noise", i))   i 回目の推論の方策の雑音   (2, i)
    script_rng(layout_seed, color, retry)     生成プログラム   (3, 色の添字, 作り直しの回数)
    inject_rng(layout_seed, color, retry)     生成での失敗の意図的な失敗   (4, 色の添字, 作り直しの回数)
"""
import numpy as np

STREAM_ID = {"layout": 0, "induce": 1, "noise": 2, "script": 3, "inject": 4, "order": 5,   # 変えない
             # センサのモデルの乱数
             "sensor_setup": 6,      # 試行ごとのセンサの値（較正誤差・深度の系統誤差・照明・フレームの位相）
             "sensor_frame": 7,      # フレームごとの雑音 (7, カメラの添字, フレームの番号)。描く順によらず同じ値
             "latency": 8}           # 推論・カメラの遅延の抽選
COLORS = ("red", "green", "blue")


def seed_sequence(seed: int, name: str, *sub: int) -> np.random.SeedSequence:
    """SeedSequence(seed, spawn_key=(STREAM_ID[name], *sub))。noise の推論 i は sub=(i,)。"""
    return np.random.SeedSequence(int(seed), spawn_key=(STREAM_ID[name],) + tuple(int(s) for s in sub))


def generator(ss: np.random.SeedSequence) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(ss))


def stream(seed: int, name: str) -> np.random.Generator:
    """np.random.Generator(np.random.PCG64(seed_sequence(seed, name)))。"""
    return generator(seed_sequence(seed, name))


def streams(seed: int) -> dict:
    """STREAM_ID の全部の名前について stream(seed, name)。"""
    return {name: stream(seed, name) for name in STREAM_ID}


def torch_seed(ss: np.random.SeedSequence) -> int:
    """torch.Generator.manual_seed に入れる 64 ビットのシード。"""
    return int(ss.generate_state(1, dtype=np.uint64)[0])


def script_rng(layout_seed: int, color: str, retry: int) -> np.random.Generator:
    """生成プログラムの乱数列。記録には (layout_seed, color, retry) を残す。"""
    return generator(seed_sequence(layout_seed, "script", COLORS.index(color), retry))


def inject_rng(layout_seed: int, color: str, retry: int) -> np.random.Generator:
    """データの生成でわざと失敗させる意図的な失敗の乱数列。"""
    return generator(seed_sequence(layout_seed, "inject", COLORS.index(color), retry))
