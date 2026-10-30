"""設定の読み込み（configs/default.yaml と、名前つきの上書き）と、<ROOT> のパス。

設定値は configs/ に置き、コードに数値を直書きしない。
上書きは default.yaml の上に辞書として再帰的に重ねる（configs/<名前>.yaml）。
"""
import copy
import functools
import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[3]          # <ROOT>（src/recovla/common/config.py から 3 つ上）
CONFIG_DIR = ROOT / "configs"


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


@functools.lru_cache(maxsize=None)
def _load_file(name: str) -> dict:
    path = CONFIG_DIR / f"{name}.yaml"
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: 最上位は辞書でなければならない")
    return data


def load(*overrides: str) -> dict:
    """default.yaml に overrides（configs/<名前>.yaml）を順に重ねた設定。呼ぶたびに新しい辞書を返す。"""
    cfg = _load_file("default")
    for name in overrides:
        cfg = _merge(cfg, _load_file(name))
    return copy.deepcopy(cfg)


def path(rel) -> pathlib.Path:
    """<ROOT> からの相対パス（設定の paths.* など）を絶対パスにする。"""
    p = pathlib.Path(rel)
    return p if p.is_absolute() else ROOT / p


V2_LAYERS = ("sensor_v1", "runtime_v2", "runtime_v2_color", "runtime_v2_derived", "runtime_v2_judge")


def load_v2() -> dict:
    """実行系の設定: センサと作動のモデル（sensor_v1）、実行系の手段（runtime_v2）、自動で決めた値（runtime_v2_derived・
    runtime_v2_judge。無ければ飛ばす）を順に重ねる。"""
    return load(*[n for n in V2_LAYERS if (CONFIG_DIR / f"{n}.yaml").is_file()])


def color_detect(cfg: dict) -> dict:
    """色の判定の閾値。センサのモデルの画像で決めた値（runtime_v2.color_detect）、無ければ学習時の値。"""
    return (cfg.get("runtime_v2") or {}).get("color_detect") or cfg["planner"]["color_detect"]
