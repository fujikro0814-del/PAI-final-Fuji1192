"""実機の Panda で測った応答に合わせたシミュレーション（v3_real）でデータを生成する入口。

    from recovla.expert import generate_real as GR
    results = GR.generate(specs, run_dir, world="v3_real", workers=8, render=True)          # generate.generate と同じ引数
    results = GR.generate(specs, run_dir, world="v3_real", switches={"b_slip": True})      # 入り・切りの上書き

中身は generate.generate そのもの。呼んでいる間だけ、generate の _init_worker・rig_config・run_attempt と
inject.Injector を差し替える（rig_kind が v3_real のときだけ働き、ほかは元の関数に渡す）。並列（spawn）の子のプロセスでは、
このモジュールの _init_worker が同じ差し替えを入れる。入り・切りの上書きは rig_kind に JSON で付けて渡す
（"v3_real|{...}"。run.json の rig_kind に残る）。エピソードの meta.json の "world" に入り・切りと値が残る。

外乱 D（expert/displace.py）: 種類 "D" の指定だけ、generate.run_attempt を包んで、近づく途中に目標を動かす。
記録は最初から。meta.json と結果に "disturb" を足す。外乱が入らなかった試みは保存しない（"disturb_<状態>"）。
"""
import contextlib
import json
import shutil

from recovla.expert import displace
from recovla.expert import generate as G
from recovla.expert import inject
from recovla.expert import inject_real
from recovla.harness import real_matched as V

_ORIG = {"init": G._init_worker, "rig_config": G.rig_config, "injector": inject.Injector, "run_attempt": G.run_attempt}


def rig_kind_of(world: str, switches: dict = None) -> str:
    return world + ("|" + json.dumps(switches, sort_keys=True) if switches else "")


def parse_rig_kind(rig_kind: str) -> tuple:
    kind, _, sw = str(rig_kind).partition("|")
    return kind, (json.loads(sw) if sw else {})


def rig_config(rig_kind: str):
    kind, switches = parse_rig_kind(rig_kind)
    if kind in V.WORLDS:
        return V.load_config(kind, generation=True, switches=switches)
    return _ORIG["rig_config"](rig_kind)


def _patch() -> None:
    G._init_worker = _init_worker
    G.rig_config = rig_config
    inject.Injector = inject_real.make_injector
    G.run_attempt = run_attempt_any


def _unpatch() -> None:
    G._init_worker = _ORIG["init"]
    G.rig_config = _ORIG["rig_config"]
    inject.Injector = _ORIG["injector"]
    G.run_attempt = _ORIG["run_attempt"]


def run_attempt_any(rig, spec, retry: int, run_dir=None, render: bool = True, handover_at_s: float = None) -> dict:
    """generate.run_attempt の代わり。種類 D だけ外乱を入れ、ほかは元のまま。"""
    if spec.kind != displace.KIND:
        return _ORIG["run_attempt"](rig, spec, retry, run_dir, render, handover_at_s)
    dis = displace.Displacer(displace.sample_params(displace.rng_for(spec.layout_seed, spec.color, retry)), rig,
                             spec.color, rig.cfg)
    own_meta = rig.__dict__.get("episode_meta")
    base_meta = rig.episode_meta
    rig.episode_meta = lambda: dict(base_meta(), disturb=dis.summary())
    dis.install()
    try:
        out = _ORIG["run_attempt"](rig, spec, retry, run_dir, render, handover_at_s)
    finally:
        dis.uninstall()
        if own_meta is None:
            rig.__dict__.pop("episode_meta", None)
        else:
            rig.episode_meta = own_meta
    out["disturb"] = dis.summary()
    if dis.status != "done" and out["success"]:      # 外乱が入らなかった: 保存しない（作り直しへ）
        if out.get("path"):
            shutil.rmtree(out.pop("path"), ignore_errors=True)
        out["success"], out["failure"] = False, f"disturb_{dis.status}"
    return out


def _init_worker(render: bool, rig_kind: str = "v1") -> None:
    """generate._init_worker の代わり。rig_kind が v3_real（"v3_real|{...}" を含む）なら V3RealGenRig を作る。"""
    kind, _ = parse_rig_kind(rig_kind)
    if kind not in V.WORLDS:
        return _ORIG["init"](render, rig_kind)
    _patch()                                         # spawn の子では、ここで初めて差し替わる
    G._RIG = V.V3RealGenRig(rig_config(rig_kind))


@contextlib.contextmanager
def patched():
    _patch()
    try:
        yield
    finally:
        _unpatch()


def generate(specs: list, run_dir, world: str = "v3_real", workers: int = 1, render: bool = True, max_retry: int = None,
             log_name: str = "generation.jsonl", switches: dict = None) -> list:
    """generate.generate を、世界 world（v3_real）で回す。"""
    if world not in V.WORLDS:
        raise ValueError(f"world {world!r} は {V.WORLDS} のどれか")
    with patched():
        return G.generate(specs, run_dir, workers=workers, render=render, max_retry=max_retry, log_name=log_name,
                          rig_kind=rig_kind_of(world, switches))


def run_attempt(rig, spec, retry: int, run_dir=None, render: bool = True) -> dict:
    """1 回の試み（generate.run_attempt）。rig は V3RealGenRig。差し替えはこの間だけ入れる。"""
    with patched():
        return G.run_attempt(rig, spec, retry, run_dir, render)
