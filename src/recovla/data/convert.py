"""記録したエピソード（recovla.record.recorder）を LeRobot v3.0 形式のデータセットに変換する。

    .venv\\Scripts\\python.exe -m recovla.data.convert
        --out DIR --name NAME (--raw-dir RAW | --manifest M.json | EPISODE_DIR ...)
    ... -m recovla.data.convert --verify DIR [--export-actions NPZ_DIR] [--check-image PNG]

元の記録は読むだけで変えない。形式は HuggingFaceVLA/libero（LeRobot v3.0、画像を埋め込み、10 fps）に合わせる。

  observation.images.image    俯瞰カメラ  } 記録の PNG を vla_image_spec.to_policy_image で方策の画像にする
  observation.images.image2   手首カメラ  } （変換の定義はそこだけ。conversion.json に記録し、--verify と評価時に確かめる）
  observation.state (15)      手先の位置 3 [m, world]、真下向きからの姿勢のずれ 3 [rad]、
                              指 2 [m]（2 本目は符号を反転、LIBERO と同じ）、関節 7 [rad]
  action (7)                  そのフレームから 0.1 s 後までの x_des の変化 3 [m, world]、回転 3 = 0、
                              グリッパ +1 閉 / -1 開（0.1 s 後の状態、LIBERO と同じ）

20 Hz の記録から 10 fps に間引く: フレーム k = 記録のフレーム 2k、行動 k = x_des[2k+2] - x_des[2k]。
後ろに 0.1 s が続かない最後のフレームは落とす。observation.state の定義は vla_state.policy_state だけにある。
"""
import argparse
import datetime
import hashlib
import json
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw

from recovla.common import config
from recovla.data import vla_image_spec as spec
from recovla.data import vla_state
# ほかのモジュール・試験から使うために再び公開する（定義は vla_state）
from recovla.data.vla_state import Q_DOWN, STATE_NAMES, orientation_deviation, quat_mul  # noqa: F401

_CFG = config.load()

# 2: 画像の変換を視点ごとに vla_image_spec から取り、conversion.json に記録する。1: 両方を左右反転、記録なし
CONVERTER_VERSION = 2
FPS = int(_CFG["convert"]["fps"])
RAW_HZ = round(1.0 / (_CFG["sim"]["record_every"] * _CFG["sim"]["timestep"]))     # 20 Hz
STRIDE = RAW_HZ // FPS
if STRIDE != int(_CFG["sim"]["stride"]):
    raise ValueError(f"configs: record rate {RAW_HZ} Hz / fps {FPS} != sim.stride {_CFG['sim']['stride']}")
IMAGE_KEYS = spec.IMAGE_KEYS
ACTION_NAMES = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]
ZERO_STD_EPS = 1e-8   # LeRobot MEAN_STD: (x - mean) / (std + eps)
# 3 色の場面のエピソード（recovla.record.episode）から conversion.json の sources に写す項目
FLAG_KEYS = ("name", "kind", "layout_kind", "target", "instruction", "pair_id", "start_pose", "layout_seed", "retry")


def episode_label(meta: dict) -> str:
    """エピソードの呼び名: 3 色の場面は名前（n_20000_red_r0 など）、名前のない記録は ep_000123。"""
    return meta["name"] if meta.get("name") else f"ep_{int(meta['episode_id']):06d}"


def _sort_key(meta: dict):
    return (1, meta["name"]) if meta.get("name") else (0, f"{int(meta['episode_id']):09d}")


# ------------------------------------------------------------- マニフェスト（変換するエピソードの一覧）

def write_manifest(path, name: str, entries: list, rule: str, base_commit=None) -> dict:
    """entries: [{"run": 回のフォルダ（<ROOT> からの相対でも可）, "key": エピソードの名前}]。"""
    comp = {}
    for e in entries:
        k = e["key"].split("_")
        comp_key = k[0] if len(k) < 3 else f"{k[0]}_{k[2]}"
        comp[comp_key] = comp.get(comp_key, 0) + 1
    manifest = {"name": name, "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                "base_commit": base_commit, "rule": rule, "entries": entries, "composition": comp}
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def manifest_episodes(manifest: dict) -> list:
    """マニフェストに載った回と名前だけを読む。1 つでも見つからなければ、変換の前に止める。"""
    paths, missing = [], []
    for e in manifest["entries"]:
        p = config.path(e["run"]) / e["key"]
        (paths if (p / "meta.json").is_file() and (p / "data.npz").is_file() else missing).append(p)
    if missing:
        raise FileNotFoundError(f"manifest {manifest['name']}: {len(missing)} episode(s) missing, e.g. {missing[:3]}")
    return paths


CUE_MODES = (None, "xyv", "xy")     # なし / 目標位置の入力 3（見えているかの旗を含む、18 次元）/ 旗を外した 2（17 次元）


def _cue_mode(target_cue):
    """真偽値（True は xyv）も受ける。"""
    mode = "xyv" if target_cue is True else (None if target_cue is False else target_cue)
    if mode not in CUE_MODES:
        raise ValueError(f"target_cue {target_cue!r} not in {CUE_MODES}")
    return mode


def state_names(target_cue=None, place_cue: bool = False) -> list:
    mode = _cue_mode(target_cue)
    return (list(STATE_NAMES) + ([] if mode is None else vla_state.cue_names(mode == "xyv"))
            + (list(vla_state.PLACE_NAMES) if place_cue else []))


def features(size: int = 256, target_cue=None, place_cue: bool = False) -> dict:
    img = {"dtype": "image", "shape": [size, size, 3], "names": ["height", "width", "channel"]}
    names = state_names(target_cue, place_cue)
    return {
        **{IMAGE_KEYS[view]: dict(img) for view in spec.CAMERAS},
        "observation.state": {"dtype": "float32", "shape": [len(names)], "names": names},
        "action": {"dtype": "float32", "shape": [len(ACTION_NAMES)], "names": ACTION_NAMES},
    }


# ------------------------------------------------------------- 目標位置の入力（俯瞰画像の指示色の位置）

CUE_THRESHOLD_SOURCES = ("training", "runtime_v2")


def cue_thresholds(source: str = "training"):
    """目標位置の入力の色の閾値。training は設定の planner.color_detect、runtime_v2 は実行系の値
    （カメラのモデルの画像で決めた runtime_v2.color_detect）。実行系と同じにするときは runtime_v2。"""
    from recovla.perception import color as PC
    if source not in CUE_THRESHOLD_SOURCES:
        raise ValueError(f"cue thresholds source {source!r} not in {CUE_THRESHOLD_SOURCES}")
    if source == "training":
        return PC.Thresholds.from_config()
    c = config.color_detect(config.load_v2())
    return PC.Thresholds(int(c["min_value"]), int(c["min_margin"]), int(c["min_pixels"]))


def cue_tracker(source: str = "training"):
    """学習と評価で同じ部品・同じ閾値（recovla.perception.color、閾値は cue_thresholds）。"""
    from recovla.perception import color as PC
    from recovla.sim import scene
    return PC.TargetCue(PC.overhead_calibration(scene.build_model("3cube")), cue_thresholds(source))


def episode_tracker(meta: dict, default):
    """エピソードが較正（cue_calibration）を記録していれば、実行系と同じく、その較正と机の面の高さで目標位置の入力を計算する。
    記録がなければ default を使う。閾値は同じ。"""
    cc = meta.get("cue_calibration")
    if cc is None or default is None:
        return default
    from recovla.runtime import cue as C
    calib = C.Calibration(np.array(cc["pos"], float), np.array(cc["rot"], float), float(cc["f"]), int(cc["width"]),
                          int(cc["height"]))
    return C.TargetCue(calib, default.thr, float(cc["plane_z"]), cc["fallback"])


def cue_record(tracker, mode: str = "xyv", source: str = "training") -> dict:
    """conversion.json の target_cue。評価の入口（vla_observation）はこれを見て同じ目標位置の入力を足す。
    names が方策の入力に入る目標位置の入力の次元（xy のときは旗を外し、旗は sources の cue_flag0_frames にだけ残す）。"""
    keep = mode == "xyv"
    return {"version": 1, "mode": mode, "names": vla_state.cue_names(keep), "thresholds": tracker.thr.to_json(),
            "thresholds_source": source,
            "plane_z": float(tracker.plane_z), "fallback_xy": [float(v) for v in tracker.fallback],
            "calibration": tracker.calib.to_json(), "fixed_stats": vla_state.CUE_FIXED_STATS if keep else {},
            "definition": "overhead raw render → pixels of the instructed color (recovla.perception.color) → "
                          "centroid → ray to the plane z = cube rest height → (x, y); flag 1 if >= min_pixels "
                          "pixels, else 0 and the last seen (x, y) is kept (region centre if never seen). "
                          "Recomputed at every 10 fps frame in episode order. No simulator truth."}


def place_label(meta: dict, data: dict) -> np.ndarray:
    """置く位置の入力の学習の値（vla_state.PLACE_NAMES）: 目標の立方体がエピソードの終わりに止まった (x, y)。箱に入っていなければ止める。"""
    i = ["red", "green", "blue"].index(meta["target"])
    if not bool(data["cube_in_box"][-1, i]):
        raise ValueError(f"episode {episode_label(meta)}: target not in the box at the end (no place label)")
    return np.asarray(data["cube_pos"][-1, i, :2], np.float32)


PLACE_RECORD = {"version": 1, "names": vla_state.PLACE_NAMES,
                "definition": "training: (x, y) [m, world] where the scripted teacher's target cube came to rest at the "
                              "end of the episode (simulator truth, a hindsight label, constant over the episode); "
                              "runtime: recovla.api.place_cue.free_cell on the estimated box and objects at the start "
                              "of every step and retry (no simulator truth, no known sizes)"}


def fix_cue_stats(out: pathlib.Path) -> None:
    """旗など、学習データで一定になりうる次元の正規化の値を固定する（vla_state.CUE_FIXED_STATS。xyv のときだけ）。"""
    path = out / "meta" / "stats.json"
    stats = json.loads(path.read_text(encoding="utf-8"))
    names = state_names("xyv")
    for name, fixed in vla_state.CUE_FIXED_STATS.items():
        i = names.index(name)
        for k, v in fixed.items():
            stats["observation.state"][k][i] = v
    path.write_text(json.dumps(stats, indent=4), encoding="utf-8")


# ------------------------------------------------------------- raw side

def load_raw(path: pathlib.Path):
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    with np.load(path / "data.npz") as npz:
        data = {k: npz[k] for k in npz.files}
    return meta, data


def raw_episodes(raw_dir: pathlib.Path) -> list:
    eps = [p for p in raw_dir.glob("*/ep_*")
           if p.is_dir() and not p.name.endswith(".partial") and (p / "meta.json").is_file()]
    return sorted(eps, key=lambda p: _sort_key(json.loads((p / "meta.json").read_text(encoding="utf-8"))))


TASK_SOURCE_KEYS = ("task", "instruction_fmt", "instruction_mode")


def task_source_keys(meta: dict, etr) -> dict:
    """別のタスクのエピソード（meta に "task" がある）だけ、sources にタスク・指示文の書式・目標位置の入力の
    平面の高さを足す（3 色の場面のエピソードの sources の形は変えない）。"""
    if not meta.get("task"):
        return {}
    out = {k: meta.get(k) for k in TASK_SOURCE_KEYS}
    out["cue_plane_z"] = None if etr is None else float(etr.plane_z)
    return out


def merge_manifests(manifests: list) -> dict:
    """マニフェストを連ねる（1 つの VLA で 2 つのタスクを学ぶデータセット用）。1 つなら、そのまま返す。
    2 つ以上なら {"name": a+b, "parts": [...], "entries": 連ねたもの}。同じエピソードが 2 回出たら止める。"""
    if len(manifests) == 1:
        return manifests[0]
    entries = [e for m in manifests for e in m["entries"]]
    keys = [(e["run"], e["key"]) for e in entries]
    if len(set(keys)) != len(keys):
        raise ValueError("manifests: 同じエピソードが 2 回出る")
    return {"name": "+".join(m["name"] for m in manifests), "parts": manifests, "entries": entries,
            "composition": {m["name"]: len(m["entries"]) for m in manifests}}


def instruction_ok(task: str, meta: dict) -> bool:
    """指示文の検査: 設定の形（convert.instruction）か、エピソードの書式（meta の instruction_fmt）を、
    メタデータの目標の色で埋めたものと一致するか。"""
    color = meta.get("target")
    fmt = meta.get("instruction_fmt")
    return task == _CFG["convert"]["instruction"].format(color=color) or (
        fmt is not None and "{color}" in fmt and task == fmt.format(color=color))


# --self-label-clean（方策が自分で動いた記録 S だけ。自分の成功を足す試みで使用）: 実行系の都合でできた形をラベルから外す。
#   始めの保持: 最初の推論を待つ間（腕が動かず、グリッパも変わらない）のフレームを落とし、動き始めのフレームから使う
#   グリッパ: grip_min_run フレーム（0.1 s ごと）より短く、すぐ戻る開閉（方策の値が 0 の近くで揺れた切り替え）を前の状態にそろえる
SELF_CLEAN = {"kinds": ["S"], "trim_lead_hold": True, "grip_min_run": 3, "hold_eps": 1e-9}


def smooth_grip(g: np.ndarray, min_run: int) -> np.ndarray:
    """端でない部分のうち min_run フレームより短いものを、前の部分の値にそろえる（そろえて新しくできた部分も見直す）。"""
    g = g.copy()
    while True:
        cut = np.flatnonzero(np.diff(g) != 0) + 1
        bounds = np.concatenate([[0], cut, [len(g)]])
        short = [i for i in range(1, len(bounds) - 2) if bounds[i + 1] - bounds[i] < min_run]
        if not short:
            return g
        i = short[0]
        g[bounds[i]:bounds[i + 1]] = g[bounds[i] - 1]


def self_clean_arrays(arr: dict, clean: dict) -> dict:
    """SELF_CLEAN の 2 つを 10 fps の配列にかける。arr["clean"] に落としたフレーム・変えたフレームの数を残す。"""
    act = arr["action"]
    lead = 0
    if clean.get("trim_lead_hold"):
        moving = np.flatnonzero(np.abs(act[:, :3]).max(axis=1) > clean["hold_eps"])
        grip_change = np.flatnonzero(act[:, 6] != act[0, 6])
        firsts = [int(v[0]) for v in (moving, grip_change) if len(v)]
        lead = min(firsts) if firsts else 0
    out = {k: v[lead:] for k, v in arr.items()}
    g0 = out["action"][:, 6]
    g1 = smooth_grip(g0, int(clean["grip_min_run"]))
    out["action"] = out["action"].copy()
    out["action"][:, 6] = g1
    out["clean"] = {"lead_trimmed": lead, "grip_changed": int((g0 != g1).sum())}
    return out


def episode_arrays(meta: dict, data: dict, clean: dict = None) -> dict:
    """10 fps の状態・行動の配列と、元の記録のフレームの番号。clean（SELF_CLEAN）があれば、その種類の
    エピソードだけ self_clean_arrays をかける。"""
    arr = _episode_arrays(meta, data)
    if clean and meta.get("kind") in clean["kinds"]:
        arr = self_clean_arrays(arr, clean)
    return arr


def _episode_arrays(meta: dict, data: dict) -> dict:
    if int(meta["record_hz"]) != RAW_HZ:
        raise ValueError(f"episode {episode_label(meta)}: record_hz {meta['record_hz']} != {RAW_HZ}")
    n = int(meta["n_frames"])
    k_frames = (n - 1) // STRIDE
    if k_frames < 1:
        raise ValueError(f"episode {episode_label(meta)}: too short ({n} raw frames)")
    idx = np.arange(k_frames) * STRIDE
    nxt = idx + STRIDE
    state = vla_state.policy_state(data["ee_pos"][idx], data["ee_quat"][idx],
                                   data["fingers"][idx], data["joints"][idx])
    grip = np.where(data["gripper_closed"][nxt], 1.0, -1.0)[:, None]
    action = np.concatenate([data["x_des"][nxt] - data["x_des"][idx],
                             np.zeros((k_frames, 3)), grip], axis=1)
    return {"raw_index": idx, "state": state, "action": action.astype(np.float32)}


def read_raw_image(path: pathlib.Path) -> np.ndarray:
    """記録の PNG を変換せずに H x W x 3 uint8 で読む。"""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB")).copy()


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------ 変換

def convert(episodes: list, out: pathlib.Path, name: str, manifest: dict = None, target_cue=None,
            cue_source: str = "training", self_clean: dict = None, place_cue: bool = False) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    mode = _cue_mode(target_cue)
    if out.exists():
        raise FileExistsError(f"{out} exists; converted datasets are rebuilt, not appended to")
    first_meta, _ = load_raw(episodes[0])
    ds = LeRobotDataset.create(repo_id=f"local/{name}", fps=FPS, features=features(
        int(first_meta["cameras"]["width"]), mode, place_cue), root=out, robot_type="panda", use_videos=False)
    tracker = cue_tracker(cue_source) if mode else None
    sources = []
    try:
        for ep_index, path in enumerate(episodes):
            meta, data = load_raw(path)
            arr = episode_arrays(meta, data, self_clean)
            place = place_label(meta, data) if place_cue else None
            flag0 = 0
            etr = episode_tracker(meta, tracker)
            if etr is not None:
                etr.reset()
            for k, i in enumerate(arr["raw_index"]):
                raw = {view: read_raw_image(path / view / f"{i:06d}.png") for view in spec.CAMERAS}
                images = spec.policy_images(raw)
                state = arr["state"][k]
                if etr is not None:
                    c = etr.update(raw["overhead"], meta["target"])
                    flag0 += int(c[2] < 0.5)
                    state = vla_state.with_cue(state, c, keep_flag=mode == "xyv")
                if place is not None:
                    state = vla_state.with_place(state, place)
                ds.add_frame({
                    **images,
                    "observation.state": state,
                    "action": arr["action"][k],
                    "task": meta["instruction"],
                })
            ds.save_episode()
            sources.append({"episode_index": ep_index, "raw_episode_id": meta.get("episode_id"),
                            "raw_episode": episode_label(meta), "run": path.parent.name,
                            **{k: meta.get(k) for k in FLAG_KEYS},
                            "raw_path": str(path), "frames": int(len(arr["raw_index"])),
                            "raw_frames": int(meta["n_frames"]),
                            "placement_id": meta.get("placement_id"),
                            "operator": meta.get("operator"), "session": meta.get("session"),
                            "success": meta.get("success"), "retakes": meta.get("retakes"),
                            "meta_sha256": sha256(path / "meta.json"),
                            "data_sha256": sha256(path / "data.npz"),
                            **({"cue_flag0_frames": flag0} if tracker is not None else {}),
                            **({"place": [float(v) for v in place]} if place is not None else {}),
                            **({"self_clean": arr["clean"]} if "clean" in arr else {}),
                            **task_source_keys(meta, etr)})
            print(f"[convert] ep {ep_index}: raw {episode_label(meta)} "
                  f"{meta['n_frames']} raw frames -> {len(arr['raw_index'])} frames")
    finally:
        ds.finalize()
    if mode == "xyv":
        fix_cue_stats(out)
    (out / "meta" / "conversion.json").write_text(json.dumps({
        "converter_version": CONVERTER_VERSION,
        "converted_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "fps": FPS, "raw_hz": RAW_HZ, "stride": STRIDE,
        **spec.spec_record(),
        "images": "PNG from raw -> vla_image_spec net transform per view (image_transforms)",
        "state": state_names(mode, place_cue), "action": ACTION_NAMES,
        **({"target_cue": cue_record(tracker, mode, cue_source)} if tracker is not None else {}),
        **({"place_cue": PLACE_RECORD} if place_cue else {}),
        "action_definition": "xyz = x_des[raw 2k+2] - x_des[raw 2k] (m, world); rot = 0; "
                             "gripper = +1 closed / -1 open at raw frame 2k+2",
        "orientation": "world-frame axis-angle of ee_quat * conj(q_down), q_down = (0,1,0,0) "
                       "(w,x,y,z); 0 = pointing straight down",
        "fingers": "finger_joint1, -finger_joint2 (LIBERO sign)",
        **({"self_label_clean": self_clean} if self_clean else {}),
        **({"manifest": manifest} if manifest else {}),
        "sources": sources,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------- 検証

def write_check_image(path: pathlib.Path, raw_by_view: dict, policy_by_view: dict, caption: str) -> None:
    """同じフレームを、視点ごとに 1 行 [記録の PNG | データセットの方策の入力] で 1 枚の PNG に並べる。"""
    size = next(iter(raw_by_view.values())).shape[0]
    label_h = 18
    canvas = Image.new("RGB", (2 * size, len(spec.CAMERAS) * (size + label_h) + label_h), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 3), caption, fill=(255, 255, 0))
    for r, view in enumerate(spec.CAMERAS):
        y = label_h + r * (size + label_h)
        draw.text((4, y + 3), f"{view}: raw PNG (as recorded / operator view)", fill=(255, 255, 255))
        draw.text((size + 4, y + 3), f"{view}: policy input ({spec.VIEW_TRANSFORMS[view]})", fill=(255, 255, 255))
        canvas.paste(Image.fromarray(raw_by_view[view]), (0, y + label_h))
        canvas.paste(Image.fromarray(policy_by_view[view]), (size, y + label_h))
    canvas.save(path)


def verify(out: pathlib.Path, export_dir, check_image=None) -> bool:
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    conv = json.loads((out / "meta" / "conversion.json").read_text(encoding="utf-8"))
    ds = LeRobotDataset(repo_id=f"local/{out.name}", root=out)
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("  ok   " if cond else "  FAIL ") + msg)
        ok &= bool(cond)

    print(f"[verify] {out}: {ds.num_episodes} episodes, {ds.num_frames} frames, fps {ds.fps}")
    try:
        spec.check_record(conv, str(out / "meta" / "conversion.json"))
        check(True, f"image spec v{spec.SPEC_VERSION} recorded == runtime {spec.VIEW_TRANSFORMS}")
    except spec.ImageSpecError as exc:
        check(False, f"image spec: {exc}")
    check(ds.fps == FPS, f"fps == {FPS}")
    check(ds.num_episodes == len(conv["sources"]), "episode count matches conversion.json")
    check(ds.num_frames == sum(s["frames"] for s in conv["sources"]), "frame count matches")

    mode = conv["target_cue"].get("mode", "xyv") if "target_cue" in conv else None
    place_cue = "place_cue" in conv
    names_state = state_names(mode, place_cue)
    check(ds.meta.features["observation.state"]["names"] == names_state and conv["state"] == names_state,
          f"state names == {len(names_state)} (target_cue {mode}, place_cue {place_cue})")
    source = conv["target_cue"].get("thresholds_source", "training") if mode else None
    tracker = cue_tracker(source) if mode else None
    if tracker is not None:
        check(conv["target_cue"]["thresholds"] == tracker.thr.to_json(),
              f"target_cue thresholds recorded == configs ({source}) {tracker.thr.to_json()}")
    fixed = vla_state.CUE_FIXED_STATS if mode == "xyv" else {}

    # 統計値: 有限で、正規化で 0 で割らない
    stats = ds.meta.stats
    for n, want in fixed.items():
        i = names_state.index(n)
        got = {k: float(np.asarray(stats["observation.state"][k])[i]) for k in want}
        check(got == want, f"observation.state {n} stats fixed to {want} (got {got})")
    for key in ("observation.state", "action"):
        mean = np.asarray(stats[key]["mean"], dtype=np.float64)
        std = np.asarray(stats[key]["std"], dtype=np.float64)
        check(np.all(np.isfinite(mean)) and np.all(np.isfinite(std)), f"{key} stats finite")
        names = ds.meta.features[key]["names"]
        tiny = [f"{n}={s:.2e}" for n, s in zip(names, std) if s < 1e-3]
        print(f"       {key} std: " + ", ".join(f"{n}={s:.4g}" for n, s in zip(names, std)))
        if tiny:
            print(f"       {key} dims with std < 1e-3 (normalization amplifies noise): {tiny}")

    # 全フレーム: 形・型・task・正規化後が有限か。元の記録と比べる
    exports = {}
    max_norm = {"observation.state": 0.0, "action": 0.0}
    # 視点ごとに、期待する変換がほかのどの変換とも違う結果になるフレームの数。0 なら（対称な画像など）
    # 変換の誤りを見つけられない
    discriminating = {view: 0 for view in spec.CAMERAS}
    check_frame = None
    for src in conv["sources"]:
        meta, data = load_raw(pathlib.Path(src["raw_path"]))
        arr = episode_arrays(meta, data, conv.get("self_label_clean"))
        if "clean" in arr or "self_clean" in src:
            check(arr.get("clean") == src.get("self_clean"), f"ep {src['episode_index']}: self label clean {arr.get('clean')}")
        if tracker is not None:            # 目標位置の入力を生の俯瞰画像から計算し直して、期待する 18 次元を作る
            etr = episode_tracker(meta, tracker)
            etr.reset()
            cues = np.array([etr.update(read_raw_image(pathlib.Path(src["raw_path"]) / "overhead" / f"{int(i):06d}.png"),
                                            meta["target"]) for i in arr["raw_index"]])
            arr["state"] = vla_state.with_cue(arr["state"], cues, keep_flag=mode == "xyv")
            check(int((cues[:, 2] < 0.5).sum()) == src.get("cue_flag0_frames", int((cues[:, 2] < 0.5).sum())),
                  f"ep {src['episode_index']}: cue flag-0 frames recorded {src.get('cue_flag0_frames')}")
        if place_cue:                      # 置く場所の正解を生の記録から取り直して、期待する状態の後ろに足す
            place = place_label(meta, data)
            check(src.get("place") == [float(v) for v in place], f"ep {src['episode_index']}: place {src.get('place')}")
            arr["state"] = vla_state.with_place(arr["state"], place)
        ep = src["episode_index"]
        start = int(ds.meta.episodes[ep]["dataset_from_index"])
        stop = int(ds.meta.episodes[ep]["dataset_to_index"])
        check(stop - start == src["frames"], f"ep {ep}: {stop - start} frames")
        acts, states = [], []
        img_err = 0
        for k in range(stop - start):
            item = ds[start + k]
            if k == 0:
                check(item["task"] == meta["instruction"], f"ep {ep}: task '{item['task']}'")
                if meta.get("target"):        # 3 色の場面: task の色がメタデータの目標の色と一致する（書式から作った文も可）
                    check(instruction_ok(item["task"], meta) and src.get("target") == meta["target"],
                          f"ep {ep}: task color == meta target '{meta['target']}'")
                check(item["observation.state"].dtype == torch.float32
                      and tuple(item["observation.state"].shape) == (len(names_state),),
                      f"ep {ep}: state float32 ({len(names_state)},)")
                h, w, c = ds.meta.features[IMAGE_KEYS["overhead"]]["shape"]
                check(tuple(item[IMAGE_KEYS["overhead"]].shape) == (c, h, w),
                      f"ep {ep}: image tensor ({c}, {h}, {w})")
            st = item["observation.state"].numpy()
            ac = item["action"].numpy()
            states.append(st)
            acts.append(ac)
            for key, v in (("observation.state", st), ("action", ac)):
                z = (v - np.asarray(stats[key]["mean"])) / (np.asarray(stats[key]["std"]) + ZERO_STD_EPS)
                max_norm[key] = max(max_norm[key], float(np.abs(z).max()))
            if k % 10 == 0 or k == stop - start - 1 or (ep == 0 and k == (stop - start) // 2):  # 抜き取りのフレーム
                i = int(arr["raw_index"][k])
                raw_imgs, got_imgs = {}, {}
                for view in spec.CAMERAS:
                    key = IMAGE_KEYS[view]
                    got = (item[key].numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
                    raw = read_raw_image(pathlib.Path(src["raw_path"]) / view / f"{i:06d}.png")
                    want = spec.to_policy_image(view, raw)
                    img_err = max(img_err, int(np.abs(got.astype(np.int16) - want.astype(np.int16)).max()))
                    others = [t for t in spec.NET_TRANSFORMS if t != spec.VIEW_TRANSFORMS[view]]
                    if all(not np.array_equal(want, spec.apply_net_transform(t, raw)) for t in others):
                        discriminating[view] += 1
                    raw_imgs[view], got_imgs[view] = raw, got
                if ep == 0 and k == (stop - start) // 2:
                    check_frame = (raw_imgs, got_imgs, f"{out.name} ep 0 frame {k} (raw frame {i})")
        states, acts = np.array(states), np.array(acts)
        check(np.allclose(states, arr["state"], atol=1e-6), f"ep {ep}: state == raw (10 fps)")
        check(np.allclose(acts, arr["action"], atol=1e-7), f"ep {ep}: action == raw x_des differences")
        check(img_err == 0, f"ep {ep}: images == vla_image_spec(raw PNG) per view (max diff {img_err})")
        check(set(np.unique(acts[:, 6]).tolist()) <= {-1.0, 1.0}, f"ep {ep}: gripper in {{-1, +1}}")
        check(np.all(acts[:, 3:6] == 0.0), f"ep {ep}: rotation actions all zero")
        jumps = int((np.abs(np.diff(states[:, 3:6], axis=0)) > 0.5).sum())
        check(jumps == 0, f"ep {ep}: orientation continuous (jumps {jumps})")
        exports[src.get("raw_episode") or episode_label(meta)] = {"action": acts, "state": states, "raw_path": src["raw_path"]}
    for view, n in discriminating.items():
        check(n > 0, f"{view}: {n} sampled frame(s) where {spec.VIEW_TRANSFORMS[view]} differs from every other "
                     f"net transform (the image check can detect a wrong transform)")
    if check_frame is not None:
        path = pathlib.Path(check_image) if check_image else out.parent / f"{out.name}_verify_raw_vs_policy.png"
        write_check_image(path, *check_frame)
        print(f"[verify] raw vs policy input image: {path}")
    for key, v in max_norm.items():
        check(np.isfinite(v), f"{key} normalized values finite (max |z| {v:.1f})")
    # LeRobot の float32 の分散（2 乗の平均 − 平均の 2 乗）が float64 と一致するか。桁落ちを全次元で見つける
    for key, field in (("observation.state", "state"), ("action", "action")):
        allv = np.concatenate([e[field] for e in exports.values()]).astype(np.float64)
        true_std = allv.std(axis=0)
        got = np.asarray(stats[key]["std"], dtype=np.float64)
        bad = [n for n, t, g in zip(ds.meta.features[key]["names"], true_std, got)
               if n not in fixed and abs(t - g) > max(1e-6, 1e-2 * t)]   # 固定した次元は上で確かめた
        check(not bad, f"{key} stats std == float64 std (mismatch: {bad})")

    if export_dir:
        export_dir = pathlib.Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        for eid, e in exports.items():
            np.savez(export_dir / f"{eid}_lerobot.npz", action=e["action"],
                     state=e["state"], fps=FPS, raw_path=e["raw_path"])
        print(f"[verify] exported actions for the 10 fps replay check to {export_dir}")
    print("[verify] " + ("PASS" if ok else "FAIL"))
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="raw episodes -> LeRobot v3.0 dataset")
    ap.add_argument("episodes", nargs="*", help="raw episode directories")
    ap.add_argument("--raw-dir", help="convert every saved episode under this raw root")
    ap.add_argument("--manifest", action="append", default=None,
                    help="convert exactly the episodes listed in this manifest (B_提案書 §6). Repeat to concatenate "
                         "manifests into one dataset (e.g. the R data + cup_plate)")
    ap.add_argument("--out", help="output dataset directory (must not exist)")
    ap.add_argument("--name", help="dataset name (repo_id local/NAME)")
    ap.add_argument("--verify", help="verify an existing converted dataset directory")
    ap.add_argument("--export-actions", help="with --verify: write per-episode action npz here")
    ap.add_argument("--check-image", help="with --verify: raw-vs-policy-input PNG path "
                                          "(default: next to the dataset, <name>_verify_raw_vs_policy.png)")
    ap.add_argument("--target-cue", action="store_true",
                    help="add the target position cue to observation.state (15 -> 18; board 0048)")
    ap.add_argument("--cue-without-flag", action="store_true",
                    help="with --target-cue: leave the visibility flag out of the input (15 -> 17; board 0050-2)")
    ap.add_argument("--cue-thresholds", default="training", choices=CUE_THRESHOLD_SOURCES,
                    help="with --target-cue: color thresholds of the cue (runtime_v2 for stage 2; board 0115)")
    ap.add_argument("--self-label-clean", action="store_true",
                    help="R2: for the policy's own episodes (kind S) drop the start hold and short gripper flips (SELF_CLEAN)")
    ap.add_argument("--place-cue", action="store_true",
                    help="R2 stage P: add the place cue (where the target came to rest, 2 values) after the target cue")
    args = ap.parse_args(argv)
    if args.verify:
        return 0 if verify(pathlib.Path(args.verify), args.export_actions, args.check_image) else 1
    episodes = [pathlib.Path(p) for p in args.episodes]
    manifest = None
    if args.manifest:
        manifest = merge_manifests([json.loads(pathlib.Path(m).read_text(encoding="utf-8")) for m in args.manifest])
        episodes += manifest_episodes(manifest)
    if args.raw_dir:
        episodes += raw_episodes(pathlib.Path(args.raw_dir))
    if not episodes or not args.out or not args.name:
        ap.error("need --out, --name and episodes (paths, --manifest or --raw-dir)")
    mode = ("xy" if args.cue_without_flag else "xyv") if args.target_cue else None
    convert(episodes, pathlib.Path(args.out), args.name, manifest, target_cue=mode, cue_source=args.cue_thresholds,
            self_clean=SELF_CLEAN if args.self_label_clean else None, place_cue=args.place_cue)
    return 0


if __name__ == "__main__":
    sys.exit(main())
