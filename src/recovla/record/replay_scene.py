"""3 色の場面のエピソードの再生確認（recovla.record.replay と同じ方式）。

保存した開始状態（data.npz の start_*、recovla.record.snapshot）を戻し、記録した x_des とグリッパの指令を、
同じ追従器と IK に通して手先の軌跡を比べる。方式（recovla.record.replay._schedule）:

    step      物理ステップごとの x_des とグリッパの指令。ビット一致するはず（開始状態と記録が揃っている確認）
    window    20 Hz の行動（x_des の差を 25 ステップに均等に配る）。手先 5 mm 以内
    actions   10 fps の行動（LeRobot から読み戻したものなど）
    last_step, shifted   誤った定義（陰性対照）。許容を超えなければならない

エピソードのフォルダには書かない。
"""
import numpy as np

from recovla.common import config
from recovla.common.seeds import COLORS
from recovla.expert.script import PhaseParams
from recovla.record import episode as E
from recovla.record import replay as legacy
from recovla.record import snapshot
from recovla.sim import frames

WINDOW_TOL_M = legacy.WINDOW_TOL_M
IMAGE_TOL = legacy.IMAGE_TOL
MODES = legacy.MODES


def replay(ep: legacy.Episode, mode: str, rig, actions=None, frame_stride: int = 1, render: bool = True) -> dict:
    """rig: recovla.sim.rig.SimRig（場面は 3 色）。返り値はフレームの配列（FRAME_FIELDS）と画像。"""
    pad = legacy.ScriptedPad(rig.controller)
    snap = snapshot.SimSnapshot.from_arrays(ep.data)
    snapshot.restore(rig.model, rig.data, rig.controller, pad, snap)
    rig.step = 0
    rig.meter.reset_window()
    target = ep.meta["target"]
    pp = PhaseParams.from_config()
    every = int(ep.meta["record_every"])
    cams = tuple(ep.meta["cameras"]["names"]) if render else ()
    out_frames, images = [], {c: [] for c in cams}

    def capture():
        f, imgs = E.capture_frame(rig, target, 0.0, pp, render=bool(cams))
        out_frames.append(f)
        for c, im in zip(rig.cameras, imgs):
            if c in images:
                images[c].append(im)

    import mujoco
    from recovla.sim.rig import quiet
    capture()
    with quiet():
        for k, (x_des, closed) in enumerate(legacy._schedule(ep, mode, actions, frame_stride), start=1):
            pad.x_cmd = np.asarray(x_des, dtype=float)
            pad.closed_target = closed
            rig.controller.update(pad)
            mujoco.mj_step(rig.model, rig.data)
            rig.step += 1
            rig.meter.on_step(rig.data)
            if k % every == 0:
                capture()
    out = {name: np.array([f[name] for f in out_frames]) for name in E.FRAME_FIELDS}
    out["images"] = images
    t = COLORS.index(target)
    out["success"] = frames.in_box(out["cube_pos"][-1, t], rig.box)
    return out


def compare(ep: legacy.Episode, rep: dict, images: bool = True) -> dict:
    d = ep.data
    n = min(ep.n_frames, len(rep["ee_pos"]))
    ee_err = np.linalg.norm(rep["ee_pos"][:n] - d["ee_pos"][:n], axis=1)
    t = COLORS.index(ep.meta["target"])
    result = {
        "n_frames": n,
        "ee_err_max_m": float(ee_err.max()), "ee_err_mean_m": float(ee_err.mean()),
        "ee_err_final_m": float(ee_err[-1]),
        "joint_err_max_rad": float(np.abs(rep["joints"][:n] - d["joints"][:n]).max()),
        "x_des_err_max_m": float(np.abs(rep["x_des"][:n] - d["x_des"][:n]).max()),
        "gripper_mismatch_frames": int((rep["gripper_closed"][:n] != d["gripper_closed"][:n]).sum()),
        "target_final_err_m": float(np.linalg.norm(rep["cube_pos"][n - 1, t] - d["cube_pos"][n - 1, t])),
        "cubes_final_err_max_m": float(np.linalg.norm(rep["cube_pos"][n - 1] - d["cube_pos"][n - 1], axis=1).max()),
        "success_recorded": bool(ep.meta["success"]), "success_replayed": bool(rep["success"]),
        "bit_exact_state": bool(np.array_equal(rep["joints"][:n], d["joints"][:n])
                                and np.array_equal(rep["ee_pos"][:n], d["ee_pos"][:n])),
    }
    if images and rep["images"]:
        for cam, frames_ in rep["images"].items():
            worst, frac = 0, 0.0
            for i in range(n):
                diff = np.abs(frames_[i].astype(int) - ep.image(cam, i).astype(int))
                worst = max(worst, int(diff.max()))
                frac = max(frac, float((diff.max(axis=2) > IMAGE_TOL).mean()))
            result[f"img_{cam}_max_diff"] = worst
            result[f"img_{cam}_worst_frame_fraction_over_tol"] = frac
    return result


def verdict(mode: str, r: dict) -> bool:
    """合格か。step はビット一致、window・actions は手先 5 mm 以内、陰性対照は 5 mm 以上（検出できた）。"""
    if mode == "step":
        return r["bit_exact_state"] and all(v <= IMAGE_TOL for k, v in r.items() if k.endswith("_max_diff"))
    if mode in ("window", "actions"):
        return r["ee_err_max_m"] < WINDOW_TOL_M
    return r["ee_err_max_m"] >= WINDOW_TOL_M
