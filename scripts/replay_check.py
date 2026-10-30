"""記録したエピソードを再生し、記録と一致するかを確かめる（recovla.record.replay を使う）。

    .venv\\Scripts\\python.exe scripts\\replay_check.py EPISODE_DIR [...]
        [--modes step window last_step shifted] [--out DIR]

再生の方式ごとに、手先の軌道と画像の差を表示する。--out を付けると、エピソードごとに report.json と
記録・再生を並べた動画も書く。エピソードのフォルダは読むだけ。
"""
import argparse
import json
import pathlib
import shutil
import sys
import tempfile

import cv2
import mujoco
import numpy as np

from recovla.record import replay
from recovla.sim import control, render

VIDEO_FPS = round(1.0 / (control.RECORD_EVERY * control.CFG["sim"]["timestep"]))    # 20（記録のフレームの速さ, fps）


def verdict(mode: str, r: dict) -> str:
    if mode == "step":
        ok = r["bit_exact_state"] and all(
            r[k] <= replay.IMAGE_TOL for k in r if k.endswith("_max_diff"))
        return "PASS (bit-exact)" if ok else "FAIL"
    if mode == "window":
        return "PASS" if r["ee_err_max_m"] < replay.WINDOW_TOL_M else "FAIL"
    # わざとずらした再生（陰性の対照）は、許容の誤差より明らかに大きくなるはず
    return ("detected (as expected)" if r["ee_err_max_m"] >= replay.WINDOW_TOL_M
            else "NOT detected -- the check is too weak")


def write_video(ep, rep, path: pathlib.Path) -> None:
    cams = list(rep["images"])
    size = (2 * control.IMAGE_SIZE, len(cams) * control.IMAGE_SIZE)  # 幅, 高さ
    # Windows の cv2.VideoWriter は ASCII 以外のパスを開けないので、一時の場所に書いて移す
    tmp = pathlib.Path(tempfile.mkdtemp()) / "replay.mp4"
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, size)
    for i in range(ep.n_frames):
        rows = [np.hstack([ep.image(c, i), rep["images"][c][i]]) for c in cams]
        vw.write(cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
    vw.release()
    shutil.move(str(tmp), str(path))


def lerobot_check(npz_files) -> int:
    """LeRobot のデータセットから読み戻した行動（10 fps、各 0.1 s）を同じ追従と逆運動学で再生する。
    手先が元の軌道に WINDOW_TOL_M 以内で従い、データセットの状態が元の記録と一致すれば合格。"""
    model = mujoco.MjModel.from_xml_path(control.SCENE_PATH)
    renderer = mujoco.Renderer(model, control.IMAGE_SIZE, control.IMAGE_SIZE)
    failed = False
    try:
        for f in npz_files:
            z = np.load(f, allow_pickle=False)
            stride = int(20 // int(z["fps"]))
            ep = replay.load_episode(str(z["raw_path"]))
            rep = replay.replay(ep, "actions", renderer, model, actions=z["action"],
                                frame_stride=stride)
            raw_idx = np.arange(len(z["action"]) + 1) * stride
            ee_err = np.linalg.norm(rep["ee_pos"][raw_idx] - ep.data["ee_pos"][raw_idx], axis=1)
            state_err = float(np.abs(z["state"][:, :3] - ep.data["ee_pos"][raw_idx[:-1]]).max())
            cube = float(np.linalg.norm(rep["cube_pos"][raw_idx[-1]] - ep.data["cube_pos"][raw_idx[-1]]))
            ok = ee_err.max() < replay.WINDOW_TOL_M and state_err < 1e-6
            failed |= not ok
            print(f"{pathlib.Path(f).name}: {len(z['action'])} actions @ {int(z['fps'])} fps  "
                  f"ee max {ee_err.max() * 1000:.3f} mm  mean {ee_err.mean() * 1000:.3f} mm  "
                  f"cube {cube * 1000:.2f} mm  success {rep['success']} "
                  f"(recorded {ep.meta['success']})  dataset state vs raw {state_err:.1e} m  "
                  f"-> {'PASS' if ok else 'FAIL'}")
    finally:
        render.close_renderer(renderer)
    return 1 if failed else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("episodes", nargs="*")
    ap.add_argument("--modes", nargs="+", default=list(replay.MODES), choices=replay.MODES)
    ap.add_argument("--out", default=None)
    ap.add_argument("--lerobot-actions", nargs="+", default=None,
                    help="npz files from convert_to_lerobot.py --verify --export-actions; "
                         "replays the 10 fps actions read back from the LeRobot dataset "
                         "against the raw episode named in each file")
    args = ap.parse_args(argv)
    if args.lerobot_actions:
        return lerobot_check(args.lerobot_actions)
    if not args.episodes:
        ap.error("give episode directories or --lerobot-actions")

    model = mujoco.MjModel.from_xml_path(control.SCENE_PATH)
    renderer = mujoco.Renderer(model, control.IMAGE_SIZE, control.IMAGE_SIZE)
    failed = False
    try:
        for ep_dir in args.episodes:
            ep = replay.load_episode(ep_dir)
            print(f"\n== {ep.path}  ({ep.n_frames} frames, "
                  f"{ep.meta['duration_s']:.2f} s, success={ep.meta['success']})")
            report = {}
            for mode in args.modes:
                rep = replay.replay(ep, mode, renderer, model)
                r = replay.compare(ep, rep, images=mode in ("step", "window"))
                r["verdict"] = verdict(mode, r)
                report[mode] = r
                failed |= r["verdict"].startswith(("FAIL", "NOT"))
                imgs = "  ".join(f"{k[4:]}={v}" for k, v in r.items() if k.startswith("img_"))
                print(f"  {mode:9s} ee max {r['ee_err_max_m'] * 1000:8.3f} mm  "
                      f"mean {r['ee_err_mean_m'] * 1000:7.3f} mm  "
                      f"joints {r['joint_err_max_rad']:.2e} rad  "
                      f"grip mismatch {r['gripper_mismatch_frames']:3d}  "
                      f"cube {r['cube_final_err_m'] * 1000:6.2f} mm  "
                      f"success {r['success_replayed']}  -> {r['verdict']}")
                if imgs:
                    print(f"            images: {imgs}")
                if args.out and mode == "window":
                    out = pathlib.Path(args.out) / ep.path.name
                    out.mkdir(parents=True, exist_ok=True)
                    write_video(ep, rep, out / "recorded_vs_window_replay.mp4")
            if args.out:
                out = pathlib.Path(args.out) / ep.path.name
                out.mkdir(parents=True, exist_ok=True)
                (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    finally:
        render.close_renderer(renderer)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
