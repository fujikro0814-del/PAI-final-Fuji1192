"""色の判定の閾値を、学習用の配置（シード 1 万台）の記録だけで決める。結果は outputs/k1/color_fit.json。

    .venv\\Scripts\\python.exe scripts\\color_fit.py [--gen gen_100] [--episodes 100] [--every 8]

1. 学習用の 100 配置（シード 10000〜10099）の記録からフレームを間引き、記録の状態から MuJoCo の状態を組み直して
   俯瞰カメラのセグメンテーションを描く（記録の PNG と描き直した画像の差で同じ状態かを確かめる）
2. 画素の分類（recovla.perception.color.color_mask）の閾値 min_value・min_margin を、3 色の F1 の平均が最大の組にする
3. 画素の数の下限 min_pixels: 写っていないのに検出された画素の数の最大 + 1 と、机の上の立方体で
   投影した位置の誤差の 95 百分位が 2 cm 以下になる最小の数の、大きい方
"""
import argparse
import json
import pathlib
import time

import mujoco
import numpy as np

from recovla.common import config
from recovla.common.seeds import COLORS
from recovla.perception import color as PC
from recovla.record.recorder import read_png
from recovla.sim import frames, scene
from recovla.sim.rig import SimRig

CFG = config.load()
OUT = config.path(CFG["paths"]["outputs"]) / "k1"
ERR_P95_MAX = 0.02
ON_TABLE_TOL = 0.005          # m。静止の高さからこの範囲なら机の上


def set_state(rig, z, i, adr) -> None:
    d = rig.scratch
    d.qpos[:] = 0.0
    d.qpos[rig.arm_qadr] = z["joints"][i]
    d.qpos[rig.finger_qadr] = z["fingers"][i]
    for k, c in enumerate(COLORS):
        d.qpos[adr[c]:adr[c] + 3] = z["cube_pos"][i, k]
        d.qpos[adr[c] + 3:adr[c] + 7] = z["cube_quat"][i, k]
    mujoco.mj_forward(rig.model, d)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gen", default="gen_100")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--every", type=int, default=8, help="何こま（20 Hz）おきに取るか")
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    g = json.loads((OUT / f"{a.gen}.json").read_text(encoding="utf-8"))
    seeds_ = g.get("layout_seeds", [10000, 10029])
    if not (10000 <= seeds_[0] and seeds_[1] <= 19999):
        raise SystemExit(f"{a.gen}: layout seeds {seeds_} are not the training band 10000-19999")
    run = config.ROOT / g["run"]
    eps = sorted(p for p in run.iterdir() if p.is_dir() and not p.name.endswith(".partial"))
    eps = eps[::max(1, len(eps) // a.episodes)][:a.episodes]
    rig = SimRig(render=True)
    m = rig.model
    adr = {c: scene.cube_qpos_adr(m, c)[0] for c in COLORS}
    gid = {c: m.geom(frames.cube_geom(c)).id for c in COLORS}
    r = rig.renderer
    calib = PC.overhead_calibration(m)
    # 2 次元の度数分布: チャンネルの値 0..255 × 差 -255..255（色ごとに、正・負の画素）
    hist = {c: {s: np.zeros((256, 511), np.int64) for s in ("pos", "neg")} for c in COLORS}
    samples = []                      # (画像, 真の画素数, 立方体の中心)。min_pixels に使う
    rgb_diff_max = 0
    for p in eps:
        z = np.load(p / "data.npz")
        for i in range(0, len(z["step"]), a.every):
            set_state(rig, z, i, adr)
            r.update_scene(rig.scratch, camera=PC.OVERHEAD)
            rgb = r.render().copy()
            img = read_png(p / "overhead" / f"{i:06d}.png")
            rgb_diff_max = max(rgb_diff_max, int(np.abs(rgb.astype(int) - img.astype(int)).max()))
            r.enable_segmentation_rendering()
            r.update_scene(rig.scratch, camera=PC.OVERHEAD)
            seg = r.render()[:, :, 0].copy()
            r.disable_segmentation_rendering()
            a16 = img.astype(np.int16)
            for c in COLORS:
                ch = PC.CHANNEL[c]
                val = a16[:, :, ch]
                margin = val - np.max(np.delete(a16, ch, axis=2), axis=2)
                pos = seg == gid[c]
                for s, msk in (("pos", pos), ("neg", ~pos)):
                    np.add.at(hist[c][s], (val[msk].ravel(), margin[msk].ravel() + 255), 1)
            samples.append((img, {c: int((seg == gid[c]).sum()) for c in COLORS},
                            {c: z["cube_pos"][i, k].copy() for k, c in enumerate(COLORS)}))
    rig.close()
    # 閾値の網羅: 値 ≥ v かつ 差 ≥ g の画素の数 = 右下の累積和
    best = None
    grid = []
    for c in COLORS:
        for s in ("pos", "neg"):
            h = hist[c][s]
            hist[c][s + "_cum"] = h[::-1, ::-1].cumsum(0).cumsum(1)[::-1, ::-1]
    for v in range(20, 256, 5):
        for gm in range(5, 256, 5):
            f1s = []
            for c in COLORS:
                tp = hist[c]["pos_cum"][v, gm + 255]
                fp = hist[c]["neg_cum"][v, gm + 255]
                fn = hist[c]["pos"].sum() - tp
                f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
            row = (float(np.mean(f1s)), v, gm, [float(x) for x in f1s])
            grid.append(row)
            if best is None or row[0] > best[0]:
                best = row
    f1, v, gm, f1s = best
    thr0 = PC.Thresholds(v, gm, 1)
    # min_pixels
    fp_max = 0
    rows = []
    for img, true_px, centers in samples:
        for c in COLORS:
            det = PC.detect(img, c, thr0)
            if true_px[c] == 0:
                fp_max = max(fp_max, det["pixels"])
            elif det["pixels"] > 0 and centers[c][2] - frames.CUBE_REST_Z < ON_TABLE_TOL:
                # 投影は机の面の高さを仮定するので、机の上の立方体だけで測る
                xy = PC.pixel_to_plane(calib, det["u"], det["v"], frames.CUBE_REST_Z)
                rows.append((det["pixels"], float(np.linalg.norm(xy - centers[c][:2]))))
    rows = np.array(rows)
    n_err = None
    for n in range(1, 400):
        e = rows[rows[:, 0] >= n, 1]
        if e.size and np.percentile(e, 95) <= ERR_P95_MAX:
            n_err = n
            break
    min_pixels = max(fp_max + 1, n_err or 1)
    res = {"gen": a.gen, "layout_seeds": seeds_, "episodes": len(eps), "every": a.every, "frames": len(samples),
           "render_vs_recorded_png_max_diff": rgb_diff_max,
           "chosen": {"min_value": v, "min_margin": gm, "min_pixels": int(min_pixels)},
           "f1_mean": f1, "f1_by_color": dict(zip(COLORS, f1s)),
           "false_positive_pixels_max_when_absent": int(fp_max), "min_pixels_for_p95_err_le_2cm": n_err,
           "positive_pixels": {c: int(hist[c]["pos"].sum()) for c in COLORS},
           "top5": [{"f1": r_[0], "min_value": r_[1], "min_margin": r_[2]} for r_ in sorted(grid, reverse=True)[:5]],
           "calibration": calib.to_json(), "wall_s": round(time.perf_counter() - t0, 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "color_fit.json").write_text(json.dumps({"check": "color_fit", "written": time.strftime("%Y-%m-%d %H:%M:%S"),
                                                    **res}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("chosen", "f1_mean", "f1_by_color", "false_positive_pixels_max_when_absent",
                                         "min_pixels_for_p95_err_le_2cm", "frames", "render_vs_recorded_png_max_diff",
                                         "wall_s")}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
