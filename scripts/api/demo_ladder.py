"""動画の場面を撮り直す。評価の入口をそのまま呼び、発表用カメラ（1280×720、30 fps、シミュレーション時刻に同期）の描画だけを足す。

    .venv\\Scripts\\python.exe scripts\\api\\demo_ladder.py record --clip <場面> --entry ladder -- run --task cup_plate --arm U0 ...
    .venv\\Scripts\\python.exe scripts\\api\\demo_ladder.py record --clip <場面> --entry v3real -- --world v3_real run ...
    .venv\\Scripts\\python.exe scripts\\api\\demo_ladder.py compare --clip <場面> --source outputs/v2eval/<実験>/<条件>

--entry ladder は scripts/api/ladder.py の main に、v3real は実機の Panda で測った応答に合わせたシミュレーションの評価の入口
（scripts/real/eval_real_matched.py）に、-- の後ろの引数をそのまま渡す（--experiment は V4DEMO、--condition は <場面> に置き換える。
--trials は 1 本だけ）。v3real で別のチェックポイントを回すときは --ckpt Rreal=<チェックポイント> を付ける（読み込んだチェックポイントの表 CKPT にだけ足す）。
描画は物理の 1 ステップの後に行い、物理と方策の入力（センサのモデル）には触れない。腕のサーボの差し替え（harness/arm_real.install）は
世界を作るときに physics_step・reset を掴むので、描画の差し込みは入口を呼ぶ前に入れる。
GPU の描画と推論の揺れのため、回し直しは元の記録とビット一致しない。compare で元の記録と同じ経過（成否・終わりの時刻）かを
判定して meta.json に書き、same_course が真のものだけを動画に使う。
出力: outputs/demo_v4/<場面>/trial_00.mp4・trial_00_frames.npz・meta.json（評価の記録は outputs/v2eval/V4DEMO/<場面>/）
"""
import argparse
import importlib.util
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from recovla.common import config  # noqa: E402

CFG = config.load_v2()
OUT = ROOT / "outputs" / "demo_v4"
EVAL = ROOT / "outputs" / "v2eval"
EXP = "V4DEMO"
FPS = 30
ENTRIES = {"ladder": ROOT / "scripts" / "api" / "ladder.py", "v3real": ROOT / "scripts" / "real" / "eval_real_matched.py"}


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def free_bodies(model) -> list:
    """自由関節を持つ体（立方体・カップ）の番号。"""
    return [int(model.jnt_bodyid[j]) for j in range(model.njnt) if int(model.jnt_type[j]) == 0]


class Presentation:
    """発表用カメラで、シミュレーション時刻 k/FPS を最初に過ぎた物理の 1 ステップの後に 1 フレーム描く。"""

    def __init__(self, rig, path):
        import av
        import mujoco
        pc = CFG["scene"]["presentation_camera"]
        self.w, self.h = int(pc["width"]), int(pc["height"])
        self.renderer = mujoco.Renderer(rig.model, self.h, self.w)
        self.cam = rig.model.camera(pc["name"]).id
        self.objs = free_bodies(rig.model)
        self.box = av.open(str(path), mode="w")
        self.stream = self.box.add_stream("libx264", rate=FPS)
        self.stream.width, self.stream.height, self.stream.pix_fmt = self.w, self.h, "yuv420p"
        self.stream.options = {"crf": "16", "preset": "medium"}
        self.rows, self.t0, self.next_i = [], None, 0

    def step(self, rig) -> None:
        import av
        from recovla.sim import frames
        t = float(rig.data.time)
        if self.t0 is None:
            self.t0 = t
        if t - self.t0 + 1e-9 < self.next_i / FPS:
            return
        s = rig.forward_scratch()
        self.renderer.update_scene(s, camera=self.cam)
        f = av.VideoFrame.from_ndarray(np.ascontiguousarray(self.renderer.render()), format="rgb24")
        for p in self.stream.encode(f):
            self.box.mux(p)
        self.rows.append({"t": t - self.t0, "fingertip": frames.fingertip_center(s, rig.hand_id).tolist(),
                          "obj_pos": s.xpos[self.objs].tolist()})
        self.next_i += 1

    def close(self, npz) -> None:
        for p in self.stream.encode():
            self.box.mux(p)
        self.box.close()
        self.renderer.close()
        r = self.rows
        np.savez(npz, t=np.array([x["t"] for x in r]), fingertip=np.array([x["fingertip"] for x in r]),
                 obj_pos=np.array([x["obj_pos"] for x in r]))


def install(clip_dir: pathlib.Path):
    """WorldRig の reset ごとに新しい動画を開き、physics_step の後に描く（評価のコードは変えない）。"""
    from recovla.harness.world import WorldRig
    state = {"pres": None, "i": 0}
    orig_reset, orig_step = WorldRig.reset, WorldRig.physics_step

    def finish():
        if state["pres"] is not None:
            state["pres"].close(clip_dir / f"trial_{state['i'] - 1:02d}_frames.npz")
            state["pres"] = None

    def reset(self, layout):
        finish()
        orig_reset(self, layout)
        state["pres"] = Presentation(self, clip_dir / f"trial_{state['i']:02d}.mp4")
        state["i"] += 1

    def physics_step(self, on_step=None):
        orig_step(self, on_step)
        if state["pres"] is not None:
            state["pres"].step(self)
    WorldRig.reset, WorldRig.physics_step = reset, physics_step
    return finish


def _replace(argv: list, flag: str, value: str) -> list:
    out, skip = [], False
    for i, x in enumerate(argv):
        if skip:
            skip = False
            continue
        if x == flag:
            skip = True
            continue
        out.append(x)
    return out + [flag, value]


def cmd_record(a, rest: list) -> int:
    clip_dir = OUT / a.clip
    if clip_dir.exists() and any(clip_dir.glob("*.mp4")):
        raise SystemExit(f"{clip_dir} にもう動画がある（消してから回す）")
    trials = rest[rest.index("--trials") + 1] if "--trials" in rest else ""
    if not trials or not trials.split(":")[-1] == "1":
        raise SystemExit("--trials は 1 本だけ（例 170012:1、natural:163005:1）")
    clip_dir.mkdir(parents=True, exist_ok=True)
    argv = _replace(_replace(rest, "--experiment", EXP), "--condition", a.clip)
    finish = install(clip_dir)
    if a.entry == "v3real":
        ev = _load(ENTRIES["v3real"], "demo_entry_v3real")
        r96 = ev.patch(ev.load_96(), "v3_real", None)
        ops, v82 = r96.load_ops(), r96.load_82(False)
        for x in a.ckpt:
            k, v = x.split("=", 1)
            v82.CKPT[k] = v
        rc = r96.cmd_main(r96.build_parser().parse_args(argv), v82, ops)
    else:
        rc = _load(ENTRIES[a.entry], f"demo_entry_{a.entry}").main(argv)
    finish()
    (clip_dir / "record.json").write_text(json.dumps({"clip": a.clip, "entry": a.entry, "argv": argv, "rc": rc},
                                                      ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"clip": a.clip, "entry": a.entry, "argv": argv, "rc": rc}, ensure_ascii=False))
    return int(rc or 0)


def outcome(m: dict) -> dict:
    """試行の記録の成否と終わりの時刻（3 色の箱・カップの皿・1 色の試行のどれでも）。"""
    if "final_in_place" in m and m.get("final_in_place"):
        ok = all(bool(v) for v in m["final_in_place"].values())
        return {"success": ok, "t": float(m.get("t_end") or 0.0), "kind": "place"}
    if "final_in_box" in m and m.get("final_in_box") and "target" not in m:
        ok = all(bool(v) for v in m["final_in_box"].values())
        return {"success": ok, "t": float(m.get("t_end") or 0.0), "kind": "task"}
    return {"success": bool(m.get("success")), "t": float(m.get("t_success") or m.get("t_end") or 0.0), "kind": "single"}


def _records(d: pathlib.Path) -> dict:
    rows = {}
    for i, p in enumerate(sorted(list(d.glob("run_[0-9][0-9][0-9][0-9].json")) + list(d.glob("trial_[0-9]*.json")))):
        m = json.loads(p.read_text(encoding="utf-8"))
        rows[(int(m["seed"]), str(m.get("target", "")))] = dict(outcome(m), trial=i)   # trial: 記録の順＝動画の trial_NN
    return rows


def cmd_compare(a) -> int:
    new, src = _records(EVAL / EXP / a.clip), _records(ROOT / a.source)
    out = []
    for k, r in new.items():
        s = src.get(k)
        same = s is not None and s["success"] == r["success"] and (not r["success"] or abs(s["t"] - r["t"]) <= a.tol_s)
        out.append({"trial": r["trial"], "seed": k[0], "target": k[1], "rerun": r, "source": s, "same_course": bool(same)})
    meta = {"clip": a.clip, "source": a.source, "tol_s": a.tol_s, "rows": out,
            "note": "同じ設定で回し直した映像。ビット一致ではない。same_course が真のものだけを動画に使う"}
    (OUT / a.clip / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=1))
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    rest = []
    if "--" in argv:
        i = argv.index("--")
        argv, rest = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("record")
    p.add_argument("--clip", required=True)
    p.add_argument("--entry", required=True, choices=sorted(ENTRIES))
    p.add_argument("--ckpt", action="append", default=[], metavar="NAME=DIR", help="v3real だけ: 82 の CKPT に足す保存点")
    p = sub.add_parser("compare")
    p.add_argument("--clip", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--tol-s", type=float, default=3.0)
    a = ap.parse_args(argv)
    return cmd_record(a, rest) if a.cmd == "record" else cmd_compare(a)


if __name__ == "__main__":
    sys.exit(main())
