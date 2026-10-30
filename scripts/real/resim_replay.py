"""再生する指令の列のシミュレーションの応答を、実機の Panda で測った応答に合わせた設定（configs/real_v1.yaml の real_v1）で計算し直す。

    python scripts/real/resim_replay.py --src outputs/real/replay_v1 --out outputs/real/replay_v1_fit
    python scripts/real/resim_replay.py --src outputs/real/replay_v1_slow --out outputs/real/replay_v1_slow_fit [--hand]

- 流す指令（q_cmd）・グリッパーの指令・時刻は元の列のまま。変えるのは q_sim・dq_sim・ee_sim・ee_sim_q・width_sim だけ。
- 指令は開ループで流す（export_replay_slow.open_loop。データの生成と同じ WorldRig.apply_joint_commands → physics_step）。
  立方体は置かない（実機と同じ）。
- 腕: harness/arm_real.py の ArmServo（real_v1.arm_servo）。最初の指令の姿勢に腕を置いて始める（実機が止まっていた姿勢）。
- --hand: ハンドも harness/hand_real.py の RealTimingHand（real_v1.gripper）に替える。付けなければ既定のハンド。
- 実機との比べは compare_replay.py（列は replay_format.md と同じ）。manifest の variant に "+real_v1" を足す。
- CPU だけで動く（データの生成・学習はしない）。
"""
import argparse
import dataclasses
import datetime
import hashlib
import json
import pathlib
import shutil
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import export_replay as X  # noqa: E402
import export_replay_slow as S  # noqa: E402
from real_to_ds import read_csv  # noqa: E402


def overlay_config() -> dict:
    """既定の設定の層に、生成プログラムの設定（expert_v3）と実機に合わせた設定（real_v1）を重ねる。"""
    from recovla.common import config
    return config.load(*[n for n in config.V2_LAYERS if (config.CONFIG_DIR / f"{n}.yaml").is_file()], "expert_v3", "real_v1")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hand", action="store_true")
    ap.add_argument("--only", default=None)
    a = ap.parse_args(argv)
    from recovla.expert import generate as G
    from recovla.harness import arm_real

    src, out = ROOT / a.src, ROOT / a.out
    if out.exists() and any(out.iterdir()):
        print(f"書き出し先 {out} が空でない")
        return 2
    out.mkdir(parents=True, exist_ok=True)
    man = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    cfg = overlay_config()
    servo = arm_real.ArmServo.from_config(cfg["real_v1"]["arm_servo"])
    G._init_worker(False, "v3")
    rig = G._RIG
    fk = X.FK(rig.motion)
    hand_info = None
    if a.hand:
        from recovla.harness import hand_real
        timing = hand_real.RealTiming.from_config(cfg["real_v1"]["gripper"])
        hand_real.install(rig, timing)
        hand_info = dataclasses.asdict(timing)
    variant = (man.get("variant") or "normal") + "+real_v1" + ("+hand" if a.hand else "")
    res_man = {"schema": "recovery_vla.real_replay/1", "variant": variant,
               "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
               "git_head": X.git_head(), "source": {"dir": a.src, "variant": man.get("variant")},
               "arm_servo": dataclasses.asdict(servo),
               "hand_timing": hand_info,
               "config_sha256": hashlib.sha256((ROOT / "configs/real_v1.yaml").read_bytes()).hexdigest(),
               "sim_response": "open loop with configs/real_v1.yaml real_v1.arm_servo (harness/arm_real.py)"
                               + (" and real_v1.gripper (harness/hand_real.py)" if a.hand else "")
                               + "; arm placed at the first streamed command; no cubes",
               "trials": []}
    # replay_source は方策の指令の manifest（export_policy_replay.py）だけにある欄
    for k in ("q1_shift_rad", "q1_range_rad", "dt_s", "ee_convention", "limits", "time_scale", "replay_source"):
        if k in man:
            res_man[k] = man[k]
    try:
        for t in man["trials"]:
            if not t.get("exported") or (a.only and t["name"] not in a.only.split(",")):
                continue
            name = t["name"]
            r = read_csv(src / t["csv"])
            q_cmd = np.stack([np.asarray(r[f"q_cmd{j}"], float) for j in range(7)], axis=1)
            ev = S.events_from_csv(src / t["gripper_csv"], 1.0)
            saved = arm_real.install(rig, servo)
            q_first = S.to_sim(q_cmd[0])
            base_reset = rig.reset

            # 配置を戻した後、腕を最初の指令の姿勢に置く
            def reset(layout, _base=base_reset, _q=q_first):
                _base(layout)
                arm_real.place_arm(rig, _q)
            rig.reset = reset
            try:
                o = S.open_loop(rig, t["seed"], t["layout_kind"], S.to_sim(q_cmd), ev, False)
            finally:
                rig.__dict__.pop("reset", None)
                arm_real.restore(rig, saved)
            n_main = len(q_cmd) - int(round(float(t["tail_s"]) / X.DT))
            res = {"cmds": S.to_sim(q_cmd), "meas": o["meas"], "n_main": n_main, "t0": o["t0"],
                   "summary": {"kind": t["kind"], "t_record_start": t.get("t_record_start")}}
            rows, cmds_r, _ = X.build_rows(res, fk)
            if np.abs(cmds_r - np.round(q_cmd, X.DECIMALS)).max() > 1e-9:
                raise RuntimeError(f"{name}: 指令の列が元と違う")
            entry = {k: t[k] for k in ("name", "kind", "seed", "color", "retry", "instruction", "layout_kind", "start_pose",
                                       "start_q_real", "start_ee_real", "stream_s", "tail_s") if k in t}
            entry.update({"source_csv_sha256": t["csv_sha256"], "t_record_start": t.get("t_record_start"),
                          "csv": f"{name}.csv", "gripper_csv": f"{name}_gripper.csv", "exported": True})
            entry["csv_sha256"] = X.write_csv(out / entry["csv"], rows, X.COLS)
            shutil.copyfile(src / t["gripper_csv"], out / entry["gripper_csv"])
            entry["gripper_csv_sha256"] = hashlib.sha256((out / entry["gripper_csv"]).read_bytes()).hexdigest()
            res_man["trials"].append(entry)
            print(f"{name}: 付け直し {len(rows)} 行", flush=True)
    finally:
        rig.close()
    res_man["n_exported"] = len(res_man["trials"])
    (out / "manifest.json").write_text(json.dumps(res_man, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{res_man['n_exported']} 本 → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
