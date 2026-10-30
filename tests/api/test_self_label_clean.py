"""convert の --self-label-clean（自分の成功のデータ（kind S）だけ、始めの静止と短い開閉をラベルから外す）。"""
import numpy as np

from recovla.data import convert as C


def test_smooth_grip_merges_short_inner_runs_only():
    g = np.array([-1, -1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, -1, 1], float)
    out = C.smooth_grip(g, 3)
    assert out.tolist() == [-1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1]   # 端の部分は残す


def test_self_clean_trims_lead_hold_and_keeps_the_rest():
    n = 10
    action = np.zeros((n, 7), np.float32)
    action[:, 6] = -1
    action[3:, 2] = -0.005                       # 3 フレーム目から動く
    action[6, 6] = 1                             # 1 フレームだけの閉
    arr = {"raw_index": np.arange(n) * 2, "state": np.arange(n)[:, None].astype(np.float32), "action": action}
    out = C.self_clean_arrays(arr, C.SELF_CLEAN)
    assert out["clean"] == {"lead_trimmed": 3, "grip_changed": 1}
    assert out["raw_index"].tolist() == [6, 8, 10, 12, 14, 16, 18]
    assert np.all(out["action"][:, 6] == -1) and np.all(out["action"][:, 2] == -0.005)


def test_only_listed_kinds_are_cleaned():
    meta = {"record_hz": C.RAW_HZ, "n_frames": 7, "kind": "n"}
    data = {"ee_pos": np.zeros((7, 3)), "ee_quat": np.tile([0, 1, 0, 0], (7, 1)).astype(float), "fingers": np.zeros((7, 2)),
            "joints": np.zeros((7, 7)), "gripper_closed": np.zeros(7, bool), "x_des": np.zeros((7, 3))}
    assert "clean" not in C.episode_arrays(meta, data, C.SELF_CLEAN)
    assert C.episode_arrays(dict(meta, kind="S"), data, C.SELF_CLEAN)["clean"]["lead_trimmed"] == 0   # 動かないなら落とさない
