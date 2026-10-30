"""前の版の説明資料（PDF）を結果のファイルから作り、照合する。数字は手で書かない。

使わない語の一覧 FORBIDDEN は、動画の照合（scripts/video_scenes.py check）と提出前の検査（scripts/check_submission.py）が読む。

    .venv\\Scripts\\python.exe scripts\\paper_pdf.py build     # 値の計算 → 図 2 → HTML → PDF（Edge のヘッドレス印刷）
    .venv\\Scripts\\python.exe scripts\\paper_pdf.py check     # 照合: 手で書いた数字がない・値が結果と一致・使わない語が 0・ページ数

原稿は paper/template.html と paper/fig1.svg。数字はすべて {{キー}} の差し込み口で、values が 1 つのサブタスクの評価の結果
（outputs/results/v2_e_report_s3.json、outputs/v2eval/V3S3、テスト用のシード 140000〜）と、設定・学習の記録・
固定したチェックポイントの一覧（docs/freeze/s3_hashes.json）・動画の場面の表（configs/demo/video_s3.yaml）から計算する。
出力: paper/build/（paper.html・図・values.json）と、paper/build/説明資料_PAI最終課題_<アカウント名>.pdf
"""
import argparse
import collections
import html
import json
import pathlib
import re
import subprocess
import sys

from recovla.common import config, goals

CFG = config.load_v2()
ROOT = config.ROOT
OUT = config.path(CFG["paths"]["outputs"])
RES = OUT / "results"
S3 = OUT / "v2eval" / "V3S3"
PAPER = ROOT / "paper"
BUILD = PAPER / "build"
VIDEO = ROOT / "configs" / "demo" / "video_s3.yaml"
FREEZE = ROOT / "docs" / "freeze" / "s3_hashes.json"
from recovla.common.account import ACCOUNT  # noqa: E402  omnicampus のアカウント名（ファイル名だけに入る。値は account.py だけに書く）
PDF_NAME = f"説明資料_PAI最終課題_{ACCOUNT}.pdf"
ALPHA = "0.05"                               # 有意水準（Holm 補正の後で p < 0.05）
GPU = "NVIDIA GeForce RTX 4080 SUPER"        # 推論時間の分布（configs/latency_v1.json）を測った機器（nvidia-smi で確認）
EXTRA = "extra_s3.json"                      # 結果を見た後に足した参考の集計（scripts/55_extra_s3.py）
EDGE = pathlib.Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
FONT = pathlib.Path(r"C:\Windows\Fonts\yumin.ttf")         # 図 2 の文字（本文と同じ游明朝）


def _j(p):
    return json.loads(pathlib.Path(p).read_text(encoding="utf-8"))


def _mod(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def pct(x):
    return f"{round(100 * x)}%"


def ci(w):
    return f"{round(100 * w[0])}〜{round(100 * w[1])}%"


def dci(d):
    lo, hi = d["diff_95ci_newcombe"]
    return f"{round(100 * lo):+d}〜{round(100 * hi):+d}".replace("+0〜", "0〜")


def pv(p):
    if p < 0.001:
        return "< 0.001"
    return f"{p:.2g}" if p < 0.1 else f"{p:.2f}"


def pv_eq(p):
    """本文で「p {{…eq}}」と書く形: 「= 0.016」または「< 0.001」。"""
    s = pv(p)
    return s if s.startswith("<") else f"= {s}"


def mmss(s):
    return f"{int(s) // 60}:{int(s) % 60:02d}"


def video_scenes() -> list:
    import yaml
    return yaml.safe_load(VIDEO.read_text(encoding="utf-8"))["scenes"]


def v3_dataset_dir() -> str:
    """立て直しデータを含む学習データ（固定したチェックポイントの一覧に入っているもの）。"""
    return _j(OUT / "f" / "data_v3.json")["datasets"]["R1"]["dataset"].replace("\\", "/")


def e7_summary() -> dict:
    """3 個の連続タスク（E7）の記録から: 3 個とも、やり直しで完了したサブタスク、完了判定の偽陰性・偽陽性、止まった位置。"""
    runs = [_j(p) for p in sorted((S3 / "E7_R1v3").glob("run_00??.json"))]
    out = {"n": len(runs), "all_three": 0, "retry_completed": 0, "false_neg": 0, "false_pos": 0, "stopped_at": collections.Counter(),
           "judged": 0, "not_judged": 0, "stopped_all_three": 0, "audit_viol": 0}
    for r in runs:
        out["all_three"] += bool(r["all_three_in_box"])
        out["audit_viol"] += int(r["audit"]["g1"]["violations"]) + int(r["audit"]["g3"]["total_violations"])
        for s in r["steps"]:
            truth = r["truth_success_t"].get(s["color"]) is not None
            if len(s["attempts"]) > 1 and truth:
                out["retry_completed"] += 1
            out["judged" if s["judged_complete"] else "not_judged"] += 1
            if s["judged_complete"] and not truth:
                out["false_pos"] += 1
            if not s["judged_complete"] and r["final_in_box"].get(s["color"]):
                out["false_neg"] += 1
        if r.get("stopped"):
            out["stopped_at"][int(r["stopped"]["step"])] += 1
            out["stopped_all_three"] += bool(r["all_three_in_box"])      # 3 個とも入っていたが、最後の完了判定が出ずに止まった
    return out


def e7_funnel() -> dict:
    """3 個の連続タスクの段ごとの内訳（箱の実際の状態 final_in_box で数える）。
    1 番目の色が箱に入った本数 → 2 番目の色も入った本数 → そのうち 2 番目の完了判定が出て 3 番目へ進んだ本数 → 3 番目の色も入った本数。
    完了判定が出ずに止まると残りの手順は行わないので、3 番目は「進んだ本」だけを分母にする。"""
    runs = [_j(p) for p in sorted((S3 / "E7_R1v3").glob("run_00??.json"))]
    order = list(runs[0]["plan"]["steps"])                               # LLM の分解（本文は「全試行で同じ順」と書く）
    assert all(list(r["plan"]["steps"]) == order for r in runs), "LLM の分解の順が試行で違う"
    assert all([s["color"] for s in r["steps"]] == order[:len(r["steps"])] for r in runs)
    assert len(order) == 3, order
    f1 = [r for r in runs if r["final_in_box"].get(order[0])]
    f2 = [r for r in f1 if r["final_in_box"].get(order[1])]
    f2go = [r for r in f2 if len(r["steps"]) >= 3]                       # 2 番目の完了判定が出て 3 番目を始めた
    assert all(r["steps"][1]["judged_complete"] for r in f2go)
    f3 = [r for r in f2go if r["final_in_box"].get(order[2])]
    return {"order": order, "n": len(runs), "f1": len(f1), "f2": len(f2), "f2go": len(f2go), "f3": len(f3)}


def e7_fn_reasons() -> collections.Counter:
    """完了判定の取りこぼし（最終の判定が未完了なのに、箱の実際の状態では入っていた手順）の理由を、記録の判定の行から分ける。
    late: 立方体が箱の中で静止した時刻（truth_success_t）がない＝持ち時間の終わりの直前に入り、待機位置での確認が間に合わなかった。
    overhead: 箱の中で静止した後、手が待機位置にある間の俯瞰カメラの画素（中央値）が閾値に届かない。
    wrist: 俯瞰の画素は届いているのに判定が出ない（記録に残らない手首カメラの画素の割合の条件で落ちたと、消去法で推定）。"""
    jcfg = CFG["runtime_v2"]["judge"]
    tol, box_min = float(jcfg["retreat_tol_m"]), int(jcfg["box_min_pixels"])
    out = collections.Counter()
    for p in sorted((S3 / "E7_R1v3").glob("run_00??.json")):
        r = _j(p)
        rows = _j(p.with_name(p.stem + "_runtime.json"))["executor"]["judge_rows"]
        for s in r["steps"]:
            if s["judged_complete"] or not r["final_in_box"].get(s["color"]):
                continue
            tt = r["truth_success_t"].get(s["color"])
            if tt is None:
                out["late"] += 1
                continue
            at = [x for x in rows if x["step"] == s["step"] and x["t"] >= tt and x["retreat_dist_m"] <= tol]
            assert at, (p.name, s["step"])
            med = lambda k: sorted(x[k] for x in at)[len(at) // 2]
            if med("box_pixels") < box_min:
                out["overhead"] += 1
            elif med("wrist_in") >= int(jcfg["wrist_min_pixels"]) and sum(x["depth_ok"] for x in at) * 2 > len(at):
                out["wrist"] += 1                                         # 記録に残る条件はすべて満たす → 残るのは手首の画素の割合
            else:
                out["other"] += 1
    assert not out["other"], out                                          # 本文の三つの分類に入らないものがあれば止める
    return out


RTC_CMP = ("A_nat", "E_nat")      # 立て直しデータありの非同期実行の、RTC なし（A）と RTC（E）。意図的な失敗なしの 99 組


def rtc_mechanism() -> dict:
    """4.3 節の仕組みの指標（全試行。テスト用の記録 outputs/v2eval/V3S3 の A_nat・E_nat の読み取りのみ）。
    close: 最初にグリッパを閉じた時点の、指先と目標の立方体の水平の差のうち、ロボットの根元から立方体へ向かう方向の成分（負＝手前）の中央値 [mm]
    ratio: 最初に閉じるまでに方策が指令した水平移動の、その方向の成分 ÷ 必要な移動（開始時の指先から閉じた時点の立方体まで）の中央値
    seam / non: 方策の指令の速度の 0.1 s ごとの変化の大きさ（試行ごとの平均の中央値）を、行動列の境界とそれ以外で分けたもの [mm/s]"""
    import numpy as np
    colors = ("red", "green", "blue")
    out = {}
    for cond in RTC_CMP:
        close, ratio, seam, non = [], [], [], []
        for jf in sorted((S3 / cond).glob("trial_*.json")):
            m = _j(jf)
            n = int(jf.stem.split("_")[1])
            rt = _j(jf.parent / f"runtime_{n:04d}.json")["runtime"]
            z = np.load(jf.with_suffix(".npz"))
            ti = colors.index(m["target"].split(">")[0])
            t, tip, cube = z["sim_time"], z["fingertip"], z["cube_pos"][:, ti]
            closed = z["gripper_closed"].astype(bool)
            acts = rt["actions"]
            A = np.array([a["a"] for a in acts], float)
            kt = np.array([a["t"] for a in acts], float)
            held = np.array([a["held"] for a in acts], bool)
            ch = np.array([(-1 if a["chunk"] is None else a["chunk"]) for a in acts])
            v = A[:, :3] * float(CFG["convert"]["fps"])               # 行動は 1 ステップごとの位置の差 → 速度
            jump = np.linalg.norm(np.diff(v, axis=0), axis=1)
            both = ~held[1:] & ~held[:-1]
            if (both & (ch[1:] != ch[:-1])).any():
                seam.append(float(jump[both & (ch[1:] != ch[:-1])].mean()))
            if (both & (ch[1:] == ch[:-1])).any():
                non.append(float(jump[both & (ch[1:] == ch[:-1])].mean()))
            if not closed.any():
                continue
            k1 = int(np.flatnonzero(closed)[0])
            u = cube[k1, :2] / np.linalg.norm(cube[k1, :2])
            close.append(float((tip[k1, :2] - cube[k1, :2]) @ u * 1e3))
            need = float((cube[k1, :2] - tip[0, :2]) @ u)
            if abs(need) > 0.02:
                ratio.append(float(A[(kt < t[k1]) & ~held, :2].sum(axis=0) @ u) / need)
        out[cond] = {"close": float(np.median(close)), "ratio": float(np.median(ratio)),
                     "seam": float(np.median(seam)) * 1e3, "non": float(np.median(non)) * 1e3, "n": len(close)}
    return out


def values() -> dict:
    ee = _mod("e_eval", ROOT / "scripts" / "compare_subtask.py")
    st3 = _mod("v2e", ROOT / "scripts" / "eval_sets.py").STAGES["s3"]
    rep = _j(RES / "v2_e_report_s3.json")
    P, sec, sets = rep["primary"], rep["secondary"], rep["sets"]
    v = {}
    # 設定・学習の記録
    fps = int(CFG["convert"]["fps"])
    rows = int(st3["sets"]["A"][3])
    v["policy_hz"] = str(fps)
    v["exec_rows"] = str(rows)
    v["exec_s"] = f"{rows / fps:g}"
    v["act_dt_s"] = f"{1 / fps:g}"                               # 行動 1 ステップの時間（4.3 節の速度の変化の刻み）
    lat = _j(ROOT / "configs" / "latency_v1.json")["kinds"]["policy"]
    v["lat_p50"], v["lat_p95"] = f"{lat['p50']:.2f}", f"{lat['p95']:.2f}"
    ck = _j(FREEZE)["checkpoints"]
    pcfg = _j(ROOT / ck["R1v3"] / "pretrained_model" / "config.json")
    v["chunk"] = str(pcfg["chunk_size"])
    info = _j(ROOT / v3_dataset_dir() / "meta" / "info.json")
    v["act_dim"] = str(info["features"]["action"]["shape"][0])
    an = info["features"]["action"]["names"]                     # dx, dy, dz, drx, dry, drz, gripper
    v["act_pos"] = str(sum(n in ("dx", "dy", "dz") for n in an))
    v["act_rot"] = str(sum(n.startswith("dr") for n in an))
    v["act_grip"] = str(sum(n == "gripper" for n in an))
    sd = info["features"]["observation.state"]["shape"][0]
    cue = sum(1 for n in info["features"]["observation.state"]["names"] if n.startswith("cue_"))
    v["state_dim"], v["cue_dim"], v["state_base"] = str(sd), str(cue), str(sd - cue)
    sn = info["features"]["observation.state"]["names"]                # 状態の内訳（関節・手先の位置・姿勢のずれ・指）
    v["st_joint"] = str(sum(n.startswith("joint") for n in sn))
    v["st_pos"] = str(sum(n in ("eef_x", "eef_y", "eef_z") for n in sn))
    v["st_rot"] = str(sum(n.startswith("eef_rot") for n in sn))
    v["st_finger"] = str(sum(n.startswith("finger") for n in sn))
    v["cam_hz"] = f"{CFG['sensor']['frame_hz']:g}"
    v["phys_ms"] = f"{float(CFG['sim']['timestep']) * 1000:g}"
    v["train_steps"] = f"{int(pathlib.Path(ck['R1v3']).name):,}"
    v3 = _j(OUT / "f" / "data_v3.json")["datasets"]
    cnt = collections.Counter(e["key"].split("_")[0] for e in _j(ROOT / v3["R1"]["manifest"])["entries"])
    v["n_normal"] = str(cnt["n"])
    v["n_recovery"] = str(sum(c for k, c in cnt.items() if k != "n"))
    v["n_n1"] = str(len(_j(ROOT / v3["N1"]["manifest"])["entries"]))
    v["n_rec_grasp"], v["n_rec_drop"], v["n_rec_place"] = str(cnt["A"]), str(cnt["B"]), str(cnt["C"])   # 立て直しデータの種類（A 掴み損ね・B 運ぶ途中の落下・C 置き損ね）
    ia = CFG["inject"]["A"]
    v["inj_lat_lo_cm"], v["inj_lat_hi_cm"] = (f"{x * 100:g}" for x in ia["lateral_offset_m"])
    v["inj_raise_lo_cm"], v["inj_raise_hi_cm"] = (f"{x * 100:g}" for x in ia["raise_close_m"])
    sc = CFG["eval"]["success"]
    v["succ_speed_cms"], v["succ_hold_s"] = f"{float(sc['rest_speed']) * 100:g}", f"{float(sc['rest_hold_s']):g}"
    tcfg = _j(ROOT / ck["R1v3"] / "pretrained_model" / "train_config.json")          # 初期値（Hugging Face のキャッシュの名前から）
    m_init = re.search(r"models--([^\\/]+?)--([^\\/]+)", tcfg["policy"]["pretrained_path"])
    v["init_model"] = f"{m_init.group(1)}/{m_init.group(2)}"
    tcfg_n = _j(ROOT / ck["N1v3"] / "pretrained_model" / "train_config.json")
    for key in ("batch_size", "steps", "seed"):                                        # 表 2 は両モデル共通の値だけを書く
        assert tcfg[key] == tcfg_n[key], key
    assert tcfg["optimizer"]["lr"] == tcfg_n["optimizer"]["lr"]
    v["batch"], v["lr"] = str(tcfg["batch_size"]), f"{tcfg['optimizer']['lr']:g}"
    v["opt"] = tcfg["optimizer"]["type"].replace("adamw", "AdamW")
    v["warmup"] = f"{tcfg['scheduler']['num_warmup_steps']:,}"
    v["gpu"] = GPU
    info_n = _j(ROOT / v3["N1"]["dataset"] / "meta" / "info.json")
    v["frames_r"], v["frames_n"] = f"{info['total_frames']:,}", f"{info_n['total_frames']:,}"
    v["eps_r"], v["eps_n"] = str(info["total_episodes"]), str(info_n["total_episodes"])
    succ_r = {e["key"] for e in _j(ROOT / v3["R1"]["manifest"])["entries"] if e["key"].startswith("n_")}
    succ_n = {e["key"] for e in _j(ROOT / v3["N1"]["manifest"])["entries"]}
    assert succ_r <= succ_n                                                            # R1 の成功データは N1 の成功データの一部
    ts = float(CFG["sim"]["timestep"])
    v["phys_hz"] = f"{1 / ts:g}"
    sf = CFG["safety_filter"]
    extra = float(CFG["runtime_v2"]["safety_extra_margin_m"])     # 実行系の安全フィルタは最小距離と対象の距離の両方に足す
    v["sf_dmin_mm"] = f"{sf['d_min_m'] * 1000:g}"
    v["sf_extra_mm"] = f"{extra * 1000:.1f}"
    v["sf_dmin_tot_mm"] = f"{(sf['d_min_m'] + extra) * 1000:.1f}"
    v["sf_detect_tot_mm"] = f"{(sf['d_detect_m'] + extra) * 1000:.1f}"
    sfd = _j(RES / "v2_safety_main_s3_V3SF.json")                  # 検証用の確定判定（規則と、検証用での成功率の低下）
    v["sf_rule_pt"] = re.search(r"(\d+(?:\.\d+)?) ポイント", sfd["rule"]).group(1)
    v["sf_val_drop"] = f"{round(sfd['drop_points'])}"
    v["alpha"] = ALPHA
    v["offset_gain_mm"] = re.search(r"OFFSET_GAIN_MM = ([\d.]+)", (ROOT / "scripts" / "check_slow_approach.py").read_text(encoding="utf-8")).group(1).rstrip("0").rstrip(".")
    lo, hi = CFG["eval"]["P1"]["lateral_offset_m"]
    v["p1_lo_cm"], v["p1_hi_cm"] = f"{lo * 100:g}", f"{hi * 100:g}"
    v["p1_win_s"], v["p1_lift_cm"] = f"{float(CFG['eval']['P1']['confirm_window_s']):g}", f"{float(CFG['eval']['P1']['confirm_lift_m']) * 100:g}"
    v["sf_dmin_src_mm"] = re.search(r"1 パーセンタイル ([\d.]+) mm", (ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")).group(1)
    v["sf_detect_mm"] = f"{sf['d_detect_m'] * 1000:g}"
    v["judge_hold_s"] = f"{float(CFG['runtime_v2']['judge']['hold_s']):g}"
    v["time_limit"] = f"{float(CFG['eval']['time_limit_s']):g}"
    # 前の 2 つの版（s1・s2）のテストの記録（本文の注記に使う）
    stages = _mod("v2e_stages", ROOT / "scripts" / "eval_sets.py")
    for st in ("s1", "s2", "s3"):                                      # テスト用のシード範囲の始まり（README の全テストの一覧）
        v[f"seed_{st}"] = str(stages.STAGES[st]["base"])
    for part in ("P1", "P2", "P3"):
        v[f"seed_s3_{part.lower()}"] = stages.trials_of("s3", part).split(":")[1]
    v["seed_s3_e6"] = _j(S3 / "e6_R1v3_both.json")["trials"].split(":")[0]
    v["seed_s3_e7"] = str(min(_j(p)["seed"] for p in (S3 / "E7_R1v3").glob("run_00??.json")))
    v["sf_val_on"], v["sf_val_off"] = (str(round(sfd["success"][k] * sfd["success"]["pairs"])) for k in ("x_rate", "y_rate"))
    v["sf_val_con_on"], v["sf_val_con_off"] = (str(round(sfd["contact"][k] * sfd["contact"]["pairs"])) for k in ("x_rate", "y_rate"))
    v["sf_val_n"] = str(sfd["success"]["pairs"])
    for st in ("s1", "s2"):
        rp = _j(RES / f"v2_e_report_{st}.json")["primary"]
        v[f"{st}_e1_k"], v[f"{st}_e1_n"] = str(rp["E1"]["successes"]), str(rp["E1"]["n"])
        v[f"{st}_e2_k"], v[f"{st}_e2_n"] = str(rp["E2"]["recovered"]), str(rp["E2"]["established"])
        hp = rp["E3"]["main_A_vs_B"]["holm_p"]
        v[f"{st}_holm"] = "1.00" if hp >= 0.995 else f"{hp:.3g}"          # 0.125 を 0.12 と丸めない（README の値と同じ書き方）
        assert rp["E3"]["main_A_vs_B"]["holm_p"] >= float(ALPHA)         # どちらも主要評価項目の基準を満たさなかった
    # テスト用の試行の数
    v["nat_n"] = str(P["E1"]["n"])
    v["nat_layouts"] = str(P["E1"]["n"] // 3)
    v["p_n"] = str(sets["A"]["P1"]["n"])
    # 主要評価項目
    v["e1_k"], v["e1_n"] = str(P["E1"]["successes"]), str(P["E1"]["n"])
    v["e1_pct"], v["e1_ci"] = pct(P["E1"]["successes"] / P["E1"]["n"]), ci(P["E1"]["success_wilson"])
    v["e2_k"], v["e2_n"] = str(P["E2"]["recovered"]), str(P["E2"]["established"])
    v["e2_pct"], v["e2_ci"] = pct(P["E2"]["recovery_rate"]), ci(P["E2"]["recovery_wilson"])

    def pair(prefix, d, holm=None):
        v[f"{prefix}_x"], v[f"{prefix}_y"] = pct(d["x_rate"]), pct(d["y_rate"])
        v[f"{prefix}_xk"], v[f"{prefix}_yk"] = str(d["x_only"] + d["both"]), str(d["y_only"] + d["both"])
        v[f"{prefix}_xo"], v[f"{prefix}_yo"] = str(d["x_only"]), str(d["y_only"])
        v[f"{prefix}_pairs"], v[f"{prefix}_ci"] = str(d["pairs"]), dci(d)
        v[f"{prefix}_p"], v[f"{prefix}_peq"] = pv(d["mcnemar_exact_p"]), pv_eq(d["mcnemar_exact_p"])
        if holm is not None:
            v[f"{prefix}_holm"], v[f"{prefix}_holmeq"] = pv(holm), pv_eq(holm)
    pair("e3a", P["E3"]["main_A_vs_B"], P["E3"]["main_A_vs_B"]["holm_p"])
    pair("e3s", P["E3"]["sync_C_vs_D"], P["E3"]["sync_C_vs_D"]["holm_p"])
    pair("nat_rn", sec["E3_main"]["nat_success"])
    pair("e4", sec["E4_naive_vs_rtc"]["P1_recovery"])
    pair("nat_nr", sec["E4_naive_vs_rtc"]["nat_success"])
    pair("con_nr", sec["E4_naive_vs_rtc"]["nat_contact"])
    jr = sec["E4_naive_vs_rtc"]["jerk_rms_all"]
    v["jerk_naive"], v["jerk_rtc"], v["jerk_pairs"] = f"{jr['x_median']:.1f}", f"{jr['y_median']:.1f}", str(jr["pairs"])
    pair("e5", sec["E5"]["nat_contact"])           # 既定の構成（フィルタなし）と フィルタあり。x が既定の構成
    pair("e5s", sec["E5"]["nat_success"])
    for part in ("P2", "P3"):
        s = sets["A"][part]
        v[f"a_{part.lower()}_k"], v[f"a_{part.lower()}_n"] = str(s["recovered"]), str(s["established"])
        v[f"a_{part.lower()}_ci"] = ci(s["recovery_wilson"])
    pair("e3p2", sec["E3_main"]["P2_recovery"])                # 落下・置き損ねからの立て直し（立て直しデータあり・なし、両モデルとも実際に失敗した組）
    pair("e3p3", sec["E3_main"]["P3_recovery"])
    v["b_p1_est"] = str(sets["B"]["P1"]["established"])
    v["e2_n_trials"] = str(sets["A"]["P1"]["n"])
    # 結果を見た後に足した参考の集計
    ex = _j(RES / EXTRA)
    pa = ex["p1_all"]
    v["p1all_n"], v["p1all_xk"], v["p1all_yk"] = str(pa["pairs"]), str(pa["x_k"]), str(pa["y_k"])
    v["p1all_peq"] = pv_eq(pa["mcnemar_exact_p"])
    ng = ex["natural_grasp_failure"]
    v["nat_gf_xk"], v["nat_gf_xn"] = str(ng["A"]["recovered"]), str(ng["A"]["grasp_failed"])
    v["nat_gf_yk"], v["nat_gf_yn"] = str(ng["B"]["recovered"]), str(ng["B"]["grasp_failed"])
    v["nat_gf_trials"] = str(ng["A"]["trials"])
    dm = ex["natural_drop_misplace"]["sets"]                    # 方策自身が起こした落下・置き損ね（A あり、B なし）
    for s, side in (("A", "x"), ("B", "y")):
        for kind, key in (("drop", "nd"), ("misplace", "nm")):
            v[f"{key}_{side}_n"], v[f"{key}_{side}_k"] = str(dm[s][kind]["n"]), str(dm[s][kind]["recovered"])
    v["nd_near_cm"] = f"{float(CFG['eval']['P2']['min_dist_from_box_m']) * 100:g}"
    # E6（目標位置の入力）
    e6, e6c = _j(S3 / "e6_R1v3_both.json"), _j(S3 / "e6_R1v3_cue_only.json")
    k6 = round(e6["accuracy_majority"] * e6["pairs"])
    v["e6_k"], v["e6_n"], v["e6_ci"] = str(k6), str(e6["pairs"]), ci(ee._wilson(k6, e6["pairs"]))
    v["e6c_cue"] = str(round(e6c["accuracy_majority"] * e6c["pairs"]))
    v["e6c_text"] = str(round(e6c["follows_instruction_text_majority"] * e6c["pairs"]))
    v["e6c_n"] = str(e6c["pairs"])
    v["e6_layouts"] = e6["trials"].split(":")[1]
    v["e6_samples"] = str(sum(e6["rows"][0]["votes"].values()))         # 1 通りあたりの生成の回数
    v["e6c_text_ci"] = ci(ee._wilson(round(e6c["follows_instruction_text_majority"] * e6c["pairs"]), e6c["pairs"]))
    # E7（3 個の連続タスク）
    s7 = e7_summary()
    v["e7_k"], v["e7_n"], v["e7_ci"] = str(s7["all_three"]), str(s7["n"]), ci(ee._wilson(s7["all_three"], s7["n"]))
    v["e7_retry"], v["e7_fn"], v["e7_fp"] = str(s7["retry_completed"]), str(s7["false_neg"]), str(s7["false_pos"])
    top_step, top_n = s7["stopped_at"].most_common(1)[0]
    v["e7_stop2"], v["e7_stop_pos"] = str(top_n), str(top_step + 1)
    v["e7_stopped"] = str(sum(s7["stopped_at"].values()))
    v["e7_judged"], v["e7_notjudged"] = str(s7["judged"]), str(s7["not_judged"])
    v["e7_stop_all3"] = str(s7["stopped_all_three"])
    v["e7_sys"] = str(s7["n"] - sum(s7["stopped_at"].values()))        # 停止せずに全サブタスクの完了判定が出た試行
    v["e7_sys_ci"] = ci(ee._wilson(s7["n"] - sum(s7["stopped_at"].values()), s7["n"]))
    v["e7_viol"] = str(s7["audit_viol"])
    fu = e7_funnel()                                                     # 段ごとの内訳（4.6 節）
    cname = {"red": "赤", "green": "緑", "blue": "青"}
    for i, c in enumerate(fu["order"], 1):
        v[f"e7_c{i}"] = cname[c]
    for k in ("f1", "f2", "f2go", "f3"):
        v[f"e7_{k}"] = str(fu[k])
    assert fu["f3"] == s7["all_three"], fu                               # 3 番目まで入った本数＝3 個とも箱に入った本数
    v["e7_f2stop"] = str(fu["f2"] - fu["f2go"])
    fr = e7_fn_reasons()
    assert sum(fr.values()) == s7["false_neg"], fr
    v["e7_fn_late"], v["e7_fn_over"], v["e7_fn_wrist"] = (str(fr[k]) for k in ("late", "overhead", "wrist"))
    # 人が手を出す回数（scripts/count_interventions.py → outputs/results/intervention_s3.json）
    iv = _j(RES / "intervention_s3.json")
    assert iv["baseline_check"]["all_match"] and iv["n"] == s7["n"] and iv["success"]["all_three_in_box"] == s7["all_three"]
    assert iv["stopped"]["n"] == sum(s7["stopped_at"].values()) and iv["stopped"]["all_three_in_box"] == s7["stopped_all_three"]
    v["iv_total"], v["iv_per_run"] = str(iv["interventions_total"]), f"{iv['interventions_per_run']:.2f}"
    v["iv_retry"] = str(iv["per_type"]["retry"]["total"])
    assert iv["interventions_total"] == iv["per_type"]["retry"]["total"]          # この評価の介入はすべて自動のやり直し
    v["iv_judge_only"], v["iv_genuine"] = str(iv["retry_breakdown"]["judge_only"]), str(iv["retry_breakdown"]["genuine"])
    v["iv_other"] = str(sum(iv["per_type"][k]["total"] for k in ("scripted_return", "replan", "judge_override")))
    v["iv_llm_per_run"] = f"{iv['llm']['calls_per_run']:g}"
    v["iv_llm_in"], v["iv_llm_out"] = f"{iv['llm']['tokens_mean']['input']:.0f}", f"{iv['llm']['tokens_mean']['output']:.0f}"
    for k in ("0", "1", "2"):
        sk = iv["success_at_k"][k]
        v[f"iv_sk{k}"], v[f"iv_sk{k}_ci"] = str(sk["k"]), ci(sk["wilson95"])
        v[f"iv_k{k}"] = str(int(k))                                       # 許したやり直しの回数（本文の「… 回まで」。手で書かない）
    sk2 = iv["success_at_k"]["2"]
    v["iv_rest"] = str(iv["n"] - sk2["k"])                               # やり直しを許しても 3 個とも片付かなかった（人が仕上げる）作業の数
    assert sk2["k"] == s7["all_three"] == iv["success_at_k"]["3"]["k"]     # やり直し 2 回までで、3 個とも入った本はすべて数えている
    # 100 回の作業に直したときの見積もり（今は本文・表で使っていない）
    v["iv_h100"] = str(round(100 * (1 - sk2["rate"])))                  # 100 回の作業に直したとき、3 個とも片付かず人が仕上げる回数
    v["iv_h100_lo"], v["iv_h100_hi"] = str(round(100 * (1 - sk2["wilson95"][1]))), str(round(100 * (1 - sk2["wilson95"][0])))
    v["iv_a100"] = str(round(100 * sk2["rate"]))
    v["iv_n"] = str(iv["n"])
    v["iv_base"] = "100"                                                # 「100 回の作業に直すと」の基数（割合を回数で言い直すためのもの）
    # 手軽さ（冒頭）: 学習 1 回の時間と GPU のメモリ（学習の記録 train_run.json）
    import datetime
    tr = _j(ROOT / pathlib.Path(ck["R1v3"]).parent.parent / "train_run.json")
    dt = datetime.datetime.fromisoformat(tr["ended_at"]) - datetime.datetime.fromisoformat(tr["started_at"])
    v["train_h"] = f"{dt.total_seconds() / 3600:.1f}"
    v["train_gib"] = f"{tr['log_summary']['gpu_mem_allocated_max_gib']:.1f}"
    v["eps_auto"] = str(tr["dataset"]["total_episodes"])
    assert v["eps_auto"] == v["eps_r"]
    # RTC の仕組みの指標（4.3 節）
    rm = rtc_mechanism()
    a_, e_ = rm[RTC_CMP[0]], rm[RTC_CMP[1]]
    v["rtc_close_a"], v["rtc_close_e"] = f"{a_['close']:.1f}", f"{e_['close']:.1f}"
    v["rtc_ratio_a"], v["rtc_ratio_e"] = f"{a_['ratio']:.2f}", f"{e_['ratio']:.2f}"
    v["rtc_seam_a"], v["rtc_non_a"] = f"{a_['seam']:.0f}", f"{a_['non']:.0f}"
    v["rtc_seam_e"], v["rtc_non_e"] = f"{e_['seam']:.0f}", f"{e_['non']:.0f}"
    v["rtc_horizon"] = str(int(CFG["runtime"]["rtc_guidance_horizon"]))
    rd = _j(RES / "rtc_decision.json")
    assert int(rd["chosen_horizon"]) == int(CFG["runtime"]["rtc_guidance_horizon"])
    v["rtc_sel_rows"] = str(int(CFG["runtime"]["exec_interval"]))        # 範囲を選んだときの実行の間隔（default.yaml の runtime）
    # E9（G1〜G3 の監査）・E10（知覚の精度）
    e9 = rep["new"]["E9"]["total"]
    v["e9_trials"] = f"{e9['trials']:,}"
    v["e9_viol"] = str(e9["g1"] + e9["g3"])           # 本文に書くのは真の状態の参照と指令の上限超過
    byc = rep["new"]["E9"]["by_condition"]            # 構成 A〜E は各 nat＋P1〜P3、H（安全フィルタあり）は nat のみ
    cfgs = sorted({k.split("_")[0] for k in byc if not k.startswith("H")})
    per = {c: sum(r["trials"] for k, r in byc.items() if k.split("_")[0] == c) for c in cfgs}
    assert len(set(per.values())) == 1, per
    v["e9_cfg_n"], v["e9_per_cfg"] = str(len(cfgs)), str(next(iter(per.values())))
    v["e9_h"] = str(sum(r["trials"] for k, r in byc.items() if k.startswith("H")))
    pa = rep["new"]["E10"]["A"]
    v["perc_med_mm"], v["perc_p95_mm"] = f"{pa['median_m'] * 1000:.0f}", f"{pa['p95_m'] * 1000:.0f}"
    # 動画の場面の時刻（本文の「動画 m:ss〜m:ss」）
    t = 0.0
    span = {}
    for sc in video_scenes():
        span[sc["id"]] = (t, t + sc["dur_s"])
        v[f"v_{sc['id']}"] = f"{mmss(t)}〜{mmss(t + sc['dur_s'])}"
        t += sc["dur_s"]
    if "task_intro" in span:                    # 実演の条件の説明から実演の終わりまでを、本文では一続きの場面とする
        v["v_task"] = f"{mmss(span['task_intro'][0])}〜{mmss(span['task'][1])}"
    v["v_total"] = mmss(t)
    # 動画の実演の条件（LLM の層の実行器の設定）
    v["step_timeout"] = f"{float(CFG['planner']['step_timeout_s']):g}"
    v["retry"] = str(int(CFG["planner"]["retry"]))
    # 6 章の実機の確かめ（参考。scripts/64_real_check.py が実機の記録の解析結果から集めたもの）
    rc = _j(RES / "real_check.json")
    v["rb_runs"] = str(rc["servo"]["n"])
    v["rb_lag_real_ms"], v["rb_lag_sim_ms"] = f"{rc['servo']['lag_real_ms']:.0f}", f"{rc['servo']['lag_sim_ms']:.0f}"
    b7 = rc["b7"]
    v["rb_n"], v["rb_inbox"], v["rb_corner"] = str(b7["n"]), str(b7["in_box"]), str(b7["corner"])
    v["rb_corner_inbox"] = str(b7["corner_in_box"])
    v["rb_ang_lo"], v["rb_ang_hi"] = f"{b7['angle_min']:g}", f"{b7['angle_max']:g}"
    return v


def fig2(v) -> bool:
    """図 2: 同じ配置での立て直しデータあり（上段）となし（下段）。configs/demo/video_s3.yaml の fig2 に場面と時刻を書く。"""
    import yaml
    spec = yaml.safe_load(VIDEO.read_text(encoding="utf-8")).get("fig2")
    if not spec:
        return False
    import av
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(str(FONT), 30)                     # 図の中の文字（30 px）
    rows = []
    for key, label in (("with", "復帰デモ\nあり"), ("without", "復帰デモ\nなし")):
        c = av.open(str(OUT / "demo_v2" / spec[key]["clip"]))
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
        c.close()
        tiles = []
        for tt in spec["times"]:
            img = Image.fromarray(frames[min(int(round(tt * 30)), len(frames) - 1)][120:600, 250:1090]).resize((420, 240))
            dr = ImageDraw.Draw(img)
            dr.rectangle((4, 4, 116, 42), fill=(255, 255, 255))
            dr.text((10, 6), f"{tt:.1f} s", fill=(20, 20, 20), font=font)
            tiles.append(img)
        row = Image.new("RGB", (420 * len(tiles) + 200, 240), (255, 255, 255))
        ImageDraw.Draw(row).multiline_text((8, 80), label, fill=(20, 20, 20), font=ImageFont.truetype(str(FONT), 31), spacing=8)
        for i, im in enumerate(tiles):
            row.paste(im, (200 + 420 * i, 0))
        rows.append(row)
    out = Image.new("RGB", (rows[0].width, 240 * 2 + 8), (255, 255, 255))
    out.paste(rows[0], (0, 0))
    out.paste(rows[1], (0, 248))
    out.save(BUILD / "fig2.png")
    return True


PIE_KINDS = [("n", "成功デモ", "#2a78d6"), ("A", "復帰デモ（把持失敗）", "#eb6834"),
             ("B", "復帰デモ（落下）", "#1baf7a"), ("C", "復帰デモ（置き損ね）", "#eda100")]   # 区別しやすい色の順（隣り合う組で確かめた）


def data_mix() -> dict:
    """立て直しデータを含む学習データの種類ごとの本数とフレーム数（変換の記録 meta/conversion.json の sources と episodes の長さ）。"""
    import pyarrow.parquet as pq
    root = ROOT / v3_dataset_dir()
    ep = pq.read_table(sorted((root / "meta" / "episodes").rglob("*.parquet"))[0], columns=["episode_index", "length"]).to_pydict()
    length = dict(zip(ep["episode_index"], ep["length"]))
    n, f = collections.Counter(), collections.Counter()
    for s in _j(root / "meta" / "conversion.json")["sources"]:
        n[s["kind"]] += 1
        f[s["kind"]] += length[s["episode_index"]]
    return {"episodes": dict(n), "frames": dict(f)}


def pie_svg() -> str:
    """図 3: 学習データの内訳の円グラフ（左が本数、右がフレーム数）。扇形の間に白い隙間、名前と割合は凡例と扇形の外に文字で書く。"""
    import math
    mix = data_mix()
    out = ['<svg viewBox="0 0 1080 360" role="img" aria-label="復帰デモありの学習データの内訳。本数とフレーム数の円グラフ。">']
    for cx, key, title in ((200, "episodes", "本数"), (560, "frames", "フレーム数")):
        tot = sum(mix[key].values())
        out.append(f'<text class="bt" x="{cx}" y="30" text-anchor="middle">{title}（計 {tot:,}）</text>')
        a0 = -math.pi / 2
        r, cy = 125, 190
        for k, _, col in PIE_KINDS:
            val = mix[key].get(k, 0)
            a1 = a0 + 2 * math.pi * val / tot
            x0, y0, x1, y1 = cx + r * math.cos(a0), cy + r * math.sin(a0), cx + r * math.cos(a1), cy + r * math.sin(a1)
            large = 1 if a1 - a0 > math.pi else 0
            out.append(f'<path d="M{cx},{cy} L{x0:.1f},{y0:.1f} A{r},{r} 0 {large} 1 {x1:.1f},{y1:.1f} z" '
                       f'style="fill:{col};stroke:#fff;stroke-width:2"/>')
            am = (a0 + a1) / 2
            lx, ly = cx + (r + 26) * math.cos(am), cy + (r + 26) * math.sin(am)
            anchor = "start" if math.cos(am) > 0.2 else ("end" if math.cos(am) < -0.2 else "middle")
            out.append(f'<text class="t" x="{lx:.1f}" y="{ly + 5:.1f}" text-anchor="{anchor}">{round(100 * val / tot)}%</text>')
            a0 = a1
    y = 120
    for k, name, col in PIE_KINDS:
        e, fr = mix["episodes"].get(k, 0), mix["frames"].get(k, 0)
        out.append(f'<rect x="790" y="{y - 14}" width="18" height="18" rx="3" style="fill:{col}"/>')
        out.append(f'<text class="t" x="818" y="{y}">{name}</text>')
        out.append(f'<text class="tm" x="818" y="{y + 22}">{e} 本・{fr:,} フレーム</text>')
        y += 58
    out.append("</svg>")
    return "\n".join(out)


BAR_COLORS = ("#2a78d6", "#eb6834")   # 立て直しデータあり・なし（PIE_KINDS と同じ色の順の 1・2 番目）


def bar_svg(v) -> str:
    """冒頭の要点の図: 主な結果の横棒（立て直しデータあり・なしの組）。値と分母は棒の右に文字で書く。"""
    rows = [("把持を意図的に失敗させたときの復帰（非同期実行、主要評価項目）", ("e3a_xk", "e3a_pairs"), ("e3a_yk", "e3a_pairs")),
            ("把持を意図的に失敗させたときの復帰（同期実行、主要評価項目）", ("e3s_xk", "e3s_pairs"), ("e3s_yk", "e3s_pairs")),
            ("方策自身が起こした把持失敗からの復帰（参考）", ("nat_gf_xk", "nat_gf_xn"), ("nat_gf_yk", "nat_gf_yn")),
            ("意図的な失敗なしの成功率", ("nat_rn_xk", "nat_rn_pairs"), ("nat_rn_yk", "nat_rn_pairs"))]
    x0, w, bh = 330, 560, 20
    out = ['<svg viewBox="0 0 1080 330" role="img" aria-label="主な結果の横棒グラフ。復帰デモありとなしの割合。">']
    out.append(f'<rect x="{x0}" y="8" width="16" height="16" rx="3" style="fill:{BAR_COLORS[0]}"/><text class="t" x="{x0 + 24}" y="22">復帰デモあり</text>')
    out.append(f'<rect x="{x0 + 160}" y="8" width="16" height="16" rx="3" style="fill:{BAR_COLORS[1]}"/><text class="t" x="{x0 + 184}" y="22">復帰デモなし</text>')
    y = 48
    for label, *pairs in rows:
        words = label.split("（")
        out.append(f'<text class="t" x="{x0 - 12}" y="{y + 16}" text-anchor="end">{words[0]}</text>')
        if len(words) > 1:
            out.append(f'<text class="tm" x="{x0 - 12}" y="{y + 38}" text-anchor="end">（{words[1]}</text>')
        for i, (k, n) in enumerate(pairs):
            kk, nn = int(v[k]), int(v[n])
            bw = w * kk / nn
            yy = y + i * (bh + 4)
            out.append(f'<path d="M{x0},{yy} h{max(bw - 4, 0):.1f} a4,4 0 0 1 4,4 v{bh - 8} a4,4 0 0 1 -4,4 h{-max(bw - 4, 0):.1f} z" '
                       f'style="fill:{BAR_COLORS[i]}"/>' if bw >= 4 else
                       f'<rect x="{x0}" y="{yy}" width="2" height="{bh}" style="fill:{BAR_COLORS[i]}"/>')
            out.append(f'<text class="t" x="{x0 + max(bw, 2) + 8:.1f}" y="{yy + 15}">{round(100 * kk / nn)}%（{kk}/{nn}）</text>')
        y += 2 * (bh + 4) + 26
    out.append(f'<line x1="{x0}" y1="40" x2="{x0}" y2="{y - 20}" style="stroke:#555c64;stroke-width:1"/>')
    out.append("</svg>")
    return "\n".join(out)


def render(v) -> str:
    def sub(text):
        def rep(m):
            k = m.group(1)
            if k not in v:
                raise KeyError(k)
            return f'<span class="num" data-k="{k}">{html.escape(v[k])}</span>'
        return re.sub(r"\{\{(\w+)\}\}", rep, text)
    def svg(name):
        return sub((PAPER / name).read_text(encoding="utf-8")).replace('<span class="num" data-k=', '<tspan data-k=').replace("</span>", "</tspan>")
    figs = {"fig1_svg": lambda: svg("fig1.svg"), "fig_rec_svg": lambda: svg("fig_recovery.svg"),   # 図 1（構成）・図 2（作り方）
            "fig_pie_svg": pie_svg,                                                                   # 図 3（学習データの内訳）
            "fig_bar_svg": lambda: bar_svg(v)}                                                        # 冒頭の要点の図
    body = (PAPER / "template.html").read_text(encoding="utf-8")
    for key in figs:
        body = body.replace("{{" + key + "}}", f"@@{key}@@")
    body = sub(body)
    for key, make in figs.items():
        body = body.replace(f"@@{key}@@", make())
    return body


def goals_or_exit() -> dict:
    """計画の文書の版（recovla.common.goals）が変更履歴にない版なら、作らない・照合しない。"""
    try:
        return goals.fingerprint()
    except goals.GoalsMismatch as e:
        print(f"目標書の照合に通らないので止める: {e}", file=sys.stderr)
        raise SystemExit(1)


def cmd_build(a) -> None:
    fp = goals_or_exit()
    BUILD.mkdir(parents=True, exist_ok=True)
    for old in ("fig3.png",):                       # 前の版の図（今は使わない）
        (BUILD / old).unlink(missing_ok=True)
    (BUILD / "goals.json").write_text(json.dumps(fp, ensure_ascii=False, indent=1), encoding="utf-8")
    v = values()
    (BUILD / "values.json").write_text(json.dumps(v, ensure_ascii=False, indent=1), encoding="utf-8")
    if not fig2(v):
        (BUILD / "fig2.png").unlink(missing_ok=True)
        print("図 2 の場面が未設定（configs/demo/video_s3.yaml の fig2）", file=sys.stderr)
    (BUILD / "paper.html").write_text(render(v), encoding="utf-8")
    pdf = BUILD / "paper.pdf"
    subprocess.run([str(EDGE), "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}",
                    (BUILD / "paper.html").as_uri()], check=True, capture_output=True, timeout=180)
    (BUILD / PDF_NAME).write_bytes(pdf.read_bytes())
    print(BUILD / PDF_NAME, "pages", pdf_pages(pdf))


def pdf_pages(p) -> int:
    return len(re.findall(rb"/Type\s*/Page(?!s)", pathlib.Path(p).read_bytes()))


# 原稿の地の文に書いてよい数字（数字が値ではなく名前や固有の決まりの一部であるもの）
ALLOWED = [r"fig2\.png", r"表 [1-3]", r"GPU 1 枚", r"学習 1 回", r"[1-3] 番目", r"(95|1) パーセンタイル",r"two3", r"Physical AI 応用 1 講座", r"Haiku 4\.5", r"SmolVLM2", r"Apache-2\.0", r"図 [1-4]",
           r"[12] 回目", r"工夫 [1-6]", r"\b[1-6]\.[1-6](?= )", r"et al\., (19|20)\d\d", r"3 (色|個)", r"95% 信頼区間", r"1 試行", r"[1-6]\. ", r"7 軸", r"画像 2 枚", r"1 行ずつ",
           r"× 3 色", r"A4", r"RGB-D", r"#[0-9a-f]{6}", r"\d+(\.\d+)?(mm|pt|px)", r"viewBox=\"[^\"]*\"", r"\b\d+(\.\d+)?%?\"",
           r"[xy][12]?=\"[^\"]*\"", r"points=\"[^\"]*\"", r"d=\"[^\"]*\"", r"rotate\([^)]*\)", r"\b(width|height|rx|dx|refX|refY|markerWidth|markerHeight)=\"[^\"]*\"",
           r"opacity:\.\d+", r"stroke-dasharray:[\d ]+", r"stroke-width:[\d.]+", r"font-size:[\d.]+(px|pt)", r"line-height:[\d.]+",
           r"\d+(\.\d+)?(fr|mm)", r"gap: \d+mm", r"margin[^;]*;", r"padding[^;]*;", r"h[123]", r"m1[co]", r"M0,0 L10,5 L0,10 z",
           r"lang=\"ja\"", r"utf-8", r"-webkit", r"0 10 10", r"font-weight: ?\d+", r"border[^;]*;", r"\bgrid-template-columns[^;]*;"]
# 使わない語（開発中に作った語・作業記録に触れる語）。「手」は手先・手首などは使うので、数と組になった形だけ
# 語の一覧は Unicode エスケープで書く（書き出しの置き換えと語の検索が、この一覧そのものに当たらないように）
FORBIDDEN = ["\u584a", "\u3053\u307e", "\u7a2e(?!\u985e)", "\u5e2f", "\u53f0\u672c", "\u8a98\u767a", "\u6210\u7acb",
             "\u7acb\u3061\u76f4", "\u624b\u304c\u304b\u308a", "\u7d99\u304e\u76ee", "\u9045\u308c",
             "\u6d41\u308c\u306e\u4e00\u81f4", "\u4fdd\u5b58\u70b9", "\u81ea\u7136", "\u901a\u3057",
             "\u4e3b\u306a\u691c\u5b9a", "\u526f\u306e\u6307\u6a19", "(?<!\u4fe1\u983c)\u533a\u9593",
             "[\u7532\u4e59\u4e19]", "\u672c\u7dda", "\u652f\u7dda", "\u76e3\u7763", "\u6c7a\u88c1",
             "\u63b2\u793a\u677f", "\u6d41\u7528\u5143", "\u5352\u7814", "\u5352\u696d\u7814\u7a76", "C:\\\\\x56LA",
             "\u5b66\u751f", "\u4e88\u5099\u5b9f\u9a13", "(\\d|\u4e07)\\s*\u624b(?![\u5148\u9996\u9806\u6cd5])",
             "\u6a21\u578b",
             # 開発中の記号（本文でも使わない）
             "(?<![A-Za-z0-9])[RNP][123](?![0-9])", "R1\\+", "[RN][12]v[23]", "(?<![A-Za-z])E(?:[1-9]|10)(?![0-9])", "(?<![A-Za-z])G[1-6](?![0-9])",
             # 失敗を起こす操作は「意図的な失敗」と書く。物理演算の扱いは開発の過程のことなので書かない
             "\u6ce8\u5165", "\u7269\u7406\u6f14\u7b97.{0,6}(\u505c\u6b62|\u6b62\u307e|\u6b62\u3081)",
             "(\u505c\u6b62|\u6b62\u307e|\u6b62\u3081).{0,6}\u7269\u7406\u6f14\u7b97",
             # \u8a9e\u306e\u7f6e\u304d\u63db\u3048:\u5bfe\u2192\u7d44\u3001\u8a2d\u7f6e\u60c5\u5831\u2192\u65e2\u77e5\u306e\u60c5\u5831\u3001\u30ad\u30e5\u30fc\u2192\u76ee\u6a19\u4f4d\u7f6e\u306e\u5165\u529b\u3001\u7279\u6a29\u60c5\u5831\u3001\u504f\u4f4d\u3001\u89e3\u653e\u3001\u30d3\u30c3\u30c8\u5358\u4f4d\u3001\u4ee5\u524d\u306e\u7248
             "\\d\\s*\u5bfe(?!\u5fdc)", "\u5bfe[\u306f\u304c\u3092]", "\u8a2d\u7f6e\u60c5\u5831", "\u30ad\u30e5\u30fc", "\u7279\u6a29\u60c5\u5831",
             "\u504f\u4f4d", "\u89e3\u653e", "\u30d3\u30c3\u30c8\u5358\u4f4d", "\u4ee5\u524d\u306e\u7248"]


def cmd_check(a) -> None:
    fp = goals_or_exit()
    v = values()
    problems = []
    built_goals = BUILD / "goals.json"
    if not built_goals.is_file() or _j(built_goals) != fp:
        problems.append("組み上げたときの目標書の版が今の版と違う（build をやり直す）")
    if not (BUILD / "fig2.png").is_file():
        problems.append("図 2 が未作成（configs/demo/video_s3.yaml の fig2）")
    # (1) 原稿に手で書いた数字がない
    for name in ("template.html", "fig1.svg", "fig_recovery.svg"):
        text = (PAPER / name).read_text(encoding="utf-8")
        text = re.sub(r"\{\{\w+\}\}", " ", text)
        text = re.sub(r"<style>.*?</style>", " ", text, flags=re.S)
        for pat in ALLOWED:
            text = re.sub(pat, " ", text)
        for m in re.finditer(r"\d", text):
            problems.append(f"{name}: 手で書いた数字 …{text[max(0, m.start() - 15):m.end() + 15]!r}")
    # (2) 組み上げた HTML の差し込み口の値が、今の結果から計算した値と一致
    built = (BUILD / "paper.html").read_text(encoding="utf-8")
    n_slots = 0
    for m in re.finditer(r'data-k="(\w+)">([^<]*)<', built):
        n_slots += 1
        if html.unescape(m.group(2)) != v[m.group(1)]:
            problems.append(f"値の食い違い {m.group(1)}: {m.group(2)} != {v[m.group(1)]}")
    # (3) 使わない語が 0
    plain = re.sub(r"<style>.*?</style>", " ", built, flags=re.S)
    plain = html.unescape(re.sub(r"<[^>]+>", " ", plain))
    hits = [(p, mm.group(0), plain[max(0, mm.start() - 12):mm.end() + 12]) for p in FORBIDDEN for mm in re.finditer(p, plain)]
    problems += [f"使わない語 {h[1]!r}: …{h[2]}…" for h in hits]
    # (4) ページ数
    pages = pdf_pages(BUILD / "paper.pdf")
    if pages < 2:                                  # 課題はページ数を指定しない（2 ページ以上だけ確かめる）
        problems.append(f"ページ数 {pages}")
    res = {"goals": fp, "slots": n_slots, "keys": len(v), "pages": pages, "forbidden_hits": len(hits), "problems": problems}
    (BUILD / "check.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    raise SystemExit(1 if problems else 0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    sub.add_parser("check")
    a = ap.parse_args(argv)
    {"build": cmd_build, "check": cmd_check}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
