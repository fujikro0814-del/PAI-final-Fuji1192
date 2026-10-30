"""実機の Panda で再生する関節の指令の一式を、実機側の Linux の PC に渡す zip にまとめる（<src の名前>/ に展開される）。

    python scripts/real/build_replay_zip.py                 # outputs/real/replay_v1 → outputs/real/replay_v1.zip
    python scripts/real/build_replay_zip.py --src outputs/real/replay_v1_slow --zip outputs/real/replay_v1_slow.zip   # 速さ 0.5 倍の版

再生するのは、生成プログラムの指令（export_replay.py）か VLA の方策の指令（export_policy_replay.py）。
入れるもの: 書き出した CSV、manifest.json（全ファイルの sha256 を files に足す）、SHA256SUMS（sha256sum -c 用）、
docs/real/replay_format.md、scripts/real/real_to_ds.py・compare_replay.py（python3 と numpy だけで動く）、README.txt。
manifest の csv_sha256 と CSV の中身が一致しなければ止める。
"""
import argparse
import hashlib
import json
import pathlib
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
README = """台本の再生（実機の試験 A8）の一式  replay_v1

1. 確かめ（全ファイルの sha256）
   cd replay_v1 && sha256sum -c SHA256SUMS

2. 流す列
   <名前>.csv          1 kHz の関節の指令（q_cmd0〜6、実機の q1）とシミュレーションの応答
   <名前>_gripper.csv  グリッパーの指令の時刻
   <名前>_ds.csv       学習データと同じ列の 10 Hz の表（台座の座標）
   開始姿勢は manifest.json の start_q_real。流し方・記録の列は replay_format.md の 2・3 節。

3. 比べ（python3 と numpy だけ）
   python3 compare_replay.py --replay n_20000_red_r0.csv --real <記録>/arm.csv \\
       --real-gripper <記録>/gripper.csv --real-meta <記録>/meta.json --out <記録>/result.json
   学習データと同じ列にするだけなら
   python3 real_to_ds.py --arm <記録>/arm.csv --gripper <記録>/gripper.csv \\
       --replay-ds n_20000_red_r0_ds.csv --out <記録>/real_ds.csv

4. 注意
   - CPU を performance にしてから流す。
   - 指先は台座の面から約 2 cm まで下りる。扇形の範囲に物がないことを確かめる。
   - 関節の加速度・躍度は上限の 0.99 倍まで使う。初回は 1 本で、停止ボタンに指をかけて流す。
"""

README_SLOW = """台本の再生の遅い版（実機の初回用）  replay_v1_slow

通常の版（replay_v1）の関節の指令を、速さ 0.5 倍（時間 2 倍）に引き延ばしたもの。関節の加速度は上限の 0.31 倍、躍度は 0.42 倍まで（通常の版は
どちらも 0.99 倍）。手先の加速度は最大 約 3.1 m/s²。グリッパーの指令の時刻も 2 倍（指の速さは同じ）。シミュレーションの応答は、同じ指令を立方体なしの
シミュレーションの腕にそのまま流したもの（台本は判断し直していない）。

1. 確かめ
   cd replay_v1_slow && sha256sum -c SHA256SUMS

2. 順番
   (1) まず通常の 1 本（n_20000_red_r0.csv）をこの遅い版で流す。停止ボタンに指をかける。
   (2) エラーなしで終わり、compare_replay.py の結果を確かめてから、通常の速さの一式（replay_v1）を流す。
   開始姿勢は manifest.json の start_q_real。流し方・記録の列は replay_format.md の 2・3 節。

3. 比べ（基準は通常の版と同じ。結果の JSON に variant: slow_0.5x と出る）
   python3 compare_replay.py --replay n_20000_red_r0.csv --real <記録>/arm.csv \\
       --real-gripper <記録>/gripper.csv --real-meta <記録>/meta.json --out <記録>/result.json

4. 学習データの列の表（_ds.csv）は入れていない。学習データは通常の速さの台本のこまなので、時間を 2 倍にした動きには
   対応するこまがない（同じ列にすると、速さが半分の別の動きになる）。学習データの列での比べは通常の版で行う。
"""


# VLA の方策の指令の一式（manifest の replay_source が "policy"）だけに使う
README_POLICY = """方策の再生（H1）の一式  {top}

シミュレーション（v3_real）で VLA の方策が出した関節の指令を、実機にそのまま流す列。物（立方体・箱）は置かない。
最初の手順（赤を箱へ）と家への戻しまで（{stream_s} s）。グリッパーの grasp は空振りになる（期待どおり）。
{variant_note}
シミュレーションの応答の列は、同じ指令を実機に合わせた腕とハンド（real_v1）に立方体なしで流したもの（比べる相手）。

1. 確かめ（全ファイルの sha256）
   cd {top} && sha256sum -c SHA256SUMS

2. 流す列
   {name}          1 kHz の関節の指令（q_cmd0〜6、実機の q1）とシミュレーションの応答
   {grip}  グリッパーの指令の時刻
   開始姿勢は manifest.json の start_q_real。流し方・記録の列は replay_format.md の 2・3 節。
   学習データの列の表（_ds.csv）は無い（方策の試行は学習データの本ではない）。

3. 比べ（python3 と numpy だけ）
   python3 compare_replay.py --replay {name} --real <記録>/arm.csv \\
       --real-gripper <記録>/gripper.csv --real-meta <記録>/meta.json --out <記録>/result.json
   「再現した」と言う基準は h1_policy_replay.md（A8 の基準より厳しい）。

4. 注意
   - CPU を performance にしてから流す。物を置かない。扇形の範囲に物・取り付け面がないことを確かめる。
   - 指先は台座の面から約 2 cm まで下りる（manifest.json の ee_z_cmd_range_m）。
   - まず遅い版（policy_replay_v1_slow_fit）を 1 本、停止ボタンに指をかけて流す。エラーなしで比べが通ってから通常の速さ。
"""


def sha256(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default="outputs/real/replay_v1")
    ap.add_argument("--zip", default="outputs/real/replay_v1.zip")
    a = ap.parse_args(argv)
    src = (ROOT / a.src) if not pathlib.Path(a.src).is_absolute() else pathlib.Path(a.src)
    zpath = (ROOT / a.zip) if not pathlib.Path(a.zip).is_absolute() else pathlib.Path(a.zip)
    manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    files = {}
    for t in manifest["trials"]:
        if not t.get("exported"):
            continue
        for key in [k for k in ("csv", "gripper_csv", "ds_csv") if k in t]:
            p = src / t[key]
            got = sha256(p)
            if got != t[f"{key}_sha256"]:
                print(f"{p.name}: sha256 が manifest と違う")
                return 2
            files[t[key]] = (p, got)
    policy = manifest.get("replay_source") == "policy"
    docs = (("docs/real/replay_format.md", "replay_format.md"), ("scripts/real/real_to_ds.py", "real_to_ds.py"),
            ("scripts/real/compare_replay.py", "compare_replay.py"))
    if policy:
        docs += (("docs/real/h1_policy_replay.md", "h1_policy_replay.md"),)
    for rel, name in docs:
        p = ROOT / rel
        files[name] = (p, sha256(p))
    slow = str(manifest.get("variant") or "").startswith("slow")
    if policy:
        t0 = [t for t in manifest["trials"] if t.get("exported")][0]
        note = ("遅い版: 通常の版の指令を速さ 0.5 倍（時間 2 倍）に引き延ばしたもの（実機の初回用。判定には使わない）。" if slow
                else "通常の速さ（判定に使う本）。関節の加速度・躍度は上限の 0.99 倍まで使う。")
        readme = README_POLICY.format(top=src.name, stream_s=t0.get("stream_s"), variant_note=note, name=t0["csv"],
                                      grip=t0["gripper_csv"]).encode("utf-8")
    else:
        readme = (README_SLOW if slow else README).encode("utf-8")
    files["README.txt"] = (None, hashlib.sha256(readme).hexdigest())
    manifest["files"] = {k: v[1] for k, v in sorted(files.items())}
    mtxt = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
    sums = "".join(f"{h}  {n}\n" for n, h in sorted([(k, v[1]) for k, v in files.items()]
                                                     + [("manifest.json", hashlib.sha256(mtxt).hexdigest())]))
    top = src.name
    zpath.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, (p, _) in sorted(files.items()):
            z.writestr(f"{top}/{name}", readme if p is None else p.read_bytes())
        z.writestr(f"{top}/manifest.json", mtxt)
        z.writestr(f"{top}/SHA256SUMS", sums.encode("utf-8"))
    print(f"{len(files) + 2} ファイル → {zpath}（sha256 {sha256(zpath)}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
