"""長い評価を回すための運用の道具（再起動の確認・環境の記録・バックアップの一覧・実行の監視）。GPU もシミュレーションも使わない。

scripts/eval_resumable.py が、空きメモリ・時刻の窓・レジストリと git の読み取り（memory_gb・window_state など）の部品として
読む。Windows の設定は読むだけで、書かない。

使い方:
    .venv\\Scripts\\python.exe scripts\\ops_tools.py wu-check [--no-log] [--no-events]
    .venv\\Scripts\\python.exe scripts\\ops_tools.py env-record
    .venv\\Scripts\\python.exe scripts\\ops_tools.py backup-manifest [--force] [--no-hash]
    .venv\\Scripts\\python.exe scripts\\ops_tools.py backup-verify --dest <コピー先> [--quick]
    .venv\\Scripts\\python.exe scripts\\ops_tools.py wait --progress <progress.json> [...] [--pid N ...]

サブコマンドと、読むもの・書くもの:
  wu-check         読む: HKLM の RebootRequired・RebootPending・PendingFileRenameOperations（Edge の一時ファイルは無視）、
                   Windows Update の履歴（COM、直近 7 日。ドライバの更新は driver_updates_7d に分ける）、
                   入っていない更新（COM、端末の控えだけ Online=false）、System イベントログ（電源・再起動、直近 7 日）、最後の起動、
                   参考として 02:00 の更新の前後の窓（--window、既定 01:45〜02:45）に入るまでの分数。
                   書く: 標準出力に JSON、outputs\\s4\\wu_check.jsonl に 1 行追記（--no-log で省く）。
                   終了コード: 再起動待ちなら 1、印はないが入っていないドライバ・再起動しうる更新があれば 3、レジストリ・更新の履歴・
                   イベントのどれかを読めなかったら 2（出力の errors）、問題なしなら 0（1・3・2・0 の順に強い）。確認用（任意）。
                   ドライバの更新は印なしで 02:00 に入って再起動することがある（出力の notes に書く）。
  env-record       読む: nvidia-smi、OS・Python・パッケージの版、git（HEAD・タグ v3-s3-freeze）、メモリ、
                   Windows Update の履歴の NVIDIA の行と V3S3 の記録の更新時刻（その評価の時点のドライバを推す）。
                   書く: outputs\\s4\\env_record.json（前の記録は env_record_history.jsonl に残す）。
  backup-manifest  読む: V3S3 の記録・outputs\\results・R1v3/N1v3 のチェックポイント（2 万ステップ）・outputs\\demo_v2・paper・docs\\freeze。
                   書く: outputs\\s4\\backup_manifest.json（SHA-256 の一覧とコピーの手順の説明）。
                   コピーは実行しない。既にあれば backup_manifest_<時刻>.json に退避してから書く。
  backup-verify    読む: 一覧とコピー先。書かない。違いがあれば終了コード 1。
  wait             読む: 実行の progress.json（eval_resumable.py が書く）か PID。書く: 標準出力だけ。
                   終わる・エラー・落ちる・止まる・固まる・時間切れのどれかになったら、要約を出して終わる。
                   背景で動かして終了の通知を受ける想定。終了コード: 0 完了、1 途中で止まった（合図・メモリ待ちの時間切れなど）、
                   2 エラー、3 異常終了（状態が最終にならないまま消えた＝再起動・強制終了）、4 途絶（更新が --stall-min 分途絶えた、
                   今の試行が固まった＝current_started からの経過が 制限時間 × --hung-factor ＋ --hung-margin-min 分を超えた、包みが子の
                   固まりを書いた、progress.json を読めない状態が --stall-min 分続いた。状態が LIVE でも最終でもない未知のものにも当てる）、
                   5 時間切れ（既定 24 時間。--timeout-min 0 で無し）。
                   再起動で止まったら、サインインの後にこれ（または progress.json）で止まった所を見て、eval_resumable.py の
                   同じコマンドで続きから回す。
この道具は成績を扱わない。判定の決まり（再起動待ちの 3 つの鍵、wait の固まりの判定の倍率 4・余裕 15 分など）は先に決めた。
"""
import argparse
import ctypes
import datetime as dt
import hashlib
import json
import os
import pathlib
import platform
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parents[1]
S4 = ROOT / "outputs" / "s4"
GIT = ROOT / ".tools" / "git" / "cmd" / "git.exe"
FREEZE_TAG = "v3-s3-freeze"
QUIET_DEFAULT = "01:45-02:45"          # 更新の適用（02:00）の前後。wu-check が参考に分数を出すだけで、実行は縛らない

FINAL_STATUS = {"done": 0, "stopped": 1, "interrupted": 1, "memory_timeout": 1, "error": 2}
LIVE_STATUS = {"starting", "loading", "running", "quiet_wait", "memory_wait"}


# ---------------------------------------------------------------- 共通
def _utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _now() -> dt.datetime:
    return dt.datetime.now()


def _dump(obj, path: pathlib.Path, tries: int = 8) -> None:
    """一時ファイルから置き換えて書く。読んでいる相手がいれば少し待ち（Windows）、置き換えられなければ例外にする
    （黙って古い中身を残さない）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    for k in range(tries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.2 * (k + 1))
    raise OSError(f"{path} を置き換えられない（ほかのプロセスが開いている）。書いた中身は {tmp} に残した")


def parse_window(s: str) -> tuple:
    m = re.fullmatch(r"(\d{1,2}):(\d{2})-(\d{1,2}):(\d{2})", s.strip())
    if not m:
        raise SystemExit(f"窓の書き方は HH:MM-HH:MM: {s!r}")
    h1, m1, h2, m2 = (int(x) for x in m.groups())
    return dt.time(h1, m1), dt.time(h2, m2)


def window_state(now: dt.datetime, window: str = QUIET_DEFAULT) -> dict:
    """窓（日をまたいでもよい）に now が入っているか、次の始まりと今の終わりまで何分か。"""
    t0, t1 = parse_window(window)
    today = now.date()
    start = dt.datetime.combine(today, t0)
    end = dt.datetime.combine(today, t1)
    if end <= start:
        end += dt.timedelta(days=1)
    # 前の日から続く窓
    prev_start, prev_end = start - dt.timedelta(days=1), end - dt.timedelta(days=1)
    if prev_start <= now < prev_end:
        return {"window": window, "in_window": True, "minutes_to_window_start": 0.0,
                "minutes_to_window_end": round((prev_end - now).total_seconds() / 60, 1)}
    if start <= now < end:
        return {"window": window, "in_window": True, "minutes_to_window_start": 0.0,
                "minutes_to_window_end": round((end - now).total_seconds() / 60, 1)}
    nxt = start if now < start else start + dt.timedelta(days=1)
    return {"window": window, "in_window": False, "minutes_to_window_start": round((nxt - now).total_seconds() / 60, 1),
            "minutes_to_window_end": None}


class _MemStatus(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]


def memory_gb() -> dict:
    ms = _MemStatus()
    ms.dwLength = ctypes.sizeof(_MemStatus)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
    g = 1024 ** 3
    return {"phys_total_gb": round(ms.ullTotalPhys / g, 2), "phys_free_gb": round(ms.ullAvailPhys / g, 2),
            "commit_limit_gb": round(ms.ullTotalPageFile / g, 2), "commit_free_gb": round(ms.ullAvailPageFile / g, 2)}


def _run(cmd, timeout=60, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, timeout=timeout, cwd=ROOT, **kw)


def _git(*args) -> str:
    exe = str(GIT) if GIT.is_file() else "git"
    try:
        r = _run([exe, "-C", str(ROOT), *args], timeout=60)
        return r.stdout.decode("utf-8", "replace").strip()
    except Exception as e:                                       # noqa: BLE001
        return f"error: {e}"


def _ps_json(script: str, timeout=120):
    """PowerShell（読み取りだけの script）を動かして JSON を受け取る。"""
    pre = "[Console]::OutputEncoding=[Text.Encoding]::UTF8; $ProgressPreference='SilentlyContinue'; "
    r = _run(["powershell", "-NoProfile", "-NonInteractive", "-Command", pre + script], timeout=timeout)
    out = r.stdout.decode("utf-8", "replace").strip()
    if r.returncode != 0 or not out:
        raise RuntimeError(f"powershell exit {r.returncode}: {r.stderr.decode('utf-8', 'replace')[:300]}")
    return json.loads(out)


# ---------------------------------------------------------------- wu-check
EDGE_TEMP = re.compile(r"\\Microsoft\\(Edge\\Temp|EdgeUpdate\\|EdgeCore\\|EdgeWebView\\Temp)", re.IGNORECASE)


def classify_pending_renames(entries) -> dict:
    """PendingFileRenameOperations（REG_MULTI_SZ）を、Edge の一時ファイル（無視）とそれ以外に分ける。
    値は「元, 先」の組が並ぶ（先が空なら削除）。元か先のどちらかが Edge の一時なら、その組は Edge の一時とみなす。"""
    ents = [str(e) for e in (entries or [])]
    ignored, other = [], []
    for i in range(0, len(ents), 2):
        pair = ents[i:i + 2]
        if not any(p.strip() for p in pair):
            continue
        (ignored if any(EDGE_TEMP.search(p) for p in pair) else other).append(pair)
    return {"entries": len(ents), "ignored_edge_pairs": len(ignored), "other_pairs": len(other),
            "other_sample": other[:10]}


def _reg_key_exists(path: str) -> bool:
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path, 0, winreg.KEY_READ | winreg.KEY_WOW64_64KEY):
            return True
    except FileNotFoundError:
        return False


def _reg_value(path: str, name: str):
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path, 0, winreg.KEY_READ | winreg.KEY_WOW64_64KEY) as k:
            return winreg.QueryValueEx(k, name)[0]
    except FileNotFoundError:
        return None


PS_HISTORY = r"""
$r = [ordered]@{}
try {
  $s = New-Object -ComObject Microsoft.Update.Session
  $h = $s.CreateUpdateSearcher()
  $n = $h.GetTotalHistoryCount()
  $cut = (Get-Date).AddDays(-7).ToUniversalTime()
  $rows = @()
  if ($n -gt 0) {
    foreach ($x in $h.QueryHistory(0, [Math]::Min($n, 200))) {
      $d = $x.Date
      if ($d -ge $cut) {
        $rows += [pscustomobject]@{ date_utc = $d.ToString('yyyy-MM-ddTHH:mm:ssZ'); title = [string]$x.Title; operation = [int]$x.Operation; result = [int]$x.ResultCode; hresult = [int]$x.HResult }
      }
    }
  }
  $r.history_total = $n
  $r.history_7d = @($rows)
} catch { $r.history_error = [string]$_.Exception.Message }
try {
  # 入っていない更新（端末の控えだけを見る Online=false。ネットには出ない。入れもしない）
  $se = $s.CreateUpdateSearcher()
  $se.Online = $false
  $pr = $se.Search("IsInstalled=0 and IsHidden=0")
  $r.pending = @($pr.Updates | ForEach-Object { [pscustomobject]@{ title = [string]$_.Title; type = [int]$_.Type;
      downloaded = [bool]$_.IsDownloaded; reboot_behavior = [int]$_.InstallationBehavior.RebootBehavior } })
} catch { $r.pending_error = [string]$_.Exception.Message }
try {
  $os = Get-CimInstance Win32_OperatingSystem
  $r.last_boot = $os.LastBootUpTime.ToString('yyyy-MM-ddTHH:mm:ss')
} catch { $r.last_boot_error = [string]$_.Exception.Message }
$r | ConvertTo-Json -Depth 5 -Compress
"""

PS_EVENTS = r"""
$cut = (Get-Date).AddDays(-7)
$ev = @()
try {
  $ev = Get-WinEvent -FilterHashtable @{ LogName = 'System'; Id = 41, 1074, 6008; StartTime = $cut } -ErrorAction Stop |
    Sort-Object TimeCreated | ForEach-Object {
      [pscustomobject]@{ time = $_.TimeCreated.ToString('yyyy-MM-ddTHH:mm:ss'); id = [int]$_.Id; provider = [string]$_.ProviderName;
                         text = ([string]$_.Message -replace '\s+', ' ').Substring(0, [Math]::Min(120, ([string]$_.Message).Length)) }
    }
} catch { }
@{ events = @($ev) } | ConvertTo-Json -Depth 4 -Compress
"""

DRIVER_TITLE = re.compile(r"driver|ドライバ|NVIDIA|Display|^[^-]+ - [^-]+ - \d+\.\d+", re.IGNORECASE)   # 最後は「会社 - 種類 - 版」の形
WIN_DRV_VER = re.compile(r"(\d+)\.(\d+)\.(\d+)\.(\d+)")


def is_driver_update(title: str) -> bool:
    return bool(DRIVER_TITLE.search(title or ""))


def nvidia_from_windows_version(v: str):
    """Windows の NVIDIA ドライバの版（例 32.0.15.6094）から nvidia-smi の版（560.94）へ。下 2 組をつなげた末尾 5 桁。"""
    m = WIN_DRV_VER.search(v or "")
    if not m:
        return None
    tail = (m.group(3) + m.group(4).zfill(4))[-5:]
    return f"{int(tail[:3])}.{tail[3:]}"


UPDATE_OP = {1: "install", 2: "uninstall", 3: "other"}
UPDATE_RESULT = {0: "not_started", 1: "in_progress", 2: "succeeded", 3: "succeeded_with_errors", 4: "failed", 5: "aborted"}


def cmd_wu_check(a) -> int:
    now = _now()
    out = {"checked": now.strftime("%Y-%m-%d %H:%M:%S"), "errors": []}
    base = r"SOFTWARE\Microsoft\Windows\CurrentVersion"
    try:
        out["RebootRequired"] = _reg_key_exists(base + r"\WindowsUpdate\Auto Update\RebootRequired")
        out["RebootPending"] = _reg_key_exists(base + r"\Component Based Servicing\RebootPending")
        pfro = _reg_value(r"SYSTEM\CurrentControlSet\Control\Session Manager", "PendingFileRenameOperations")
        out["PendingFileRenameOperations"] = classify_pending_renames(pfro)
    except Exception as e:                                       # noqa: BLE001
        print(json.dumps({"checked": out["checked"], "fatal": f"レジストリを読めない: {e}"}, ensure_ascii=False))
        return 2
    out["pending_file_rename_blocks"] = out["PendingFileRenameOperations"]["other_pairs"] > 0
    out["reboot_pending"] = bool(out["RebootRequired"] or out["RebootPending"] or out["pending_file_rename_blocks"])
    try:
        info = _ps_json(PS_HISTORY, timeout=120)
        hist = []
        for h in info.get("history_7d", []) or []:
            h = dict(h)
            try:                                                 # COM の時刻は UTC。ローカルの時刻も添える
                h["date_local"] = (dt.datetime.strptime(h["date_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
                                   .astimezone().strftime("%Y-%m-%dT%H:%M:%S"))
            except Exception:                                    # noqa: BLE001
                pass
            h["operation"] = UPDATE_OP.get(h.get("operation"), h.get("operation"))
            h["result"] = UPDATE_RESULT.get(h.get("result"), h.get("result"))
            hist.append(h)
        out["update_history_7d"] = hist
        out["update_history_total"] = info.get("history_total")
        out["driver_updates_7d"] = [h for h in hist if is_driver_update(h.get("title", ""))]
        pend = [dict(p) for p in (info.get("pending") or [])]
        for p in pend:
            p["is_driver"] = p.get("type") == 2 or is_driver_update(p.get("title", ""))
        out["pending_updates"] = pend
        out["pending_reboot_risk"] = [p for p in pend if p["is_driver"] or p.get("reboot_behavior", 0) != 0]
        out["last_boot"] = info.get("last_boot")
        for k in ("history_error", "last_boot_error", "pending_error"):
            if info.get(k):
                out["errors"].append(f"{k}: {info[k]}")
    except Exception as e:                                       # noqa: BLE001
        out["errors"].append(f"更新の履歴（COM）を読めない: {e}")
    if not out.get("last_boot"):                                 # COM が使えなくても最後の起動は出す
        up = ctypes.windll.kernel32.GetTickCount64() / 1000.0
        out["last_boot"] = (now - dt.timedelta(seconds=up)).strftime("%Y-%m-%dT%H:%M:%S")
        out["last_boot_source"] = "GetTickCount64（高速スタートアップで実際より古く出ることがある）"
    if not a.no_events:
        try:
            out["power_events_7d"] = _ps_json(PS_EVENTS, timeout=120).get("events", [])
        except Exception as e:                                   # noqa: BLE001
            out["errors"].append(f"System イベントを読めない: {e}")
    out.update(window_state(now, a.window))
    risk = bool(out.get("pending_reboot_risk"))
    resume_note = ("再起動で止まったら、サインインの後に ops_tools.py wait（または progress.json）で止まった所を見て、"
                   "eval_resumable.py の同じコマンドで続きから回す（切れた試行は _incomplete_* に退避して同じ種で回し直す）。")
    if out["reboot_pending"]:
        out["advice"] = ("再起動待ち。02:00 などに再起動しうる。実行は続きから回せる包み（eval_resumable.py）だけで行う。"
                         "再起動すると実行中のプロセスと会話は止まり、サインイン画面で止まるので、人がサインインして再開する。" + resume_note)
    elif risk:
        out["advice"] = ("再起動待ちの印はないが、入っていない更新（ドライバか再起動しうるもの）が端末の控えにある。"
                         "02:00 に入って再起動する恐れがある。実行は続きから回せる包み（eval_resumable.py）だけで行う。"
                         "再起動の後は env-record でドライバの版を確かめる。" + resume_note)
    else:
        out["advice"] = "再起動待ちなし。" + resume_note
    out["notes"] = [
        "ドライバの更新は再起動待ちの印（RebootRequired・RebootPending）を出さずに 02:00 に入り、そのまま再起動することがある。"
        "実例: 10/08 01:38 の確認では両方 False だったが、02:00 に NVIDIA のドライバ（32.0.16.1088 = nvidia-smi 610.88）が入り、"
        "MoUsoCoreWorker が再起動した（サインインは 02:38、人の手）。再起動は更新が入った日だけで曜日の決まりがない"
        "（過去 120 日で 10/02 と 10/08 の 2 回）ので、時間帯を避ける規則は置かず、止まったら続きから回す（作者の決定 10/08）。"
        "window・in_window・minutes_to_window_start は参考。",
        "COM の更新の履歴の時刻は UTC（date_utc）。date_local が日本時間。",
        "pending_updates は端末の控え（Online=false）だけを見る。最後の検索より後に配られた更新は出ない。",
    ]
    if out.get("driver_updates_7d"):
        out["notes"].append("直近 7 日にドライバの更新がある: GPU のドライバが変わっていれば、以後の GPU の測定は別の環境として env-record に残す。")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    if not a.no_log:
        S4.mkdir(parents=True, exist_ok=True)
        brief = {k: out[k] for k in ("checked", "RebootRequired", "RebootPending", "reboot_pending", "last_boot",
                                      "minutes_to_window_start", "in_window") if k in out}
        brief["pending_file_rename_other_pairs"] = out["PendingFileRenameOperations"]["other_pairs"]
        brief["n_updates_7d"] = len(out.get("update_history_7d", []))
        brief["n_driver_updates_7d"] = len(out.get("driver_updates_7d", []))
        brief["n_pending_reboot_risk"] = len(out.get("pending_reboot_risk", []))
        with open(S4 / "wu_check.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(brief, ensure_ascii=False) + "\n")
    if out["reboot_pending"]:
        return 1
    if risk:
        return 3
    return 2 if out["errors"] else 0                             # 読めなかったものがあれば「問題なし」にしない


# ---------------------------------------------------------------- env-record
def _pkg_version(name: str):
    from importlib import metadata
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _sha256_file(p: pathlib.Path, bufsize=1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


PS_DRIVER_HISTORY = r"""
$r = [ordered]@{}
try {
  $s = New-Object -ComObject Microsoft.Update.Session
  $h = $s.CreateUpdateSearcher()
  $n = $h.GetTotalHistoryCount()
  $rows = @()
  if ($n -gt 0) {
    foreach ($x in $h.QueryHistory(0, [Math]::Min($n, 1000))) {
      if ([string]$x.Title -match 'NVIDIA') {
        $rows += [pscustomobject]@{ date_utc = $x.Date.ToString('yyyy-MM-ddTHH:mm:ssZ'); title = [string]$x.Title; result = [int]$x.ResultCode }
      }
    }
  }
  $r.rows = @($rows)
} catch { $r.error = [string]$_.Exception.Message }
$r | ConvertTo-Json -Depth 4 -Compress
"""


def _utc_to_local(s: str) -> dt.datetime:
    return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc).astimezone().replace(tzinfo=None)


def stage3_window() -> dict:
    """評価 V3S3 の試行の記録の書かれた時刻の幅（ファイルの更新時刻を読むだけ）。"""
    d = ROOT / "outputs" / "v2eval" / "V3S3"
    ts = [p.stat().st_mtime for p in d.glob("*/trial_[0-9][0-9][0-9][0-9].json")]
    ts += [p.stat().st_mtime for p in d.glob("*/run_[0-9][0-9][0-9][0-9].json")]
    if not ts:
        return {}
    f = lambda x: dt.datetime.fromtimestamp(x).strftime("%Y-%m-%d %H:%M:%S")       # noqa: E731
    return {"first": f(min(ts)), "last": f(max(ts)), "n_files": len(ts), "source": "outputs/v2eval/V3S3/*/trial_*.json・run_*.json の更新時刻"}


def driver_history() -> dict:
    """Windows Update の履歴（COM、読むだけ）から NVIDIA のドライバの入れ替えを並べ、評価 V3S3 の時点の版を推す。"""
    info = _ps_json(PS_DRIVER_HISTORY, timeout=180)
    if info.get("error"):
        return {"error": info["error"]}
    rows = []
    for r in info.get("rows", []) or []:
        loc = _utc_to_local(r["date_utc"])
        rows.append({"date_local": loc.strftime("%Y-%m-%d %H:%M:%S"), "title": r["title"],
                     "result": UPDATE_RESULT.get(r["result"], r["result"]), "nvidia_smi_version": nvidia_from_windows_version(r["title"])})
    rows.sort(key=lambda x: x["date_local"])
    res = {"source": "Windows Update の履歴（COM Microsoft.Update.Session、QueryHistory）。時刻は日本時間に直した", "installs": rows}
    w = stage3_window()
    res["stage3_eval_window"] = w
    if w:
        ok = [r for r in rows if r["result"] == "succeeded" and r["date_local"] <= w["first"]]
        after = [r for r in rows if r["result"] == "succeeded" and w["first"] < r["date_local"] <= w["last"]]
        res["stage3_driver"] = {
            "nvidia_smi_version": ok[-1]["nvidia_smi_version"] if ok else None,
            "installed": ok[-1]["date_local"] if ok else None,
            "changed_during_stage3": bool(after),
            "how": "段階 3 の評価の最初の記録より前に成功した最後の NVIDIA の更新（推定。nvidia-smi の当時の出力は残っていない）"}
    return res


def cmd_env_record(a) -> int:
    rec = {"recorded": _now().strftime("%Y-%m-%d %H:%M:%S"), "host": platform.node()}
    try:
        r = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,pstate", "--format=csv,noheader,nounits"], 30)
        name, drv, mt, mu, ps = [x.strip() for x in r.stdout.decode("utf-8", "replace").strip().splitlines()[0].split(",")]
        rec["gpu"] = {"name": name, "driver": drv, "memory_total_mib": int(mt), "memory_used_mib": int(mu), "pstate": ps}
        full = _run(["nvidia-smi"], 30).stdout.decode("utf-8", "replace")
        m = re.search(r"CUDA (?:UMD )?Version:\s*([\d.]+)", full)          # 610 系は「CUDA UMD Version」と出す
        rec["gpu"]["cuda_driver_api"] = m.group(1) if m else None
    except Exception as e:                                       # noqa: BLE001
        rec["gpu"] = {"error": str(e)}
    try:
        rec["gpu_driver_history"] = driver_history()
    except Exception as e:                                       # noqa: BLE001
        rec["gpu_driver_history"] = {"error": str(e)}
    s3 = (rec["gpu_driver_history"] or {}).get("stage3_driver") or {}
    rec["gpu_driver_note"] = (f"段階 3 の評価（V3S3）はドライバ {s3.get('nvidia_smi_version')}（推定）、段階 4 の測定は "
                              f"{(rec.get('gpu') or {}).get('driver')}。10/08 02:00 の Windows Update でドライバが入れ替わったので、"
                              "段階 4 の GPU の測定（物差し K を含む）は段階 3 と別の環境として扱う。")
    os_info ={"platform": platform.platform(), "version": platform.version(), "machine": platform.machine()}
    for k in ("ProductName", "DisplayVersion", "EditionID", "CurrentBuildNumber", "UBR"):
        try:
            os_info[k] = _reg_value(r"SOFTWARE\Microsoft\Windows NT\CurrentVersion", k)
        except Exception:                                        # noqa: BLE001
            os_info[k] = None
    rec["os"] = os_info
    rec["cpu"] = {"processor": platform.processor(), "logical_cpus": os.cpu_count()}
    rec["python"] = {"version": sys.version, "executable": sys.executable}
    names = ["torch", "torchvision", "lerobot", "mujoco", "numpy", "transformers", "safetensors", "psutil", "pytest"]
    rec["packages"] = {n: _pkg_version(n) for n in names}
    try:
        r = _run([sys.executable, "-I", "-c",
                  "import torch,json;print(json.dumps({'cuda_available':torch.cuda.is_available(),'torch_cuda':torch.version.cuda,"
                  "'cudnn':torch.backends.cudnn.version(),'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))"], 180)
        rec["torch_runtime"] = json.loads(r.stdout.decode("utf-8", "replace").strip().splitlines()[-1])
    except Exception as e:                                       # noqa: BLE001
        rec["torch_runtime"] = {"error": str(e)}
    rec["git"] = {"head": _git("rev-parse", "HEAD"), "head_subject": _git("log", "-1", "--format=%s"),
                  "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
                  "tag": FREEZE_TAG, "tag_commit_sha": _git("rev-parse", f"{FREEZE_TAG}^{{commit}}"),
                  "tag_object_sha": _git("rev-parse", FREEZE_TAG),
                  "dirty_files": [ln for ln in _git("status", "--short").splitlines() if ln.strip()]}
    rec["git"]["head_is_tag"] = rec["git"]["head"] == rec["git"]["tag_commit_sha"]
    hashes = {}
    for rel in ("scripts/eval_trials.py", "scripts/eval_sets.py", "scripts/compare_subtask.py", "configs/default.yaml",
                "docs/freeze/s3_hashes.json", "scripts/ops_tools.py", "scripts/eval_resumable.py"):
        p = ROOT / rel
        hashes[rel] = _sha256_file(p) if p.is_file() else None
    rec["file_sha256"] = hashes
    rec["memory"] = memory_gb()
    try:
        du = __import__("shutil").disk_usage(str(ROOT))
        rec["disk"] = {"free_gb": round(du.free / 1024 ** 3, 1), "total_gb": round(du.total / 1024 ** 3, 1)}
    except Exception:                                            # noqa: BLE001
        pass
    S4.mkdir(parents=True, exist_ok=True)
    dst = S4 / "env_record.json"
    if dst.is_file():                                            # 前の記録は消さず、1 行の履歴に残す
        try:
            old = json.loads(dst.read_text(encoding="utf-8"))
            with open(S4 / "env_record_history.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(old, ensure_ascii=False) + "\n")
        except Exception:                                        # noqa: BLE001
            pass
    _dump(rec, dst)
    print(json.dumps({k: rec[k] for k in ("recorded", "gpu", "git", "memory")}, ensure_ascii=False, indent=1)[:2500])
    print(f"-> {dst}")
    return 0


# ---------------------------------------------------------------- backup-manifest
def _latest_train_run(model: str):
    """学習の出力 train_<モデル>_2* のうち最後のもの。"""
    runs = sorted((ROOT / "outputs" / "train").glob(f"train_{model}_2*"))
    return runs[-1] if runs else None


def backup_groups() -> dict:
    """バックアップの最小の組。値は (説明, [(相対パス, 'dir'|'file', 除外する下位の名前の集合)])。"""
    g = {
        "v3s3_records": ("段階 3 の最終評価の記録（試行ごとの json・npz・runtime、集計の元）",
                         [("outputs/v2eval/V3S3", "dir"), ("outputs/v2eval/V3S3_logs", "dir"),
                          ("outputs/v2eval/V3S3_queue.json", "file"), ("outputs/v2eval/V3S3_run.log", "file"),
                          ("outputs/v2eval/V3S3_report.log", "file"), ("outputs/v2eval/V3S3_report_H.log", "file")]),
        "results": ("集計の結果（outputs/results）", [("outputs/results", "dir")]),
        "demo_v2": ("発表用の動画の素材（outputs/demo_v2）", [("outputs/demo_v2", "dir")]),
        "paper": ("説明資料（作者が編集中。コピーの直前に作り直す）", [("paper", "dir")]),
        "docs_freeze": ("凍結のハッシュ（docs/freeze）", [("docs/freeze", "dir")]),
    }
    for m in ("R1v3", "N1v3"):
        run = _latest_train_run(m)
        if run is None:
            g[f"ckpt_{m}"] = (f"{m} のチェックポイント（見つからない）", [])
            continue
        rel = run.relative_to(ROOT).as_posix()
        items = [(f"{rel}/checkpoints/020000", "dir")]
        for name in ("conversion.json", "loss.csv", "loss.png", "train_launch_config.json", "train_run.json"):
            if (run / name).is_file():
                items.append((f"{rel}/{name}", "file"))
        g[f"ckpt_{m}"] = (f"{m} のチェックポイント（評価に使う 2 万手。ほかの保存点と last は含めない）", items)
    return g


def _walk_files(rel: str, kind: str):
    p = ROOT / rel
    if kind == "file":
        if p.is_file():
            yield rel
        return
    for dp, dns, fns in os.walk(p):
        dns[:] = [d for d in dns if d != "__pycache__"]
        for fn in fns:
            if fn.endswith((".pyc", ".tmp")):
                continue
            fp = pathlib.Path(dp) / fn
            if fp.is_symlink():
                continue
            yield fp.relative_to(ROOT).as_posix()


def cmd_backup_manifest(a) -> int:
    groups = backup_groups()
    res = {"schema": "s4_backup_manifest_v1", "created": _now().strftime("%Y-%m-%d %H:%M:%S"), "root": str(ROOT),
           "git_head": _git("rev-parse", "HEAD"), "groups": {}, "missing_roots": []}
    t0 = time.time()
    for gname, (desc, items) in groups.items():
        paths = []
        if not items:                                            # 元が見つからない組（例: チェックポイント）も見落とさない
            res["missing_roots"].append(f"{gname}: {desc}")
        for rel, kind in items:
            if not (ROOT / rel).exists():
                res["missing_roots"].append(rel)
                continue
            paths += list(_walk_files(rel, kind))
        paths = sorted(set(paths))

        def one(rel):
            fp = ROOT / rel
            return {"path": rel, "bytes": fp.stat().st_size, "sha256": None if a.no_hash else _sha256_file(fp)}
        with ThreadPoolExecutor(max_workers=4) as ex:
            files = list(ex.map(one, paths))
        gh = hashlib.sha256("\n".join(f"{f['path']}\t{f['bytes']}\t{f['sha256']}" for f in files).encode("utf-8")).hexdigest()
        res["groups"][gname] = {"description": desc, "roots": [r for r, _ in items], "n_files": len(files),
                                "bytes": sum(f["bytes"] for f in files), "group_sha256": gh, "files": files}
        print(f"{gname}: {len(files)} ファイル {sum(f['bytes'] for f in files) / 1024 ** 2:,.0f} MB（{time.time() - t0:.0f} s）", flush=True)
    res["total_files"] = sum(g["n_files"] for g in res["groups"].values())
    res["total_bytes"] = sum(g["bytes"] for g in res["groups"].values())
    res["manifest_sha256"] = hashlib.sha256("".join(g["group_sha256"] for g in res["groups"].values()).encode()).hexdigest()
    res["copy_instructions"] = {
        "status": "コピーは実行していない。置き場所（外付けか共有の NAS か）は作者が決める。",
        "steps": [
            "1. 置き場所 <DEST>（このフォルダの外。例 E:\\recovery_vla_backup_2026-10）を作者が決める。空きは total_bytes の 1.2 倍以上。",
            "2. paper は編集中なので、コピーの直前に backup-manifest --force で一覧を作り直す（--force は古い一覧を退避する）。",
            "3. 組ごとにコピー（例。実行は作者）: robocopy <ROOT>\\<相対パスの根> <DEST>\\<相対パスの根> /E /COPY:DAT /DCOPY:DAT /R:1 /W:1 /NP /LOG+:<DEST>\\robocopy.log",
            "   根は各組の roots を見る。ファイル 1 個の根（例 outputs\\v2eval\\V3S3_queue.json）は robocopy <親フォルダ> <DEST>\\<親フォルダ> <ファイル名> /COPY:DAT。",
            "4. 照合: .venv\\Scripts\\python.exe scripts\\ops_tools.py backup-verify --dest <DEST>（終了コード 0 で全ファイル一致）。",
            "5. 一覧 outputs\\s4\\backup_manifest.json 自体も <DEST> に置く（manifest_sha256 を別の場所のメモにも書く）。",
        ],
        "excluded": ["outputs/sealed（開かない）", "R1v3/N1v3 の 2 万手以外の保存点と last", "outputs/datasets・outputs/train の他の実行（再生成できる）",
                     "__pycache__・*.pyc・*.tmp・シンボリックリンク"],
    }
    S4.mkdir(parents=True, exist_ok=True)
    dst = S4 / "backup_manifest.json"
    if dst.is_file():
        if not a.force:
            raise SystemExit(f"{dst} が既にある。作り直すなら --force（古い一覧は退避する）")
        os.replace(dst, S4 / f"backup_manifest_{_now().strftime('%Y%m%d-%H%M%S')}.json")
    _dump(res, dst)
    print(f"合計 {res['total_files']} ファイル {res['total_bytes'] / 1024 ** 3:.2f} GB、manifest_sha256 {res['manifest_sha256']}")
    if res["missing_roots"]:
        print("見つからない根:", res["missing_roots"])
    print(f"-> {dst}（コピーは実行していない）")
    return 0 if not res["missing_roots"] else 1


def cmd_backup_verify(a) -> int:
    man = json.loads(pathlib.Path(a.manifest).read_text(encoding="utf-8"))
    dest = pathlib.Path(a.dest)
    bad = {"missing": [], "size": [], "hash": [], "no_hash_in_manifest": []}
    n = 0
    for gname, g in man["groups"].items():
        for f in g["files"]:
            n += 1
            p = dest / f["path"]
            if not p.is_file():
                bad["missing"].append(f["path"])
            elif p.stat().st_size != f["bytes"]:
                bad["size"].append(f["path"])
            elif not a.quick and not f["sha256"]:
                bad["no_hash_in_manifest"].append(f["path"])     # --no-hash の一覧ではハッシュを照合できない（黙って飛ばさない）
            elif not a.quick and _sha256_file(p) != f["sha256"]:
                bad["hash"].append(f["path"])
    ok = not any(bad.values())
    print(json.dumps({"checked": n, "ok": ok, **{k: {"n": len(v), "first": v[:10]} for k, v in bad.items()}}, ensure_ascii=False, indent=1))
    return 0 if ok else 1


# ---------------------------------------------------------------- wait
def _alive(pid, create_time=None) -> bool:
    try:
        import psutil
        if not psutil.pid_exists(pid):
            return False
        p = psutil.Process(pid)
        if create_time is not None and abs(p.create_time() - float(create_time)) > 2.0:
            return False                                         # PID が別のプロセスに使われている
        return p.status() != psutil.STATUS_ZOMBIE
    except ImportError:
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
        if not h:
            return False
        try:
            code = ctypes.c_ulong()
            ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
            if code.value != 259:
                return False
            if create_time is not None:                          # psutil が無くても PID の使い回しを見分ける
                ct = _win_create_time(h)
                if ct is not None and abs(ct - float(create_time)) > 2.0:
                    return False
            return True
        finally:
            ctypes.windll.kernel32.CloseHandle(h)


def _win_create_time(h):
    """プロセスのハンドルから作られた時刻（エポック秒。psutil.Process.create_time と同じ GetProcessTimes の値）。"""
    ft = [ctypes.c_ulonglong() for _ in range(4)]                # 作成・終了・カーネル・ユーザーの FILETIME（100 ns、1601 年から）
    if not ctypes.windll.kernel32.GetProcessTimes(h, *[ctypes.byref(x) for x in ft]):
        return None
    return ft[0].value / 1e7 - 11644473600.0


def _create_time(pid):
    try:
        import psutil
        return psutil.Process(int(pid)).create_time()
    except ImportError:
        pass
    except Exception:                                            # noqa: BLE001
        return None
    try:
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
    except Exception:                                            # noqa: BLE001
        return None
    if not h:
        return None
    try:
        return _win_create_time(h)
    finally:
        ctypes.windll.kernel32.CloseHandle(h)


def _read_progress(path: pathlib.Path):
    """progress.json を読む。無ければ None、3 回読めなければ {"status": "unreadable", "_unreadable": True}。
    BOM 付き（PowerShell 5.1 の Out-File などで書いた形）も読む。"""
    for _ in range(3):
        try:
            d = json.loads(path.read_text(encoding="utf-8-sig"))
            if isinstance(d, dict):
                return d
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            pass
        time.sleep(0.5)
    return {"status": "unreadable", "_unreadable": True}


def _age_min(ts: str) -> float:
    try:
        return (_now() - dt.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")).total_seconds() / 60
    except Exception:                                            # noqa: BLE001
        return 0.0


def _progress_age_min(pr: dict, path=None) -> float:
    """最後の更新からの分。updated が読めなければファイルの更新時刻で測る（未知の状態の progress にも途絶の判定を当てるため）。"""
    try:
        return (_now() - dt.datetime.strptime(pr["updated"], "%Y-%m-%d %H:%M:%S")).total_seconds() / 60
    except Exception:                                            # noqa: BLE001
        pass
    try:
        return (time.time() - pathlib.Path(path).stat().st_mtime) / 60
    except Exception:                                            # noqa: BLE001
        return 0.0


# 固まった試行の判定。eval_resumable.py の心拍の別スレッドは、本体が GPU の推論などで固まっても updated を
# 30 s ごとに書き続けるので、updated では見つからない。そこで今の試行の始め（progress の current_started。試行ごとに本体だけが
# 書く）からの経過が「制限時間 × 実時間の倍率 ＋ 余裕」を超えたら固まったとみなす。倍率は実測（run 1.6〜1.86 倍、
# task 約 3.1 倍）に、3 本並行の遅延と LLM の API の 1 回の回し直しを見込んで 4 倍、余裕は世界の作り直し・API の待ち
# の分で 15 分（run 60 s なら 19 分、連続タスクの全体 200 s なら約 28 分）。判定するのは status が running のときだけ（loading・
# memory_wait などの待ちは current_started を更新しないので当てない）。結果を見る前に決めた値（成績は読まない）。
HUNG_FACTOR = 4.0
HUNG_MARGIN_MIN = 15.0
HUNG_DEFAULT_LIMIT_S = 200.0          # progress に制限時間が無いとき（最長の、連続タスクの全体 200 s）


def trial_limit_s(pr: dict):
    """progress の time_limits から 1 試行の打ち切りの秒（task は全体の打ち切り）。無ければ None。"""
    lim = (pr or {}).get("time_limits") or {}
    for k in ("task_time_limit_s", "time_limit_s"):
        v = lim.get(k) if isinstance(lim, dict) else None
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
            return float(v)
    return None


def hung_check(pr: dict, now=None, factor: float = HUNG_FACTOR, margin_min: float = HUNG_MARGIN_MIN,
               default_limit_s: float = HUNG_DEFAULT_LIMIT_S):
    """固まった試行なら {"current", "current_started", "elapsed_min", "allowed_min", ...}、そうでなければ None。
    factor が 0 なら判定しない。"""
    if not pr or pr.get("status") != "running" or not factor:
        return None
    cs = pr.get("current_started")
    try:
        t = dt.datetime.strptime(str(cs), "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return None
    el = ((now or _now()) - t).total_seconds() / 60
    lim = trial_limit_s(pr) or float(default_limit_s)
    allow = lim * float(factor) / 60 + float(margin_min)
    if el <= allow:
        return None
    return {"condition": pr.get("condition"), "current": pr.get("current"), "current_seed": pr.get("current_seed"),
            "current_started": cs, "elapsed_min": round(el, 1), "allowed_min": round(allow, 1), "limit_s": lim,
            "rule": f"今の試行の始めからの経過 > 制限時間 {lim:g} s × {factor:g} + {margin_min:g} 分"}


def _tail(path, n=15) -> list:
    try:
        return pathlib.Path(path).read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
    except Exception:                                            # noqa: BLE001
        return []


def evaluate_target(t: dict, a) -> None:
    """t を 1 回調べ、終わっていれば t['final'] に {'outcome','code'} を入れる。
    途絶（コード 4）は次の 4 つ: 更新が --stall-min 分途絶えた（stalled。LIVE でない未知の状態にも当てる）、今の試行が固まった
    （hung。hung_check）、包み（98_s4_d_plan bundle）が子の固まりを状態に書いた（hung_child）、progress.json を読めない状態が
    --stall-min 分続いた（unreadable）。"""
    if t["kind"] == "pid":
        if not _alive(t["pid"], t.get("create_time")):          # 始めに控えた create_time で PID の使い回しを見分ける
            t["final"] = {"outcome": "pid_gone", "code": 0}
        return
    pr = _read_progress(t["path"])
    if pr is not None and pr.get("_unreadable"):
        t.setdefault("unreadable_since", time.time())
        t["unreadable_n"] = t.get("unreadable_n", 0) + 1
        if (time.time() - t["unreadable_since"]) / 60 > a.stall_min:
            t["final"] = {"outcome": "unreadable", "code": 4}
        return
    t.pop("unreadable_since", None)
    t["progress"] = pr
    if pr is None:
        if (time.time() - t["t0"]) / 60 > a.appear_min:
            t["final"] = {"outcome": "no_progress_file", "code": 3}
        return
    st = pr.get("status")
    if st in FINAL_STATUS:
        t["final"] = {"outcome": st, "code": FINAL_STATUS[st]}
        return
    pid = pr.get("pid")
    if pid and not _alive(pid, pr.get("proc_create_time")):
        pr2 = _read_progress(t["path"])                           # 最終の状態を書いた直後に消えた場合
        if pr2 and pr2.get("status") in FINAL_STATUS:
            t["progress"] = pr2
            t["final"] = {"outcome": pr2["status"], "code": FINAL_STATUS[pr2["status"]]}
        else:
            t["final"] = {"outcome": "crashed", "code": 3}
        return
    hung = hung_check(pr, factor=getattr(a, "hung_factor", HUNG_FACTOR), margin_min=getattr(a, "hung_margin_min", HUNG_MARGIN_MIN))
    if hung:
        t["hung"] = [hung]
        t["final"] = {"outcome": "hung", "code": 4}
        return
    if pr.get("hung_children"):                                  # 包みが見つけた子の固まり（98_s4_d_plan.py bundle が書く）
        t["hung"] = pr["hung_children"]
        t["final"] = {"outcome": "hung_child", "code": 4}
        return
    if _progress_age_min(pr, t["path"]) > a.stall_min:           # LIVE の状態も未知の状態も
        t["final"] = {"outcome": "stalled", "code": 4}


def cmd_wait(a) -> int:
    if not a.progress and not a.pid:
        raise SystemExit("--progress か --pid が要る")
    targets = [{"kind": "progress", "path": pathlib.Path(p), "t0": time.time()} for p in a.progress or []]
    targets += [{"kind": "pid", "pid": int(p), "t0": time.time(), "create_time": _create_time(p)} for p in a.pid or []]
    t0, last_report, timed_out = time.time(), 0.0, False
    print(f"[wait] 開始 {_now():%H:%M:%S}: {len(targets)} 件を監視（間隔 {a.interval} s、途絶 {a.stall_min} 分、固まり 制限時間 × "
          f"{a.hung_factor:g} + {a.hung_margin_min:g} 分、時間切れ {a.timeout_min if a.timeout_min else 'なし'} 分）", flush=True)
    while True:
        for t in targets:
            if "final" not in t:
                evaluate_target(t, a)
        if all("final" in t for t in targets):
            break
        if a.timeout_min and (time.time() - t0) / 60 > a.timeout_min:
            timed_out = True
            break
        if (time.time() - last_report) / 60 >= a.report_min:
            last_report = time.time()
            for t in targets:
                if t["kind"] == "progress" and t.get("progress"):
                    p = t["progress"]
                    print(f"[wait] {_now():%H:%M:%S} {p.get('condition')} {p.get('status')} {p.get('done')}/{p.get('total')} "
                          f"成功 {p.get('successes')} 空き {p.get('free_phys_gb')} GB", flush=True)
        time.sleep(a.interval)
    rows, code = [], 0
    for t in targets:
        fin = t.get("final") or {"outcome": "timeout", "code": 5}
        code = max(code, fin["code"])
        row = {"target": str(t.get("path") or t.get("pid")), "outcome": fin["outcome"], "code": fin["code"]}
        p = t.get("progress")
        if p:
            for k in ("experiment", "condition", "status", "done", "total", "skipped_complete", "ran_this_session", "successes",
                      "started", "updated", "elapsed_min", "last_trial_wall_s", "free_phys_gb", "error", "stop_reason", "pid"):
                if k in p:
                    row[k] = p[k]
        if t.get("hung"):
            row["hung"] = t["hung"]
        if t.get("unreadable_n"):
            row["unreadable_reads"] = t["unreadable_n"]
        if fin["outcome"] in ("crashed", "error", "stalled", "hung", "hung_child", "unreadable") and a.log:
            row["log_tail"] = _tail(a.log)
        rows.append(row)
    summary = {"finished": _now().strftime("%Y-%m-%d %H:%M:%S"), "waited_min": round((time.time() - t0) / 60, 1),
               "exit_code": code, "timed_out": timed_out, "targets": rows}
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return code


# ---------------------------------------------------------------- main
def main(argv=None) -> int:
    _utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("wu-check", help="再起動待ちと更新の履歴（終了コード 1＝再起動待ち）")
    p.add_argument("--window", default=QUIET_DEFAULT, help="参考に分数を出す窓 HH:MM-HH:MM（実行は縛らない）")
    p.add_argument("--no-log", action="store_true", help="outputs/s4/wu_check.jsonl に追記しない")
    p.add_argument("--no-events", action="store_true", help="System イベントログを読まない")
    p = sub.add_parser("env-record", help="環境の記録 outputs/s4/env_record.json")
    p = sub.add_parser("backup-manifest", help="バックアップの SHA-256 の一覧（コピーはしない）")
    p.add_argument("--force", action="store_true", help="既にある一覧を退避して作り直す")
    p.add_argument("--no-hash", action="store_true", help="試験用: ハッシュを取らず大きさだけ")
    p = sub.add_parser("backup-verify", help="コピー先を一覧と照合する（読み取りだけ）")
    p.add_argument("--dest", required=True)
    p.add_argument("--manifest", default=str(S4 / "backup_manifest.json"))
    p.add_argument("--quick", action="store_true", help="大きさだけ照合する")
    p = sub.add_parser("wait", help="実行の終了・エラーを待って要約を出す")
    p.add_argument("--progress", nargs="*", default=[], help="progress.json（複数可）")
    p.add_argument("--pid", nargs="*", default=[], help="PID（複数可。終了だけを見る）")
    p.add_argument("--interval", type=float, default=30.0, help="確認の間隔 [s]")
    p.add_argument("--stall-min", type=float, default=30.0,
                   help="progress.json の更新がこの分数途絶えたら（読めない状態がこの分数続いても）止まったとみなす")
    p.add_argument("--hung-factor", type=float, default=HUNG_FACTOR,
                   help="今の試行の始めからの経過が 制限時間 × これ + --hung-margin-min 分を超えたら固まったとみなす（0 で判定しない）")
    p.add_argument("--hung-margin-min", type=float, default=HUNG_MARGIN_MIN)
    p.add_argument("--timeout-min", type=float, default=1440.0,
                   help="時間切れ（既定 24 時間。束の全体を見るときは長くする。0 なら時間切れなし）")
    p.add_argument("--appear-min", type=float, default=10.0, help="progress.json がこの分数のうちに現れなければエラー")
    p.add_argument("--report-min", type=float, default=15.0, help="途中経過の行を出す間隔 [分]")
    p.add_argument("--log", default=None, help="異常のときに末尾を要約へ入れるログ")
    a = ap.parse_args(argv)
    return {"wu-check": cmd_wu_check, "env-record": cmd_env_record, "backup-manifest": cmd_backup_manifest,
            "backup-verify": cmd_backup_verify, "wait": cmd_wait}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
