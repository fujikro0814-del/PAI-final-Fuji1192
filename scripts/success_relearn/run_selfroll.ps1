# Collect the policy's own successes for retraining (docs/api/r2_protocol.md 2). Runs Rdag on seed range S_A (PC A) or S_B (PC B),
# split into -Lanes shards that run in parallel on the GPU. Output folders are fixed, so running this again continues where it stopped.
# Start detached:  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\success_relearn\run_selfroll.ps1 -Machine A -Lanes 3 -Quota 250 -Until "2026-10-24 06:00"
# Stop: create outputs\r2\runs\STOP_R2S (checked before starting; to stop running shards, wait for -Until or kill python).
# ASCII only (Windows PowerShell 5.1 reads BOM-less files as ANSI).
param(
    [ValidateSet('A', 'B')][string]$Machine = 'A',
    [int]$Lanes = 3,
    [int]$Quota = 250,
    [string]$Until = '',
    [string]$Root = 'C:\PAI\wt_r2'
)
$py = 'C:\PAI\recovery_vla\.venv\Scripts\python.exe'
$dir = Join-Path $Root 'outputs\r2\runs'
New-Item -ItemType Directory -Force $dir | Out-Null
$log = Join-Path $dir "run_r2_selfroll_$Machine.log"
$stop = Join-Path $dir 'STOP_R2S'
$band = "S_$Machine"
function L($m) { "$(Get-Date -Format s) $m" | Out-File -FilePath $log -Encoding utf8 -Append }
$env:PYTHONUTF8 = '1'; $env:HF_HUB_OFFLINE = '1'; $env:PYTHONPATH = "$Root\src;$Root"
Set-Location $Root
L "start pid $PID machine $Machine band $band lanes $Lanes quota $Quota until '$Until'"
if (Test-Path $stop) { L 'STOP file found'; exit 1 }

$per = [int][math]::Ceiling($Quota / $Lanes)
$procs = @
for ($i = 0; $i -lt $Lanes; $i++) {
    $o = Join-Path $dir "selfroll_${Machine}_s$i.out"; $e = Join-Path $dir "selfroll_${Machine}_s$i.err"
    $a = @('scripts\success_relearn\selfroll_data.py', 'selfroll', '--band', $band, '--quota', "$per", '--shard', "$i/$Lanes",
           '--out', "outputs\gen\R2S_${band}_s${i}of${Lanes}_main")
    if ($Until) { $a += @('--until', "`"$Until`"") }
    $procs += Start-Process -FilePath $py -ArgumentList $a -NoNewWindow -PassThru -RedirectStandardOutput $o -RedirectStandardError $e
    $null = $procs[-1].Handle   # keep the handle, or PS 5.1 may lose ExitCode
    L "selfroll $band shard $i/$Lanes quota $per pid $($procs[-1].Id)"
    Start-Sleep -Seconds 60     # stagger the model loads (memory)
}
$procs | ForEach-Object { $_.WaitForExit }
$rcs = ($procs | ForEach-Object { $_.ExitCode }) -join ','
L "selfroll rc $rcs"
if (($procs | Where-Object { $_.ExitCode -ne 0 }).Count) { L "STOP: a shard failed (see selfroll_${Machine}_s*.err)"; exit 3 }
L 'ALL DONE'
