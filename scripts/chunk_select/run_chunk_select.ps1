# Learning how to pick action chunks (noise-space RL DSRL, or a learned success score; docs/api/r2c_plan.md): start the workers
# (and, on the learner PC, the learner) for one run.
# Workers run seed shards of range C_train (80000-89999) in parallel; the learner reads all episode files and rewrites actor.pt.
# The base policy is Rdag by default; for R2a pass -Model R2a -Ckpt <pretrained_model folder> -CkptSha <sha256 of model.safetensors>.
#   PC B (learner + 3 workers):
#     powershell -NoProfile -ExecutionPolicy Bypass -File scripts\chunk_select\run_chunk_select.ps1 -Machine B -Learner -Shards 0,1,2 -Total 5 -Shared <folder> -Until "2026-10-27 20:00"
#   PC A (2 workers that write to the shared folder and read the mirrored actor):
#     powershell -NoProfile -ExecutionPolicy Bypass -File scripts\chunk_select\run_chunk_select.ps1 -Machine A -Shards 3,4 -Total 5 -Shared <folder> -Until "..."
# Running it again continues (finished seeds are skipped, the learner resumes from state.pt). Stop: create <run>\STOP.
# ASCII only (Windows PowerShell 5.1 reads BOM-less files as ANSI).
param(
    [ValidateSet('A', 'B')][string]$Machine = 'B',
    [switch]$Learner,
    [string]$Shards = '0,1,2',      # a string: powershell -File does not pass arrays
    [int]$Total = 5,
    [string]$Shared = '',
    [string]$Run = 'run3',          # run folder under outputs\r2c (earlier runs are kept)
    [switch]$Select,                # make K action chunks from ordinary noise and pick by the learned success score
    [ValidateSet('C_train', 'C_pilot')][string]$Band = 'C_train',
    [ValidateSet('Rdag', 'R2a')][string]$Model = 'Rdag',
    [string]$Ckpt = '',
    [string]$CkptSha = '',
    [string]$Until = '',
    [string]$Root = 'C:\PAI\wt_r2c'
)
$py = 'C:\PAI\recovery_vla\.venv\Scripts\python.exe'
$dir = Join-Path $Root 'outputs\r2c\runs'
New-Item -ItemType Directory -Force $dir | Out-Null
$log = Join-Path $dir "run_r2c_$Machine.log"
function L($m) { "$(Get-Date -Format s) $m" | Out-File -FilePath $log -Encoding utf8 -Append }
$env:PYTHONUTF8 = '1'; $env:HF_HUB_OFFLINE = '1'; $env:PYTHONPATH = "$Root\src;$Root"; $env:HF_HOME = "$Root\models\hf_home"
Set-Location $Root
$shardList = @($Shards -split ',' | ForEach-Object { [int]$_.Trim })
L "start pid $PID model $Model ckpt '$Ckpt' band $Band machine $Machine learner $Learner shards $($shardList -join ',')/$Total shared '$Shared' run $Run select $Select until '$Until'"
$wfile = if ($Select) { 'scorer.pt' } else { 'actor.pt' }
$local = "outputs\r2c\$Run"
$sharedRun = if ($Shared) { Join-Path $Shared $Run } else { '' }
if ($Machine -eq 'A' -and -not $sharedRun) { L 'STOP: machine A needs -Shared'; exit 2 }
if ($sharedRun) { New-Item -ItemType Directory -Force (Join-Path $sharedRun 'A\episodes') | Out-Null }
# seeds already used by earlier runs are not run again (each training seed once)
if ($Machine -eq 'A') { $prevDirs = @(Get-ChildItem $Shared -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -ne $Run } | ForEach-Object { Join-Path $_.FullName 'A\episodes' }) }
else { $prevDirs = @(Get-ChildItem (Join-Path $Root 'outputs\r2c') -Directory -Filter 'run*' -ErrorAction SilentlyContinue | Where-Object { $_.Name -ne $Run } | ForEach-Object { Join-Path $_.FullName 'episodes' }) }
$prevDirs = @($prevDirs | Where-Object { Test-Path $_ })
L "skip seeds used in: $($prevDirs -join '; ')"
$procs = @
if ($Learner) {
    $lcmd = if ($Select) { 'learner-select' } else { 'learner' }
    $a = @('scripts\chunk_select\dsrl_run.py', $lcmd, '--out', $local, '--episodes-glob', "$local\episodes\*.npz")
    if ($sharedRun) { $a += @("`"$sharedRun\A\episodes\*.npz`"", '--mirror', "`"$sharedRun\$wfile`"") }
    if ($Until) { $a += @('--until', "`"$Until`"") }
    $p = Start-Process -FilePath $py -ArgumentList $a -NoNewWindow -PassThru -RedirectStandardOutput (Join-Path $dir "learner_${Machine}_$Run.out") -RedirectStandardError (Join-Path $dir "learner_${Machine}_$Run.err")
    $null = $p.Handle
    $procs += $p
    L "learner pid $($p.Id)"
    Start-Sleep -Seconds 20     # write the first actor.pt before the workers start
}
foreach ($i in $shardList) {
    if ($Machine -eq 'A') { $out = "`"$sharedRun\A`""; $actor = "`"$sharedRun\$wfile`"" } else { $out = $local; $actor = "$local\$wfile" }
    $a = @('scripts\chunk_select\dsrl_run.py', 'worker', '--band', $Band, '--shard', "$i/$Total", '--out', $out, '--model', $Model)
    if ($Select) { $a += @('--select', '--scorer', $actor) } else { $a += @('--actor', $actor) }
    if ($Ckpt) { $a += @('--ckpt', "`"$Ckpt`"", '--ckpt-sha256', $CkptSha) }
    if ($Until) { $a += @('--until', "`"$Until`"") }
    if ($prevDirs.Count) { $a += @('--skip-seeds-from') + @($prevDirs | ForEach-Object { "`"$_`"" }) }
    $p = Start-Process -FilePath $py -ArgumentList $a -NoNewWindow -PassThru -RedirectStandardOutput (Join-Path $dir "worker_${Machine}_${Run}_s$i.out") -RedirectStandardError (Join-Path $dir "worker_${Machine}_${Run}_s$i.err")
    $null = $p.Handle
    $procs += $p
    L "worker shard $i/$Total pid $($p.Id)"
    Start-Sleep -Seconds 60     # stagger the model loads (memory)
}
$procs | ForEach-Object { $_.WaitForExit }
$rcs = ($procs | ForEach-Object { $_.ExitCode }) -join ','
L "rc $rcs"
if (($procs | Where-Object { $_.ExitCode -ne 0 }).Count) { L "STOP: a process failed (see worker_*_$Run*.err / learner_*_$Run.err)"; exit 3 }
L 'ALL DONE'
