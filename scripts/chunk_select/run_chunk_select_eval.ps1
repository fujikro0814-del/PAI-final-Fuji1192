# Evaluation launcher for condition Uc (R2a + pick among action chunks by the learned success score; docs/api/r2_protocol.md
# 6.1 and 7). PC A runs R2CE, R2CE2, R2CE3 and PC B runs R2CE_B, R2CE_B2, R2CE_B3 (one condition each, so one
# progress keeper per experiment, one lane each, all at the same time).
# Start it detached and hidden, for example:
#   $si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
#   Invoke-CimMethod Win32_Process Create @{CommandLine='powershell.exe -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass
#     -File C:\PAI\wt_r2c\scripts\chunk_select\run_chunk_select_eval.ps1 -Machine A -HeadWant <40-hex commit> -ScorerSha <64-hex>';
#     CurrentDirectory='C:\PAI\wt_r2c'; ProcessStartupInformation=$si}
# Trial run (seed 193592 on A, 193593 on B; no HEAD check):  ... run_chunk_select_eval.ps1 -Machine A -ScorerSha <64-hex> -Smoke
# Stop: create outputs\r2c\runs\STOP_R2CEVAL (or STOP_R2CEVAL_SMOKE). Running trials finish first, nothing new starts.
# Continue: delete that STOP file and run the same command again. Finished trials are kept and skipped.
# ASCII only (Windows PowerShell 5.1 reads BOM-less files as ANSI).
param([ValidateSet('A', 'B')][string]$Machine = 'A', [string]$HeadWant = '', [string]$ScorerSha = '', [double]$MinCommitGb = 6,
      [switch]$Smoke, [ValidateSet('C', 'D')][string]$Stage = 'C')
# -Stage D (protocol 6.2): condition Udc = Rdag + the scorer of learning run 4; experiments R2DE*, trial-run seeds 193594 (A) / 193595 (B).
$ErrorActionPreference = 'Continue'
$root = 'C:\PAI\wt_r2c'
$main = 'C:\PAI\recovery_vla'
$py = Join-Path $main '.venv\Scripts\python.exe'
$gitdir = Join-Path $main '.tools\git\cmd'
$git = Join-Path $gitdir 'git.exe'
$dir = Join-Path $root 'outputs\r2c\runs'
$rot = Join-Path $root 'outputs\real\rotate'
$dyn = Join-Path $rot 'rotate_dyn.py'
$TAG = if ($Stage -eq 'D') { 'R2DEVAL' } else { 'R2CEVAL' }
$MARK = "$TAG launcher stop mark"
$sfx = if ($Smoke) { '_SMOKE' } else { '' }
$lname = if ($Stage -eq 'D') { 'run_r2d_eval' } else { 'run_r2c_eval' }
$log = Join-Path $dir "$($lname)_$($Machine.ToLower)$($sfx.ToLower).log"
$stopFile = Join-Path $dir "STOP_$TAG$sfx"
$R2A = 'outputs\r2\train\train_R2as1000_20261023-221311_20261023-221311\checkpoints\010000\pretrained_model'
$R2A_SHA = '2ac83ec4bfc83954e957c4a748b16fa94bdf89e9c1e1f530a437d418b0121544'
$SCORER = 'outputs\r2c\eval\scorer_run3.pt'
if ($Stage -eq 'D') {             # the base is Rdag, read through the same relative path as the Ud records of R2E
    $R2A = 'outputs\r1\train\train_Rdags1000_20261021-155742_20261021-155742\checkpoints\040000\pretrained_model'
    $R2A_SHA = '93f74cc86e81a00b6c8610b8fbb5c576e10fa536abda7db646a960e8f0c60233'
    $SCORER = 'outputs\r2c\eval\scorer_run4.pt'
}
function L($m) { $line = "$(Get-Date -Format s) $m"; Write-Host $line; $line | Out-File -FilePath $log -Encoding utf8 -Append }
function Mem {
    $os = Get-CimInstance Win32_OperatingSystem
    return ('phys free {0:N1} GB, commit free {1:N1} GB' -f ($os.FreePhysicalMemory / 1MB), ($os.FreeVirtualMemory / 1MB))
}

New-Item -ItemType Directory -Force $dir | Out-Null
$env:PYTHONUTF8 = '1'; $env:HF_HUB_OFFLINE = '1'; $env:PYTHONPATH = "$root\src;$root"
$env:HF_HOME = Join-Path $root 'models\hf_home'
$env:PATH = "$gitdir;$env:PATH"          # the evaluation records git rev-parse HEAD
Set-Location $root
L "launcher pid $PID machine $Machine stage $Stage smoke $([bool]$Smoke)"
if (Test-Path $stopFile) { L "STOP file found ($stopFile). Delete it to continue."; exit 1 }
if ($ScorerSha -notmatch '^[0-9a-f]{64}$') { L 'STOP: pass -ScorerSha <64-hex> (SHA-256 of the last run3 scorer.pt)'; exit 2 }

# --- the run must not already be going (nor other GPU jobs on this machine)
$busy = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'rotate_dyn|ladder\.py.* run|selfroll_data\.py|train_r2a\.py|dsrl_run\.py|train_wrapped' }
if ($busy) { L "STOP: python runs are still going (pid $(($busy | ForEach-Object { $_.ProcessId }) -join ',')). Wait for them to end."; exit 3 }

# --- code
$head = (& $git -C $root -c safe.directory=* rev-parse HEAD).Trim
$dirty = & $git -C $root -c safe.directory=* status --porcelain --untracked-files=no
L "HEAD $head, tracked changes $(@($dirty).Count)"
if (-not $Smoke) {
    if ($HeadWant -notmatch '^[0-9a-f]{40}$') { L 'STOP: pass -HeadWant <40-hex commit> (the fixed protocol commit)'; exit 4 }
    if ($head -ne $HeadWant) { L "STOP: HEAD must be $HeadWant"; exit 4 }
    if ($dirty) { L 'STOP: tracked files changed'; exit 4 }
}
if (-not (Test-Path $dyn)) { L "STOP: $dyn missing"; exit 4 }
L ("rotate_dyn sha256 " + (Get-FileHash $dyn -Algorithm SHA256).Hash.ToLower)

# --- R2a checkpoint and the scorer
$f = Join-Path $root (Join-Path $R2A 'model.safetensors')
if (-not (Test-Path $f)) { L "STOP: R2a checkpoint missing ($f)"; exit 5 }
$sha = (Get-FileHash $f -Algorithm SHA256).Hash.ToLower
L "R2a sha256 $sha"
if ($sha -ne $R2A_SHA) { L 'STOP: base checkpoint SHA-256 differs'; exit 5 }
$sp = Join-Path $root $SCORER
if (-not (Test-Path $sp)) { L "STOP: scorer missing ($sp)"; exit 5 }
$ssha = (Get-FileHash $sp -Algorithm SHA256).Hash.ToLower
L "scorer sha256 $ssha"
if ($ssha -ne $ScorerSha) { L 'STOP: scorer SHA-256 differs from -ScorerSha'; exit 5 }

# --- API key for the start-of-run planner (presence only, never printed)
& $py -c "import sys; from recovla.planner import decompose as D; sys.exit(0 if D._api_key else 1)"
if ($LASTEXITCODE -ne 0) { L 'STOP: ANTHROPIC_API_KEY not found (user environment)'; exit 6 }
try { $gpu = (& nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader) -join '; ' } catch { $gpu = '' }
if (-not $gpu) { L 'STOP: nvidia-smi did not answer'; exit 6 }
L "gpu: $gpu"
L (Mem)

# --- plans
if ($Smoke) {
    $pdir = Join-Path $dir 'plans'
    $tag = $Machine.ToLower
    if ($Stage -eq 'D') {
        $seed = if ($Machine -eq 'A') { 193594 } else { 193595 }
        & $py scripts\api\plan_success_relearn.py --stage-d --scorer-sha256 $ScorerSha --smoke $seed --tag $tag --out $pdir | ForEach-Object { L "  plan $_" }
        $plans = @((Join-Path $pdir "plan_r2dsmoke_$tag.json"))
    } else {
        $seed = if ($Machine -eq 'A') { 193592 } else { 193593 }
        & $py scripts\api\plan_success_relearn.py --stage-c --scorer-sha256 $ScorerSha --smoke $seed --tag $tag --r2a-ckpt ($R2A -replace '\\', '/') --out $pdir | ForEach-Object { L "  plan $_" }
        $plans = @((Join-Path $pdir "plan_r2csmoke_$tag.json"))
    }
    if ($LASTEXITCODE -ne 0) { L 'STOP: plan_success_relearn.py --smoke failed'; exit 7 }
} else {
    $pre = if ($Stage -eq 'D') { 'r2de' } else { 'r2ce' }
    $names = if ($Machine -eq 'A') { $pre, "${pre}2", "${pre}3" } else { "${pre}_b", "${pre}_b2", "${pre}_b3" }
    $plans = @($names | ForEach-Object { Join-Path $root "configs\rotate\plan_$_.json" })
}
$info = @
foreach ($pl in $plans) {
    if (-not (Test-Path $pl)) { L "STOP: plan missing $pl"; exit 7 }
    $j = Get-Content -Raw -Encoding UTF8 $pl
    $name = [regex]::Match($j, '"name":\s*"([^"]+)"').Groups[1].Value
    $exp = [regex]::Match($j, '"experiment":\s*"([^"]+)"').Groups[1].Value
    $conds = @([regex]::Matches($j, '"([A-Za-z0-9]+)":\s*\{\s*"n"') | ForEach-Object { $_.Groups[1].Value })
    if ($j -notmatch [regex]::Escape(($R2A -replace '\\', '/'))) { L "STOP: $pl does not use the base checkpoint $R2A"; exit 7 }
    if ($j -notmatch $ScorerSha) { L "STOP: $pl does not use the scorer $ScorerSha"; exit 7 }
    L ("plan $pl name $name experiment $exp conditions $($conds -join ',') sha256 " + (Get-FileHash $pl -Algorithm SHA256).Hash.ToLower)
    $info += , @($pl, $name, $exp, $conds)
}

# --- remove only the stop marks this launcher wrote before
foreach ($it in $info) {
    $marks = @((Join-Path $rot "$($it[1]).STOP")) + @($it[3] | ForEach-Object { Join-Path $root "outputs\v2eval\$($it[2])\$_\STOP" })
    foreach ($m in $marks) {
        if (Test-Path $m) {
            $txt = (Get-Content -Raw $m).Trim
            if ($txt -eq $MARK) { Remove-Item $m -Force -Confirm:$false; L "removed old stop mark $m" }
            else { L "STOP: $m exists and was not written by this launcher. Look at it first."; exit 8 }
        }
    }
}

function DoneCounts($exp, $conds) {
    ($conds | ForEach-Object { "$_ " + @(Get-ChildItem (Join-Path $root "outputs\v2eval\$exp\$_") -Filter 'run_????.json' -ErrorAction SilentlyContinue).Count }) -join ', '
}

# --- the plans at once, one lane each (the progress keeper runs at most one block per condition)
$jobs = @
for ($i = 0; $i -lt $info.Count; $i++) {
    $pl, $name, $exp, $conds = $info[$i]
    "1" | Out-File -FilePath (Join-Path $rot "$name.lanes") -Encoding ascii
    $o = Join-Path $rot "$($name)_dyn.out"
    if (Test-Path $o) { $o = Join-Path $rot "$($name)_dyn_$(Get-Date -Format yyyyMMdd-HHmmss).out" }
    $e = $o -replace '\.out$', '.err'
    $p = Start-Process -FilePath $py -ArgumentList @($dyn, $pl, '--lanes', '1', '--min-commit-gb', "$MinCommitGb") -NoNewWindow -PassThru -RedirectStandardOutput $o -RedirectStandardError $e
    $null = $p.Handle   # keep the handle, or PS 5.1 may lose ExitCode
    L "rotate $name ($exp) pid $($p.Id), out $o"
    $jobs += , @{ p = $p; name = $name; exp = $exp; conds = $conds; out = $o; done = $false; rc = $null }
    if ($i -lt $info.Count - 1) { Start-Sleep -Seconds 95 }   # the progress keeper staggers its own starts by 90 s
}
$k = 0; $sent = $false; $failed = @
while (@($jobs | Where-Object { -not $_.done }).Count -gt 0) {
    Start-Sleep -Seconds 30
    $k++
    if ((-not $sent) -and (Test-Path $stopFile)) {
        foreach ($jb in $jobs) {
            $MARK | Out-File -FilePath (Join-Path $rot "$($jb.name).STOP") -Encoding ascii
            foreach ($c in $jb.conds) {
                $cd = Join-Path $root "outputs\v2eval\$($jb.exp)\$c"
                if (Test-Path $cd) { $MARK | Out-File -FilePath (Join-Path $cd 'STOP') -Encoding ascii }
            }
        }
        $sent = $true
        L 'STOP file seen: stop marks written; running trials finish first'
    }
    foreach ($jb in $jobs) {
        if ($jb.done -or -not $jb.p.HasExited) { continue }
        $jb.p.WaitForExit
        $jb.rc = $jb.p.ExitCode
        $jb.done = $true
        L "rotate $($jb.name) rc $($jb.rc); done: $(DoneCounts $jb.exp $jb.conds); $(Mem)"
        if ($jb.rc -ne 0) { $failed += $jb.name }
    }
    if ($k % 20 -eq 0) {
        foreach ($jb in $jobs) { if (-not $jb.done) { L "  $($jb.name) done: $(DoneCounts $jb.exp $jb.conds)" } }
        L "  $(Mem)"
    }
}
if ($sent -or (Test-Path $stopFile)) { L 'stopped by the STOP file. Delete it and run again to continue.'; exit 1 }
if ($failed.Count -gt 0) { L "STOP: rotate ended with an error for $($failed -join ', ') (see the _dyn.out files)"; exit 9 }
L 'ALL DONE'
exit 0
