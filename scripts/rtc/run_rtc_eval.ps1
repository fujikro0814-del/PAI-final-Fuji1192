# R2 stage R evaluation launcher: arm Lp plus RTC (docs/api/r2_protocol.md 6.5 and 7). 18 units (configs\rotate\plan_r2ra*.json,
# plan_r2rb*.json, plan_r2rc*.json; one RTC setting each, about 33 seeds each, seeds 171400-171599). Same claim scheme as
# run_place_eval.ps1: any machine (A, B) takes the next free unit from a shared claim folder and runs -Slots units at the same time.
# Lp is not run again (the stage P records R2PP1-6 are read).
# Start it detached and hidden, for example (machine A):
#   $si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
#   Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='powershell.exe -NoProfile -WindowStyle Hidden
#     -ExecutionPolicy Bypass -File C:\PAI\wt_r2c\scripts\rtc\run_rtc_eval.ps1 -Machine A -HeadWant <40-hex> -RplaceDir <dir>
#     -RplaceSha <64-hex> -ClaimDir <folder>'; CurrentDirectory='C:\PAI\wt_r2c'; ProcessStartupInformation=$si}
# Smoke (one seed of each RTC setting; seed 193310 on A, 193311 on B; no HEAD check, no claims):
#   ... run_rtc_eval.ps1 -Machine A -RplaceDir <dir> -RplaceSha <64-hex> -Smoke
# Stop: create outputs\r2c\runs\STOP_R2REVAL. Running trials finish first, nothing new starts.
# Continue (also after a PC restart): delete that STOP file and run the same command again. Units this machine claimed and did
# not finish are taken up first; finished trials are kept and skipped. To move a unit to another machine, delete its .claim.
# ASCII only (Windows PowerShell 5.1 reads BOM-less files as ANSI).
param([ValidateSet('A', 'B')][string]$Machine = 'A', [string]$HeadWant = '', [string]$RplaceDir = '', [string]$RplaceSha = '',
      [string]$ClaimDir = '', [int]$Slots = 3, [double]$MinCommitGb = 6, [switch]$Smoke)
$ErrorActionPreference = 'Continue'
$root = 'C:\PAI\wt_r2c'
$main = 'C:\PAI\recovery_vla'
$py = Join-Path $main '.venv\Scripts\python.exe'
$gitdir = Join-Path $main '.tools\git\cmd'
$git = Join-Path $gitdir 'git.exe'
$dir = Join-Path $root 'outputs\r2c\runs'
$rot = Join-Path $root 'outputs\real\rotate'
$dyn = Join-Path $rot 'rotate_dyn.py'
$TAG = 'R2REVAL'
$MARK = "$TAG launcher stop mark"
$sfx = if ($Smoke) { '_SMOKE' } else { '' }
$log = Join-Path $dir "run_r2r_eval_$($Machine.ToLower)$($sfx.ToLower).log"
$stopFile = Join-Path $dir "STOP_$TAG$sfx"
# claim order: the three RTC settings mixed, so a partial run stays balanced
$ORDER = @
for ($i = 1; $i -le 6; $i++) { $ORDER += "r2ra$i", "r2rb$i", "r2rc$i" }
$CANDS = @('LrE44', 'LrL10', 'LrZ')
function L($m) { $line = "$(Get-Date -Format s) $m"; Write-Host $line; $line | Out-File -FilePath $log -Encoding utf8 -Append }
function Mem {
    $os = Get-CimInstance Win32_OperatingSystem
    return ('phys free {0:N1} GB, commit free {1:N1} GB' -f ($os.FreePhysicalMemory / 1MB), ($os.FreeVirtualMemory / 1MB))
}
function TryClaim($u) {
    $p = Join-Path $ClaimDir "$u.claim"
    try {
        $fs = [System.IO.File]::Open($p, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write)
        $b = [System.Text.Encoding]::ASCII.GetBytes("$Machine $(Get-Date -Format s)")
        $fs.Write($b, 0, $b.Length); $fs.Close
        return $true
    } catch { return $false }
}
function Owner($u) {
    $p = Join-Path $ClaimDir "$u.claim"
    if (-not (Test-Path $p)) { return '' }
    try { return ((Get-Content $p -TotalCount 1).Trim -split ' ')[0] } catch { return '?' }
}
function IsDone($u) { return (Test-Path (Join-Path $ClaimDir "$u.done")) }

New-Item -ItemType Directory -Force $dir | Out-Null
$env:PYTHONUTF8 = '1'; $env:HF_HUB_OFFLINE = '1'; $env:PYTHONPATH = "$root\src;$root"
$env:HF_HOME = Join-Path $root 'models\hf_home'
$env:PATH = "$gitdir;$env:PATH"          # 96 records git rev-parse HEAD
Set-Location $root
L "launcher pid $PID machine $Machine slots $Slots smoke $([bool]$Smoke) claims $ClaimDir"
if (Test-Path $stopFile) { L "STOP file found ($stopFile). Delete it to continue."; exit 1 }
if ($RplaceSha -notmatch '^[0-9a-f]{64}$') { L 'STOP: pass -RplaceSha <64-hex> (SHA-256 of the Rplace model.safetensors)'; exit 2 }
if (-not $RplaceDir) { L 'STOP: pass -RplaceDir <pretrained_model folder, relative to the repo>'; exit 2 }
if (-not $Smoke) {
    if (-not $ClaimDir) { L 'STOP: pass -ClaimDir <shared folder>'; exit 2 }
    New-Item -ItemType Directory -Force $ClaimDir | Out-Null
    if (-not (Test-Path $ClaimDir)) { L "STOP: claim folder $ClaimDir not reachable"; exit 2 }
}

# --- the run must not already be going (nor other GPU jobs on this machine)
$busy = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'rotate_dyn|ladder\.py.* run|selfroll_data\.py|train_r2a\.py|train_place\.py|dsrl_run\.py|train_wrapped' }
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

# --- Rplace checkpoint
$f = Join-Path $root (Join-Path $RplaceDir 'model.safetensors')
if (-not (Test-Path $f)) { L "STOP: Rplace checkpoint missing ($f)"; exit 5 }
$sha = (Get-FileHash $f -Algorithm SHA256).Hash.ToLower
L "Rplace sha256 $sha"
if ($sha -ne $RplaceSha) { L 'STOP: Rplace checkpoint SHA-256 differs from -RplaceSha'; exit 5 }

# --- API key for the planner and the replanner (presence only, never printed)
& $py -c "import sys; from recovla.planner import decompose as D; sys.exit(0 if D._api_key else 1)"
if ($LASTEXITCODE -ne 0) { L 'STOP: ANTHROPIC_API_KEY not found (user environment)'; exit 6 }
try { $gpu = (& nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader) -join '; ' } catch { $gpu = '' }
if (-not $gpu) { L 'STOP: nvidia-smi did not answer'; exit 6 }
L "gpu: $gpu"
L (Mem)

function PlanInfo($pl) {
    if (-not (Test-Path $pl)) { L "STOP: plan missing $pl"; exit 7 }
    $j = Get-Content -Raw -Encoding UTF8 $pl
    $name = [regex]::Match($j, '"name":\s*"([^"]+)"').Groups[1].Value
    $exp = [regex]::Match($j, '"experiment":\s*"([^"]+)"').Groups[1].Value
    $conds = @([regex]::Matches($j, '"([A-Za-z0-9]+)":\s*\{\s*"n"') | ForEach-Object { $_.Groups[1].Value })
    if ($j -notmatch [regex]::Escape(($RplaceDir -replace '\\', '/'))) { L "STOP: $pl does not use the Rplace checkpoint $RplaceDir"; exit 7 }
    if ((@($conds) | Where-Object { $CANDS -notcontains $_ }).Count -gt 0) { L "STOP: $pl conditions are not RTC settings ($($conds -join ','))"; exit 7 }
    foreach ($m in @((Join-Path $rot "$name.STOP")) + @($conds | ForEach-Object { Join-Path $root "outputs\v2eval\$exp\$_\STOP" })) {
        if (Test-Path $m) {
            $txt = (Get-Content -Raw $m).Trim
            if ($txt -eq $MARK) { Remove-Item $m -Force -Confirm:$false; L "removed old stop mark $m" }
            else { L "STOP: $m exists and was not written by this launcher. Look at it first."; exit 8 }
        }
    }
    L ("plan $pl name $name experiment $exp conditions $($conds -join ',') sha256 " + (Get-FileHash $pl -Algorithm SHA256).Hash.ToLower)
    return , @($pl, $name, $exp, $conds)
}
function DoneCounts($exp, $conds) {
    ($conds | ForEach-Object { "$_ " + @(Get-ChildItem (Join-Path $root "outputs\v2eval\$exp\$_") -Filter 'run_????.json' -ErrorAction SilentlyContinue).Count }) -join ', '
}
function StartUnit($it, $lanes) {
    $pl, $name, $exp, $conds = $it
    "$lanes" | Out-File -FilePath (Join-Path $rot "$name.lanes") -Encoding ascii
    $o = Join-Path $rot "$($name)_dyn.out"
    if (Test-Path $o) { $o = Join-Path $rot "$($name)_dyn_$(Get-Date -Format yyyyMMdd-HHmmss).out" }
    $e = $o -replace '\.out$', '.err'
    $p = Start-Process -FilePath $py -ArgumentList @($dyn, $pl, '--lanes', "$lanes", '--min-commit-gb', "$MinCommitGb") -NoNewWindow -PassThru -RedirectStandardOutput $o -RedirectStandardError $e
    $null = $p.Handle   # keep the handle, or PS 5.1 may lose ExitCode
    L "rotate $name ($exp) pid $($p.Id), out $o"
    return @{ p = $p; name = $name; exp = $exp; conds = $conds; done = $false; rc = $null; t0 = Get-Date }
}

# --- smoke: one plan with each RTC setting on one seed
if ($Smoke) {
    $pdir = Join-Path $dir 'plans'
    $tag = $Machine.ToLower
    $seed = @{ a = 193310; b = 193311 }[$tag]
    & $py scripts\api\plan_success_relearn.py --stage-r --rplace-ckpt ($RplaceDir -replace '\\', '/') --smoke $seed --tag $tag --out $pdir | ForEach-Object { L "  plan $_" }
    if ($LASTEXITCODE -ne 0) { L 'STOP: plan_success_relearn.py --smoke failed'; exit 7 }
    $jb = StartUnit (PlanInfo (Join-Path $pdir "plan_r2rsmoke_$tag.json")) 3
    $jb.p.WaitForExit
    L "rotate $($jb.name) rc $($jb.p.ExitCode); done: $(DoneCounts $jb.exp $jb.conds)"
    if ($jb.p.ExitCode -ne 0) { exit 9 }
    L 'SMOKE ALL DONE'
    exit 0
}

# --- the units: claim, run up to $Slots at once, mark done
$queue = New-Object System.Collections.ArrayList
foreach ($u in $ORDER) {
    if ((Owner $u) -eq $Machine -and -not (IsDone $u) -and -not $queue.Contains($u)) { [void]$queue.Add($u); L "resume own unit $u" }
}
function NextUnit {
    if ($queue.Count -gt 0) { $u = $queue[0]; $queue.RemoveAt(0); return $u }
    foreach ($u in $ORDER) {
        if (IsDone $u) { continue }
        if (TryClaim $u) { L "claimed $u"; return $u }
    }
    return $null
}
$running = @; $failed = @; $sent = $false; $k = 0; $noMore = $false
while ($true) {
    if ((-not $sent) -and (Test-Path $stopFile)) {
        foreach ($jb in $running) {
            $MARK | Out-File -FilePath (Join-Path $rot "$($jb.name).STOP") -Encoding ascii
            foreach ($c in $jb.conds) {
                $cd = Join-Path $root "outputs\v2eval\$($jb.exp)\$c"
                if (Test-Path $cd) { $MARK | Out-File -FilePath (Join-Path $cd 'STOP') -Encoding ascii }
            }
        }
        $sent = $true
        L 'STOP file seen: stop marks written; running trials finish first'
    }
    foreach ($jb in $running) {
        if ($jb.done -or -not $jb.p.HasExited) { continue }
        $jb.p.WaitForExit
        $jb.rc = $jb.p.ExitCode
        $jb.done = $true
        $mins = [math]::Round(((Get-Date) - $jb.t0).TotalMinutes)
        L "rotate $($jb.name) rc $($jb.rc) after $mins min; done: $(DoneCounts $jb.exp $jb.conds); $(Mem)"
        if ($jb.rc -eq 0 -and -not $sent) {
            try { "$Machine $(Get-Date -Format s)" | Out-File -FilePath (Join-Path $ClaimDir "$($jb.name).done") -Encoding ascii } catch { L "could not write $($jb.name).done ($_)" }
        } elseif ($jb.rc -ne 0) { $failed += $jb.name }
    }
    $running = @($running | Where-Object { -not $_.done })
    if ($failed.Count -gt 0 -and $running.Count -eq 0) { break }
    if ((-not $sent) -and $failed.Count -eq 0 -and -not $noMore) {
        while ($running.Count -lt $Slots) {
            if (-not (Test-Path $ClaimDir)) { L "claim folder $ClaimDir not reachable; trying again in 60 s"; break }
            $u = NextUnit
            if (-not $u) { $noMore = $true; L 'no unit left to claim'; break }
            $running += , (StartUnit (PlanInfo (Join-Path $root "configs\rotate\plan_$u.json")) 1)
            if ($running.Count -lt $Slots) { Start-Sleep -Seconds 95 }   # stagger the starts (the progress keeper waits 90 s too)
        }
    }
    if ($running.Count -eq 0 -and ($sent -or $noMore)) { break }
    Start-Sleep -Seconds 30
    $k++
    if ($k % 20 -eq 0) {
        foreach ($jb in $running) { L "  $($jb.name) done: $(DoneCounts $jb.exp $jb.conds)" }
        L "  $(Mem)"
    }
}
if ($sent -or (Test-Path $stopFile)) { L 'stopped by the STOP file. Delete it and run again to continue.'; exit 1 }
if ($failed.Count -gt 0) { L "STOP: rotate ended with an error for $($failed -join ', ') (see the _dyn.out files)"; exit 9 }
L 'ALL DONE (this machine)'
exit 0
