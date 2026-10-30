# 実行環境を <ROOT> の下に作る。管理者権限は使わず、PATH・レジストリ・計算機全体の設定は変えない。
# 使い方: powershell -ExecutionPolicy Bypass -File env\setup_env.ps1 [-Stage tools|python|torch|lock|extras|check|editable|all]
# 各段は、済んでいれば何もしない（やり直すときは該当の置き場所を消してから）。
param(
    [ValidateSet('tools', 'python', 'torch', 'lock', 'extras', 'check', 'editable', 'all')]
    [string]$Stage = 'all'
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$Root = Split-Path -Parent $PSScriptRoot
$Tools = Join-Path $Root '.tools'
$Lock = Join-Path $Root 'env\requirements-lock.txt'
$Venv = Join-Path $Root '.venv'
$VenvPy = Join-Path $Venv 'Scripts\python.exe'
$Uv = Join-Path $Tools 'uv\uv.exe'

# 道具と Python の版
$UvVersion = '0.12.15'
$MinGitTag = 'v2.55.0.windows.5'
$MinGitAsset = 'MinGit-2.55.0.5-64-bit.zip'
$GhVersion = '2.101.0'
$PythonVersion = '3.12.14'
$TorchIndex = 'https://download.pytorch.org/whl/cu126'
$Extras = @('scipy', 'matplotlib', 'anthropic')

# このプロセスだけの環境変数
$env:UV_PYTHON_INSTALL_DIR = Join-Path $Root '.python'
$env:UV_CACHE_DIR = Join-Path $Root '.cache\uv'
$env:UV_SYSTEM_CERTS = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'

function Get-Download([string]$Url, [string]$OutFile) {
    New-Item -ItemType Directory -Force (Split-Path -Parent $OutFile) | Out-Null
    Invoke-WebRequest -Uri $Url -OutFile $OutFile -UseBasicParsing -Headers @{ 'User-Agent' = 'recovla-setup' }
}

function Assert-Sha256([string]$File, [string]$Expected) {
    $actual = (Get-FileHash -Algorithm SHA256 $File).Hash.ToLower
    if ($actual -ne $Expected.ToLower) { throw "SHA-256 が一致しない: $File`n  期待 $Expected`n  実際 $actual" }
    Write-Host "  SHA-256 一致: $(Split-Path -Leaf $File) $actual"
}

function Invoke-Checked([string]$Exe, [string[]]$ArgList) {
    & $Exe @ArgList
    if ($LASTEXITCODE -ne 0) { throw "失敗 (exit $LASTEXITCODE): $Exe $($ArgList -join ' ')" }
}

function Install-Tools {
    $dl = Join-Path $Root '.cache\download'
    # uv（配布元の .sha256 と照合）
    if (-not (Test-Path $Uv)) {
        $base = "https://github.com/astral-sh/uv/releases/download/$UvVersion/uv-x86_64-pc-windows-msvc.zip"
        $zip = Join-Path $dl 'uv.zip'
        Get-Download $base $zip
        Get-Download "$base.sha256" "$zip.sha256"
        Assert-Sha256 $zip ((Get-Content "$zip.sha256" -Raw).Trim -split '\s+')[0]
        Expand-Archive $zip (Join-Path $Tools 'uv') -Force
    }
    Write-Host "uv: $(& $Uv --version)"

    # MinGit（GitHub の API が返す資産の digest と照合）
    $git = Join-Path $Tools 'git\cmd\git.exe'
    if (-not (Test-Path $git)) {
        $rel = Invoke-RestMethod "https://api.github.com/repos/git-for-windows/git/releases/tags/$MinGitTag" -Headers @{ 'User-Agent' = 'recovla-setup' }
        $asset = $rel.assets | Where-Object name -eq $MinGitAsset
        if (-not $asset.digest) { throw "MinGit の digest が API にない" }
        $zip = Join-Path $dl $MinGitAsset
        Get-Download $asset.browser_download_url $zip
        Assert-Sha256 $zip ($asset.digest -replace '^sha256:', '')
        Expand-Archive $zip (Join-Path $Tools 'git') -Force
    }
    Write-Host "git: $(& $git --version)"

    # gh（配布元の checksums.txt と照合）
    $gh = Join-Path $Tools 'gh\bin\gh.exe'
    if (-not (Test-Path $gh)) {
        $name = "gh_${GhVersion}_windows_amd64.zip"
        $base = "https://github.com/cli/cli/releases/download/v$GhVersion"
        $zip = Join-Path $dl $name
        Get-Download "$base/$name" $zip
        Get-Download "$base/gh_${GhVersion}_checksums.txt" "$dl\gh_checksums.txt"
        $line = Get-Content "$dl\gh_checksums.txt" | Where-Object { $_ -match "\s$([regex]::Escape($name))$" }
        Assert-Sha256 $zip (($line -split '\s+')[0])
        Expand-Archive $zip (Join-Path $Tools 'gh') -Force
    }
    Write-Host "gh: $((& $gh --version)[0])"
}

function Install-Python {
    # --no-bin・--no-registry: <ROOT> の外（~\.local\bin と HKCU\Software\Python）に書かせない
    Invoke-Checked $Uv @('python', 'install', $PythonVersion, '--no-bin', '--no-registry')
    if (-not (Test-Path $VenvPy)) {
        Invoke-Checked $Uv @('venv', $Venv, '--python', $PythonVersion)
    }
    Write-Host "python: $(& $VenvPy --version)"
}

function Get-LockLines([switch]$WithoutTorch) {
    $lines = Get-Content $Lock -Encoding UTF8 | Where-Object { $_ -and -not $_.StartsWith('#') }
    if ($WithoutTorch) { $lines = $lines | Where-Object { $_ -notmatch '^(torch|torchvision)==' } }
    return $lines
}

function Install-Torch {
    $pins = Get-LockLines | Where-Object { $_ -match '^(torch|torchvision)==' }
    Invoke-Checked $Uv (@('pip', 'install', '--python', $VenvPy, '--no-deps', '--index-url', $TorchIndex) + $pins)
}

function Install-Lock {
    $tmp = Join-Path $Root '.cache\lock-without-torch.txt'
    [IO.File]::WriteAllLines($tmp, [string[]](Get-LockLines -WithoutTorch), (New-Object Text.UTF8Encoding $false))
    Invoke-Checked $Uv @('pip', 'install', '--python', $VenvPy, '--no-deps', '-r', $tmp)
}

function Install-Extras {
    # lock の版を動かさずに解き、新たに入った版を lock の末尾に追記する
    $constraint = Join-Path $Root '.cache\lock-constraint.txt'
    [IO.File]::WriteAllLines($constraint, [string[]](Get-LockLines -WithoutTorch), (New-Object Text.UTF8Encoding $false))
    $before = @(& $Uv pip freeze --python $VenvPy)
    Invoke-Checked $Uv (@('pip', 'install', '--python', $VenvPy, '--constraint', $constraint) + $Extras)
    $after = @(& $Uv pip freeze --python $VenvPy)
    $added = $after | Where-Object { $before -notcontains $_ } | Sort-Object
    $known = Get-LockLines
    $new = $added | Where-Object { $known -notcontains $_ }
    if ($new) {
        $text = @('', "# 追加（$(Get-Date -Format 'yyyy-MM-dd')、setup_env.ps1 の extras の段で $($Extras -join '・') を入れたときに入ったもの）") + $new
        [IO.File]::AppendAllLines($Lock, [string[]]$text, (New-Object Text.UTF8Encoding $false))
        Write-Host "lock に追記: $($new -join ', ')"
    }
}

function Test-Env {
    Invoke-Checked $Uv @('pip', 'check', '--python', $VenvPy)
    $freeze = @(& $Uv pip freeze --python $VenvPy | Where-Object { $_ -notmatch '^-e |^recovla' })
    $lock = @(Get-LockLines)
    $diff = Compare-Object ($lock | Sort-Object) ($freeze | Sort-Object)
    if ($diff) { $diff | Format-Table -AutoSize | Out-String | Write-Host; throw '入った一覧が lock と一致しない' }
    Write-Host "入った一覧は lock と一致（$($lock.Count) 件）"
}

function Install-Editable {
    # --no-build-isolation: lock の setuptools で組み立てる（ビルド用に別の版を取りに行かない）
    Invoke-Checked $Uv @('pip', 'install', '--python', $VenvPy, '--no-deps', '--no-build-isolation', '-e', $Root)
}

$stages = if ($Stage -eq 'all') { @('tools', 'python', 'torch', 'lock', 'extras', 'check', 'editable') } else { @($Stage) }
foreach ($s in $stages) {
    Write-Host "=== $s"
    switch ($s) {
        'tools' { Install-Tools }
        'python' { Install-Python }
        'torch' { Install-Torch }
        'lock' { Install-Lock }
        'extras' { Install-Extras }
        'check' { Test-Env }
        'editable' { Install-Editable }
    }
}
