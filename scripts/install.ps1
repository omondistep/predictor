# ============================================================================
# Football Match Predictor — Installer (Windows / PowerShell)
#
# Replicates the entire system on a new machine:
#   git clone <repo> predictor
#   cd predictor
#   powershell -ExecutionPolicy Bypass -File scripts\install.ps1
#
# - DB (history.db), calibration params and trained ML models come from git
# - Regenerated state (data\pending_results.json, leagues_db.json,
#   predictions.json) is restored from seed\data\
# - The `pr` command is installed as pr.cmd / pr.ps1 in %USERPROFILE%\.local\bin
#   and that directory is added to the user PATH
# ============================================================================
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$DIR = [System.IO.Path]::GetFullPath((Join-Path $ScriptDir ".."))
Set-Location $DIR

Write-Host "==> Setting up Football Match Predictor in $DIR"

# ── Python ────────────────────────────────────────────────────────────────
$py = $null
foreach ($c in @("py", "python", "python3")) {
    if (Get-Command $c -ErrorAction SilentlyContinue) { $py = $c; break }
}
if (-not $py) {
    Write-Error "Python 3 not found. Install it from https://www.python.org/downloads/ and re-run."
}

# ── Virtual environment ──────────────────────────────────────────────────
if (-not (Test-Path "$DIR\.venv\Scripts\python.exe")) {
    Write-Host "==> Creating virtual environment..."
    if ($py -eq "py") { py -3 -m venv "$DIR\.venv" } else { & $py -m venv "$DIR\.venv" }
}
$venvPy  = "$DIR\.venv\Scripts\python.exe"
$venvPip = "$DIR\.venv\Scripts\pip.exe"

# ── Dependencies ─────────────────────────────────────────────────────────
Write-Host "==> Installing dependencies..."
& $venvPip install --upgrade pip --quiet
& $venvPip install --quiet -r "$DIR\requirements.txt"

# ── Restore regenerated state (data\ is gitignored) ──────────────────────
Write-Host "==> Restoring data files..."
New-Item -ItemType Directory -Force -Path "$DIR\data" | Out-Null
foreach ($f in @("pending_results.json", "leagues_db.json", "predictions.json")) {
    $dest = Join-Path "$DIR\data" $f
    $src  = Join-Path "$DIR\seed\data" $f
    if (-not (Test-Path $dest) -and (Test-Path $src)) {
        Copy-Item $src $dest
        Write-Host "    restored data\$f"
    }
}

# ── ML models (trained models ship in git; only retrain if broken/missing) ─
if (Test-Path "$DIR\ml_models\ml_predictor\meta.json") {
    Write-Host "==> Verifying ML models..."
    & $venvPy -c "from ml_model import MLPredictor; m=MLPredictor.load(auto_train=False); assert m is not None and m.is_trained" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "    models OK (loaded from git)"
    } else {
        Write-Host "    models missing/incompatible — retraining from history.db..."
        & $venvPy "$DIR\ml_model.py" --train
    }
} else {
    Write-Host "==> Training ML model..."
    & $venvPy "$DIR\ml_model.py" --train
}

# ── pr command (alias) ───────────────────────────────────────────────────
# .cmd shims in %USERPROFILE%\.local\bin work from cmd.exe, PowerShell and
# Git-Bash. No pr.ps1 is written there on purpose: PowerShell would prefer it
# over pr.cmd, and a Restricted execution policy would then refuse to run `pr`
# at all instead of falling back to the .cmd.
$bindir = Join-Path $HOME ".local\bin"
New-Item -ItemType Directory -Force -Path $bindir | Out-Null

function Write-CmdShim([string]$Name, [string]$ScriptRel, [string[]]$PreArgs) {
    $shim = Join-Path $bindir "$Name.cmd"
    $scriptPath = Join-Path $DIR $ScriptRel
    $argLine = ""
    if ($PreArgs) { $argLine = " " + ($PreArgs -join " ") }
    $body = @"
@echo off
rem Football Match Predictor
"$venvPy" "$scriptPath"$argLine %*
"@
    Set-Content -Path $shim -Value $body -Encoding ASCII
    Write-Host "==> Alias created: $shim"
}

Write-CmdShim "pr"        "predict.py"            @()
Write-CmdShim "predict"   "predict.py"            @()
Write-CmdShim "predictor" "predict.py"            @()
Write-CmdShim "pr2"       "predict.py"            @("--analysis", "pipeline")
Write-CmdShim "book"      "scripts\bettor.py"     @("card")

# Add .local\bin to the FRONT of the user PATH (so `pr` is unambiguous)
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
$parts = @()
if ($userPath) {
    $parts = $userPath -split ';' | Where-Object { $_ -and ($_ -ne $bindir) }
}
$parts = @($bindir) + $parts
[Environment]::SetEnvironmentVariable("Path", ($parts -join ';'), "User")
$env:Path = $bindir + ";" + $env:Path
Write-Host "==> $bindir is first on the user PATH"

# ── PowerShell aliases (predict, predict-calibrate, book, output, ...) ────
# Dot-source scripts\aliases.ps1 from the profile so every shell gets the
# full alias set. .cmd shims above still cover cmd.exe / no-profile sessions.
# Target both PowerShell editions (Windows PowerShell 5.1 and pwsh 7+) so
# whichever one the user opens picks the aliases up.
$aliasesPath = Join-Path $DIR "scripts\aliases.ps1"
$dotSource = ". '$aliasesPath'"
$profiles = @()
if ($PROFILE) {
    foreach ($p in @($PROFILE.CurrentUserAllHosts, $PROFILE.CurrentUserCurrentHost)) {
        if ($p) { $profiles += $p }
    }
}
$docs = [Environment]::GetFolderPath("MyDocuments")
foreach ($p in @(
    (Join-Path $docs "WindowsPowerShell\Microsoft.PowerShell_profile.ps1"),
    (Join-Path $docs "PowerShell\Microsoft.PowerShell_profile.ps1")
)) {
    $profiles += $p
}
$profiles = $profiles | Where-Object { $_ } | Select-Object -Unique

foreach ($prof in $profiles) {
    $dir = Split-Path -Parent $prof
    # Only create profiles for shells that have been used (dir exists) or
    # that this session is actually running under.
    $isCurrent = $false
    if ($PROFILE) {
        $isCurrent = (($prof -eq $PROFILE.CurrentUserAllHosts) -or ($prof -eq $PROFILE.CurrentUserCurrentHost))
    }
    if (-not $isCurrent -and -not (Test-Path $dir)) { continue }

    if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    $existing = ""
    if (Test-Path $prof) { $existing = Get-Content $prof -Raw }
    if ($existing -notmatch [regex]::Escape($aliasesPath)) {
        Add-Content -Path $prof -Value "`n# Football Predictor aliases`n$dotSource"
        Write-Host "==> Sourced aliases.ps1 from $prof"
    } else {
        Write-Host "==> aliases.ps1 already sourced from $prof"
    }
}

# ── Verify install ───────────────────────────────────────────────────────
Write-Host "==> Verifying install (DB + models)..."
& $venvPy -c "from database import DB_PATH; from ml_model import MLPredictor; m=MLPredictor.load(auto_train=False); print('    DB :', DB_PATH); print('    ML :', 'trained' if (m and m.is_trained) else 'NOT trained')"

Write-Host ""
Write-Host "Ready! Usage:  pr https://www.forebet.com/en/football/matches/..."
Write-Host "    pr <file>       Predict matches from a links file"
Write-Host "    pr results      Scrape Forebet for scores -> update DB + HTML"
Write-Host "    pr --calibrate  Show accuracy stats"
Write-Host ""
Write-Host "NOTE: ml_models (XGB/LGB) and seed\data\ must be committed to git for"
Write-Host "      a fresh clone to carry them over.  See 'git status' for untracked files."
