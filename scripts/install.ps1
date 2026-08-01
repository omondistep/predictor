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
$bindir = Join-Path $HOME ".local\bin"
New-Item -ItemType Directory -Force -Path $bindir | Out-Null

# pr.cmd shim (works from cmd, PowerShell and Git-Bash)
$prCmd = Join-Path $bindir "pr.cmd"
$escPy  = $venvPy.Replace("'", "''")
$escPredict = (Join-Path $DIR "predict.py").Replace("'", "''")
@"
@echo off
rem Football Match Predictor
"%escPy%" "%escPredict%" %*
"@ | Set-Content -Path $prCmd -Encoding ASCII
Write-Host "==> Alias created: $prCmd"

# pr.ps1 (PowerShell-native wrapper, hardcoded to this repo)
$prPs1 = Join-Path $bindir "pr.ps1"
$escPy  = $venvPy.Replace("'", "''")
$escPredict = (Join-Path $DIR "predict.py").Replace("'", "''")
@"
`$Python  = "$escPy"
`$Predict = "$escPredict"
& `$Python `$Predict `$args
"@ | Set-Content -Path $prPs1 -Encoding ASCII
Write-Host "==> Alias created: $prPs1"

# Add .local\bin to user PATH
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath -notlike "*$bindir*") {
    [Environment]::SetEnvironmentVariable("Path", "$userPath;$bindir", "User")
    Write-Host "==> Added $bindir to user PATH (open a new terminal to use 'pr')"
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
