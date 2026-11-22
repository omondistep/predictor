# ============================================================================
# Football Predictor — PowerShell Aliases (Windows)
# ============================================================================
# Dot-source this file in your PowerShell profile:
#   . C:\path\to\predictor\scripts\aliases.ps1
#
# The installer (scripts\install.ps1) appends that line to your profile and
# drops pr.cmd / predict.cmd / predictor.cmd / pr2.cmd / book.cmd into
# %USERPROFILE%\.local\bin so the same commands also work from cmd.exe.
#
# Predictor root = parent of this script's directory (scripts\).
# All commands use the repo's .venv python and absolute paths, so every
# command works from any terminal / directory (no cd needed).
# ============================================================================

$script:_PredAliasesDir = $PSScriptRoot
if (-not $script:_PredAliasesDir) {
    $script:_PredAliasesDir = Split-Path -Parent $MyInvocation.MyCommand.Path
}
$global:PREDICTOR_DIR = Split-Path -Parent $script:_PredAliasesDir
Remove-Variable _PredAliasesDir -Scope Script -ErrorAction SilentlyContinue

$_predVenvPy = Join-Path $global:PREDICTOR_DIR ".venv\Scripts\python.exe"
if (Test-Path $_predVenvPy) {
    $global:PREDICTOR_PY = $_predVenvPy
} else {
    Write-Warning "predictor: no .venv found (run scripts\install.ps1) — using system python"
    $global:PREDICTOR_PY = "python"
}
Remove-Variable _predVenvPy -ErrorAction SilentlyContinue

function Invoke-Predictor {
    & $global:PREDICTOR_PY (Join-Path $global:PREDICTOR_DIR "predict.py") @args
}

# ── Main predictor ──────────────────────────────────────────────────────────
function pr             { Invoke-Predictor @args }
function predict        { Invoke-Predictor @args }

function pr2            { Invoke-Predictor --analysis pipeline @args }

function predict-calibrate  { Invoke-Predictor --calibrate @args }
function predict-review     { Invoke-Predictor --review @args }
function predict-autoreview { Invoke-Predictor --review --auto @args }
function predict-high       { Invoke-Predictor --high-only @args }
function predict-json       { Invoke-Predictor --json @args }
function predict-url        { Invoke-Predictor @args }
function predict-test       { Invoke-Predictor (Join-Path $global:PREDICTOR_DIR "test.txt") @args }
function predict-test-high  { Invoke-Predictor --high-only (Join-Path $global:PREDICTOR_DIR "test.txt") @args }

# ── Reports ─────────────────────────────────────────────────────────────────
function predict-report {
    Invoke-Predictor --calibrate @args
    Write-Host ""
    Write-Host "── League profiles ──────────────────────────────"
    Push-Location $global:PREDICTOR_DIR
    try {
        & $global:PREDICTOR_PY -c @'
from predict import LEAGUE_PROFILES
for name, p in sorted(LEAGUE_PROFILES.items()):
    vol = p["volatility"]
    tag = "*" if vol >= 0.25 else "~" if vol >= 0.15 else " "
    print(f"{tag} {name:<30s} avg_g={p['avg_goals']:.1f}  vol={vol:.2f}")
'@
    } finally {
        Pop-Location
    }
}

# ── Auto-Learn (continuous learning pipeline) ───────────────────────────────
function auto-learn   { & $global:PREDICTOR_PY (Join-Path $global:PREDICTOR_DIR "scripts\auto_learn.py") @args }
function learn-status { & $global:PREDICTOR_PY (Join-Path $global:PREDICTOR_DIR "scripts\auto_learn.py") --status @args }
function learn-daemon { & $global:PREDICTOR_PY (Join-Path $global:PREDICTOR_DIR "scripts\auto_learn.py") --daemon @args }

# ── Bets / Bettor agent tools ───────────────────────────────────────────────
# book <links.txt | preds.json>   — bettor's own card: de-vigged edge, EV%,
#                                   Quarter-Kelly stakes, league reliability.
#                                   Pass a links file and it predicts first.
function book         { & $global:PREDICTOR_PY (Join-Path $global:PREDICTOR_DIR "scripts\bettor.py") card @args }
function bettor-audit { & $global:PREDICTOR_PY (Join-Path $global:PREDICTOR_DIR "scripts\bettor.py") audit @args }

# ── Confidence updater ──────────────────────────────────────────────────────
function cnf      { & $global:PREDICTOR_PY (Join-Path $global:PREDICTOR_DIR "scripts\cnfupdate.py") @args }
function cnfupdate { & $global:PREDICTOR_PY (Join-Path $global:PREDICTOR_DIR "scripts\cnfupdate.py") @args }

# ── Results scraper ─────────────────────────────────────────────────────────
function scrape-results { & $global:PREDICTOR_PY (Join-Path $global:PREDICTOR_DIR "scripts\scrape_results.py") @args }

# ── Output directory ─────────────────────────────────────────────────────────
# output — open the predictions folder in Explorer and list its HTML files.
function output {
    $predDir = Join-Path $global:PREDICTOR_DIR "predictions"
    if (-not (Test-Path $predDir)) {
        Write-Host "No predictions directory yet: $predDir"
        return
    }
    if ($Host.Name -eq "ConsoleHost") {
        try { Start-Process explorer.exe $predDir } catch { }
    }
    Write-Host "DIR  $predDir"
    Write-Host ""
    Get-ChildItem -Path $predDir -Filter *.html -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime |
        Select-Object -Last 20 |
        Format-Table Name, @{n = "Size"; e = { "{0:N0} KB" -f ($_.Length / 1KB) } }, LastWriteTime -AutoSize
}

# ── Historical calibration ──────────────────────────────────────────────────
function histcal              { & $global:PREDICTOR_PY (Join-Path $global:PREDICTOR_DIR "scripts\historical_calibrate.py") @args }
function historical-calibrate { & $global:PREDICTOR_PY (Join-Path $global:PREDICTOR_DIR "scripts\historical_calibrate.py") @args }
