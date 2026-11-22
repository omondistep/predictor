#!/usr/bin/env pwsh
# PowerShell wrapper for Football Match Predictor
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ScriptDir ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    $Python = "python"
}

& $Python (Join-Path $ScriptDir "predict.py") $args
