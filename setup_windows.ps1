$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

New-Item -ItemType Directory -Force -Path ".tmp" | Out-Null

$env:PYTHONPATH = ""
$env:TEMP = (Resolve-Path ".tmp").Path
$env:TMP = (Resolve-Path ".tmp").Path

if (-not (Test-Path ".venv\\Scripts\\python.exe")) {
    python -m venv .venv --without-pip
    .\.venv\Scripts\python.exe -m ensurepip --upgrade
}

.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.train_model

Write-Host ""
Write-Host "Setup complete."
Write-Host "Run the app with:"
Write-Host ".\\.venv\\Scripts\\streamlit.exe run app\\app.py"
