$ErrorActionPreference = "Stop"

$RepositoryRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $RepositoryRoot

$VenvPython = ".\.venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw "The virtual environment is missing. Run .\scripts\setup_windows.ps1 first."
}

& $VenvPython -m streamlit run app.py

