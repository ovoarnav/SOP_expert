$ErrorActionPreference = "Stop"

$RepositoryRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepositoryRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Create the local virtual environment first with scripts/setup_windows.ps1."
}

& $Python -m pip install -e "$RepositoryRoot[dev]"
& $Python (Join-Path $PSScriptRoot "download_reranker.py") `
    --destination (Join-Path $RepositoryRoot "models\bge-reranker-v2-m3")

Write-Host "Citation-aware local RAG setup is complete. Restart Streamlit to load the reranker."
