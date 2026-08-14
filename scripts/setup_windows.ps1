$ErrorActionPreference = "Stop"

$RepositoryRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $RepositoryRoot

$VenvPython = Join-Path $RepositoryRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $VenvPython)) {
    $PythonLauncher = Get-Command "py" -ErrorAction SilentlyContinue
    if ($null -ne $PythonLauncher) {
        & $PythonLauncher.Source -3.11 -m venv .venv
    }
    else {
        $SystemPython = Get-Command "python" -ErrorAction SilentlyContinue
        if ($null -eq $SystemPython) {
            throw "Python 3.11 or later was not found. Install Python for Windows and rerun this script."
        }
        & $SystemPython.Source -m venv .venv
    }
}

& $VenvPython -m pip install -e ".[dev]"

$EnvPath = Join-Path $RepositoryRoot ".env"
$EnvExamplePath = Join-Path $RepositoryRoot ".env.example"
if (-not (Test-Path -LiteralPath $EnvPath)) {
    Copy-Item -LiteralPath $EnvExamplePath -Destination $EnvPath
    Write-Host "Created .env from .env.example."
}
else {
    Write-Host "Existing .env preserved."
}

$TesseractCommand = Get-Command "tesseract" -ErrorAction SilentlyContinue
$TesseractPath = if ($null -ne $TesseractCommand) {
    $TesseractCommand.Source
}
else {
    @(
        "C:\Program Files\Tesseract-OCR\tesseract.exe",
        "C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        (Join-Path $env:LOCALAPPDATA "Programs\Tesseract-OCR\tesseract.exe")
    ) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}

if (-not $TesseractPath) {
    Write-Warning "Tesseract OCR was not found. Run: winget install --id tesseract-ocr.tesseract --exact"
}
else {
    Write-Host "Tesseract OCR found at $TesseractPath."
}
