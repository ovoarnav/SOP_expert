$ErrorActionPreference = "Stop"

$RepositoryRoot = Split-Path -Parent $PSScriptRoot
$ModelDirectory = Join-Path $RepositoryRoot "models"
$EmbeddingFile = Join-Path $ModelDirectory "bge-small-en-v1.5-q8_0.gguf"
$AnswerFile = Join-Path $ModelDirectory "Qwen3-0.6B-Q8_0.gguf"

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    throw "Ollama is required. Install Ollama for Windows, start it, and run this script again."
}

New-Item -ItemType Directory -Force -Path $ModelDirectory | Out-Null

if (-not (Test-Path -LiteralPath $EmbeddingFile)) {
    Write-Host "Downloading the 37 MB BGE semantic-retrieval model..."
    Invoke-WebRequest `
        -Uri "https://huggingface.co/ggml-org/bge-small-en-v1.5-Q8_0-GGUF/resolve/main/bge-small-en-v1.5-q8_0.gguf" `
        -OutFile $EmbeddingFile `
        -TimeoutSec 600
}

if (-not (Test-Path -LiteralPath $AnswerFile)) {
    Write-Host "Downloading the 639 MB Qwen answer model..."
    Invoke-WebRequest `
        -Uri "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf" `
        -OutFile $AnswerFile `
        -TimeoutSec 1800
}

Write-Host "Registering local models with Ollama..."
& ollama create archbold-bge-small:latest -f (Join-Path $ModelDirectory "bge-small.Modelfile")
& ollama create archbold-qwen3:0.6b -f (Join-Path $ModelDirectory "qwen3-0.6b.Modelfile")

Write-Host "Local AI models are ready."
& ollama list
