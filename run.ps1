param(
    [switch]$Cpu
)

$ErrorActionPreference = "Stop"

# Activate venv
& "$PSScriptRoot\.venv\Scripts\Activate.ps1"

if ($Cpu) {
    $env:FORCE_CPU = "true"
    Write-Host "Running in CPU mode (FORCE_CPU=true)" -ForegroundColor Yellow
}

# Run the FastAPI app
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
