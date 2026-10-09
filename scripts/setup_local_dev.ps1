# ============================================================================
# VA-AI-CAC Local Development Setup Script (PowerShell)
# Quick setup: Creates venv, installs dependencies, downloads model
# Usage: .\scripts\setup_local_dev.ps1
# ============================================================================

$ErrorActionPreference = "Stop"

# Navigate to project root (one level above scripts/)
$ProjectRoot = Split-Path $PSScriptRoot -Parent
Set-Location $ProjectRoot

Write-Host "VA-AI-CAC Local Development Setup" -ForegroundColor Cyan
Write-Host "===================================`n" -ForegroundColor Cyan
Write-Host "Working directory: $ProjectRoot`n" -ForegroundColor Gray

# 1. Create virtual environment if it doesn't exist
$VenvPath = Join-Path $ProjectRoot ".venv"
if (-not (Test-Path $VenvPath)) {
    Write-Host "[1/3] Creating virtual environment at: $VenvPath" -ForegroundColor Yellow
    python -m venv $VenvPath
    Write-Host "  Created .venv`n" -ForegroundColor Green
} else {
    Write-Host "[1/3] Virtual environment already exists at: $VenvPath`n" -ForegroundColor Green
}

# 2. Activate and install dependencies
Write-Host "[2/3] Installing dependencies..." -ForegroundColor Yellow
& "$VenvPath\Scripts\Activate.ps1"
pip install --upgrade pip
pip install -r requirements-windows-cpu.txt
Write-Host "  Dependencies installed`n" -ForegroundColor Green

# 3. Download model
$ModelDir = Join-Path $ProjectRoot "model"
New-Item -ItemType Directory -Path $ModelDir -Force | Out-Null
Write-Host "[3/3] Downloading model to: $ModelDir" -ForegroundColor Yellow
& "$PSScriptRoot\download_model.ps1"

# 4. Create storage directories
Write-Host "`n[4/4] Creating storage directories..." -ForegroundColor Yellow
$StorageDirs = @(
    "storage/incoming",
    "storage/outputs/masks",
    "storage/outputs/debug"
)
foreach ($dir in $StorageDirs) {
    $dirPath = Join-Path $ProjectRoot $dir
    New-Item -ItemType Directory -Path $dirPath -Force | Out-Null
}
Write-Host "  Storage directories created`n" -ForegroundColor Green

# 5. Create .env file if it doesn't exist
$EnvFile = Join-Path $ProjectRoot ".env"
$EnvExampleFile = Join-Path $ProjectRoot ".env.example"

if (-not (Test-Path $EnvFile)) {
    if (Test-Path $EnvExampleFile) {
        Write-Host "Creating .env file from .env.example..." -ForegroundColor Yellow
        Copy-Item $EnvExampleFile $EnvFile
        Write-Host "  .env file created - Please update with your values`n" -ForegroundColor Green
    } else {
        Write-Host "Creating default .env file..." -ForegroundColor Yellow
        @"
# AI-CAC Configuration
# S3 Storage (set USE_S3_STORAGE=true to enable)
USE_S3_STORAGE=false
S3_BUCKET_NAME=
AWS_REGION=us-east-1
S3_PREFIX=RADFLOW/outputs/ai-cac-outputs
S3_URL_EXPIRATION=3600

# AWS Credentials (if using S3)
# AWS_ACCESS_KEY_ID=
# AWS_SECRET_ACCESS_KEY=

# Model Configuration
MODEL_CHECKPOINT_FILE=model/va_non_gated_ai_cac_model.pth
INFERENCE_BATCH_SIZE=16
DATALOADER_NUM_WORKERS=4
"@ | Out-File -FilePath $EnvFile -Encoding UTF8
        Write-Host "  .env file created - Please update with your values`n" -ForegroundColor Green
    }
} else {
    Write-Host ".env file already exists`n" -ForegroundColor Green
}

Write-Host "`nSetup complete!" -ForegroundColor Green
Write-Host "  Activate venv with: $VenvPath\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host "`nConfiguration:" -ForegroundColor Cyan
Write-Host "  - Edit .env file to configure S3 storage" -ForegroundColor Gray
Write-Host "  - Set USE_S3_STORAGE=true to enable S3" -ForegroundColor Gray
Write-Host "  - Set S3_BUCKET_NAME and AWS credentials" -ForegroundColor Gray
Write-Host "`nRun the app:" -ForegroundColor Cyan
Write-Host "  python app.py" -ForegroundColor Gray

