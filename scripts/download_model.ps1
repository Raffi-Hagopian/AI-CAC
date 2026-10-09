# ---------------------------------------------------------------------------
# download_model.ps1
#
# Downloads the VA-AI-CAC model weights into the project's model/ directory.
# Only downloads if the file doesn't exist or is outdated.
#
# Usage (run from anywhere in the project):
# powershell -ExecutionPolicy Bypass -File scripts\download_model.ps1
#
# Or if execution policy allows:
# .\scripts\download_model.ps1
#
# The script is also called automatically during container builds.
# ---------------------------------------------------------------------------

$ErrorActionPreference = "Stop"

$MODEL_URL = "https://github.com/Raffi-Hagopian/AI-CAC/releases/download/v1.0.0/va_non_gated_ai_cac_model.pth"
$MODEL_FILENAME = "va_non_gated_ai_cac_model.pth"

# Resolve model/ relative to this script's location so it works from any cwd
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$MODEL_DIR = Join-Path $SCRIPT_DIR "..\model"
$MODEL_PATH = Join-Path $MODEL_DIR $MODEL_FILENAME

# Create model directory if it doesn't exist
if (-not (Test-Path $MODEL_DIR)) {
    New-Item -ItemType Directory -Path $MODEL_DIR -Force | Out-Null
}

Write-Host "Checking model weights at: $MODEL_DIR" -ForegroundColor Cyan
Write-Host "Remote URL: $MODEL_URL" -ForegroundColor Cyan

# Check if file exists and get remote file info
$shouldDownload = $true

if (Test-Path $MODEL_PATH) {
    $localFile = Get-Item $MODEL_PATH
    
    try {
        # Get remote file headers to check size/last modified
        $remoteInfo = Invoke-WebRequest -Uri $MODEL_URL -Method Head -UseBasicParsing
        $remoteSize = [long]$remoteInfo.Headers.'Content-Length'
        
        if ($localFile.Length -eq $remoteSize) {
            Write-Host "Model file already exists and matches remote size." -ForegroundColor Green
            Write-Host "Skipping download." -ForegroundColor Green
            $shouldDownload = $false
        } else {
            Write-Host "Local file size differs from remote. Re-downloading..." -ForegroundColor Yellow
        }
    } catch {
        Write-Host "Could not check remote file info. Will re-download..." -ForegroundColor Yellow
    }
}

if ($shouldDownload) {
    Write-Host "Downloading model weights..." -ForegroundColor Yellow
    Write-Host "This may take a few minutes (file size: ~600 MB)" -ForegroundColor Yellow
    
    try {
        # Download with progress
        $ProgressPreference = 'Continue'
        Invoke-WebRequest -Uri $MODEL_URL -OutFile $MODEL_PATH -UseBasicParsing
        
        Write-Host "`nDownload complete!" -ForegroundColor Green
    } catch {
        Write-Host "`nDownload failed: $_" -ForegroundColor Red
        throw
    }
}

# Display final status
$finalFile = Get-Item $MODEL_PATH
$fileSizeMB = [math]::Round($finalFile.Length / 1MB, 2)

Write-Host "`nModel is ready:" -ForegroundColor Green
Write-Host "  Path: $MODEL_PATH" -ForegroundColor White
Write-Host "  Size: $fileSizeMB MB" -ForegroundColor White
