# ============================================================================
# VA-AI-CAC Prediction Script
# Usage: .\scripts\run_prediction.ps1 -ZipPath "path\to\study.zip"
# 
# Examples:
#   .\scripts\run_prediction.ps1 -ZipPath "study.zip"
#   .\scripts\run_prediction.ps1 -ZipPath "study.zip" -ApiUrl "http://server:25000/predict"
#   .\scripts\run_prediction.ps1 -ZipPath "study.zip" -SaveMasks $false
# ============================================================================

param(
    [Parameter(Mandatory=$true)]
    [string]$ZipPath,
    
    [Parameter(Mandatory=$false)]
    [string]$ApiUrl = "http://localhost:25000/predict",
    
    [Parameter(Mandatory=$false)]
    [bool]$SaveMasks = $true,
    
    [Parameter(Mandatory=$false)]
    [switch]$HealthCheck
)

# Health check only
if ($HealthCheck) {
    Write-Host "Checking AI-CAC Server Health" -ForegroundColor Cyan
    Write-Host "============================`n" -ForegroundColor Cyan
    
    $healthUrl = $ApiUrl -replace '/predict$', '/health'
    Write-Host "Health URL: $healthUrl`n" -ForegroundColor Gray
    
    try {
        $health = Invoke-RestMethod -Uri $healthUrl -Method GET
        
        Write-Host "Server Status:" -ForegroundColor Green
        Write-Host "  Status: $($health.status)" -ForegroundColor White
        Write-Host "  Model Loaded: $($health.model_loaded)" -ForegroundColor White
        Write-Host "  Device: $($health.device)" -ForegroundColor White
        Write-Host "  Storage Mode: $($health.storage_mode)" -ForegroundColor White
        
        if ($health.storage_mode -eq "s3") {
            Write-Host "`nS3 Configuration:" -ForegroundColor Cyan
            Write-Host "  Bucket: $($health.s3_bucket)" -ForegroundColor White
            Write-Host "  Prefix: $($health.s3_prefix)" -ForegroundColor White
        } else {
            Write-Host "`nLocal Storage:" -ForegroundColor Cyan
            Write-Host "  Mask Root: $($health.mask_root)" -ForegroundColor White
            Write-Host "  Debug Root: $($health.debug_root)" -ForegroundColor White
        }
        
        Write-Host "`nServer is healthy!" -ForegroundColor Green
    }
    catch {
        Write-Host "ERROR: Failed to connect to server" -ForegroundColor Red
        Write-Host $_.Exception.Message -ForegroundColor Red
        exit 1
    }
    exit 0
}

# Check if file exists
if (-not (Test-Path $ZipPath)) {
    Write-Host "ERROR: File not found: $ZipPath" -ForegroundColor Red
    exit 1
}

Write-Host "Running CAC Prediction" -ForegroundColor Cyan
Write-Host "=====================`n" -ForegroundColor Cyan
Write-Host "Study ZIP: $ZipPath" -ForegroundColor Gray
Write-Host "API URL:   $ApiUrl" -ForegroundColor Gray
Write-Host "Save masks: $SaveMasks`n" -ForegroundColor Gray

# Run prediction
Write-Host "Submitting prediction request...`n" -ForegroundColor Yellow

try {
    $result = curl -X POST $ApiUrl `
        -F "study_zip=@$ZipPath" `
        -F "save_masks=$($SaveMasks.ToString().ToLower())"
    
    # Parse JSON response
    $jsonResult = $result | ConvertFrom-Json
    
    # Display results
    Write-Host "Response:" -ForegroundColor Green
    Write-Host "  Success: $($jsonResult.success)" -ForegroundColor White
    Write-Host "  Request ID: $($jsonResult.request_id)" -ForegroundColor White
    Write-Host "  Number of Studies: $($jsonResult.num_studies)" -ForegroundColor White
    
    if ($jsonResult.results) {
        Write-Host "`nStudy Results:" -ForegroundColor Cyan
        foreach ($study in $jsonResult.results) {
            Write-Host "  Study: $($study.study_id)" -ForegroundColor Yellow
            Write-Host "    AI-CAC Score: $($study.ai_cac)" -ForegroundColor White
            
            if ($study.mask_urls) {
                Write-Host "    Mask URLs: $($study.mask_urls.Count) files" -ForegroundColor White
            } elseif ($study.mask_files) {
                Write-Host "    Mask Files: $($study.mask_files.Count) files" -ForegroundColor White
            }
        }
    }
    
    # Save to file
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $outputFile = "prediction_result_$timestamp.json"
    $result | Out-File -FilePath $outputFile -Encoding UTF8
    
    Write-Host "`nFull results saved to: $outputFile" -ForegroundColor Cyan
    
    # Show storage mode info
    if ($jsonResult.PSObject.Properties.Name -contains "storage_mode" -or 
        ($jsonResult.results -and $jsonResult.results[0].mask_urls)) {
        Write-Host "`nNote: Outputs are stored in S3" -ForegroundColor Gray
        Write-Host "Mask URLs are presigned and expire after configured duration" -ForegroundColor Gray
    }
}
catch {
    Write-Host "`nERROR: Prediction failed" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
