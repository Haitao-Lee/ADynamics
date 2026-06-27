# stop-training.ps1
# =================
# Kill all running python.exe training processes.
# Usage: .\stop-training.ps1

$ErrorActionPreference = "Continue"

Write-Host "python.exe processes:" -ForegroundColor Cyan
$procs = Get-Process -Name python -ErrorAction SilentlyContinue
if ($procs) {
    $procs | Format-Table Id, @{N="WS_MB";E={[math]::Round($_.WorkingSet64 / 1MB, 1)}}, StartTime -AutoSize
} else {
    Write-Host "  (none running)"
}

Write-Host "Killing python.exe /F /T ..." -ForegroundColor Yellow
& taskkill /F /T /IM python.exe 2>$null

Start-Sleep -Seconds 3
$remaining = Get-Process -Name python -ErrorAction SilentlyContinue
if ($remaining) {
    Write-Host "Still alive:" -ForegroundColor Yellow
    $remaining | Format-Table Id, WS -AutoSize
} else {
    Write-Host "All python.exe killed." -ForegroundColor Green
}

Write-Host "`nGPU state:" -ForegroundColor Cyan
& nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
