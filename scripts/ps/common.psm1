# common.psm1
# ===========
# Shared functions for ADynamics PowerShell launchers.

function Get-PythonExe {
    return "C:\SoftwareInstallationFile\Anaconda\envs\ADynamics\python.exe"
}

function Initialize-ADynamicsEnv {
    <# Set env vars needed by all stages. #>
    $env:PYTHONIOENCODING        = "utf-8"
    $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
}

function Test-StageRunning {
    <# Return $true if a training process matching $Pattern is already running. #>
    param([string]$Pattern)
    $existing = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
        Where-Object {
            $_.CommandLine -and
            $_.CommandLine -like "*$Pattern*" -and
            $_.WorkingSetSize -gt 500MB
        }
    return [bool]$existing
}

function Stop-ExistingStage {
    <# Kill existing training process matching $Pattern, with user prompt. #>
    param([string]$Pattern, [string]$Label)
    $existing = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
        Where-Object {
            $_.CommandLine -and
            $_.CommandLine -like "*$Pattern*" -and
            $_.WorkingSetSize -gt 500MB
        }
    if ($existing) {
        Write-Host "$Label already running (PID=$($existing.ProcessId)):" -ForegroundColor Yellow
        $existing | Format-Table ProcessId, @{N="WS_GB";E={[math]::Round($_.WorkingSet64/1GB,2)}}, CommandLine -AutoSize
        $ans = Read-Host "Kill it and start fresh? [y/N]"
        if ($ans -eq 'y') {
            & taskkill /F /T /IM python.exe 2>$null
            Start-Sleep -Seconds 3
        } else {
            Write-Host "Aborted."
            exit 1
        }
    }
}

function Start-DetachedTraining {
    <# Launch Python as a detached process with redirected stdout/stderr. #>
    param(
        [string[]]$ArgList,
        [string]$Stdout,
        [string]$Stderr,
        [string]$Label
    )
    if (Test-Path $Stdout) { Remove-Item $Stdout -Force }
    if (Test-Path $Stderr) { Remove-Item $Stderr -Force }

    $Py = Get-PythonExe
    $proc = Start-Process -FilePath $Py `
        -ArgumentList $ArgList `
        -RedirectStandardOutput $Stdout `
        -RedirectStandardError  $Stderr `
        -WindowStyle Hidden `
        -PassThru

    Write-Host "Started $Label (PID=$($proc.Id))" -ForegroundColor Green
    Write-Host ""
    Write-Host "Monitor:  Get-Content '$Stderr' -Wait"
    Write-Host "Stop:     .\stop-training.ps1"
}
