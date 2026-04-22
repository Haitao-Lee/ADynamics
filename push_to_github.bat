@echo off
REM ADynamics GitHub Push Script for Windows
REM
REM Prerequisites:
REM 1. Install GitHub CLI: winget install GitHub.cli
REM 2. Authenticate: gh auth login
REM
REM Usage: Double-click this file or run from cmd: push_to_github.bat

echo ==================================================
echo   ADynamics GitHub Push Script
echo ==================================================

REM Check if gh is installed
where gh >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] GitHub CLI (gh) is not installed.
    echo.
    echo Please install it first:
    echo   winget install GitHub.cli
    echo.
    echo After installation, run:
    echo   gh auth login
    echo   push_to_github.bat
    echo.
    pause
    exit /b 1
)

REM Set your GitHub username and repository name
set GITHUB_USER=Haitao-Lee
set REPO_NAME=ADynamics

REM Navigate to script directory
cd /d "%~dp0"

REM Check git remote
git remote get-url origin >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo.
    echo [STEP 1] Creating GitHub repository...
    gh repo create %REPO_NAME% --public --source=. --push
) else (
    echo.
    echo [INFO] Remote 'origin' already exists
    git remote -v
)

echo.
echo [STEP 2] Pushing to GitHub...
git push -u origin main

echo.
echo ==================================================
echo   Done! Your repo should be live at:
echo   https://github.com/%GITHUB_USER%/%REPO_NAME%
echo ==================================================
pause
