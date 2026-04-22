#!/bin/bash
# ADynamics GitHub Push Script
# Run this script after installing gh CLI or manually creating the repo on GitHub

# Set your GitHub username and repository name
GITHUB_USER="Haitao-Lee"
REPO_NAME="ADynamics"

echo "=================================================="
echo "  ADynamics GitHub Push Script"
echo "=================================================="

# Check if gh is installed
if ! command -v gh &> /dev/null; then
    echo ""
    echo "[ERROR] GitHub CLI (gh) is not installed."
    echo ""
    echo "Please install it first:"
    echo "  Windows: winget install GitHub.cli"
    echo "  or download from: https://github.com/cli/cli/releases"
    echo ""
    echo "After installation, run:"
    echo "  gh auth login"
    echo "  ./push_to_github.sh"
    exit 1
fi

# Navigate to ADynamics directory
cd "$(dirname \"$0\")"

# Check git remote
if git remote get-url origin &> /dev/null; then
    echo ""
    echo "[INFO] Remote 'origin' already exists"
    git remote -v
else
    echo ""
    echo "[STEP 1] Creating GitHub repository..."
    gh repo create $REPO_NAME --public --source=. --push
fi

echo ""
echo "[STEP 2] Pushing to GitHub..."
git push -u origin main

echo ""
echo "=================================================="
echo "  Done! Your repo should be live at:"
echo "  https://github.com/$GITHUB_USER/$REPO_NAME"
echo "=================================================="
