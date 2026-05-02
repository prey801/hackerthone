#!/bin/bash
set -e

# Change to the directory where the script is located
cd "$(dirname "$0")"

echo "======================================"
echo "HackerRank Orchestration Triage Runner"
echo "======================================"

# Check if .env exists
if [ ! -f "code/.env" ]; then
    echo " WARNING: code/.env file not found."
    if [ -f "code/.env.example" ]; then
        echo "Creating .env from .env.example..."
        cp code/.env.example code/.env
        echo "Please ensure you add your API keys to code/.env before running."
        exit 1
    fi
fi

# Set up virtual environment if it doesn't exist
if [ ! -d "code/.venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv code/.venv
fi

# Activate virtual environment
source code/.venv/bin/activate

# Ensure pip is up-to-date and installed
python -m ensurepip --upgrade > /dev/null 2>&1 || true

# Install dependencies
echo " Checking dependencies..."
python -m pip install -r code/requirements.txt -q

# Run the batch processor
echo "🚀 Starting the batch pipeline..."
cd code
python run_batch.py "$@"

echo "✅ Finished!"
