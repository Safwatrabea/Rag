#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate 2>/dev/null || {
    echo "Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
}
python3 ingest.py
echo "---------------------------------------------------"
echo "✅ Brain Updated! You can close this window now."
echo "---------------------------------------------------"
read -p "Press [Enter] to close..."
