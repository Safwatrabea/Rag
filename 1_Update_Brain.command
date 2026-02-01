#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate 2>/dev/null || true # Try to activate venv if it exists, otherwise assume global
python3 ingest.py
echo "---------------------------------------------------"
echo "✅ Brain Updated! You can close this window now."
echo "---------------------------------------------------"
read -p "Press [Enter] to close..."
