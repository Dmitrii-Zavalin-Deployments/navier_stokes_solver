#!/usr/bin/env bash
# ==============================================================================
# Diagnostic Script: Inspect Isotropic Accelerated Flow Test Failure
# ==============================================================================
set -euo pipefail

echo "================================================================================"
echo "[DIAGNOSTIC START] Running pytest with full stdout capture and traceback..."
echo "================================================================================"

# 1. Run pytest with live stdout (-s) and long tracebacks (--tb=long) to inspect 
#    every print statement from the test execution and matrix shapes/statistics.
pytest -s --tb=long tests/test_scenario_isotropic_accelerated_flow.py || true

echo ""
echo "================================================================================"
echo "[ARTIFACT SEARCH] Locating temporary test workspaces and output files..."
echo "================================================================================"

# 2. Automatically find the most recent pytest temporary directory in /tmp
TMP_ROOT="/tmp"
RECENT_WORKSPACE=$(find "$TMP_ROOT" -type d -name "pytest-of-*" 2>/dev/null | sort | tail -n 1)

if [ -n "$RECENT_WORKSPACE" ]; then
    echo "-> Found active pytest temp directory: $RECENT_WORKSPACE"
    
    echo "-> Searching for generated manifests, JSON inputs, and ZIP archives:"
    find "$RECENT_WORKSPACE" -type f \( -name "*.json" -o -name "*.zip" -o -name "*.npy" \) | while read -r file; do
        echo "   Found artifact: $file"
    done
    
    # 3. If a manifest or input json exists, print its contents for quick verification
    MANIFEST=$(find "$RECENT_WORKSPACE" -name "*manifest*.json" | tail -n 1)
    if [ -n "$MANIFEST" ]; then
        echo ""
        echo "-> Inspecting manifest summary contents ($MANIFEST):"
        python3 -c "import json; data = json.load(open('$MANIFEST')); print(json.dumps(data, indent=2))"
    fi
else
    echo "-> No /tmp pytest workspace found. Checking local project directory for artifacts..."
    find . -maxdepth 3 -name "*manifest*.json" -o -name "*.zip"
fi

echo ""
echo "================================================================================"
echo "[INSPECTION COMPLETE] Review stdout logs above to isolate mask or force anomalies."
echo "================================================================================"