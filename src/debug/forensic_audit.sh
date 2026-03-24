#!/bin/bash
# src/debug/forensic_audit.sh

echo "🔍 STEP 1: Probing for Shell Argument Corruption..."
# This identifies if the CI environment is leaking the 'FAILED' string into variables
env | grep "FAILED" || echo "No leaked 'FAILED' strings in environment."

echo "🔍 STEP 2: Verifying Test File Existence..."
TARGET_FILE="tests/property_integrity/test_heavy_elasticity_lifecycle.py"
if [ -f "$TARGET_FILE" ]; then
    echo "✅ Found $TARGET_FILE"
else
    echo "❌ $TARGET_FILE is missing!"
    exit 1
fi

echo "🛠️ REPAIR: Forcing a Clean Pytest Execution..."
# Rule 7: We use the full path and clear the cache to prevent 'FAILED' ghosting.
# We also use -k to ensure we target Scenario 3 specifically.

rm -rf .pytest_cache
python3 -m pytest "$TARGET_FILE" -k "test_scenario_3_terminal_failure" -vv

echo "📉 STEP 3: Checking if the test STILL fails (Post-Syntax Fix)..."
# If the test runs but fails, we check the 'AUDIT [Limit]' vs 'AUDIT [Explosion]' logic.
grep "AUDIT \[" src/common/solver_state.py