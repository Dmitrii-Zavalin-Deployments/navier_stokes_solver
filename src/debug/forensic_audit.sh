#!/bin/bash

LOG_FILE="simulation.log"
TEST_FILE="tests/property_integrity/test_heavy_elasticity_lifecycle.py"
MANAGER_FILE="src/common/elasticity.py"
CONFIG_FILE="config.json"

echo "===================================================="
echo "🛡️  ELASTICITY INTEGRITY AUDIT (PHASE C COMPLIANT)"
echo "===================================================="

# Ensure log exists
touch $LOG_FILE

# --- 1. LOG AUDIT: Did it retry the configured amount? ---
echo "Step 1: Auditing Log Sequence..."
# We now pull the expected count from the config file itself to remain deterministic
EXPECTED_RETRIES=$(grep "ppe_max_retries" "$CONFIG_FILE" | tr -dc '0-9')

if [ -z "$EXPECTED_RETRIES" ]; then
    echo "❌ ERROR: ppe_max_retries not found in $CONFIG_FILE"
    exit 1
fi

RETRY_COUNT=$(grep -c "Instability" "$LOG_FILE")

if [ "$RETRY_COUNT" -eq "$EXPECTED_RETRIES" ]; then
    echo "✅ PASS: Found exactly $RETRY_COUNT stabilization retries (Config Match)."
else
    echo "❌ FAIL: Expected $EXPECTED_RETRIES retries from Config, but found $RETRY_COUNT in logs."
fi

# --- 2. CODE AUDIT: Is it using the Config SSoT? ---
echo "Step 2: Checking Code SSoT Compliance..."

# Check if ElasticManager is pulling retries from config (Rule 5)
if grep -q "self.config.ppe_max_retries" "$MANAGER_FILE"; then
    echo "✅ PASS: ElasticManager is pulling ppe_max_retries from Config SSoT."
else
    echo "❌ FAIL: ElasticManager is using hardcoded retry limits."
fi

# Check if we removed the global logging (Rule 8)
if grep -q "logging.warning" "$MANAGER_FILE"; then
     # We check if it's the global one vs the instance one
     GLOBAL_LOG_COUNT=$(grep -c "logging.warning" "$MANAGER_FILE")
     if [ "$GLOBAL_LOG_COUNT" -gt 0 ]; then
        echo "⚠️  ADVISORY: Ensure all logging uses self.logger to avoid redundant API calls."
     fi
fi

# --- 3. INPUT AUDIT: Schema Integrity ---
echo "Step 3: Checking Test Input Schema Compliance..."
if grep -q "initial_conditions_override" "$TEST_FILE"; then
    echo "❌ FAIL: Illegal field 'initial_conditions_override' found in $TEST_FILE."
else
    echo "✅ PASS: Test input follows the official JSON schema."
fi

echo "===================================================="
echo "🔍 SYSTEM STATE DUMP"
echo "===================================================="

# A. Verify Elasticity range calculation
echo "--- Elasticity Logic (Wiring) ---"
cat -n src/common/elasticity.py | sed -n '15,22p'

# B. Verify stabilization logging and iteration
echo "--- Stabilization Implementation ---"
cat -n src/common/elasticity.py | sed -n '35,45p'

# C. Verify Main Loop Catch Mechanism
echo "--- Main Solver Exception Handling ---"
cat -n src/main_solver.py | sed -n '110,116p'