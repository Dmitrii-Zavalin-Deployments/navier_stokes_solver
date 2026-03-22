#!/bin/bash

LOG_FILE="simulation.log"
TEST_FILE="tests/property_integrity/test_heavy_elasticity_lifecycle.py"
MANAGER_FILE="src/common/elasticity.py"

echo "===================================================="
echo "🛡️  ELASTICITY INTEGRITY AUDIT"
echo "===================================================="

# --- 1. LOG AUDIT: Did it retry 10 times? ---
echo "Step 1: Auditing Log Sequence..."
if [ ! -f "$LOG_FILE" ]; then
    echo "❌ ERROR: $LOG_FILE not found. Run: pytest $TEST_FILE > $LOG_FILE 2>&1"
    exit 1
fi

RETRY_COUNT=$(grep -c "Instability" "$LOG_FILE")
if [ "$RETRY_COUNT" -eq 10 ]; then
    echo "✅ PASS: Found exactly 10 stabilization retries."
else
    echo "❌ FAIL: Expected 10 retries, but found $RETRY_COUNT."
fi

# --- 2. CODE AUDIT: Is it using the Config SSoT? ---
echo "Step 2: Checking Code SSoT Compliance..."
if grep -q "5000" "$MANAGER_FILE"; then
    echo "❌ FAIL: Hardcoded '5000' iteration boost found in $MANAGER_FILE."
else
    echo "✅ PASS: No hardcoded iteration boosts in ElasticManager."
fi

if grep -q "config.ppe_max_iter" "src/main_solver.py"; then
    echo "✅ PASS: Main loop is pulling ppe_max_iter from config."
else
    echo "❌ FAIL: Main loop is not using config-driven iterations."
fi

# --- 3. INPUT AUDIT: Is the 'Bomb' Schema-Compliant? ---
echo "Step 3: Checking Test Input Schema Compliance..."
if grep -q "initial_conditions_override" "$TEST_FILE"; then
    echo "❌ FAIL: Illegal field 'initial_conditions_override' found in $TEST_FILE."
else
    echo "✅ PASS: Test input follows the official JSON schema."
fi

echo "===================================================="

# A. See the Elasticity logic (should only be about dt)
cat -n src/common/elasticity.py | sed -n '27,41p'

# B. See the Main Loop (should show the ppe_max_iter gear)
cat -n src/main_solver.py | sed -n '87,96p'

# C. See the PPE solver (the 'fuse' that triggers the ArithmeticError)
cat -n src/step3/orchestrate_step3.py | sed -n '42,48p'