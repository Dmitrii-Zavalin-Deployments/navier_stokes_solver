#!/bin/bash
# src/debug/forensic_audit.sh

echo "🔍 STEP 1: Auditing Scenario 3 Input Data..."
# Scenario 3 usually relies on 'test_terminal_fail.json'. Let's verify it exists and check the threshold.
if [ -f "test_terminal_fail.json" ]; then
    cat test_terminal_fail.json | grep -E "divergence_threshold|ppe_atol"
else
    echo "❌ test_terminal_fail.json not found in root!"
fi

echo "🔍 STEP 2: Smoking-Gun Audit of Test Scenario 3..."
# We need to see how the test expects the failure to happen.
# We are looking for the 'pytest.raises' block in scenario 3.
cat -n tests/property_integrity/test_heavy_elasticity_lifecycle.py | grep -A 20 "test_scenario_3_terminal_failure"

echo "🔍 STEP 3: Checking Orchestrator Logic for Terminal Signals..."
# Does the orchestrator actually raise an error that the test can catch?
cat -n src/step3/orchestrate_step3.py | grep -A 5 "solve_pressure_poisson_step"

echo "🛠️ REPAIR: Fixing the CI Runner Pathing..."
# The error 'file not found: FAILED' suggests a shell variable leak.
# We will inject a fix to the pytest command to ensure it only targets the specific test.

# 1. Ensure ppe_solver.py has the correct threshold logic we wrote earlier
# (Already verified, but good for logs)
grep "divergence_threshold" src/step3/ppe_solver.py

echo "📉 STEP 4: Targeted Execution of Scenario 3..."
# We use -k to filter for only the Terminal Failure scenario
python3 -m pytest tests/property_integrity/test_heavy_elasticity_lifecycle.py -k "test_scenario_3_terminal_failure" -vv

echo "✅ Audit Complete."