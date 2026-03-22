#!/bin/bash
echo "============================================================"
echo "🎯 PHASE C: LOGGER HIERARCHY & NUMERICAL STRESS AUDIT"
echo "============================================================"

# --- [Audit 1] Logger Hierarchy Check ---
echo "--- [Audit 1] Verifying Logger Propagation ---"
# Check if propagate is set to False (which would block caplog)
grep -r "propagate" src/

# --- [Audit 2] Namespace Audit ---
echo "--- [Audit 2] Checking all logger definitions ---"
grep -r "getLogger(" src/

# --- [Audit 3] Test Stress Audit ---
echo "--- [Audit 3] Checking if simulation duration is too short to diverge ---"
grep -E "total_time|time_step" test_recovery_input.json

# --- [Audit 4] Error Trap Integrity ---
echo "--- [Audit 4] Checking if ArithmeticError is reachable ---"
# Check if Step 3 or Step 4 has its own try/except that swallows errors
grep -r "except" src/step3/ src/step4/

# --- [5] AUTOMATED REPAIRS (Candidate Injections) ---

# REPAIR A: Force the test to use the specific logger name
# sed -i 's/logger=""/logger="Solver.Main"/' tests/property_integrity/test_heavy_elasticity_lifecycle.py

# REPAIR B: Force the solver to use the root logger (Rule 4 fix)
# This ensures all logs are captured regardless of the test's logger name filter.
# sed -i 's/getLogger("Solver.Main")/getLogger()/' src/main_solver.py

# REPAIR C: Stress the physics further by reducing the stability limit
# sed -i 's/dt_min_limit": 0.0001/dt_min_limit": 1.0/' tests/property_integrity/test_heavy_elasticity_lifecycle.py

echo "============================================================"
echo "✅ Audit Complete. Use REPAIR A or B to fix Log Capture."