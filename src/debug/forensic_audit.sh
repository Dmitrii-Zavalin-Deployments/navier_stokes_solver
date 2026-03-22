#!/bin/bash
echo "============================================================"
echo "🎯 FORENSIC AUDIT: FORCING EXCEPTION BUBBLE-UP"
echo "============================================================"

# --- [1] Audit: Step 3 "Silent Death" Check ---
echo "--- [Audit 1] Checking if Step 3 returns instead of raising ---"
# We need to see if there is an 'except' block between lines 54 and 70.
cat -n src/step3/orchestrate_step3.py | sed -n '50,75p'

# --- [2] Audit: Elasticity Signal Verification ---
echo "--- [Audit 2] Checking Elasticity Manager logic ---"
# Verifying if the stabilization method itself has a silent fail.
cat -n src/common/elasticity.py | grep -A 10 "def stabilization"

# --- [3] Audit: Logger Propagation Fix ---
echo "--- [Audit 3] Checking Logger Propagation ---"
# If propagation is True but caplog is empty, we need to check the Level.
grep "logger.setLevel" src/main_solver.py || echo "⚠️ Logger level might be blocking WARNINGS."

# --- [4] AUTOMATED REPAIRS ---

# REPAIR A: Force Step 3 to be "Transparent"
# Removing the try/finally in Step 3 ensures exceptions hit the Main Solver immediately.
# Rule 7: Fail-Fast.
# sed -i '41d' src/step3/orchestrate_step3.py
# sed -i '54,56d' src/step3/orchestrate_step3.py

# REPAIR B: Synchronize Test Logger Scope
# Standardizing the test to listen to the specific 'Solver.Main' channel.
# sed -i 's/logger=""/logger="Solver.Main"/g' tests/property_integrity/test_heavy_elasticity_lifecycle.py

# REPAIR C: Inject High-Visibility "Smoking Gun" Print
# Bypassing the logger to prove the Main Solver's catch block is reachable.
# sed -i '125i \                print("CORE_RECOVERY_SIGNAL: EXCEPTION_TRAPPED")' src/main_solver.py

# REPAIR D: Ensure NumPy Raise is Global
# Moving the seterr to the absolute top of the run_solver function.
# sed -i '57i \        import numpy as np; np.seterr(all="raise")' src/main_solver.py

echo "============================================================"
echo "✅ Audit Block Configured. Run to bridge the exception gap."