#!/bin/bash
echo "============================================================"
echo "🎯 REPAIR PLAN: LOG STRING & IDENTITY ALIGNMENT"
echo "============================================================"

# --- [1] Diagnostic: Check Logger definition in Main Solver ---
echo "--- [Audit 1] Identifying main_solver.py logger name ---"
head -n 20 src/main_solver.py | grep "getLogger"

# --- [2] Diagnostic: Exact String Match ---
echo "--- [Audit 2] Comparing 'Instability' vs 'Instability detected' ---"
grep "Instability" src/main_solver.py
grep "Instability" src/common/elasticity.py

# --- [3] Diagnostic: Execution Check ---
echo "--- [Audit 3] Verifying if recovery path is actually reached ---"
# We inject a temporary counter to verify the exception branch is hit
# sed -i '126i \            print("DEBUG: Recovery branch entered")' src/main_solver.py

# --- [4] AUTOMATED REPAIRS ---

# REPAIR A: Align Main Solver logger with SSoT (Rule 4)
# This ensures main_solver uses the same "Solver.Main" identity as Elasticity
# sed -i 's/getLogger(__name__)/getLogger("Solver.Main")/g' src/main_solver.py

# REPAIR B: Standardize the Log String (Rule 8)
# The test looks for "Instability". Let's make sure the main_solver uses the exact keyword.
# sed -i 's/Instability detected/Instability/g' src/main_solver.py

# REPAIR C: Robust Test Assertion (Rule 6)
# Update the test to be case-insensitive and look for the keyword anywhere in the record.
# sed -i 's/"Instability" in r.message/ "instability" in r.message.lower()/g' tests/property_integrity/test_heavy_elasticity_lifecycle.py

echo "============================================================"
echo "✅ Audit Complete. Suggest REPAIR A & B to unify the Audit Trail."

cat -n src/main_solver.py