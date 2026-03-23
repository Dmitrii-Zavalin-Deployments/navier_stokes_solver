#!/bin/bash
# forensic_audit.sh - Rule 4 (Routing) & Rule 7 (Scientific Audit)

echo "============================================================"
echo "🔍 DIAGNOSING: Why the Recovery Circuit stayed silent"
echo "============================================================"

# 1. Check if the Predictor Step actually validates its output
echo "--- [PREDICTOR AUDIT: src/main_solver.py] ---"
# Check if we are missing a check that would actually raise ArithmeticError
grep -C 5 "predictor" src/main_solver.py

# 2. Check the Audit Slice implementation (Rule 7)
# If FI.VX_STAR isn't being audited for finite values, the loop continues with NaNs
echo -e "\n--- [FIELD AUDIT LOGIC: src/common/solver_state.py] ---"
cat -n src/common/solver_state.py | grep -A 10 "def audit_fields"

# 3. Check for RuntimeWarning redirection
# NumPy often emits warnings instead of errors; we must ensure they are errors
echo -e "\n--- [NUMPY ERROR CONFIG: src/main_solver.py] ---"
grep "np.seterr" src/main_solver.py

echo -e "\n============================================================"
echo "🛠️ AUTOMATED REPAIR: Ensuring High-Fidelity Triggering"
echo "============================================================"

# Fix 1: Ensure NumPy treats 'invalid' (NaN) and 'over' (Inf) as Errors (Rule 0)
# This forces the ArithmeticError to actually be raised in Python
# sed -i '/import numpy as np/a np.seterr(all="raise")' src/main_solver.py

# Fix 2: Explicitly trigger an audit after the predictor step (Rule 7)
# This ensures the 'Physical anomaly' is caught before the loop progresses
# sed -i '/predictor.execute/a \            state.audit_fields() # Rule 7: Prove physics before proceeding' src/main_solver.py

# Fix 3: Standardize the Logger Name (Rule 8 Alignment)
# Ensure the logger is strictly named to match the test's caplog filter
# sed -i 's/getLogger(__name__)/getLogger("Solver.Main")/' src/main_solver.py

echo "Audit Complete. Diagnostic data collected and repair sequence prepared."