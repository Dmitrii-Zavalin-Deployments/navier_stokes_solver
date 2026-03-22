#!/bin/bash
echo "============================================================"
echo "🎯 FORENSIC AUDIT: FORCING THE ARITHMETIC TRAP"
echo "============================================================"

# --- [1] Diagnostic: Verification of NumPy Error Config ---
echo "--- [Audit 1] Checking global NumPy error state ---"
grep -r "np.seterr" src/

# --- [2] Diagnostic: Rule 7 Compliance Check ---
echo "--- [Audit 2] Checking for manual NaN/Inf checks in ops ---"
# If Rule 7 is violated, kernels return non-finite values instead of raising errors.
grep -E "isinf|isnan|isfinite" src/step3/ops/advection.py || echo "⚠️ VIOLATION: No finite-math guards found in advection.py"

# --- [3] Audit: Orchestration Flow ---
echo "--- [Audit 3] Checking if step3 catches its own math errors ---"
cat -n src/step3/orchestrate_step3.py | grep -A 5 "try:"

# --- [4] AUTOMATED REPAIRS ---

# REPAIR A: Force NumPy to 'raise' on all anomalies (Rule 5)
# This ensures that 1e15 * 1e15 triggers a Python exception immediately.
# sed -i 's/np.seterr(all="warn")/np.seterr(all="raise")/g' src/main_solver.py
# sed -i 's/np.seterr(all="ignore")/np.seterr(all="raise")/g' src/main_solver.py

# REPAIR B: Inject Rule 7 "Fail-Fast" Guard into Advection
# We force an explicit check before returning to ensure the recovery path is hit.
# sed -i '43i \        if not np.isfinite(advection_val): raise ArithmeticError("In-flight instability detected in advection")' src/step3/ops/advection.py

# REPAIR C: Fix caplog scope in the test
# Rule 6: Ensure the test listens to the correct logger identity.
# sed -i 's/logger=""/logger="Solver.Main"/g' tests/property_integrity/test_heavy_elasticity_lifecycle.py

# REPAIR D: Broaden Trap in Main Solver
# Rule 2: Ensure we don't miss RuntimeWarnings that have been promoted.
# sed -i 's/except (ArithmeticError, FloatingPointError, ValueError):/except (ArithmeticError, FloatingPointError, ValueError, RuntimeWarning):/g' src/main_solver.py

echo "============================================================"
echo "✅ Forensic Script Ready. Run to bridge the 'Math-Logic' gap."