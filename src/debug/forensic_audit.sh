#!/bin/bash
echo "============================================================"
echo "🎯 PHASE C: KERNEL TRAP & LOGGER PROPAGATION AUDIT"
echo "============================================================"

# --- [Audit 1] Trap Verification ---
echo "--- [Audit 1] Checking NumPy error mode in runtime ---"
# Verify if something is overriding 'raise' back to 'ignore'
grep -r "np.seterr" src/

# --- [Audit 2] Logger Hierarchy Audit ---
echo "--- [Audit 2] Checking Logger naming (Rule 4 Compliance) ---"
# If logger = getLogger(__name__), but run_solver is called from a test,
# the hierarchy might be breaking log capture.
grep "logging.getLogger" src/main_solver.py

# --- [Audit 3] Smoking Gun: The 'Try' Block Scope ---
echo "--- [Audit 3] Inspecting Main Loop for swallowed Exceptions ---"
cat -n src/main_solver.py | sed -n '100,140p'

# --- [Audit 4] Physics Audit: Boundary vs. Field ---
echo "--- [Audit 4] Checking for hardcoded velocity caps ---"
grep -rE "clip|min\(|max\(" src/step3/ops/

# --- [5] AUTOMATED REPAIRS (Candidate Injections) ---

# REPAIR A: Force Logger to root to ensure pytest 'caplog' visibility
# sed -i 's/getLogger(__name__)/getLogger()/' src/main_solver.py

# REPAIR B: Inject a manual instability trigger for values > 1e10
# This catches "stable" overflows that don't trigger ArithmeticError
# sed -i '/state = orchestrate_step3/a \                if np.max(np.abs(state.stencil_matrix[0].field.u)) > 1e10: raise ArithmeticError("Velocity Explosion")' src/main_solver.py

# REPAIR C: Fix potential "underflow=ignore" masking broader issues
# sed -i 's/under="ignore"/under="raise"/g' src/main_solver.py

echo "============================================================"
echo "✅ Audit Complete. If Audit 2 shows a specific name, caplog may need that name."