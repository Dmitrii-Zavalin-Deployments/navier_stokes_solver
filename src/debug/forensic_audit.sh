#!/bin/bash
echo "============================================================"
echo "🎯 PHASE C: BOUNDARY INJECTION & LOG PROPAGATION AUDIT"
echo "============================================================"

# --- [Audit 1] Boundary Value Injection Check ---
echo "--- [Audit 1] Checking if 1e15 is actually reaching the solver ---"
grep -A 5 "values" test_recovery_input.json

# --- [Audit 2] Logger Visibility Check ---
echo "--- [Audit 2] Checking if 'logger' in main_solver is root or local ---"
head -n 20 src/main_solver.py | grep "logger ="

# --- [Audit 3] Boundary Applier Logic ---
echo "--- [Audit 3] Inspecting inflow application in Step 4 ---"
cat -n src/step4/boundary_applier.py | grep -A 10 "inflow"

# --- [Audit 4] Floating Point Reality Check ---
echo "--- [Audit 4] Checking for hidden 'clip' or 'minimum' functions ---"
grep -r "np.clip" src/
grep -r "min(" src/step3/

# --- [5] AUTOMATED REPAIRS (Candidate Injections) ---

# REPAIR A: Force an immediate crash if VX exceeds a stability threshold (e.g., 1000)
# This bypasses the need for an ArithmeticError if the values are "stable but wrong"
# sed -i '/delta = solve_pressure_poisson_step/i \    if np.max(np.abs(block.field.u)) > 1000: raise ValueError("Velocity Divergence")' src/step3/orchestrate_step3.py

# REPAIR B: Force the Logger to propagate to the root (Ensures caplog sees it)
# sed -i 's/logger = logging.getLogger(__name__)/logger = logging.getLogger()/' src/main_solver.py

# REPAIR C: Ensure boundary conditions aren't being overwritten by initial conditions
# sed -i '/apply_boundary_conditions/i \    print(f"DEBUG: Max U before BC: {np.max(state.stencil_matrix[0].field.u)}")' src/main_solver.py

echo "============================================================"
echo "✅ Audit Complete. If Audit 3 shows a clipping function, that is your root cause."