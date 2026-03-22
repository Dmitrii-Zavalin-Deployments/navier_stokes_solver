#!/bin/bash
echo "============================================================"
echo "🎯 PHASE H: RESIDUAL SENSITIVITY & TRAP-DOOR AUDIT"
echo "============================================================"

# --- [Audit 1] PPE Residual Logic ---
echo "--- [Audit 1] Checking how 'delta' is calculated in Step 3 ---"
# If delta = abs(new - old) / old, and old is 1e15, delta becomes 0.0 (Silent Pass)
grep -n "delta =" src/step3/ppe_solver.py

# --- [Audit 2] Logic Ordering Audit ---
echo "--- [Audit 2] Confirming BC application relative to PPE ---"
# If BC is applied AFTER PPE, the PPE uses the state from the PREVIOUS step.
cat -n src/main_solver.py | sed -n '105,115p'

# --- [Audit 3] Force Divergence Trigger ---
echo "--- [Audit 3] Checking if 'ArithmeticError' is actually raisable ---"
python3 -c "import numpy as np; np.seterr(all='raise'); print(np.array([1e300]) * 1e300)" || echo "Traps working."

# --- [Audit 4] Manual Injection Verification ---
echo "--- [Audit 4] Verifying if 1e15 is even in the block after Step 4 ---"
# Check if Step 5 wipes the data before the next loop starts
grep -r "clear_field" src/step5/

# --- [5] AUTOMATED REPAIRS (The "Signal Fix") ---

# REPAIR A: Force an instability if velocity is physically impossible (> 1e6)
# This ensures we don't rely on hardware floating point overflows.
# sed -i '/orchestrate_step4/a \                if np.max(np.abs(block.center.get_field(FI.VX))) > 1e10: raise ArithmeticError("Velocity Explosion")' src/main_solver.py

# REPAIR B: Move BC application BEFORE the PPE solver
# sed -i '111d; 109a \                    orchestrate_step4(block, context, state.grid, state.boundary_conditions)' src/main_solver.py

# REPAIR C: Log all warnings to a file to ensure caplog isn't just failing to capture
# sed -i '23a logging.basicConfig(filename="audit_debug.log", level=logging.WARNING)' src/main_solver.py

echo "============================================================"
echo "✅ Audit Complete. Apply REPAIR A to force the recovery path signal."