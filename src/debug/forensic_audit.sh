#!/bin/bash
echo "============================================================"
echo "🎯 FORENSIC AUDIT: BREAKING THE SILENT TRY-BLOCK"
echo "============================================================"

# --- [1] Audit: The "Swallow" Reveal ---
echo "--- [Audit 1] Checking for silent exception swallowing in Step 3 ---"
# We need to see what happens after Line 54. If it's "except: return block", we found it.
cat -n src/step3/orchestrate_step3.py | sed -n '50,70p'

# --- [2] Audit: Main Solver Recovery Logic ---
echo "--- [Audit 2] Checking the catch block in main_solver.py ---"
# Verifying if the main loop actually has the logic to catch the bubble-up.
cat -n src/main_solver.py | grep -C 5 "except"

# --- [3] Audit: Logger Propagation ---
echo "--- [Audit 3] Checking if Solver.Main propagates to Root ---"
grep "propagate" src/main_solver.py || echo "⚠️ Logger propagation not explicitly set."

# --- [4] AUTOMATED REPAIRS ---

# REPAIR A: Fix the "Swallow" in Step 3
# If there is an 'except' block catching math errors, we force it to re-raise.
# Rule 7: Fail-Fast.
# sed -i '/except ArithmeticError:/a \            raise' src/step3/orchestrate_step3.py
# sed -i '/except Exception:/a \            raise' src/step3/orchestrate_step3.py

# REPAIR B: Synchronize Test Logger with Code Logger
# Rule 6: Listen to the correct pipe.
# sed -i 's/logger=""/logger="Solver.Main"/g' tests/property_integrity/test_heavy_elasticity_lifecycle.py

# REPAIR C: Force Finite-Math Guard (The "Hard Stop")
# Injecting a manual finite check in the predictor to ensure we don't return 'inf' blocks.
# sed -i '45i \            if not np.isfinite(delta): raise ArithmeticError("PPE Diverged: non-finite delta")' src/step3/orchestrate_step3.py

# REPAIR D: Emergency Print Signal
# Direct stdout bypasses all logger/caplog issues to prove the recovery path was hit.
# sed -i '/logger.warning/a \            print("CI_SIGNAL: INSTABILITY_CAUGHT_REDUCING_DT")' src/main_solver.py

echo "============================================================"
echo "✅ Audit Block Configured. Run to expose the 'return' instead of 'raise'."
echo "============================================================"
cat -n src/step3/orchestrate_step3.py
echo "============================================================"
cat -n src/main_solver.py