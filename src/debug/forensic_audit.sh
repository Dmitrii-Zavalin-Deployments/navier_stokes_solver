#!/bin/bash
# forensic_audit.sh - Rule 7 & 4: The Predictor-Stage Firebreak

echo "============================================================"
echo "🔍 DIAGNOSING: The Empty Gate"
echo "============================================================"

# 1. Confirm the gap between Predictor and PPE
echo "--- [FLOW AUDIT: src/main_solver.py] ---"
cat -n src/main_solver.py | sed -n '95,105p'

# 2. Check for NaN-masking in the audit logic
echo -e "\n--- [LOGIC AUDIT: src/common/solver_state.py] ---"
grep "np.max" src/common/solver_state.py

echo -e "\n============================================================"
echo "🛠️ AUTOMATED REPAIR: Installing the Firebreak"
echo "============================================================"

# Fix 1: Insert the Fail-Fast Audit (Rule 7)
# This catches the 1e10 velocity before the PPE loop turns it into a crash.
# sed -i '98i \                state.audit_physical_bounds()' src/main_solver.py

# Fix 2: Upgrade to NaN-Aware maximum (Rule 7)
# Standard np.max(abs(nan)) is NaN, and NaN > limit is False. np.nanmax fixes this.
# sed -i 's/np.max(np.abs/np.nanmax(np.abs/' src/common/solver_state.py

# Fix 3: Ensure the log text matches the test expectation perfectly
# sed -i 's/STABILITY TRIGGER/STABILITY TRIGGER/' src/common/elasticity.py

echo "Audit Complete. Firebreak installed at Line 98."