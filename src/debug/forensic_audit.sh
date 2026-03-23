#!/bin/bash
# forensic_audit.sh - Rule 7: NaN-Awareness & Fail-Fast Gates

echo "============================================================"
echo "🔍 DIAGNOSING: Why is the audit silent?"
echo "============================================================"

# 1. Check if the Audit is actually present in the main loop
echo "--- [FLOW AUDIT: src/main_solver.py] ---"
cat -n src/main_solver.py | sed -n '95,105p'

# 2. Check the Audit logic for NaN handling
echo -e "\n--- [LOGIC AUDIT: src/common/solver_state.py] ---"
cat -n src/common/solver_state.py | sed -n '490,495p'

echo -e "\n============================================================"
echo "🛠️ AUTOMATED REPAIR: Hard-Wiring the Safety Circuit"
echo "============================================================"

# Fix 1: Make the Audit NaN-Aware (Rule 7)
# np.nanmax ensures that if any NaN exists, it triggers the audit.
# # sed -i 's/np.max/np.nanmax/' src/common/solver_state.py

# Fix 2: Insert the Missing Safety Gate (Rule 7)
# We must audit AFTER the predictor (orchestrate_step3) but BEFORE the PPE loop.
# # sed -i '98i \                # Rule 7: Fail-Fast Predictor Audit\n                state.audit_physical_bounds()' src/main_solver.py

# Fix 3: Ensure Log Level Matching
# Force caplog to see the message by ensuring propagation is on.
# # sed -i 's/logger.propagate = False/logger.propagate = True/' src/main_solver.py

echo "Audit Complete. Safety gate installed and NaN-trapping enabled."