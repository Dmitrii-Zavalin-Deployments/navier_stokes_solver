#!/bin/bash
# forensic_audit.sh - Rule 7 & 9: Predictor Awareness and Recovery Orchestration

echo "============================================================"
echo "🔍 DIAGNOSING: The Disconnected Safety Circuit"
echo "============================================================"

# 1. Confirm Predictor fields vs Foundation fields in the audit
echo "--- [SOURCE AUDIT: src/common/solver_state.py] ---"
cat -n src/common/solver_state.py | sed -n '490,498p'

# 2. Confirm missing stabilization call in the exception handler
echo -e "\n--- [ORCHESTRATION AUDIT: src/main_solver.py] ---"
cat -n src/main_solver.py | grep -A 10 "except ArithmeticError:"

echo -e "\n============================================================"
echo "🛠️ AUTOMATED REPAIR: Connecting Predictors to the Safety Ladder"
echo "============================================================"

# Fix 1: Expand Audit Scope to include intermediate *_STAR fields (Rule 7)
sed -i '492s/\[FI.VX, FI.VY, FI.VZ\]/\[FI.VX, FI.VY, FI.VZ, FI.VX_STAR, FI.VY_STAR, FI.VZ_STAR\]/' src/common/solver_state.py

# Fix 2: Delegate recovery to the Elasticity Manager (Rule 4)
# We replace the manual iteration/time decrement with a formal stabilization call.
sed -i '/except ArithmeticError:/!b;n;n;n;n;c\            elasticity.stabilization(is_needed=True)' src/main_solver.py

echo "Audit Complete. Un-comment # sed lines to synchronize the physics engine."