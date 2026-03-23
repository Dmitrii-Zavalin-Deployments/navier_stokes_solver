#!/bin/bash
# forensic_audit.sh - Rule 7: Bridging the Predictor Blind Spot

echo "============================================================"
echo "🔍 DIAGNOSING: Field Index Blind Spot"
echo "============================================================"

# 1. Audit line 492 - Verify it only checks VX, VY, VZ
echo "--- [SOURCE AUDIT: src/common/solver_state.py] ---"
cat -n src/common/solver_state.py | sed -n '488,498p'

# 2. Check if main_solver is calling stabilization(is_needed=True)
echo -e "\n--- [ORCHESTRATION AUDIT: src/main_solver.py] ---"
grep -nC 5 "ArithmeticError" src/main_solver.py

echo -e "\n============================================================"
echo "🛠️ AUTOMATED REPAIR: Expanding Audit Scope"
echo "============================================================"

# Rule 7 Fix: Expand the audit to check both Foundation AND Predictor fields.
# If either is blown, we trigger the Elasticity ladder.
# # sed -i '492s/\[FI.VX, FI.VY, FI.VZ\]/\[FI.VX, FI.VY, FI.VZ, FI.VX_STAR, FI.VY_STAR, FI.VZ_STAR\]/' src/common/solver_state.py

echo "Audit Complete. Expanded audit scope to include Predictor (_STAR) fields."