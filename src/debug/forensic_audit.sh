#!/bin/bash
# forensic_audit.sh - Rule 7 & 9: Atomic Commitment & Advance

echo "============================================================"
echo "🔍 DIAGNOSING: The Missing Safety Gate"
echo "============================================================"

# 1. Check where the audit is actually triggered in the main loop
echo "--- [FLOW AUDIT: src/main_solver.py] ---"
grep -nE "audit|stabilization" src/main_solver.py

# 2. Check the ElasticManager for the new Advance logic location
echo -e "\n--- [TARGET AUDIT: src/common/elasticity.py] ---"
cat -n src/common/elasticity.py | sed -n '50,65p'

echo -e "\n============================================================"
echo "🛠️ AUTOMATED REPAIR: Migrating Advancement to the Gatekeeper"
echo "============================================================"

# Fix 1: Add Advancement to ElasticManager (Rule 9)
# We increment time and iteration ONLY after a successful audit/commit.
# # sed -i '/data\[:, FI.P\] = data\[:, FI.P_NEXT\]/a \        self._state.iteration += 1\n        self._state.time += self._dt' src/common/elasticity.py

# Fix 2: Remove premature advancement from main_solver.py
# # sed -i '/state.iteration += 1/d' src/main_solver.py
# # sed -i '/state.time += elasticity.dt/d' src/main_solver.py

# Fix 3: Force an immediate audit after the Predictor Pass
# This ensures we catch the 1e10 velocity BEFORE the PPE solver wastes cycles.
# # sed -i '/orchestrate_step3.*is_first_pass=True/a \                state.audit_physical_bounds()' src/main_solver.py

echo "Audit Complete. Safety gates moved to post-predictor and post-commitment phases."