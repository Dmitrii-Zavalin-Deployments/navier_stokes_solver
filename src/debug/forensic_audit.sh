#!/bin/bash
# forensic_audit.sh - Diagnosing State Mapping Discontinuity

echo "============================================================"
echo "🔍 DIAGNOSING: State Initialization Gap"
echo "============================================================"

# 1. Audit the handover logic in Step 1
echo "--- [HANDOVER AUDIT: src/step1/orchestrate_step1.py] ---"
if grep -q "physical_constraints" src/step1/orchestrate_step1.py; then
    cat -n src/step1/orchestrate_step1.py | grep -C 3 "physical_constraints"
else
    echo "⚠️ SMOKING GUN: physical_constraints is never assigned in orchestrate_step1.py"
    cat -n src/step1/orchestrate_step1.py
fi

# 2. Verify SolverState property setter
echo -e "\n--- [PROPERTY AUDIT: src/common/solver_state.py] ---"
cat -n src/common/solver_state.py | grep -A 5 "def physical_constraints(self, value):"

echo -e "\n============================================================"
echo "🛠️ AUTOMATED REPAIR: Mapping Injection"
echo "============================================================"

# This sed command injects the missing assignment into orchestrate_step1.
# It looks for the assignment of external_forces and appends the physical_constraints assignment after it.

# # sed -i '/state.external_forces =/a \    state.physical_constraints = input_data.physical_constraints' src/step1/orchestrate_step1.py

# Ensure the SolverState class-level attribute is ready to receive it
# # sed -i '/self._external_forces =/a \        self._physical_constraints = None' src/common/solver_state.py

echo "Audit Complete. Un-comment # sed lines to repair the state handover."