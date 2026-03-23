#!/bin/bash
# src/debug/forensic_audit.sh

echo "============================================================"
echo "🔍 DIAGNOSING: SolverState Initialization Gap"
echo "============================================================"

# 1. Audit SolverState Definition
echo "--- [STATE DEFINITION: src/common/solver_state.py] ---"
grep -n "_physical_constraints" src/common/solver_state.py

# 2. Audit Step 1 Orchestration (The likely culprit)
echo -e "\n--- [STEP 1 AUDIT: src/step1/orchestrate_step1.py] ---"
cat -n src/step1/orchestrate_step1.py | grep -C 5 "physical_constraints" || echo "⚠️ MISSING: Step 1 is not mapping constraints to State!"

# 3. Audit the Gate Keeper
echo -e "\n--- [GATE AUDIT: src/common/solver_state.py] ---"
cat -n src/common/solver_state.py | sed -n '510,520p'

echo -e "\n============================================================"
echo "🛠️ AUTOMATED REPAIR: State Mapping Injection"
echo "============================================================"

# The fix requires ensuring orchestrate_step1 actually copies the values from 
# the input object into the state object.

# # sed -i '/state.external_forces =/a \    state.physical_constraints = input_data.physical_constraints' src/step1/orchestrate_step1.py

# Also ensure SolverState initializes the private attribute to None in __init__ 
# to satisfy the hasattr() check in _get_safe.
# # sed -i '/self._external_forces =/a \        self._physical_constraints = None' src/common/solver_state.py

echo "Audit Complete. Un-comment # sed lines to bridge the Input -> State gap."