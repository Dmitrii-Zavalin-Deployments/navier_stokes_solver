#!/bin/bash
# src/debug/forensic_audit.sh

echo "============================================================"
echo "🔍 DIAGNOSING: Step 1 Handover Failure"
echo "============================================================"

# 1. Confirm the omission in Section 2 of Orchestrator
echo "--- [ORCHESTRATOR AUDIT: src/step1/orchestrate_step1.py] ---"
cat -n src/step1/orchestrate_step1.py | sed -n '53,60p'

# 2. Verify the Manager exists in SolverState (Dependency Check)
echo -e "\n--- [DEPENDENCY AUDIT: src/common/solver_state.py] ---"
grep "class PhysicalConstraintsManager" src/common/solver_state.py || echo "⚠️ Warning: PhysicalConstraintsManager might be missing from solver_state.py"

echo -e "\n============================================================"
echo "🛠️ AUTOMATED REPAIR: Mapping Injection"
echo "============================================================"

# This sed command finds Section 2 (Physical Context) and injects the 
# PhysicalConstraintsManager initialization and mapping.

# # sed -i '/state.external_forces.force_vector =/a \ \n        state.physical_constraints = PhysicalConstraintsManager()\n        state.physical_constraints.min_velocity = float(input_data.physical_constraints.min_velocity)\n        state.physical_constraints.max_velocity = float(input_data.physical_constraints.max_velocity)\n        state.physical_constraints.min_pressure = float(input_data.physical_constraints.min_pressure)\n        state.physical_constraints.max_pressure = float(input_data.physical_constraints.max_pressure)' src/step1/orchestrate_step1.py

# Ensure PhysicalConstraintsManager is included in the imports if it isn't already
# # sed -i '/    SolverState,/a \    PhysicalConstraintsManager,' src/step1/orchestrate_step1.py

echo "Audit Complete. Un-comment # sed lines to repair the state handover."