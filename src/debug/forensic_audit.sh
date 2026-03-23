#!/bin/bash
# forensic_audit.sh - Redirecting Elasticity to SSoT

echo "============================================================"
echo "🔍 DIAGNOSING: SSoT Violation in Elasticity"
echo "============================================================"

# 1. Audit the smoking gun in ElasticManager
echo "--- [SOURCE AUDIT: src/common/elasticity.py] ---"
cat -n src/common/elasticity.py | sed -n '20,30p'

# 2. Verify the correct path in SimulationParameterManager
echo -e "\n--- [CONTAINER AUDIT: src/common/solver_state.py] ---"
grep -A 5 "class SimulationParameterManager" src/common/solver_state.py | grep "time_step"

echo -e "\n============================================================"
echo "🛠️ AUTOMATED REPAIR: Hierarchy Alignment"
echo "============================================================"

# This sed fixes the ElasticManager __init__ to use the proper container path
# # sed -i 's/self._dt = state.dt/self._dt = state.simulation_parameters.time_step/' src/common/elasticity.py

# Optional: Check if other parts of elasticity.py are using the incorrect 'state.dt'
# # grep -l "state.dt" src/common/elasticity.py | xargs -I {} sed -i 's/state.dt/state.simulation_parameters.time_step/g' {}

echo "Audit Complete. Un-comment # sed lines to align with Rule 4 (SSoT)."