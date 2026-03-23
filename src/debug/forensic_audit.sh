#!/bin/bash
# forensic_audit.sh - Investigating the Silent Physical Breach

echo "============================================================"
echo "🔍 DIAGNOSING: Why did $10^{10}$ velocity not trigger a fail?"
echo "============================================================"

# 1. Audit the Audit: Is SolverState actually checking VX, VY, VZ?
echo "--- [SOURCE AUDIT: src/common/solver_state.py] ---"
cat -n src/common/solver_state.py | grep -A 15 "def audit_physical_bounds"

# 2. Audit the Loop: Is the main solver catching the ArithmeticError?
echo -e "\n--- [ORCHESTRATION AUDIT: src/main_solver.py] ---"
cat -n src/main_solver.py | grep -A 20 "try:" | grep -B 5 -A 10 "ArithmeticError"

# 3. Verify the field indices being used in the audit
echo -e "\n--- [SCHEMA CHECK: src/common/field_schema.py] ---"
grep -E "VX|VY|VZ|P" src/common/field_schema.py

echo -e "\n============================================================"
echo "🛠️ AUTOMATED REPAIR: Reconnecting the Safety Circuit"
echo "============================================================"

# Fix 1: Ensure the audit actually raises the error (Rule 7)
# # sed -i '/v_max_current > pc.max_velocity/!b;n;c\            raise ArithmeticError(f"PHYSICAL EXPLOSION: Velocity {v_max_current}")' src/common/solver_state.py

# Fix 2: Ensure main_solver passes the error to stabilization
# # sed -i 's/except ArithmeticError:/except ArithmeticError:\n            elasticity.stabilization(is_needed=True)/' src/main_solver.py

echo "Audit Complete. Check if the audit is looking at *_STAR fields or foundation fields."