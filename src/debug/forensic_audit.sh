#!/bin/bash
echo "============================================================"
echo "🔍 DIAGNOSTIC: Investigating Pressure Reconstruction Neutralization"
echo "============================================================"

# 1. Source Audit: Check how audit_physical_bounds handles the reference pressure
echo "--- [SOURCE AUDIT: solver_state.py Pressure Logic] ---"
cat -n src/common/solver_state.py | sed -n '580,610p'

# 2. Test Audit: Check the current failure point in the test
echo "--- [SOURCE AUDIT: test_solver_state.py] ---"
cat -n tests/common/test_solver_state.py | sed -n '85,95p'

# 3. Root Cause Verification:
# The test sets a uniform 10M. Reconstructed pressure becomes:
# p_real = 10M (cell) - 10M (ref_mean) + 0.0 (ref_bc) = 0.0 -> This will ALWAYS pass.

echo "--- [REPAIR PLAN]: Injecting localized pressure spike to trigger violation ---"

# Automated Repair: Update the test to use a non-uniform field that reconstruction won't neutralize
sed -i '91s/populated_state.fields.data\[:, FI.P_NEXT\] = 10_000_000.0/populated_state.fields.data\[:, FI.P_NEXT\] = 0.0; populated_state.fields.data\[0, FI.P_NEXT\] = 20_000_000.0/' tests/common/test_solver_state.py

echo "✅ Repair Instructions generated. Spike at index 0 will ensure p_real > 1e6."