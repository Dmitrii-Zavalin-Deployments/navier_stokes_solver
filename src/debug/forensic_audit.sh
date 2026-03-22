#!/bin/bash
echo "🔍 STARTING DEEP FORENSIC AUDIT: THE LOGGING & OVERFLOW GHOST"

# 1. Audit NumPy's Error State in the Solver
echo "--- [1] Checking for Rule 5 Compliance: NumPy Runtime Configuration ---"
grep -r "np.seterr" src/main_solver.py || echo "⚠️ VIOLATION: NumPy error behavior is undefined."

# 2. Smoking-Gun Audit: Trace the Exception Path
# We need to see if the (ArithmeticError, FloatingPointError, ValueError) block is correctly implemented.
echo "--- [2] Source Audit: Exception Handling in main_solver.py ---"
cat -n src/main_solver.py | sed -n '110,130p'

# 3. Logger Identity Check
# Testing if the logger name matches what the Test Suite expects.
echo "--- [3] Logger Name Audit: elasticity.py ---"
grep "logging.getLogger" src/common/elasticity.py

# 4. REPAIR: Force NumPy to 'raise' (Ensures the 'except' block is reachable)
# # sed -i 's/np.seterr(all="raise", under="ignore")/np.seterr(all="raise")/g' src/main_solver.py

# 5. REPAIR: Harmonize Logger name for pytest visibility
# Ensures the test's caplog can catch the 'Instability' message.
# # sed -i 's/logging.getLogger("Elasticity")/logging.getLogger("src.common.elasticity")/g' src/common/elasticity.py

# 6. REPAIR: Ensure the retry block catches specific Linear Algebra failures
# # sed -i 's/except ArithmeticError:/except (ArithmeticError, FloatingPointError, ValueError):/g' src/main_solver.py

echo "--- [4] Verification: Checking for 'Instability' string in source ---"
grep -r "Instability" src/common/elasticity.py

echo "✅ Audit Complete. Review the 'Source Audit' above to confirm the catch block logic."