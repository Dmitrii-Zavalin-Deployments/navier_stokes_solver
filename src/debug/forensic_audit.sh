#!/bin/bash
echo "🔍 STARTING DEEP FORENSIC AUDIT: THE SILENT OVERFLOW GHOST"

# 1. Audit NumPy's Error State in the Solver
echo "--- [1] Checking for NumPy Error Configuration ---"
grep -r "np.seterr" src/main_solver.py || echo "⚠️ Rule 5 Violation: NumPy error behavior is undefined (Silent Defaults)."

# 2. Trace the exception path in main_solver
echo "--- [2] Smoking-Gun Audit: main_solver.py ---"
cat -n src/main_solver.py | sed -n '90,120p'

# 3. Check for Logger naming drift
echo "--- [3] Logger Name Check ---"
grep "logging.getLogger" src/common/elasticity.py

# 4. REPAIR: Force NumPy to raise ArithmeticError on all math anomalies
# This ensures the 'except ArithmeticError' block is actually triggered.
# # sed -i '1s/^/import numpy as np\nnp.seterr(all="raise")\n/' src/main_solver.py

# 5. REPAIR: Harmonize Logger name for pytest caplog visibility
# Changing from "Elasticity" to the module path ensures standard log capture.
# # sed -i 's/logging.getLogger("Elasticity")/logging.getLogger("src.common.elasticity")/g' src/common/elasticity.py

# 6. REPAIR: Add ValueError to the catch block (common for linear algebra failures)
# # sed -i 's/except ArithmeticError:/except (ArithmeticError, ValueError):/g' src/main_solver.py

echo "✅ Audit Complete. Repairs staged in sed comments."