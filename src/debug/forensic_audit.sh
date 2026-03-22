#!/bin/bash
echo "🔍 STARTING DEEP FORENSIC AUDIT: ELASTICITY DISCONNECT"

# 1. Audit the Catch Block in Main Solver
echo "--- [1] Checking main_solver.py Exception Handling ---"
cat -n src/main_solver.py | sed -n '100,125p'

# 2. Audit the Logger Configuration in Elasticity
echo "--- [2] Checking Elasticity Logger Name and Level ---"
cat -n src/common/elasticity.py | grep -A 5 "self.logger ="

# 3. Check for "Silent" Arithmetic Errors (NaNs in output but no raise)
echo "--- [3] Checking for NaN signatures in latest output ---"
if [ -f "navier_stokes_output/snapshot_0001.h5" ]; then
    echo "Snapshot found. Checking for data corruption..."
    # If h5py is available, we could probe here, otherwise check logs for NaN prints
fi

# 4. Verify test_input.json parameters actually reached the solver
echo "--- [4] Verifying Test Input generation ---"
cat test_input.json | jq '.simulation_parameters, .boundary_conditions[0]'

# 5. Automated Repair: Ensure main_solver uses the correct exception catching
# # sed -i 's/except ArithmeticError:/except (ArithmeticError, ValueError):/g' src/main_solver.py

# 6. Automated Repair: Ensure Elasticity Logger matches Caplog expectations
# # sed -i 's/logging.getLogger("Elasticity")/logging.getLogger("src.common.elasticity")/g' src/common/elasticity.py