#!/bin/bash
echo "============================================================"
echo "🎯 PHASE C: LOGGER INTERCEPTION & ITERATION DEPTH AUDIT"
echo "============================================================"

# --- [Audit 1] Logger Interception Check ---
echo "--- [Audit 1] Verifying if caplog can see Solver.Main ---"
# Check if the test file has been updated to use the named logger
grep "logger=" tests/property_integrity/test_heavy_elasticity_lifecycle.py

# --- [Audit 2] Loop Boundary Check ---
echo "--- [Audit 2] Inspecting Main Solver loop termination logic ---"
cat -n src/main_solver.py | sed -n '120,130p'

# --- [Audit 3] Step 3 Exception Bubbling ---
echo "--- [Audit 3] Checking if Step 3 returns NaNs instead of raising ---"
# If Step 3 returns a NaN 'delta' instead of raising, the loop in main_solver won't catch it
grep -n "return.*np.nan" src/step3/ppe_solver.py

# --- [Audit 4] Runtime Field Audit ---
echo "--- [Audit 4] Checking for 'None' or 'Inf' in memory ---"
# Check if elasticity.dt ever actually changes during the run
grep "dt reduction" src/common/elasticity.py

# --- [5] AUTOMATED REPAIRS (Candidate Injections) ---

# REPAIR A: Point caplog to the correct logger name
sed -i 's/logger=""/logger="Solver.Main"/' tests/property_integrity/test_heavy_elasticity_lifecycle.py

# REPAIR B: Force the solver to run longer so the instability can "bloom"
sed -i 's/"total_time": 0.2/"total_time": 2.0/' tests/property_integrity/test_heavy_elasticity_lifecycle.py

# REPAIR C: Ensure the logger name is consistent across the project
# sed -i 's/getLogger("Solver.Main")/getLogger("Solver")/' src/main_solver.py
# sed -i 's/logger="Solver.Main"/logger="Solver"/' tests/property_integrity/test_heavy_elasticity_lifecycle.py

echo "============================================================"
echo "✅ Audit Complete. Suggest applying REPAIR A and REPAIR B together."