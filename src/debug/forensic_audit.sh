echo "============================================================"
echo "🔍 DIAGNOSING LOG SIGNAL MISMATCH: ELASTICITY RETRY"
echo "============================================================"

# --- [1] Diagnostic: Verify Logger Name and Level ---
echo "--- [Audit 1] main_solver.py Logger Configuration ---"
# Check if the logger name matches what the test expect or if it's being silenced
grep -n "logger =" src/main_solver.py
grep -n "logger.warning" src/main_solver.py

# --- [2] Diagnostic: Check NumPy Error State ---
echo "--- [Audit 2] NumPy Error Handling Policy ---"
# Ensure np.seterr is actually inside the runtime config as required by Rule 5
grep -A 5 "def _configure_numerical_runtime" src/main_solver.py

# --- [3] Smoking Gun: View the catch block in main_solver ---
echo "--- [Audit 3] main_solver.py Recovery Block Audit ---"
# Verify the exact string being logged during a failure
cat -n src/main_solver.py | sed -n '100,130p'

# --- [4] Automated Repairs (Drafts) ---

# REPAIR A: Ensure the logger uses the specific 'Solver.Main' handle instead of root logging
# This aligns the code with Rule 4 and ensures caplog picks it up.
# sed -i 's/import logging; logging.warning/logger.warning/g' src/main_solver.py

# REPAIR B: Force the log message to match the test's exact regex if 'Instability' is missing
# sed -i 's/Arithmetic anomaly/Instability detected: Arithmetic anomaly/g' src/main_solver.py

# REPAIR C: Ensure the test input actually triggers the raise by checking dt_min_limit
# If dt_min_limit is too high, elasticity might fail to reduce dt enough to succeed.
# sed -i 's/"dt_min_limit": 0.0001/"dt_min_limit": 1e-9/g' config.json

echo "============================================================"
echo "✅ Forensic Audit and Repair Script Ready"