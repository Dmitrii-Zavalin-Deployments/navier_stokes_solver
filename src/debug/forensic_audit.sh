echo "============================================================"
echo "🔍 DIAGNOSING LOG CAPTURE FAILURE: Solver.Main vs Caplog"
echo "============================================================"

# --- [1] Audit: Check Logger propagation and level ---
echo "--- [Audit 1] main_solver.py Logger definition ---"
cat -n src/main_solver.py | grep -B 2 -A 2 "getLogger"

# --- [2] Audit: Inspect Test capture logic ---
echo "--- [Audit 2] Test file capture configuration ---"
cat -n tests/property_integrity/test_heavy_elasticity_lifecycle.py | grep -A 10 "caplog.at_level"

# --- [3] Audit: Verify Numerical Config is actually called ---
# If this is missing, ArithmeticError is never raised, so the log is never hit.
echo "--- [Audit 3] Search for _configure_numerical_runtime call ---"
grep -n "_configure_numerical_runtime(context)" src/main_solver.py

# --- [4] AUTOMATED REPAIRS ---

# REPAIR A: Update the test to listen specifically to the named logger 'Solver.Main'
# This is the most likely fix for 'assert 0 > 0' when the code clearly shows the log line.
# sed -i 's/caplog.at_level(logging.WARNING):/caplog.at_level(logging.WARNING, logger="Solver.Main"):/' tests/property_integrity/test_heavy_elasticity_lifecycle.py

# REPAIR B: Fallback - Force propagation to root so default caplog catches it
# sed -i '/logger = logging.getLogger("Solver.Main")/a logger.propagate = True' src/main_solver.py

# REPAIR C: If the error isn't triggering, ensure the numerical runtime is configured
# sed -i '/context = _load_simulation_context(input_path)/a \    _configure_numerical_runtime(context)' src/main_solver.py

echo "============================================================"
echo "✅ Forensic Audit and Repair Script Ready"