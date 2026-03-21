# 1. ARCHITECTURAL AUDIT: Verify the Nested Try-Except and Circuit Breaker
echo "--- AUDITING: src/main_solver.py ---"
grep -A 8 "except ArithmeticError" src/main_solver.py

echo -e "\n--- AUDITING: src/common/elasticity.py ---"
grep -A 5 "def apply_panic_mode" src/common/elasticity.py

# 2. BEHAVIORAL VERIFICATION: Run the specific "Panic" test
# This test is designed to force the solver into a floor-crash scenario.
# We expect a 'PASSED' result because pytest.raises(RuntimeError) will now catch the error.
echo -e "\n--- RUNNING TARGETED RECOVERY TEST ---"
pytest tests/property_integrity/test_heavy_elasticity_lifecycle.py -v

# 3. TRACEBACK INSPECTION (Simulation of a real failure)
# We run the solver directly with the known-bad input to see the "ABORT" log and Traceback.
# We use '|| true' so the script doesn't stop if the command (rightfully) fails.
echo -e "\n--- LIVE EXECUTION TRACEBACK CHECK ---"
python3 src/main_solver.py integration_input.json || true

# 4. LOG INTEGRITY CHECK
# Ensure the "ABORT" message we injected is actually hitting the logs.
if [ -f "solver.log" ]; then
    echo -e "\n--- CHECKING LOG FILE FOR FATAL SIGNALS ---"
    grep -E "PANIC|ABORT|FATAL" solver.log | tail -n 5
else
    echo -e "\n[Note] No solver.log found; standard output was used."
fi