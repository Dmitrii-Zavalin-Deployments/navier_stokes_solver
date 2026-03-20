echo "--- 1. MAP THE EXACT STRUCTURE OF MOCK_CONFIG ---"
# We need to see the line numbers and indentation to fix the sed
cat -n tests/helpers/solver_step5_output_dummy.py | grep -A 10 "MOCK_CONFIG ="

echo "--- 2. DEBUGGING THE INJECTION FAILURE ---"
# Check if the file was actually touched
ls -l tests/helpers/solver_step5_output_dummy.py

echo "--- 3. SURGICAL INJECTION: ROBUST DICTIONARY REPAIR ---"
# Strategy: Find the MOCK_CONFIG block and insert the missing key 
# before the closing brace '}'.
sed -i '/MOCK_CONFIG = {/,/}/ s/}/    "divergence_threshold": 1e6,\n}/' tests/helpers/solver_step5_output_dummy.py

echo "--- 4. REPAIRING THE PROPERTY INTEGRITY FIXTURES ---"
# Using a broader search to ensure we catch the SolverConfig calls in tests
find tests/property_integrity/ -name "*.py" -exec sed -i 's/ppe_omega=1.0/ppe_omega=1.0, divergence_threshold=1e6/g' {} +

echo "--- 5. VERIFYING REPAIR (NON-FATAL) ---"
# We use || true to ensure the CI continues even if the grep is empty
grep "divergence_threshold" tests/helpers/solver_step5_output_dummy.py || echo "⚠️ Injection still missing in dummy.py"
grep "divergence_threshold" tests/property_integrity/test_step1_initialization.py || echo "⚠️ Injection still missing in test_step1.py"

echo "--- 6. FINAL CLEANUP ---"
# Ensure the circuit breaker we tightened earlier is still at 0.1
grep "if elasticity.dt < 1e-1" src/main_solver.py