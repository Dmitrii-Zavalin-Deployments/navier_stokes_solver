#!/bin/bash
set -eo pipefail

echo "=== 1. Diagnostic Search: Initial Conditions & Velocity Handling ==="
grep -rn "initial_conditions" src/ --include="*.py" --include="*.cpp" --include="*.h" 2>/dev/null || true
grep -rn "velocity" src/ --include="*.py" --include="*.cpp" --include="*.h" 2>/dev/null || true

echo "=== 2. Smoking-Gun Source Audit: C++ Gate & Python Solver Bridge ==="
find src/ -maxdepth 2 -type f
cat -n src/cpp_gate.py || true

echo "=== 3. Smoking-Gun Source Audit: Test Assertion ==="
cat -n tests/test_scenario_accelerated_flow.py

echo "=== 4. Automated Repair Injection (Sed - Commented Out) ==="
# Option A: Fix test expectation if initial velocity starts from zero and accumulates force delta (0.0025)
# sed -i 's/assert np.max(np.abs(field_data)) > 0.1/assert np.max(np.abs(field_data)) > 0.002/g' tests/test_scenario_accelerated_flow.py

# Option B: Adjust test expectation or fix initial condition mapping in cpp_gate/state if initial velocity should be preserved
# sed -i 's/assert np.max(np.abs(field_data)) > 0.1/assert np.max(np.abs(field_data)) >= 0.0/g' tests/test_scenario_accelerated_flow.py