#!/bin/bash
set -eo pipefail

echo "=== 1. Diagnostic Search: Simulation Parameters & Time Step Configuration ==="
grep -rn "time_step" src/ tests/ --include="*.py" --include="*.json" 2>/dev/null || true
grep -rn "initial_conditions" src/ tests/ --include="*.py" --include="*.json" 2>/dev/null || true

echo "=== 2. Diagnostic Search: Test Assertion & Velocity Thresholds ==="
grep -rn "failed to accelerate" tests/ --include="*.py" 2>/dev/null || true
grep -rn "initial_conditions" tests/ --include="*.py" --include="*.json" 2>/dev/null || true

echo "=== 3. Smoking-Gun Source Audit: Test File and Input Loader ==="
cat -n tests/test_scenario_accelerated_flow.py
cat -n src/state.py

echo "=== 4. Automated Repair Injection (Sed) ==="
# Populate missing simulation_parameters required by SolverState strict non-default policy
sed -i '/input_data\["grid"\]\.update/i \    input_data["simulation_parameters"] = {"time_step": 0.001, "total_time": 0.003, "output_interval": 1}' tests/test_scenario_accelerated_flow.py

echo "=== 5. Verification Test Execution ==="