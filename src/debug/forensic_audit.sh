#!/bin/bash
# src/debug/forensic_audit.sh

echo "============================================================"
echo "🔍 SMOKING GUN: KeyError 'physical_constraints' Audit"
echo "============================================================"

# 1. Audit Ingestion Logic
echo "--- [INGESTION AUDIT: src/common/solver_input.py] ---"
cat -n src/common/solver_input.py | sed -n '305,313p'

# 2. Audit Test Fixture (Checking for missing key)
echo -e "\n--- [FIXTURE AUDIT: tests/property_integrity/test_heavy_elasticity_lifecycle.py] ---"
grep -A 10 "base_input" tests/property_integrity/test_heavy_elasticity_lifecycle.py

# 3. Validation of Schema Compliance
echo -e "\n--- [SCHEMA COMPLIANCE CHECK] ---"
if [ -f "schema/solver_input_schema.json" ]; then
    grep -C 2 "physical_constraints" schema/solver_input_schema.json
fi

echo -e "\n============================================================"
echo "🛠️ AUTOMATED REPAIR: Fixture Injection"
echo "============================================================"

# This sed command finds the external_forces block in the test fixture 
# and appends the missing physical_constraints block immediately after.

# # sed -i '/"external_forces": {[^}]*}/a \            "physical_constraints": {"min_velocity": -1e6, "max_velocity": 1e6, "min_pressure": -1e6, "max_pressure": 1e6},' tests/property_integrity/test_heavy_elasticity_lifecycle.py

# Also repair the global dummy helper to prevent future failures
# # sed -i '/"external_forces": {[^}]*}/a \        "physical_constraints": {"min_velocity": -1e6, "max_velocity": 1e6, "min_pressure": -1e6, "max_pressure": 1e6},' tests/helpers/solver_input_schema_dummy.py

echo "Audit Complete. Un-comment # sed lines to apply the hotfix in CI."