#!/bin/bash
# forensic_audit.sh - Automated Diagnostics and Repair for Contract Violations

echo "============================================================"
echo "🔍 DIAGNOSING: KeyError 'physical_constraints'"
echo "============================================================"

# 1. Audit the Source Code (Line 308)
echo "--- [SOURCE AUDIT: src/common/solver_input.py] ---"
cat -n src/common/solver_input.py | sed -n '300,315p'

# 2. Audit the Test Suite Input
echo -e "\n--- [TEST AUDIT: tests/property_integrity/test_heavy_elasticity_lifecycle.py] ---"
grep -A 5 "base_input" tests/property_integrity/test_heavy_elasticity_lifecycle.py

# 3. Check for Schema alignment
if [ -f "schema/solver_input_schema.json" ]; then
    echo -e "\n--- [SCHEMA AUDIT] ---"
    grep "physical_constraints" schema/solver_input_schema.json || echo "⚠️ Warning: physical_constraints missing from JSON Schema"
fi

echo -e "\n============================================================"
echo "🛠️ AUTOMATED REPAIR: Injecting missing key into test suite"
echo "============================================================"

# The following sed commands will inject physical_constraints into the fixtures
# to satisfy the new SolverInput requirement.

# Repair test_heavy_elasticity_lifecycle.py
# sed -i '/"external_forces":/a \            "physical_constraints": {"velocity_max": 1e6, "pressure_max": 1e6},' tests/property_integrity/test_heavy_elasticity_lifecycle.py

# Repair solver_input_schema_dummy.py (if applicable)
# sed -i '/"external_forces":/a \        "physical_constraints": {"velocity_max": 1e6, "pressure_max": 1e6},' tests/helpers/solver_input_schema_dummy.py

echo "Repair instructions generated. Un-comment sed lines in CI to apply."