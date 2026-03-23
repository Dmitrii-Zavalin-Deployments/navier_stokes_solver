#!/bin/bash
echo "============================================================"
echo "🕵️  FINAL AUDIT: Locating the Inflow Blocks"
echo "============================================================"

# 1. Search the pytest output directly (don't rely on simulation.log)
# We look for the 'inflow' string which SHOULD be there if the config loaded.
echo "--- Step 1: Checking for Inflow presence in logs ---"
pytest tests/property_integrity/test_heavy_elasticity_lifecycle.py -s | grep "Boundary: inflow"

# 2. Audit the BC Manager initialization
echo "--- Step 2: Mapping Logic Audit ---"
grep -A 15 "class BoundaryConditionManager" src/step3/boundaries/manager.py

# 3. Check the loop that calls the applier
echo "--- Step 3: Orchestration Loop Audit ---"
cat -n src/step3/orchestrate_step3.py | grep -C 5 "apply_boundary_values"

# ============================================================
# 🛠️ AUTOMATED REPAIR INJECTIONS
# ============================================================

# Repair 1: Force the Orchestrator to handle ALL boundary types
# Sometimes 'solid' is hardcoded in the loop; this ensures 'inflow' is processed.
# sed -i 's/if rule.get("type") == "solid":/if True:  # Process all BC types/' src/step3/orchestrate_step3.py

# Repair 2: Redirect stdout to a file so grep works in the next CI step
# sed -i 's/run_solver(input_filename)/run_solver(input_filename) > simulation.log 2>\&1/' tests/property_integrity/test_heavy_elasticity_lifecycle.py

echo "✅ Diagnostic commands ready. Run Step 1 manually to see if 'inflow' exists."