#!/bin/bash
echo "============================================================"
echo "🎯 PHASE C: INDENTATION & STRUCTURAL REPAIR"
echo "============================================================"

# --- [Audit 1] Space Counting ---
echo "--- [Audit 1] Exact Space Count for Lines 99-105 ---"
# This identifies exactly how many leading spaces exist
sed -n '99,105p' tests/property_integrity/test_heavy_elasticity_lifecycle.py | grep -o '^ *' | awk '{ print length, "spaces" }'

# --- [Audit 2] Structural Alignment ---
echo "--- [Audit 2] Visualizing Block Alignment ---"
# Using cat -A to ensure there are no hidden characters causing the shift
cat -A tests/property_integrity/test_heavy_elasticity_lifecycle.py | sed -n '98,110p'

# --- [Audit 3] Physics Foundation Check ---
echo "--- [Audit 3] Verifying HDF5 usage in Step 5 (Output) ---"
grep -r "h5py.File" src/step5/

# --- [4] AUTOMATED REPAIRS (The "Indentation Hammer") ---

# REPAIR A: Fix Scenario 2 indentation (Lines 100 to 150)
# This adds 4 extra spaces to every line from the docstring to the end of the test function
# sed -i '100,150s/^    /        /' tests/property_integrity/test_heavy_elasticity_lifecycle.py

# REPAIR B: Global Tab-to-Space conversion (Safety Net)
# sed -i 's/\t/    /g' tests/property_integrity/test_heavy_elasticity_lifecycle.py

# REPAIR C: Fix the 'import' block specifically if sed A misses it
# sed -i '104,107s/^import/    import/' tests/property_integrity/test_heavy_elasticity_lifecycle.py

echo "============================================================"
echo "✅ Audit Complete. If Audit 1 showed equal space counts for 99 and 100, use REPAIR A."