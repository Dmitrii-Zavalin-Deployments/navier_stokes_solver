#!/bin/bash
echo "============================================================"
echo "🔍 PHASE C: FORENSIC STRUCTURAL & PHYSICS AUDIT"
echo "============================================================"

# --- [Audit 1] Whitespace & Indentation Smoking Gun ---
echo "--- [Audit 1] Checking for Mixed Tabs/Spaces in Tests ---"
# Shows non-printable characters (Tabs as ^I, Line endings as $)
cat -A tests/property_integrity/test_heavy_elasticity_lifecycle.py | sed -n '95,110p'

# --- [Audit 2] Source Integrity Check ---
echo "--- [Audit 2] Line-Numbered Source Audit (Rule 7) ---"
cat -n tests/property_integrity/test_heavy_elasticity_lifecycle.py | sed -n '98,115p'

# --- [Audit 3] Search for Swallowed Exceptions ---
echo "--- [Audit 3] Scanning for 'except:' blocks that lack logging ---"
grep -r "except:" src/ | grep -v "logger"

# --- [Audit 4] Physics Kernel Sync Check ---
echo "--- [Audit 4] Verifying dt property setter in StencilBlock ---"
grep -A 5 "@dt.setter" src/common/stencil_block.py

# --- [5] AUTOMATED REPAIRS (Candidate Injections) ---

# REPAIR A: Force 4-space indentation and remove Tabs (Fixes the Ruff Error)
# sed -i 's/\t/    /g' tests/property_integrity/test_heavy_elasticity_lifecycle.py

# REPAIR B: Inject a hard sync-check into the main loop for CI debugging
# sed -i '/state.iteration += 1/i \            assert all(b.dt == elasticity.dt for b in state.stencil_matrix), "STENCIL_SYNC_FAILURE"' src/main_solver.py

# REPAIR C: Ensure NumPy traps are active in the test environment
# sed -i '104i \        import numpy as np; np.seterr(all="raise")' tests/property_integrity/test_heavy_elasticity_lifecycle.py

echo "============================================================"
echo "✅ Audit Complete. Review the 'cat -A' output for ^I (Tabs)."