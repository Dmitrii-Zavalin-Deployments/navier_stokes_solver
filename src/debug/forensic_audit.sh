#!/usr/bin/env bash
# File: src/debug/forensic_audit.sh
# Purpose: Post-test forensic audit for boundary / MMS / BC-collision failures.

set -euo pipefail

echo "=== Forensic audit: starting ==="

# --------------------------------------------------------------------
# 1. Basic context: show failing tests (if pytest log is available)
# --------------------------------------------------------------------
if [ -f ".pytest_cache/v/cache/lastfailed" ]; then
  echo "=== Pytest lastfailed cache ==="
  cat .pytest_cache/v/cache/lastfailed || true
fi

# If GitHub Actions stored a test log, dump the tail for quick context
if [ -f "pytest.log" ]; then
  echo "=== Tail of pytest.log ==="
  tail -n 200 pytest.log || true
fi

# --------------------------------------------------------------------
# 2. MMS FAILURE: Boundary Applier ignored its own center cell
#    Message: "MMS FAILURE: Boundary Applier ignored its own center cell. Block Block_808 (Mask=0) remained poisoned."
#    Assertion: assert 1.0 != 1.0
# --------------------------------------------------------------------
echo "=== Searching for MMS / Boundary Applier messages ==="
grep -RIn --color=always "Boundary Applier ignored its own center cell" . || true
grep -RIn --color=always "Block_808" . || true
grep -RIn --color=always "MMS FAILURE" tests src || true

# Likely MMS / boundary applier implementation
echo "=== Candidate MMS / boundary applier sources (numbered) ==="
for f in \
  src/*mms*py \
  src/*boundary*py \
  src/**/mms*.py \
  src/**/boundary*.py \
  2>/dev/null; do
  echo "--- cat -n $f ---"
  cat -n "$f" | sed -n '1,260p'
done || true

# Smoking-gun search for "center" / "mask" logic
echo "=== Grep for center/mask logic in src ==="
grep -RIn --color=always "center" src || true
grep -RIn --color=always "mask" src || true
grep -RIn --color=always "poison" src || true

# Suggested automated repair hooks (commented out; for manual review)
# - Example: ensure center cell is included in boundary application loop
# - Example: clear poison flag for center when mask == 0
# sed -i 's/for cell in neighbors:/for cell in neighbors + [center_cell]:/' src/path/to/boundary_applier.py
# sed -i 's/if cell.is_boundary:/if cell.is_boundary or cell.is_center:/' src/path/to/boundary_applier.py
# sed -i 's/if mask == 0:/if mask == 0: center_cell.poisoned = False/' src/path/to/boundary_applier.py

# --------------------------------------------------------------------
# 3. KeyError: 'Missing boundary definition for x_min' / 'z_min'
# --------------------------------------------------------------------
echo "=== Searching for missing boundary definition KeyErrors ==="
grep -RIn --color=always "Missing boundary definition for x_min" . || true
grep -RIn --color=always "Missing boundary definition for z_min" . || true
grep -RIn --color=always "Missing boundary definition" src tests || true

# Likely boundary-condition dispatcher / registry
echo "=== Candidate boundary-condition dispatcher sources (numbered) ==="
for f in \
  src/**/boundary*.py \
  src/**/bc*.py \
  src/**/dispatcher*.py \
  2>/dev/null; do
  echo "--- cat -n $f ---"
  cat -n "$f" | sed -n '1,260p'
done || true

# Also inspect any boundary configuration / schema files
echo "=== Candidate boundary configuration files (numbered) ==="
for f in \
  config/**/boundary*.* \
  config/**/bc*.* \
  2>/dev/null; do
  echo "--- cat -n $f ---"
  cat -n "$f" | sed -n '1,260p'
done || true

# Suggested automated repair hooks (commented out; for manual review)
# - Example: ensure x_min / z_min keys exist in boundary registry
# sed -i "s/BOUNDARY_MAP = {/BOUNDARY_MAP = { 'x_min': DEFAULT_WALL_BC,/" src/path/to/bc_dispatcher.py
# sed -i "s/BOUNDARY_MAP = {/BOUNDARY_MAP = { 'z_min': DEFAULT_WALL_BC,/" src/path/to/bc_dispatcher.py
# - Example: provide symmetric defaults for missing faces
# sed -i "s/def get_bc(face)/def get_bc(face):\n    if face not in BOUNDARY_MAP: return DEFAULT_WALL_BC/" src/path/to/bc_dispatcher.py

# --------------------------------------------------------------------
# 4. Sensitivity gate: missing wall config collision
#    Failed: DID NOT RAISE <class 'KeyError'>
# --------------------------------------------------------------------
echo "=== Searching for missing wall config collision tests ==="
grep -RIn --color=always "missing_wall_config_collision" tests || true
grep -RIn --color=always "test_gate_3a_missing_wall_config_collision" tests || true

# Inspect the specific test file
if [ -f "tests/quality_gates/sensitivity_gate/test_bc_collisions.py" ]; then
  echo "--- cat -n tests/quality_gates/sensitivity_gate/test_bc_collisions.py ---"
  cat -n tests/quality_gates/sensitivity_gate/test_bc_collisions.py
fi

# Search for collision / wall config handling in src
echo "=== Grep for collision / wall config handling in src ==="
grep -RIn --color=always "collision" src || true
grep -RIn --color=always "wall_config" src || true
grep -RIn --color=always "missing wall" src || true

# Suggested automated repair hooks (commented out; for manual review)
# - Example: enforce KeyError when wall config is missing
# sed -i "s/config.get('wall')/config['wall']  # enforce KeyError on missing wall/" src/path/to/collision_handler.py
# - Example: add explicit guard raising KeyError
# sed -i "/def resolve_collision/a\    if 'wall' not in config: raise KeyError('Missing wall config')" src/path/to/collision_handler.py

# --------------------------------------------------------------------
# 5. Summary markers for GitHub Actions logs
# --------------------------------------------------------------------
echo "=== Forensic audit: completed ==="
echo "Check above for:"
echo " - MMS center-cell / mask handling"
echo " - Boundary dispatcher x_min / z_min definitions"
echo " - Missing wall config collision behavior vs tests"
