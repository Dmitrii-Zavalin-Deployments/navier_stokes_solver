#!/usr/bin/env bash
set -euo pipefail

echo "=============================================="
echo " FORENSIC AUDIT: Boundary / Stencil Integrity "
echo "=============================================="

echo
echo "---- 1. Inspect Cell class definition ----"
grep -Rni "class Cell" -n src || true
echo
echo "---- Cell source (annotated) ----"
cat -n src/common/*.py | sed -n '/class Cell/,/class /p' || true

echo
echo "---- 2. Search for boundary metadata fields ----"
grep -Rni "is_boundary" src || true
grep -Rni "location" src || true
grep -Rni "boundary" src/common || true

echo
echo "---- 3. Inspect StencilBlock definitions ----"
grep -Rni "class StencilBlock" -n src || true
cat -n src/common/*.py | sed -n '/class StencilBlock/,/class /p' || true

echo
echo "---- 4. Inspect stencil_matrix construction ----"
grep -Rni "stencil_matrix" -n src || true
grep -Rni "build" -n src/common || true

echo
echo "---- 5. Dump boundary_conditions from test input ----"
echo "Input file:"
ls -l *.json || true
echo
echo "Boundary conditions:"
grep -Rni "\"boundary_conditions\"" -n . || true

echo
echo "---- 6. Inspect solver_state audit block ----"
grep -Rni "audit_physical_bounds" -n src/common/solver_state.py || true
cat -n src/common/solver_state.py | sed -n '/audit_physical_bounds/,/validate_physical_readiness/p'

echo
echo "---- 7. Inspect elasticity recovery logs ----"
grep -Rni "ROLLBACK" -n . || true
grep -Rni "Explosion" -n . || true
grep -Rni "P_real_range" -n . || true

echo
echo "---- 8. Inspect failing test ----"
grep -Rni "test_scenario_2_retry_and_recover" -n tests || true
cat -n tests/property_integrity/test_heavy_elasticity_lifecycle.py | sed -n '1,200p'

echo
echo "---- 9. Suggested automated repair (commented out) ----"
echo "# sed -i 's/class Cell:/class Cell:\\n    __slots__ = [\"index\", \"fields_buffer\", \"is_boundary\", \"location\"]/' src/common/cell.py"
echo "# sed -i 's/self.index = idx/self.index = idx; self.is_boundary = False; self.location = None/' src/common/cell.py"
echo "# sed -i 's/block.center = cell/block.center = cell; cell.is_boundary = True; cell.location = bc.location/' src/common/stencil_builder.py"

echo
echo "---- 10. Final status ----"
echo "Forensic audit completed."
