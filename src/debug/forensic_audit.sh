#!/usr/bin/env bash
# File: src/debug/forensic_audit.sh
# Purpose: Post-test forensic audit for dispatcher / elasticity / MMS failures.
set -euo pipefail

echo "=== Forensic audit: starting ==="

# --------------------------------------------------------------------
# 0. Reconfirm failing tests (last run)
# --------------------------------------------------------------------
if [ -f "pytest.log" ]; then
  echo "=== Tail of pytest.log ==="
  tail -n 200 pytest.log || true
fi

echo "=== Pytest lastfailed cache (if any) ==="
if [ -f ".pytest_cache/v/cache/lastfailed" ]; then
  cat .pytest_cache/v/cache/lastfailed || true
fi

# --------------------------------------------------------------------
# 1. MMS center-mutation failure (Boundary Applier / dispatcher wiring)
# --------------------------------------------------------------------
echo
echo "=== MMS center-mutation probe (test_step3_mms) ==="
echo ">>> Smoking-gun: VX_STAR remains 1.0 on masked center cell"
grep -RIn "test_logic_gate_3_center_mutation_audit" tests/quality_gates/logic_gate || true
echo
cat -n tests/quality_gates/logic_gate/test_step3_mms.py | sed -n '51,95p' || true

echo
echo "=== Orchestrator + boundary applier wiring ==="
cat -n src/step3/orchestrate_step3.py || true
echo
cat -n src/step3/boundaries/applier.py || true

echo
echo "=== Probe: center mask / VX_STAR behavior around failing block ==="
grep -RIn "VX_STAR before" -n . || true
grep -RIn "Block id:" -n . || true

# --------------------------------------------------------------------
# 2. Dispatcher: spatial vs mask priority + logging contract
# --------------------------------------------------------------------
echo
echo "=== Dispatcher: implementation vs tests ==="
cat -n src/step3/boundaries/dispatcher.py || true

echo
echo "=== Dispatcher tests (spatial + mask symmetry) ==="
cat -n tests/step3/test_dispatcher.py || true
echo
cat -n tests/quality_gates/sensitivity_gate/test_bc_collisions.py || true

echo
echo "=== Grep: ghost-based domain detection & mask logging ==="
grep -RIn "_get_domain_location_type" src/step3 || true
grep -RIn "is_ghost" src/step3 || true
grep -RIn "DISPATCH [Mask]" -n src tests || true
grep -RIn "treated as Wall (mask -1)" -n tests || true

# --------------------------------------------------------------------
# 3. Elasticity lifecycle: expected RuntimeError not raised
# --------------------------------------------------------------------
echo
echo "=== Elasticity lifecycle tests ==="
cat -n tests/property_integrity/test_heavy_elasticity_lifecycle.py || true

echo
echo "=== Grep: elasticity / terminal failure hooks in src ==="
grep -RIn "Elasticity" src || true
grep -RIn "terminal_failure" src || true
grep -RIn "RuntimeError" src/common src/step4 src || true

# --------------------------------------------------------------------
# 4. Step 1 / Step 2 topology & mask wiring (for dispatcher context)
# --------------------------------------------------------------------
echo
echo "=== Step 1: mask + padded mask wiring ==="
cat -n src/step1/helpers.py || true
grep -RIn "padded_mask" -n src/step1 src/common || true
cat -n src/step1/orchestrate_step1.py | sed -n '70,105p' || true

echo
echo "=== Step 2: stencil assembly (ghost vs core wiring) ==="
cat -n src/step2/stencil_assembler.py || true
cat -n tests/helpers/solver_step2_output_dummy.py || true

# --------------------------------------------------------------------
# 5. Domain configuration / domain_type drift (INTERNAL vs EXTERNAL)
# --------------------------------------------------------------------
echo
echo "=== Domain configuration wiring (INTERNAL / EXTERNAL) ==="
grep -RIn "domain_configuration" -n src tests || true
grep -RIn "domain_type" -n src tests || true
grep -RIn "\"type\": \"INTERNAL\"" -n tests src || true
grep -RIn "\"type\": \"EXTERNAL\"" -n tests src || true

# --------------------------------------------------------------------
# 6. Suggested automated repair hooks (commented sed lines)
#    These are NOT executed automatically; they are explicit, reviewable patches.
# --------------------------------------------------------------------
echo
echo "=== Suggested sed patches (commented; manual opt-in) ==="

# --- A. Restore ghost-based spatial detection in dispatcher ---
# sed -i 's/^def _get_domain_location_type(block: StencilBlock, grid) -> str:.*/def _get_domain_location_type(block: StencilBlock, grid) -> str:/' src/step3/boundaries/dispatcher.py
# sed -i 's/^    \"\"\"$/    \"\"\"/' src/step3/boundaries/dispatcher.py
# sed -i 's/^    # Primary & Only Authority: Ghost Neighbor Detection.*/    # Primary & Only Authority: Ghost Neighbor Detection/' src/step3/boundaries/dispatcher.py
# sed -i 's/^    if block.i_minus.is_ghost.*/    if block.i_minus.is_ghost: return "x_min"/' src/step3/boundaries/dispatcher.py
# sed -i 's/^    if block.i_plus.is_ghost.*/    if block.i_plus.is_ghost:  return "x_max"/' src/step3/boundaries/dispatcher.py
# sed -i 's/^    if block.j_minus.is_ghost.*/    if block.j_minus.is_ghost: return "y_min"/' src/step3/boundaries/dispatcher.py
# sed -i 's/^    if block.j_plus.is_ghost.*/    if block.j_plus.is_ghost:  return "y_max"/' src/step3/boundaries/dispatcher.py
# sed -i 's/^    if block.k_minus.is_ghost.*/    if block.k_minus.is_ghost: return "z_min"/' src/step3/boundaries/dispatcher.py
# sed -i 's/^    if block.k_plus.is_ghost.*/    if block.k_plus.is_ghost:  return "z_max"/' src/step3/boundaries/dispatcher.py
# sed -i 's/^    # If no neighbors are ghosts.*/    # If no neighbors are ghosts, fall back to mask axioms in caller/' src/step3/boundaries/dispatcher.py
# sed -i 's/^    return "none"/    return "none"/' src/step3/boundaries/dispatcher.py

# --- B. Enrich mask logging to match tests (wall / solid text) ---
# sed -i 's/LOGGER.debug(f"DISPATCH \

\[Mask\\]

: wall")/logger.debug(f"DISPATCH [Mask]: wall - treated as Wall (mask -1)")/' src/step3/boundaries/dispatcher.py
# sed -i 's/LOGGER.debug(f"DISPATCH \

\[Mask\\]

: solid")/logger.debug(f"DISPATCH [Mask]: solid - treated as Solid (mask 0)")/' src/step3/boundaries/dispatcher.py

# --- C. Ensure missing spatial config raises KeyError as tests expect ---
# sed -i 's/return _find_config(boundary_cfg, b_type)/return _find_config(boundary_cfg, b_type)/' src/step3/boundaries/dispatcher.py
# sed -i 's/except KeyError:/except KeyError:  # Rule 5: spatial must fail-fast before mask/' src/step3/boundaries/dispatcher.py
# sed -i 's/raise KeyError(f"Missing boundary definition for {b_type}") from None/raise KeyError(f"Missing boundary definition for {b_type}") from None/' src/step3/boundaries/dispatcher.py

# --- D. Elasticity lifecycle: force RuntimeError on terminal failure hooks ---
# (Exact hook depends on archive / elasticity implementation; these are placeholders.)
# sed -i 's/if not terminal_failure:/if not terminal_failure:/' src/common/archive_service.py
# sed -i 's/# ELASTICITY_TERMINAL_FAILURE/raise RuntimeError("Elasticity Terminal Failure")  # ELASTICITY_TERMINAL_FAILURE/' src/common/archive_service.py

echo "=== Forensic audit: complete ==="
