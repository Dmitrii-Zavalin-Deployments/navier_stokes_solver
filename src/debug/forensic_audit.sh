#!/usr/bin/env bash
# src/debug/forensic_audit.sh
#
# Post-test forensic audit for elasticity / PPE failures in CI.
# Intended to be run from the repo root in GitHub Actions after pytest.

set -euo pipefail

echo "=== FORENSIC AUDIT: START ==="

###############################################################################
# 1. Pytest summary (if available)
###############################################################################
echo
echo "=== PYTEST SUMMARY (if available) ==="
if [ -f ".pytest_cache/v/cache/lastfailed" ]; then
  cat .pytest_cache/v/cache/lastfailed || true
else
  echo "(no lastfailed cache found)"
fi

###############################################################################
# 2. Scan repo for instability markers
###############################################################################
echo
echo "=== GREP: CRITICAL INSTABILITY / AUDIT / STABILITY TRIGGER ==="
grep -RIn \
  -e "CRITICAL INSTABILITY" \
  -e "STABILITY TRIGGER" \
  -e "AUDIT \

\[Explosion\\]

" \
  -e "AUDIT \

\[Limit\\]

" \
  tests src || true

###############################################################################
# 3. Show the heavy elasticity lifecycle test
###############################################################################
echo
echo "=== CAT -n: tests/property_integrity/test_heavy_elasticity_lifecycle.py ==="
if [ -f "tests/property_integrity/test_heavy_elasticity_lifecycle.py" ]; then
  cat -n tests/property_integrity/test_heavy_elasticity_lifecycle.py
else
  echo "(file missing)"
fi

###############################################################################
# 4. Elasticity engine (dt ladder + retry logic)
###############################################################################
echo
echo "=== CAT -n: src/common/elasticity.py ==="
if [ -f "src/common/elasticity.py" ]; then
  cat -n src/common/elasticity.py
else
  echo "(file missing)"
fi

###############################################################################
# 5. Physical audit (pressure corridor, rollback)
###############################################################################
echo
echo "=== CAT -n: src/common/solver_state.py (audit_physical_bounds) ==="
if [ -f "src/common/solver_state.py" ]; then
  cat -n src/common/solver_state.py | sed -n '560,700p'
else
  echo "(file missing)"
fi

###############################################################################
# 6. PPE solver (SOR + Rhie–Chow)
###############################################################################
echo
echo "=== CAT -n: src/step3/ppe_solver.py ==="
if [ -f "src/step3/ppe_solver.py" ]; then
  cat -n src/step3/ppe_solver.py
else
  echo "(file missing)"
fi

###############################################################################
# 7. Main solver orchestration (error propagation)
###############################################################################
echo
echo "=== CAT -n: src/main_solver.py ==="
if [ -f "src/main_solver.py" ]; then
  cat -n src/main_solver.py
else
  echo "(file missing)"
fi

###############################################################################
# 8. Search junit/logs for failing test
###############################################################################
echo
echo "=== GREP: failing test name in junit/logs (if present) ==="
grep -RIn "test_scenario_4_retry_and_recover" . || true

###############################################################################
# 9. Suggested sed repair hooks (commented out)
###############################################################################
echo
echo "=== SUGGESTED sed HOOKS (COMMENTED) FOR LOCAL REPAIR ==="
echo "# Example: make Scenario 4 easier to recover"
echo "# sed -i 's/u = np.float64(15.0)/u = np.float64(12.0)/' tests/property_integrity/test_heavy_elasticity_lifecycle.py"
echo "# sed -i 's/dt_initial = np.float64(0.04)/dt_initial = np.float64(0.02)/' tests/property_integrity/test_heavy_elasticity_lifecycle.py"
echo "# sed -i 's/\"ppe_max_retries\": 5/\"ppe_max_retries\": 8/' tests/property_integrity/test_heavy_elasticity_lifecycle.py"
echo "# sed -i 's/\"dt_min_limit\": 1e-4/\"dt_min_limit\": 1e-5/' tests/property_integrity/test_heavy_elasticity_lifecycle.py"

echo
echo "=== FORENSIC AUDIT: END ==="
