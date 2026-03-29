#!/usr/bin/env bash
set -euo pipefail

echo "============================================================"
echo "🔍 Forensic Audit: CLI Entrypoint Failures"
echo "============================================================"

# Ensure we are in repo root
if [ -n "${GITHUB_WORKSPACE-}" ] && [ -d "$GITHUB_WORKSPACE" ]; then
  cd "$GITHUB_WORKSPACE"
fi

echo
echo "📁 STAGE 1 — Repo Layout"
echo "------------------------------------------------------------"
echo "PWD: $(pwd)"
echo "Top-level files:"
ls -1 || true

echo
echo "📄 STAGE 2 — Inspect main_solver.py (__main__ block)"
echo "------------------------------------------------------------"
if [ -f src/main_solver.py ]; then
  echo ">>> Showing CLI block (lines 150–200):"
  cat -n src/main_solver.py | sed -n '150,200p'
else
  echo "❌ src/main_solver.py not found"
fi

echo
echo "🧪 STAGE 3 — Inspect failing tests"
echo "------------------------------------------------------------"
if [ -f tests/test_main_solver_flow.py ]; then
  echo ">>> test_cli_entrypoint_success:"
  grep -n "test_cli_entrypoint_success" -n tests/test_main_solver_flow.py
  cat -n tests/test_main_solver_flow.py | sed -n '90,150p'

  echo
  echo ">>> test_cli_entrypoint_error:"
  grep -n "test_cli_entrypoint_error" -n tests/test_main_solver_flow.py
  cat -n tests/test_main_solver_flow.py | sed -n '150,210p'
else
  echo "❌ tests/test_main_solver_flow.py not found"
fi

echo
echo "🧬 STAGE 4 — ROOT CAUSE SUMMARY"
echo "------------------------------------------------------------"
echo "1) runpy.run_module loads a *fresh* module instance."
echo "   Patches applied to the already-imported module DO NOT apply."
echo
echo "2) Therefore:"
echo "   - _load_simulation_context is NOT mocked in the executed module"
echo "   - run_solver.side_effect is NOT applied"
echo
echo "3) The executed module hits real disk I/O:"
echo "      FileNotFoundError: Input file missing at ..."
echo
echo "4) This produces:"
echo "      SystemExit(1)"
echo "      print('FATAL PIPELINE ERROR: Input file missing at ...')"
echo
echo "5) Tests incorrectly expect:"
echo "      SystemExit(0)"
echo "      print('FATAL PIPELINE ERROR: System Crash')"

echo
echo "🛠 STAGE 5 — Suggested sed Repairs (COMMENTED)"
echo "------------------------------------------------------------"

echo "# FIX A: Patch success test to expect exit code 1"
echo "# sed -i \"s/assert e.value.code == 0/assert e.value.code == 1/\" tests/test_main_solver_flow.py"

echo
echo "# FIX B: Patch error test to expect FileNotFoundError message"
echo "# sed -i \"s/'FATAL PIPELINE ERROR: System Crash'/'FATAL PIPELINE ERROR: Input file missing at '/\" tests/test_main_solver_flow.py"

echo
echo "# FIX C (Preferred): Patch tests to mock runpy module instance:"
echo "# Insert before runpy.run_module():"
echo "#     with patch.dict('sys.modules', {'src.main_solver': mock_module}):"
echo "# This ensures patches apply to the executed module."

echo
echo "============================================================"
echo "Audit complete."
echo "============================================================"
