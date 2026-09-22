#!/usr/bin/env bash
# ==============================================================================
# Forensic Audit & Automated Repair Script for Navier-Stokes Solver
# Target Issue: JSON Schema Validation Failure ('x_min' missing under 'grid')
# ==============================================================================

set -euo pipefail

echo "================================================================================"
echo "[PHASE 1] Diagnostic Code & Schema Inspections (grep / cat)"
echo "================================================================================"

echo "-> Inspecting schema definition for grid requirements:"
if [ -f "schema/solver_input_schema.json" ]; then
    grep -A 15 -B 2 "grid" schema/solver_input_schema.json || true
else
    echo "Warning: schema/solver_input_schema.json not found in current directory."
fi

echo -n "-> Checking python test file existence: "
if [ -f "tests/test_free_slip_symmetry.py" ]; then
    echo "FOUND"
else
    echo "MISSING"
    exit 1
fi

echo "================================================================================"
echo "[PHASE 2] Smoking-Gun Source Audit (cat -n)"
echo "================================================================================"
echo "-> Printing line-numbered source of tests/test_free_slip_symmetry.py around grid definition:"
cat -n tests/test_free_slip_symmetry.py | grep -C 10 "grid" || cat -n tests/test_free_slip_symmetry.py

echo "================================================================================"
echo "[PHASE 3] Automated Repair Injections (sed)"
echo "================================================================================"
echo "-> The following sed commands can be uncommented to patch the missing required grid bounds:"