#!/usr/bin/env bash
# ==============================================================================
# Script Name: forensic_audit.sh
# Description: Post-test forensic diagnostic and automated repair script for 
#              Navier-Stokes solver schema validation and grid dimension errors.
# ==============================================================================

set -Eeuo pipefail

echo "================================================================================"
echo "[FORENSIC AUDIT START] Analyzing grid/nz schema validation failure"
echo "================================================================================"

echo "[1/4] Inspecting JSON schema definition for grid/nz minimum constraints..."
SCHEMA_FILE="./schema/solver_input_schema.json"
if [ -f "$SCHEMA_FILE" ]; then
    echo "Found schema file: $SCHEMA_FILE"
    grep -C 4 "nz" "$SCHEMA_FILE"
else
    echo "WARNING: Explicit schema path not found, searching via find..."
    SCHEMA_FILE=$(find . -path "*/schema/*input*.json" -o -name "solver_input_schema.json" | head -n 1)
    if [ -n "$SCHEMA_FILE" ]; then
        echo "Found schema file: $SCHEMA_FILE"
        grep -C 4 "nz" "$SCHEMA_FILE"
    else
        echo "ERROR: solver_input_schema.json could not be located."
    fi
fi

echo "--------------------------------------------------------------------------------"
echo "[2/4] Locating grid definition in test files..."
grep -rn "nx, ny, nz =" tests/ || grep -rn "nz =" tests/

echo "--------------------------------------------------------------------------------"
echo "[3/4] Performing smoking-gun source audit via cat -n on test file..."
TEST_FILE="tests/test_plane_poiseuille_flow.py"
if [ -f "$TEST_FILE" ]; then
    echo "Inspecting lines containing grid initialization in $TEST_FILE:"
    cat -n "\(TEST_FILE" | grep -C 5 "nz" || cat -n "\)TEST_FILE" | sed -n '35,65p'
else
    echo "WARNING: $TEST_FILE not found."
fi

echo "--------------------------------------------------------------------------------"
echo "[4/4] Automated Repair Instructions:"
echo "The schema enforces a minimum of 4 for grid/nz (to prevent stencil cancellation)."
echo "To automatically fix the test configuration, uncomment and run the sed command below:"
echo ""
# sed -i 's/nx, ny, nz = 16, 16, 3/nx, ny, nz = 16, 16, 4/g' tests/test_plane_poiseuille_flow.py
# sed -i 's/"nz": 3/"nz": 4/g' tests/test_plane_poiseuille_flow.py

echo "================================================================================"
echo "[FORENSIC AUDIT COMPLETE]"
echo "================================================================================"