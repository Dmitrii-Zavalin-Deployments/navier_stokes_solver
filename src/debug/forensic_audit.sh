#!/bin/bash
# Description: Deep Forensic Audit for Navier-Stokes Numerical Stability
# Targets: Rule 7 (Scientific Rigor) and Rule 5 (Deterministic Initialization)

echo "--- [DEEP FORENSIC AUDIT @ $(date +%T)] ---"

# 1. Check if the test files even exist and what's in them
echo "Checking Test Artifacts:"
if [ -f "config.json" ]; then
    echo "✅ config.json found: $(cat config.json)"
else
    echo "❌ config.json MISSING"
fi

# 2. Inspect the Solver Logic for 'ArithmeticError' Handling
echo "--- [CODE INSPECTION: INSTABILITY CATCHING] ---"
grep -nE "try:|except ArithmeticError:|raise RuntimeError" src/main_solver.py src/physics/elasticity.py 2>/dev/null || echo "No explicit stability catch found."

# 3. Trace the Log Output for Retries
echo "--- [EXECUTION TRACE: STABILIZATION ATTEMPTS] ---"
# We look for the 'Instability' keyword to see if the loop actually ran
if [ -f "pytest_log.txt" ]; then
    grep -i "Instability" pytest_log.txt || echo "No stabilization logs found in output."
else
    echo "Hint: Run with '$TEST_COMMAND > pytest_log.txt 2>&1' to capture logs for this script."
fi

# 4. Check for 'Ghost' Properties (Rule 8 Violation)
echo "--- [SLOT INTEGRITY CHECK] ---"
grep -r "self.ppe_max_retries =" src/common/
grep -r "def ppe_max_retries" src/common/

# 5. Numerical 'Smoking Gun' - Check for NaN/Inf in any saved state
echo "--- [NUMERICAL SANITY] ---"
if [ -d "output" ]; then
    grep -rE "NaN|Infinity" output/ || echo "No NaN/Inf detected in output files."
fi

echo "--- [AUDIT COMPLETE] ---"