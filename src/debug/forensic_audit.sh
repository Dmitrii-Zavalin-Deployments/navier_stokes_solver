#!/bin/bash

echo "--- 1. ARCHITECTURAL INTEGRITY: src/common/elasticity.py ---"
# Check if __slots__ are present to prevent memory leaks/accidental attributes
grep -n "__slots__" src/common/elasticity.py

# Ensure all 3 stability knobs are properties (Read-Only to the outside world)
echo -e "\nChecking Stability Knobs (Properties):"
grep -nE "@property|def (dt|omega|max_iter)" src/common/elasticity.py

echo -e "\n--- 2. LOGIC AUDIT: PANIC & RECOVERY GATES ---"
# Verify the Panic Trigger (Must set is_in_panic, reset iteration, and reduce dt)
echo "Auditing Panic Branch (Should be around line 40):"
cat -n src/common/elasticity.py | grep -A 10 "if not is_sane:"

# Verify the Success Path (Must increment iteration)
echo -e "\nAuditing Success Path (Should be around line 55):"
cat -n src/common/elasticity.py | grep -B 2 -A 2 "self._iteration += 1"

# Verify the Recovery Gate (The Ratchet check)
echo -e "\nAuditing Recovery Logic (Should be around line 61):"
cat -n src/common/elasticity.py | grep -A 12 "if self.is_in_panic and self._iteration >= 10:"

echo -e "\n--- 3. SMOKING GUN: MAX_ITER & OMEGA RESET ---"
# This specifically catches the error where max_iter doesn't reset until full dt recovery
echo "Checking SSoT for ppe_max_iter reset location:"
grep -n "self._max_iter = self.config.ppe_max_iter" src/common/elasticity.py

echo -e "\n--- 4. SYNTAX & UNIT TEST VALIDATION ---"
python3 -m py_compile src/common/elasticity.py && echo "✅ AST Syntax Valid"

# Run the unit tests we just created to confirm the fixes
if [ -f "tests/common/test_elasticity_manager.py" ]; then
    pytest tests/common/test_elasticity_manager.py -vv
else
    echo "⚠️ Unit test file not found at tests/common/test_elasticity_manager.py"
fi

echo -e "\n--- 5. FULL FILE DUMP FOR MANUAL VERIFICATION ---"
cat -n src/common/elasticity.py