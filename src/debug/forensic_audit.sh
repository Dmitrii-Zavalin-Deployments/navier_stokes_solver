#!/bin/bash

# --- COLOR CODES FOR GITHUB ACTIONS LOGS ---
BLUE='\033[0;34m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}        NAV-STOKES ELASTICITY MANAGER: POST-RUN CI AUDIT              ${NC}"
echo -e "${BLUE}======================================================================${NC}"

echo -e "\n${GREEN}--- 1. ARCHITECTURAL INTEGRITY (Memory & API) ---${NC}"
# Check __slots__ (Mandatory Rule 0)
if grep -q "__slots__" src/common/elasticity.py; then
    echo -e "✅ __slots__ Defined: Memory Protection Active."
    grep -n "__slots__" src/common/elasticity.py
else
    echo -e "${RED}❌ ERROR: __slots__ Missing! Rule 0 Violation.${NC}"
fi

# Check Read-Only Properties
echo -e "\nChecking Stability Knob Properties:"
grep -nE "@property|def (dt|omega|max_iter)" src/common/elasticity.py

echo -e "\n${GREEN}--- 2. LOGIC AUDIT: PANIC BRANCH (Line 40-52) ---${NC}"
# Use sed to grab the exact panic block
cat -n src/common/elasticity.py | sed -n '39,52p' | while read line; do
    echo "  $line"
done

echo -e "\n${GREEN}--- 3. LOGIC AUDIT: RECOVERY GATE (Line 60-71) ---${NC}"
# Check if the max_iter reset is correctly "Flatted" (outside the else)
cat -n src/common/elasticity.py | sed -n '59,71p'

echo -e "\n${GREEN}--- 4. SSoT PERFORMANCE CHECK (The Smoking Gun) ---${NC}"
# If this returns 2 lines, and one is inside the recovery gate, it's correct.
RESET_COUNT=$(grep -c "self._max_iter = self.config.ppe_max_iter" src/common/elasticity.py)
if [ "$RESET_COUNT" -ge 2 ]; then
    echo -e "✅ Found $RESET_COUNT reset points. (Init + Recovery Gate)"
else
    echo -e "${RED}⚠️ WARNING: Only $RESET_COUNT reset point found. Performance may be trapped in Panic mode.${NC}"
fi

echo -e "\n${GREEN}--- 5. COVERAGE & UNIT TEST EXECUTION ---${NC}"
if [ -f "tests/common/test_elasticity_manager.py" ]; then
    # Run tests with coverage and verbose output
    pytest tests/common/test_elasticity_manager.py -vv --cov=src.common.elasticity --cov-report=term-missing
else
    echo -e "${RED}❌ ERROR: Test file missing at tests/common/test_elasticity_manager.py${NC}"
    exit 1
fi

echo -e "\n${GREEN}--- 6. FULL SOURCE SNAPSHOT (For Traceability) ---${NC}"
cat -n src/common/elasticity.py

echo -e "\n${BLUE}======================================================================${NC}"
echo -e "${BLUE}              AUDIT COMPLETE: VERIFY LOGS ABOVE                       ${NC}"
echo -e "${BLUE}======================================================================${NC}"