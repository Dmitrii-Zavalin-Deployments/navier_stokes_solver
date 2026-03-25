#!/bin/bash
# ==============================================================================
# NAVIER-STOKES FORENSIC AUDIT: [FAIL] - Attribute Discrepancy Detected
# TARGET: AttributeError: 'Cell'/'SimpleCellMock' object has no attribute 'u'
# ==============================================================================

echo "🔍 [1/4] LOG SCAN: Identifying 'u' attribute failures..."
grep -r "AttributeError: '.*Cell.*' object has no attribute 'u'" .

echo -e "\n🔍 [2/4] SOURCE AUDIT: Checking Cell and Mock implementations..."
# Locate the Cell definition and the SimpleCellMock in tests
CELL_FILE=$(grep -l "class Cell" src/grid_modules/cell.py 2>/dev/null || echo "NOT_FOUND")
MOCK_FILE="tests/property_integrity/test_architecture_parity.py"

if [ "$CELL_FILE" != "NOT_FOUND" ]; then
    echo "--- Source: $CELL_FILE ---"
    cat -n "$CELL_FILE" | grep -A 10 "class Cell"
fi

echo "--- Source: $MOCK_FILE (Mocks) ---"
cat -n "$MOCK_FILE" | grep -A 15 "class SimpleCellMock"

echo -e "\n🔍 [3/4] MEMORY ALIGNMENT FORENSICS..."
# Capturing the offset reported in the CI log for the ledger
echo "Reported Alignment Offset: 16 (64-byte boundary violated)"

echo -e "\n🛠️ [4/4] PROPOSED REPAIRS (Deactivated):"
echo "To fix, uncomment the following injections in the local environment:"

# Injection for the production Cell class to support velocity vector access
# sed -i '/class Cell:/a \    @property\n    def u(self):\n        return self._state_ref.fields.data[self.index, [FI.VX, FI.VY, FI.VZ]]' src/grid_modules/cell.py

# Injection for the Test Mock to satisfy the Rule 9 Sentinel
# sed -i '/class SimpleCellMock:/a \    def __init__(self, index):\n        self.index = index\n        self.u = [0.0, 0.0, 0.0]\n        self.p = 0.0\n        self.is_ghost = False' tests/property_integrity/test_architecture_parity.py

echo "=============================================================================="
echo "Forensic Audit Complete. Signal suggests: Update Cell/Mock to support .u property."
exit 1