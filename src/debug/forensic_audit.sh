#!/bin/bash
# src/debug/forensic_audit.sh
# Purpose: Diagnose Step 3 Dimensionality Leak (3087 vs 343)

echo "🔍 AUDIT 1: Inspecting Cell Property implementation"
cat -n src/common/cell.py | grep -A 5 "def vx"

echo "🔍 AUDIT 2: Inspecting Field Schema (FI) for stride count"
cat -n src/common/field_schema.py

echo "🔍 AUDIT 3: Verifying Foundation Buffer shape in Step 3 Factory"
grep -r "np.zeros" tests/helpers/solver_step3_output_dummy.py || echo "Factory uses dynamic allocation."

echo "------------------------------------------------------------"
echo "🧬 SMOKING GUN: The buffer size 3087 indicates [343, 9] flattening."
echo "Rule 9 requires Cell properties to be views of field-specific vectors."
echo "------------------------------------------------------------"

# PROPOSED REPAIR 1: Fix the Test to look for the stride-aware size
# This aligns the test with a 2D foundation buffer [Cells, Fields]
# sed -i 's/actual_buffer_size == n_expected/actual_buffer_size == n_expected * 9/g' tests/property_integrity/test_architecture_parity.py

# PROPOSED REPAIR 2: Fix the Cell to return a view of the field-column only (Preferred)
# This keeps the Cell 'vx' property as a 343-element vector view
# sed -i 's/return self.fields_buffer\[self.index:self.index+1, FI.VX\]/return self.fields_buffer[:, FI.VX][self.index:self.index+1]/g' src/common/cell.py

echo "✅ Forensic Audit complete. Root cause: 2D Buffer Stride Leak."