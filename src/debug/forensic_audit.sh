#!/bin/bash
echo "============================================================"
echo "🔍 SMOKING GUN: Data-Flow Disconnect Detected"
echo "============================================================"

# 1. Audit the BC Dispatcher: Why is 1e10 being ignored?
echo "--- Step 1: Source Audit of BC Applier (Rule 5/9 Check) ---"
cat -n src/step3/boundaries/applier.py | grep -A 20 "def apply_inflow"

# 2. Audit the Field Mapping: Is VX actually being written to?
echo "--- Step 2: Verification of FI.VX index usage ---"
grep -r "FI.VX" src/step3/

# 3. Audit the Orchestration: Is the state.boundary_conditions list populated?
echo "--- Step 3: Main Solver Orchestration (Data-Wiring) ---"
cat -n src/main_solver.py | grep -C 5 "boundary_conditions"

# 4. Check for hardcoded "Safe" limits that might be clipping the input
echo "--- Step 4: Search for Clipping/Clamping Logic ---"
grep -r "np.clip" src/step3/

# ============================================================
# 🛠️ AUTOMATED REPAIR INJECTIONS
# ============================================================

# Repair 1: Force the BC Applier to use the explicit value from the config (Rule 5)
# sed -i 's/target_field\[index\] = 0.1/target_field\[index\] = bc.values\["u"\]/' src/step3/boundaries/applier.py

# Repair 2: Ensure the logger name is exactly "Solver.Main" for caplog parity
# sed -i "s/getLogger(__name__)/getLogger('Solver.Main')/" src/main_solver.py