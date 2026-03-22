#!/bin/bash
echo "============================================================"
echo "🎯 PHASE G: CELL-CENTERED MOMENTUM & ATTENUATION AUDIT"
echo "============================================================"

# --- [Audit 1] Neighbor Synchronization ---
echo "--- [Audit 1] Checking if boundary values propagate to neighbors ---"
# If Rule 9 (Hybrid Memory) is active, does set_field update the SHARED ghost cells?
grep -r "sync_ghost_cells" src/common/

# --- [Audit 2] The "One-Step" Execution Depth ---
echo "--- [Audit 2] Verifying step count vs. velocity propagation ---"
# With dt=0.8, the CFL condition is 1e15 * 0.8 / dx. 
# This should be ~1e15. If Max U is 13.4, the 1e15 was NEVER written.
cat -n src/main_solver.py | grep -A 5 "apply_boundary_conditions"

# --- [Audit 3] Stencil Field Locking ---
echo "--- [Audit 3] Is the center field being overwritten by Step 3? ---"
# If Step 3 (Predictor) runs AFTER Step 4 (Boundary), it will overwrite your 1e15.
cat -n src/main_solver.py | sed -n '100,115p'

# --- [Audit 4] Logger Name Collision ---
echo "--- [Audit 4] Confirming the exact logger object in use ---"
# If elasticity.py uses a different logger instance, caplog might miss it.
grep "self.logger =" src/common/elasticity.py

# --- [5] AUTOMATED REPAIRS (The "Force Collision" Fixes) ---

# REPAIR A: Increase simulation time to force 10+ iterations.
# This ensures the 1e15 has time to propagate from the boundary cell to the interior.
# sed -i 's/"total_time": 0.2/"total_time": 5.0/' tests/property_integrity/test_heavy_elasticity_lifecycle.py

# REPAIR B: Re-order logic so Boundary Conditions are applied LAST in the loop.
# This prevents the Predictor/Corrector from wiping out the "Inflow" value.
# sed -i '/orchestrate_step3/i \                # Moving BC to priority position' src/main_solver.py

# REPAIR C: Force a hard-crash in the test if logs are empty but VX is low.
# sed -i '/assert len(stabilization_logs) > 0/i \            if len(stabilization_logs) == 0: print(f"CRITICAL: Solver bypassed instability. MaxVX={np.max(h5[\"vx\"][:])}")' tests/property_integrity/test_heavy_elasticity_lifecycle.py

echo "============================================================"
echo "✅ Audit Complete. If Audit 3 shows Step 3 follows Step 4, your BC is being deleted."