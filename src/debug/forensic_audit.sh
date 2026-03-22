#!/bin/bash
echo "============================================================"
echo "🎯 PHASE E: ITERATION DEPTH & TRAP-DOOR AUDIT"
echo "============================================================"

# --- [Audit 1] Propagation Check ---
echo "--- [Audit 1] Checking if time-step logic allows 'Instant Finish' ---"
cat -n src/main_solver.py | sed -n '120,135p'

# --- [Audit 2] PPE Divergence Audit ---
echo "--- [Audit 2] Checking for silent convergence with NaNs ---"
# If max_delta remains 0.0 despite 1e15 input, the BC is being ignored.
grep -A 5 "max_delta =" src/main_solver.py

# --- [Audit 3] Test Artifact Forensic ---
echo "--- [Audit 3] Checking if the 'successful' HDF5 is actually garbage ---"
python3 -c "import h5py, numpy as np; f=h5py.File('navier_stokes_output/snapshot_0001.h5','r'); print(f'Max U: {np.max(f[\"vx\"][:])}')"

# --- [Audit 4] Logger Capture Verification ---
echo "--- [Audit 4] Verification of Logger capture scope ---"
grep "caplog.at_level" tests/property_integrity/test_heavy_elasticity_lifecycle.py

# --- [5] AUTOMATED REPAIRS (The Smoking Gun Fixes) ---

# REPAIR A: Force the simulation to run longer. 
# This gives the instability time to propagate through the grid and trigger a crash.
# sed -i 's/"total_time": 0.2/"total_time": 5.0/' tests/property_integrity/test_heavy_elasticity_lifecycle.py

# REPAIR B: Force a manual trap for non-physical velocities.
# This ensures we don't rely on the hardware to throw a FloatingPointError.
# sed -i '/orchestrate_step5/i \                if np.max(np.abs(state.stencil_matrix[0].field.u)) > 1e12: raise ArithmeticError("Instability")' src/main_solver.py

# REPAIR C: Ensure the Test captures the right logger namespace.
# sed -i 's/logger=""/logger="Solver.Main"/' tests/property_integrity/test_heavy_elasticity_lifecycle.py

echo "============================================================"
echo "✅ Audit Complete. Apply REPAIR A and REPAIR B for a guaranteed signal."