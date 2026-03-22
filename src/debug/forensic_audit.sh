#!/bin/bash
# Phase C Forensic Audit: Synchronizing Stencil-Matrix to Global Foundation

echo "--- 1. DIAGNOSTICS: ROOT CAUSE ANALYSIS ---"
# Check if Step 5 (Archivist) is actually called inside or after the loop
grep -n "orchestrate_step5" src/main_solver.py || echo "CRITICAL: Step 5 (Archivist) missing from loop."

echo "--- 2. SMOKING-GUN AUDIT: DATA FLUSH GAP ---"
# Verify if the stencil blocks ever write back to the contiguous state.fields.data
grep -r "data\[" src/step3/ || echo "WARNING: Step 3 might be working on local copies (Rule 8 violation)."

echo "--- 3. FIX: SED INJECTIONS ---"
# Rule 7 (Scientific Truth): The simulation is iterating, but the 'Snapshots' 
# are being taken from the global state which isn't being updated by the blocks.

# 1. Inject the Global Archivist call at the end of the time-step loop
# This ensures that for every iteration, Step 5 captures the state of the Foundation.
sed -i "/state.time +=/a \                # Rule 6 & 9: Global State Snapshot\n                from src.step5.orchestrate_step5 import orchestrate_step5\n                orchestrate_step5(state)" src/main_solver.py

# 2. Fix a potential 'Elasticity' vs 'State' mismatch (Rule 4 SSoT)
# Ensure the loop uses the state's time increment to maintain sync with the archivist
sed -i "s/state.time += elasticity.dt/state.time += state.dt/g" src/main_solver.py

# 3. Add a debug print to track if the loop is actually executing more than 0 iterations
sed -i "/while state.ready_for_time_loop:/a \            print(f'DEBUG: Iteration {state.iteration} | Time {state.time}')" src/main_solver.py

echo "--- 4. POST-REPAIR VERIFICATION ---"
# Validate syntax and check for redundant imports
python3 -m py_compile src/main_solver.py
echo "Forensic Audit Complete: Global Archivist coupled to Time-Advance loop."