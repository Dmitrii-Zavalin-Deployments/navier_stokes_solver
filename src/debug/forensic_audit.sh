echo "--- 1. EXPECTED VS ACTUAL FILENAME ---"
# Check what the test is trying to open/unzip
grep -r "zip" tests/property_integrity/test_heavy_elasticity_lifecycle.py | grep ".zip"

echo "--- 2. ARCHIVER OUTPUT VERIFICATION ---"
# Confirm the exact string the archiver uses for the final file
grep "final_destination =" src/common/archive_service.py

echo "--- 3. ZIP INTEGRITY AUDIT ---"
# Check if the zip actually contains the expected 'simulation_state.json'
# We use the path found in your previous audit
ACTUAL_ZIP="data/testing-input-output/navier_stokes_output.zip"
if [ -f "$ACTUAL_ZIP" ]; then
    unzip -l "$ACTUAL_ZIP"
else
    echo "❌ Zip file not found at expected path: $ACTUAL_ZIP"
fi

echo "--- 4. SIMULATION END-STATE ---"
# Verify the solver reached the line that sets the loop to False
grep -C 2 "state.ready_for_time_loop = False" src/main_solver.py