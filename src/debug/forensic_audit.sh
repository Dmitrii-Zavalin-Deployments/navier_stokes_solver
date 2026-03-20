echo "--- 1. PROOF OF SUCCESSFUL COMPLETION ---"
# If the solver reached the end, it should have a 'time' near 1.0. 
# We audit the source to see where it prints its progress.
grep -n "print(f\"DEBUG" src/main_solver.py || echo "DEBUG prints are disabled in source."

echo "--- 2. AUDIT THE LOOP TERMINATION ---"
# Let's see exactly what happens when state.time >= total_time
# This shows if the solver simply 'finishes' and returns to the test.
sed -n '120,135p' src/main_solver.py

echo "--- 3. ARCHIVER LOGIC CHECK ---"
# Since the test failed to find a ZIP, we check if the archiver actually does anything.
cat src/common/archive_service.py

echo "--- 4. DATA DIRECTORY RECONNAISSANCE ---"
# Check if Step 5 actually wrote the state files before the archiver was called.
python3 -c "import os; from pathlib import Path; from src.main_solver import BASE_DIR; \
p = Path(BASE_DIR) / 'data' / 'testing-input-output'; \
print(f'Path: {p}'); print(f'Files found: {os.listdir(p) if p.exists() else \"DIR NOT FOUND\"}')"

echo "--- 5. RECOVERY THRESHOLD VERIFICATION ---"
# Check the current dt floor. If 6.25e-2 is > floor, the exception is never raised.
grep "if elasticity.dt <" src/main_solver.py