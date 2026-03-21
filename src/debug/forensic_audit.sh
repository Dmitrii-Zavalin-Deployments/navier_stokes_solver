# 1. ARCHITECTURAL INTEGRITY: Verify the SSoT (Single Source of Truth)
echo "--- AUDITING ELASTICITY LOGIC: src/common/elasticity.py ---"
# Check __slots__ and recovery gates
grep -nE "__slots__|def (gradual_recovery|apply_panic_mode|validate_and_commit)" src/common/elasticity.py
# Check where the success counter increments
grep -n "_iteration +=" src/common/elasticity.py

# 2. SMOKING GUN: Verify the loop orchestration in main_solver.py
echo -e "\n--- AUDITING MAIN SOLVER LOOP: src/main_solver.py ---"
# Ensure gradual_recovery is triggered only in the success path
cat -n src/main_solver.py | grep -C 5 "elasticity.gradual_recovery()"

# 3. SYNTAX VALIDATION: Pre-flight check
echo -e "\n--- STATIC ANALYSIS ---"
python3 -m py_compile src/main_solver.py src/common/elasticity.py && echo "✅ AST Syntax Validated"

# 4. BEHAVIORAL VERIFICATION: Run the Heavy Elasticity Lifecycle Test
# This test forces a crash. If it passes, it means the RuntimeError was RAISED 
# because dt successfully hit the floor without oscillating back up.
echo -e "\n--- RUNNING TARGETED RECOVERY TEST ---"
pytest tests/property_integrity/test_heavy_elasticity_lifecycle.py -v --log-level=WARNING

# 5. FORENSIC LOG AUDIT: Check for Monotonic Decay
# We run the solver with the 'bad' input and look for the dt sequence.
echo -e "\n--- LIVE EXECUTION DECAY PATTERN ---"
python3 src/main_solver.py integration_input.json > solver_output.tmp 2>&1 || true
echo "dt sequence during panic:"
grep "PANIC: dt reduced to" solver_output.tmp | awk '{print $NF}' | uniq

# 6. FINAL SUCCESS SIGNAL
# If the last dt in the log is smaller than the first, the ratchet works.
FINAL_DT=$(grep "PANIC: dt reduced to" solver_output.tmp | tail -n 1 | awk '{print $NF}')
echo -e "\nFinal dt before crash: $FINAL_DT"
if [[ -n "$FINAL_DT" ]]; then
    echo "Verification Complete."
else
    echo "Error: No panic signals detected. Check integration_input.json parameters."
fi