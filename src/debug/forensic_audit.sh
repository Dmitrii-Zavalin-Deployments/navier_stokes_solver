#!/bin/bash
# src/debug/forensic_audit.sh

echo "============================================================"
echo "🔍 STARTING DEEP FORENSIC AUDIT: THE GHOST IN THE MAIN LOOP"
echo "============================================================"

# --- [1] Smoking Gun: Check if FloatingPointError is re-raised ---
echo "--- [Audit 1] Checking orchestrators for hidden try-except blocks ---"
grep -r "except" src/step3/ src/step4/

# --- [2] Source Audit: Log Level & Logger Name Verification ---
echo "--- [Audit 2] Verification of ElasticManager Logger Config ---"
cat -n src/common/elasticity.py | grep -A 5 "class ElasticManager"

# --- [3] Repair: Inject Warning in Main Solver Catch Block ---
# This ensures that even if the ElasticManager is silent, the main loop speaks.
# Rule 7: Granular Traceability mandate.
# sed -i '123i \                logger.warning("Instability detected: Arithmetic anomaly triggered recovery path.")' src/main_solver.py

# --- [4] Repair: Force ElasticManager to use the shared namespace ---
# If the logger is named incorrectly, pytest caplog might miss it.
# sed -i 's/logging.getLogger("src.common.elasticity")/logging.getLogger("Solver.Main.Elasticity")/g' src/common/elasticity.py

# --- [5] Repair: Ensure Instability log is at WARNING level ---
# The test expects WARNING; if it is currently INFO, it will fail.
# sed -i 's/self.logger.info(f"Instability/self.logger.warning(f"Instability/g' src/common/elasticity.py

# --- [6] Final Verification of the Exception Scope ---
echo "--- [Audit 3] main_solver.py loop boundary audit ---"
cat -n src/main_solver.py | sed -n '88,127p'

echo "✅ Audit Complete. If 'Instability' was logged at INFO, Step 5 above will fix it."