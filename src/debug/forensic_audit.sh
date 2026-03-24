#!/bin/bash
# src/debug/forensic_audit.sh

LOG_FILE="caplog.txt"

echo "----------------------------------------------------------------"
echo "🔍 FORENSIC AUDIT: RHIE-CHOW DECOUPLING ANALYSIS"
echo "----------------------------------------------------------------"

# 1. VELOCITY EXPLOSION PROFILE
echo "STEP 1: Velocity Divergence Profile (Final Value vs Trial Number)"
grep "AUDIT \[Limit\]: Velocity" $LOG_FILE | awk -F'=' '{print $NF}' | nl -v 0

# 2. PRESSURE GRADIENT STAGNATION CHECK
# If Velocity is climbing but Pressure is static, Rhie-Chow is dead.
echo -e "\nSTEP 2: Pressure Gradient vs. Velocity Correlation"
grep -E "AUDIT \[Limit\]: (Velocity|Pressure)" $LOG_FILE | tail -n 10

# 3. PPE SOLVER AUDIT
echo -e "\nSTEP 3: Solver State - Rhie-Chow Implementation (src/step3/ppe_solver.py)"
# Capturing the Laplacian calculation we identified as the 'Zero-Trap'
cat -n src/step3/ppe_solver.py | sed -n '32,45p'

# 4. ELASTICITY LADDER AUDIT
echo -e "\nSTEP 4: Elasticity Step-Down Logic (src/common/elasticity.py)"
# Checking how it generates the _dt_range ladder
cat -n src/common/elasticity.py | grep -A 5 "_dt_range ="

# 5. REPAIR PLAN: PROPOSED INJECTIONS
echo -e "\nSTEP 5: Proposed Stability Injections..."

# Repair 1: Inject Pressure Noise to break the Zero-Laplacian Symmetry
echo "Proposed: sed -i '57s/cell.p = init.pressure/cell.p = init.pressure + 1e-6 * (i + j + k)/' src/step2/factory.py"

# Repair 2: Upgrade Linear Ladder to Geometric Decay (More aggressive recovery)
echo "Proposed: sed -i 's/i \* (self.dt_floor - self._dt) \/ self._runs/self._dt * (0.5 ** i)/' src/common/elasticity.py"

echo "----------------------------------------------------------------"
echo "✅ Forensic Data Collected. Compare Step 1 and Step 2 to verify decoupling."