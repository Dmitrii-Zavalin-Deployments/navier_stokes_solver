# src/debug/forensic_audit.sh

echo "🔍 DIAGNOSTIC: Investigating Type-Safety Collision in SolverState..."

# 1. Smoking Gun: Check the setter logic in SolverState
echo "--- [SOURCE AUDIT: solver_state.py Setters] ---"
cat -n src/common/solver_state.py | sed -n '438,445p'

# 2. Identify the illegal Mock usage in the test file
echo "--- [SOURCE AUDIT: test_solver_state.py Fixture] ---"
cat -n tests/common/test_solver_state.py | sed -n '15,30p'

# 3. PROPOSED REPAIR: Replace MockManager with actual GridManager/PhysicalConstraintsManager
# This ensures isinstance() checks pass while keeping the tests lightweight.

# REPAIR STEP A: Inject real GridManager into the test fixture
sed -i 's/state.grid = MockManager(nx=2, ny=2, nz=2)/from src.common.solver_state import GridManager; gm = GridManager(); gm.nx=2; gm.ny=2; gm.nz=2; gm.x_min=0.0; gm.x_max=1.0; gm.y_min=0.0; gm.y_max=1.0; gm.z_min=0.0; gm.z_max=1.0; state.grid = gm/' tests/common/test_solver_state.py

# REPAIR STEP B: Fix the Velocity Violation test to use real PhysicalConstraintsManager
sed -i 's/class MockConstraints:/from src.common.solver_state import PhysicalConstraintsManager; pc = PhysicalConstraintsManager();/' tests/common/test_solver_state.py

echo "============================================================"
echo "✅ DIAGNOSIS COMPLETE: fixture 'populated_state' requires real Manager instances."
echo "The Constitution (SolverState) forbids non-validated citizens (Mocks)."