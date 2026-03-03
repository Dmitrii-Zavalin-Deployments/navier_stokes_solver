# src/step4/orchestrate_step4.py

from src.solver_state import SolverState
from .ghost_manager import initialize_ghost_fields
from .boundary_filler import fill_ghost_boundaries
from .audit_diagnostics import run_preflight_audit

# Global Debug Toggle
DEBUG = True

def orchestrate_step4(state: SolverState) -> SolverState:
    """
    Step 4 Orchestrator: Ghost Padding & Audit.
    Rule 5 Compliance: Strict progression. No silent failures in the final audit.
    """
    if DEBUG:
        print(f"\nDEBUG [Step 4 Orchestrator]: Finalizing Step 4 (Time: {state.time})")

    # 1. Expand fields to include halos (Allocation)
    initialize_ghost_fields(state)
    if DEBUG:
        print("DEBUG [Step 4 Orchestrator]: Ghost fields initialized and mapped.")

    # 2. Synchronize boundaries across ghosts (Enforcement)
    fill_ghost_boundaries(state)
    if DEBUG:
        print(f"DEBUG [Step 4 Orchestrator]: Boundaries synchronized (Passed: {state.diagnostics.bc_verification_passed})")
    
    # 3. Perform diagnostic audit (Stability Check)
    run_preflight_audit(state)
    if DEBUG:
        print(f"DEBUG [Step 4 Orchestrator]: Pre-flight audit complete.")
        print(f"DEBUG [Step 4 Orchestrator]: Calculated CFL limit: {state.diagnostics.initial_cfl_dt:.6e}")

    # 4. Final Lock: State is now ready for iterative cycling
    # We only flip this bit if we reached this point without a RuntimeError
    state.ready_for_time_loop = True
    
    if DEBUG:
        print(f"DEBUG [Step 4 Orchestrator]: State marked 'ready_for_time_loop'. HANDSHAKE SUCCESSFUL.")
    
    return state