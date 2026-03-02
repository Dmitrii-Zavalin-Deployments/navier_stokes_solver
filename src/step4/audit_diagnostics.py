# src/step4/audit_diagnostics.py

from src.solver_state import SolverState

def run_preflight_audit(state: SolverState) -> None:
    """
    Step 4.3: Pre-flight Audit. Calculates memory and stability limits.
    """
    nx, ny, nz = state.grid.nx, state.grid.ny, state.grid.nz
    
    # Calculate approximate memory footprint (float64 = 8 bytes)
    total_voxels = (nx+2)*(ny+2)*(nz+2) 
    state.diagnostics.memory_footprint_gb = (total_voxels * 8 * 4) / 1e9
    
    # CFL Limit based on current max velocity. 
    # Use dx facade or calculate if missing.
    dx = getattr(state.grid, 'dx', 1.0/nx)
    max_u = getattr(state.health, 'max_u', 0.0)
    
    if max_u > 0:
        state.diagnostics.initial_cfl_dt = 0.5 * dx / max_u
    else:
        # Fallback to ppe_tolerance via simulation_parameters dict
        params = state.config.simulation_parameters
        state.diagnostics.initial_cfl_dt = params.get("ppe_tolerance", 1e-6)
