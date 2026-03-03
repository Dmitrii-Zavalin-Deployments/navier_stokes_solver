# src/step4/audit_diagnostics.py

from src.solver_state import SolverState

# Global Debug Toggle
DEBUG = True

def run_preflight_audit(state: SolverState) -> None:
    """
    Step 4.3: Pre-flight Audit. Calculates memory and stability limits.
    Rule 5 Compliance: No defaults. Uses actual grid and health data.
    """
    nx, ny, nz = state.grid.nx, state.grid.ny, state.grid.nz
    
    if DEBUG:
        print(f"DEBUG [Step 4 Audit]: Auditing {nx}x{ny}x{nz} grid.")

    # 1. Memory Calculation
    # float64 = 8 bytes. We track P, U, V, W + temporary star fields (~8 fields total)
    total_voxels_with_ghosts = (nx + 2) * (ny + 2) * (nz + 2) 
    state.diagnostics.memory_footprint_gb = (total_voxels_with_ghosts * 8 * 8) / 1e9
    
    # 2. Stability Analysis (CFL)
    # No getattr fallbacks. These must exist in the state/config.
    dx = state.grid.dx
    max_u = state.health.max_u
    
    if DEBUG:
        print(f"DEBUG [Step 4 Audit]: dx={dx:.4e}, max_u={max_u:.4e}")

    if max_u > 0:
        # Standard CFL: dt < dx / u
        state.diagnostics.initial_cfl_dt = 0.5 * dx / max_u
    else:
        # If fluid is at rest, CFL is not the limiting factor. 
        # We use the configured time_step as the diagnostic target.
        state.diagnostics.initial_cfl_dt = state.dt

    # 3. Diffusion Limit (Von Neumann Stability)
    # dt < 0.5 * dx^2 / nu
    nu = state.viscosity / state.density
    diffusion_dt = 0.5 * (dx**2) / nu
    
    if DEBUG:
        print(f"DEBUG [Step 4 Audit]: Memory Footprint: {state.diagnostics.memory_footprint_gb:.6f} GB")
        print(f"DEBUG [Step 4 Audit]: CFL dt Limit: {state.diagnostics.initial_cfl_dt:.6e}")
        print(f"DEBUG [Step 4 Audit]: Diffusion dt Limit: {diffusion_dt:.6e}")

    # 4. Safety Check
    if state.dt > state.diagnostics.initial_cfl_dt:
        if DEBUG:
            print(f"!!! WARNING: Configured dt ({state.dt}) exceeds CFL limit !!!")