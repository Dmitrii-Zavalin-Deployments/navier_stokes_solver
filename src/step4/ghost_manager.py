# src/step4/ghost_manager.py

import numpy as np
from src.solver_state import SolverState

# Global Debug Toggle
DEBUG = True

def initialize_ghost_fields(state: SolverState) -> None:
    """
    Step 4.1: Allocation & Synchronization. 
    Creates halo regions for BC enforcement.
    Rule 5 Compliance: No silent shape mismatches.
    """
    nx, ny, nz = state.grid.nx, state.grid.ny, state.grid.nz

    if DEBUG:
        print(f"DEBUG [Step 4 Ghost]: Initializing extended fields for {nx}x{ny}x{nz}")

    # 1. Allocate Extended Fields (nx+2 for P, nx+3 for staggered U)
    # Using 'F' order to maintain consistency with the operators
    state.fields.P_ext = np.zeros((nx + 2, ny + 2, nz + 2), order='F')
    state.fields.U_ext = np.zeros((nx + 3, ny + 2, nz + 2), order='F')
    state.fields.V_ext = np.zeros((nx + 2, ny + 3, nz + 2), order='F')
    state.fields.W_ext = np.zeros((nx + 2, ny + 2, nz + 3), order='F')

    if DEBUG:
        print(f"DEBUG [Step 4 Ghost]: Allocated P_ext shape: {state.fields.P_ext.shape}")

    # 2. Map Interior Data
    # We use explicit slicing to ensure the interior is centered within the halo
    try:
        state.fields.P_ext[1:-1, 1:-1, 1:-1] = state.fields.P
        state.fields.U_ext[1:-1, 1:-1, 1:-1] = state.fields.U
        state.fields.V_ext[1:-1, 1:-1, 1:-1] = state.fields.V
        state.fields.W_ext[1:-1, 1:-1, 1:-1] = state.fields.W
    except ValueError as e:
        if DEBUG:
            print(f"!!! CRITICAL: Shape mismatch during ghost mapping: {e} !!!")
        raise RuntimeError(f"Ghost Initialization Failed: Interior/Exterior field dimension mismatch.")

    if DEBUG:
        # Verify a sample interior value made it into the extended field
        sample_p = state.fields.P[0,0,0]
        ext_p = state.fields.P_ext[1,1,1]
        print(f"DEBUG [Step 4 Ghost]: Mapping check - Interior P[0,0,0]({sample_p:.2e}) == Ext P[1,1,1]({ext_p:.2e})")
        print(f"DEBUG [Step 4 Ghost]: Memory allocated for P_ext: {state.fields.P_ext.nbytes / 1024:.2f} KB")