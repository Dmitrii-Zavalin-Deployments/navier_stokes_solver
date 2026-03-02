# src/step4/ghost_manager.py

import numpy as np
from src.solver_state import SolverState

def initialize_ghost_fields(state: SolverState) -> None:
    """
    Step 4.1: Allocation. Creates the halo regions for BC enforcement.
    """
    nx, ny, nz = state.grid.nx, state.grid.ny, state.grid.nz

    # Allocate Extended Fields
    state.fields.P_ext = np.zeros((nx + 2, ny + 2, nz + 2))
    state.fields.U_ext = np.zeros((nx + 3, ny + 2, nz + 2))
    state.fields.V_ext = np.zeros((nx + 2, ny + 3, nz + 2))
    state.fields.W_ext = np.zeros((nx + 2, ny + 2, nz + 3))

    # Copy Interior data with shape validation
    state.fields.P_ext[1:-1, 1:-1, 1:-1] = state.fields.P
    state.fields.U_ext[1:-1, 1:-1, 1:-1] = state.fields.U
    state.fields.V_ext[1:-1, 1:-1, 1:-1] = state.fields.V
    state.fields.W_ext[1:-1, 1:-1, 1:-1] = state.fields.W
