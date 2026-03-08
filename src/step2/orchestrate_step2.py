# src/step2/orchestrate_step2.py

import numpy as np

from src.core.solver_state import SolverState

from .compiler import GET_CELL_ATTRIBUTES, cell_to_numpy_row
from .factory import build_core_cell, build_ghost_cell, get_initialization_context


def orchestrate_step2(state: SolverState) -> SolverState:
    # 1. Dynamic Attribute Mapping
    attributes = GET_CELL_ATTRIBUTES()
    num_attributes = len(attributes)
    
    # Grid dimensions (Core)
    nx, ny, nz = state.grid.nx, state.grid.ny, state.grid.nz
    
    # Total cells including the ghost layer (-1, nx; -1, ny; -1, nz)
    total_cells = (nx + 2) * (ny + 2) * (nz + 2)
    
    # 2. Pre-allocate local buffer
    local_cell_matrix = np.zeros((total_cells, num_attributes), dtype=np.float64)
    
    # 3. Hoist constants
    ctx = get_initialization_context(state)

    # 4. Main Processing Loop
    cursor = 0
    # Loop range: -1 to nx (inclusive of indices)
    for i in range(-1, nx + 1):
        for j in range(-1, ny + 1):
            for k in range(-1, nz + 1):
                
                # Check if we are inside the core domain
                is_core = (0 <= i < nx) and (0 <= j < ny) and (0 <= k < nz)
                
                if is_core:
                    # Cell needs the full state to read the masks
                    cell = build_core_cell(i, j, k, state, ctx)
                else:
                    # Ghost cell needs only coordinates and context
                    cell = build_ghost_cell(i, j, k, ctx)
                
                # Compiler writes to row
                local_cell_matrix[cursor] = cell_to_numpy_row(cell)
                cursor += 1

    # 5. Final Commit
    state.cell_matrix = local_cell_matrix
    
    # Set ready_for_time_loop to True now that topography is defined
    state.ready_for_time_loop = True
    
    return state