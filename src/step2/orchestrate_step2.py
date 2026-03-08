# src/step2/orchestrate_step2.py

import numpy as np

from src.core.solver_state import SolverState

from .compiler import GET_CELL_ATTRIBUTES, cell_to_numpy_row
from .factory import build_cell, get_initialization_context


def orchestrate_step2(state: SolverState) -> SolverState:
    # 1. Dynamic Attribute Mapping
    # GET_CELL_ATTRIBUTES = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'p', 'mask', 'is_ghost']
    attributes = GET_CELL_ATTRIBUTES()
    num_attributes = len(attributes)
    
    total_cells = state.grid.nx * state.grid.ny * state.grid.nz
    
    # 2. Pre-allocate the LOCAL buffer
    local_cell_matrix = np.zeros((total_cells, num_attributes), dtype=np.float64)
    
    # 3. Initialization Context (Physical constants)
    ctx = get_initialization_context(state)

    # 4. The Main Processing Loop
    cursor = 0
    for i in range(state.grid.nx):
        for j in range(state.grid.ny):
            for k in range(state.grid.nz):
                # Factory creates DTO
                cell = build_cell(i, j, k, state, ctx)
                
                # Compiler converts DTO to row based on the same attribute list
                local_cell_matrix[cursor] = cell_to_numpy_row(cell)
                cursor += 1

    # 5. Final Commit
    state.cell_matrix = local_cell_matrix
    return state