# src/step2/factory.py

from src.common.cell import Cell
from src.common.solver_state import SolverState

# Rule 7: Granular Traceability
DEBUG = True

def build_core_cell(i: int, j: int, k: int, state: SolverState) -> Cell:
    """
    Creates a View-based Cell (Logic Wiring).
    Uses local caching of SSoT pointers for high-performance access.
    """
    # 1. Local caching of SSoT containers (Rule 4)
    grid = state.grid
    fields = state.fields
    init = state.initial_conditions
    mask_grid = state.mask.mask

    # 2. Index calculation and topology (Rule 9)
    # The indices (i, j, k) are expected to be 1-based relative to the buffer
    # to account for the ghost halo at index 0.
    nx_buf, ny_buf = grid.nx + 2, grid.ny + 2
    index = i + nx_buf * (j + ny_buf * k)
    
    # 3. Instantiate the Cell (The Wiring)
    cell = Cell(index=index, fields_buffer=fields.data, is_ghost=False)
    
    if DEBUG:
        print(f"DEBUG [Step 2.1]: Core Cell Created -> Index: {index} at ({i}, {j}, {k})")
    
    # 4. Initialize physical fields and topological mask (Rule 9)
    # The Cell writes directly to the NumPy-managed Foundation buffer.
    cell.vx, cell.vy, cell.vz = init.velocity
    cell.p = init.pressure
    
    # Temporary Diagnostic
    print(f"DEBUG: Mapping loop indices ({i}, {j}, {k}) to mask shape {mask_grid.shape}")

    # Rule 9: Map internal mask (0-indexed) to buffer coordinate (1-indexed)
    # We subtract 1 from i, j, k to access the physical mask correctly.
    cell.mask = int(mask_grid[i-1, j-1, k-1])
    
    return cell

def build_ghost_cell(i: int, j: int, k: int, state: SolverState) -> Cell:
    """
    Creates a View-based virtual cell on the perimeter with valid buffer indexing.
    Provides unique memory slots for boundary values.
    """
    grid = state.grid
    nx_buf, ny_buf = grid.nx + 2, grid.ny + 2
    
    # RULE 9: Calculate actual index to provide a dedicated memory slot in the halo.
    # This avoids the "Sentinel -1" overlap which would corrupt the last physical cell.
    index = i + nx_buf * (j + ny_buf * k)
    
    cell = Cell(index=index, fields_buffer=state.fields.data, is_ghost=True)
    
    if DEBUG:
        print(f"DEBUG [Step 2.1]: Ghost Cell Created -> ({i}, {j}, {k}) at Index: {index}")
    
    # RULE 5: Explicitly zero all fields to ensure a clean state for the Poisson solver.
    cell.vx, cell.vy, cell.vz = 0.0, 0.0, 0.0
    cell.p = 0.0
    
    # Ghost cells are topological insulators (mask=0)
    cell.mask = 0
    
    return cell