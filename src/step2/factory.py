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
    # Retrieve the mask from the SSoT (the MaskManager)
    mask_grid = state.mask.mask

    # 2. Index calculation and topology (Rule 9)
    # Use buffer dimensions (grid.nx + 2) to ensure the flat index 
    # accounts for the ghost cell halo.
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
    # Explicitly set the mask into the foundation via the Cell view
    cell.mask = int(mask_grid[i, j, k])
    
    return cell

def build_ghost_cell(i: int, j: int, k: int, state: SolverState) -> Cell:
    """
    Creates a View-based virtual cell on the perimeter with sentinel indexing.
    """
    # Sentinel index -1 for ghost regions (Rule 9)
    cell = Cell(index=-1, fields_buffer=state.fields.data, is_ghost=True)
    
    if DEBUG:
        print(f"DEBUG [Step 2.1]: Ghost Cell Created -> ({i}, {j}, {k}) mapped to sentinel index -1")
    
    # Initialize physical fields to zero state (Explicit zeroing per Rule 5)
    cell.vx, cell.vy, cell.vz = 0.0, 0.0, 0.0
    cell.p = 0.0
    # Ghost cells are explicitly non-fluid (0)
    cell.mask = 0
    
    return cell