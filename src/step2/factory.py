# src/step2/factory.py

from src.common.cell import Cell
from src.common.solver_state import SolverState
from src.common.grid_math import get_flat_index

# Rule 7: Granular Traceability
DEBUG = True 

# Explicit constants for ghost cell initialization
GHOST_VELOCITY = (0.0, 0.0, 0.0)
GHOST_PRESSURE = 0.0
GHOST_MASK = 0

def get_cell(i: int, j: int, k: int, state: SolverState) -> Cell:
    """
    Pure allocator: Creates a new Cell instance. 
    Responsibility for caching/identity has been delegated to the Assembler.
    """
    grid = state.grid
    
    # Identify if the coordinate is in the core domain or ghost region
    is_core = (0 <= i < grid.nx) and (0 <= j < grid.ny) and (0 <= k < grid.nz)
    
    if is_core:
        cell = _build_core_cell(i, j, k, state)
    else:
        cell = _build_ghost_cell(i, j, k, state)
    
    if DEBUG:
        print(f"DEBUG [Factory]: Allocated new {'Core' if is_core else 'Ghost'} "
              f"Cell at ({i}, {j}, {k}) | ID: {id(cell)}")
        
    return cell

def _build_core_cell(i: int, j: int, k: int, state: SolverState) -> Cell:
    """Creates a View-based Core Cell."""
    grid = state.grid
    fields = state.fields
    init = state.initial_conditions
    mask_grid = state.mask.mask

    nx_buf, ny_buf = grid.nx + 2, grid.ny + 2
    index = get_flat_index(i, j, k, nx_buf, ny_buf)
    
    cell = Cell(index=index, fields_buffer=fields.data, nx_buf=nx_buf, ny_buf=ny_buf, is_ghost=False)
    cell.vx, cell.vy, cell.vz = init.velocity
    cell.p = init.pressure
    cell.mask = int(mask_grid[i, j, k])
    
    return cell

def _build_ghost_cell(i: int, j: int, k: int, state: SolverState) -> Cell:
    """Creates a View-based Ghost Cell."""
    grid = state.grid
    nx_buf, ny_buf = grid.nx + 2, grid.ny + 2
    
    index = get_flat_index(i, j, k, nx_buf, ny_buf)
    
    cell = Cell(index=index, fields_buffer=state.fields.data, nx_buf=nx_buf, ny_buf=ny_buf, is_ghost=True)
    cell.vx, cell.vy, cell.vz = GHOST_VELOCITY
    cell.p = GHOST_PRESSURE
    cell.mask = GHOST_MASK
    
    return cell