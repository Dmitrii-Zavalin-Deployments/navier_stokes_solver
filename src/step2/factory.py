# src/step2/factory.py

from src.common.cell import Cell
from src.common.solver_state import SolverState

# Rule 7: Granular Traceability
DEBUG = True # Enabled for diagnostic tracking

# Centralized cache for Flyweight pattern
_CELL_CACHE = {}

# Explicit constants for ghost cell initialization (Rule 5 compliance)
GHOST_VELOCITY = (0.0, 0.0, 0.0)
GHOST_PRESSURE = 0.0
GHOST_MASK = 0

def get_cell(i: int, j: int, k: int, state: SolverState) -> Cell:
    """
    Unified entry point for Cell retrieval. 
    Implements Flyweight caching to ensure topological identity.
    """
    coord = (i, j, k)
    
    if coord in _CELL_CACHE:
        cell = _CELL_CACHE[coord]
        if DEBUG:
            print(f"DEBUG: Returning CACHED {coord} at {id(cell)}")
        return cell
    
    # Cache MISS
    grid = state.grid
    # Determine if we are building a Core or Ghost cell based on grid bounds
    is_core = (0 <= i < grid.nx) and (0 <= j < grid.ny) and (0 <= k < grid.nz)
    
    if is_core:
        cell = _build_core_cell(i, j, k, state)
    else:
        cell = _build_ghost_cell(i, j, k, state)
        
    _CELL_CACHE[coord] = cell
    
    if DEBUG:
        print(f"DEBUG: Created NEW {coord} at {id(cell)}")
        
    return cell

def _build_core_cell(i: int, j: int, k: int, state: SolverState) -> Cell:
    """
    Creates a View-based Cell (Logic Wiring).
    Uses local caching of SSoT pointers for high-performance access.
    """
    grid = state.grid
    fields = state.fields
    init = state.initial_conditions
    mask_grid = state.mask.mask

    nx_buf, ny_buf = grid.nx + 2, grid.ny + 2
    # 1-based indexing to account for ghost halo
    index = (i + 1) + nx_buf * ((j + 1) + ny_buf * (k + 1))
    
    cell = Cell(index=index, fields_buffer=fields.data, is_ghost=False)
    
    # 4. Initialize physical fields and topological mask (Rule 9)
    cell.vx, cell.vy, cell.vz = init.velocity
    cell.p = init.pressure
    cell.mask = int(mask_grid[i, j, k])
    
    return cell

def _build_ghost_cell(i: int, j: int, k: int, state: SolverState) -> Cell:
    """
    Creates a View-based virtual cell on the perimeter.
    """
    grid = state.grid
    nx_buf, ny_buf = grid.nx + 2, grid.ny + 2
    
    # Calculate index with ghost padding
    index = (i + 1) + nx_buf * ((j + 1) + ny_buf * (k + 1))
    
    cell = Cell(index=index, fields_buffer=state.fields.data, is_ghost=True)
    
    # RULE 5: Explicitly zero all fields to ensure a clean state
    cell.vx, cell.vy, cell.vz = GHOST_VELOCITY
    cell.p = GHOST_PRESSURE
    cell.mask = GHOST_MASK
    
    return cell

def clear_cell_cache():
    """Utility to reset cache between simulation steps."""
    if DEBUG:
        print(f"DEBUG: Clearing cache. Current size: {len(_CELL_CACHE)}")
    _CELL_CACHE.clear()