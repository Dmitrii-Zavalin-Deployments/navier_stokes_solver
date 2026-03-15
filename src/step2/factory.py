# src/step2/factory.py

from src.common.cell import Cell
from src.common.solver_state import SolverState

# Rule 7: Granular Traceability
DEBUG = True 

# Centralized cache for Flyweight pattern
_CELL_CACHE = {}

# Explicit constants for ghost cell initialization
GHOST_VELOCITY = (0.0, 0.0, 0.0)
GHOST_PRESSURE = 0.0
GHOST_MASK = 0

def get_cell(i: int, j: int, k: int, state: SolverState) -> Cell:
    """
    Unified entry point for Cell retrieval. 
    Implements Flyweight caching with state-aware keys and type-normalized coordinates.
    """
    # Root Cause Fix: Normalize coordinates to standard Python ints to ensure 
    # hash consistency across different sources (NumPy scalars vs standard ints)
    i, j, k = int(i), int(j), int(k)
    
    state_id = id(state)
    coord = (i, j, k)
    cache_key = (coord, state_id)
    
    if cache_key in _CELL_CACHE:
        return _CELL_CACHE[cache_key]
    
    # Cache MISS
    grid = state.grid
    is_core = (0 <= i < grid.nx) and (0 <= j < grid.ny) and (0 <= k < grid.nz)
    
    if is_core:
        cell = _build_core_cell(i, j, k, state)
    else:
        cell = _build_ghost_cell(i, j, k, state)
        
    _CELL_CACHE[cache_key] = cell
    
    if DEBUG:
        # Only log misses to identify topology build-out without log flooding
        print(f"DEBUG: CACHE MISS {coord} | New Cell Created.")
        
    return cell

def _build_core_cell(i: int, j: int, k: int, state: SolverState) -> Cell:
    """Creates a View-based Core Cell."""
    grid = state.grid
    fields = state.fields
    init = state.initial_conditions
    mask_grid = state.mask.mask

    nx_buf, ny_buf = grid.nx + 2, grid.ny + 2
    index = (i + 1) + nx_buf * ((j + 1) + ny_buf * (k + 1))
    
    cell = Cell(index=index, fields_buffer=fields.data, nx_buf=nx_buf, ny_buf=ny_buf, is_ghost=False)
    cell.vx, cell.vy, cell.vz = init.velocity
    cell.p = init.pressure
    cell.mask = int(mask_grid[i, j, k])
    
    return cell

def _build_ghost_cell(i: int, j: int, k: int, state: SolverState) -> Cell:
    """Creates a View-based Ghost Cell."""
    grid = state.grid
    nx_buf, ny_buf = grid.nx + 2, grid.ny + 2
    
    index = (i + 1) + nx_buf * ((j + 1) + ny_buf * (k + 1))
    
    cell = Cell(index=index, fields_buffer=state.fields.data, nx_buf=nx_buf, ny_buf=ny_buf, is_ghost=True)
    cell.vx, cell.vy, cell.vz = GHOST_VELOCITY
    cell.p = GHOST_PRESSURE
    cell.mask = GHOST_MASK
    
    return cell

def clear_cell_cache():
    """Utility to reset cache between simulation steps."""
    if DEBUG and len(_CELL_CACHE) > 0:
        print(f"DEBUG: Clearing cache. Size: {len(_CELL_CACHE)}")
    _CELL_CACHE.clear()