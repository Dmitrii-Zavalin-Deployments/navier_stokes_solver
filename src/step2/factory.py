# src/step2/factory.py

import numpy as np
from src.common.cell import Cell
from src.common.field_schema import FI
from src.core.solver_state import SolverState

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
    masks = state.masks

    # 2. Index calculation and topology (Rule 9)
    index = i + grid.nx * (j + grid.ny * k)
    mask = int(masks.mask[i, j, k])
    
    # 3. Instantiate the Cell (The Wiring)
    cell = Cell(index=index, fields_buffer=fields.data)
    
    if DEBUG:
        print(f"DEBUG [Step 2.1]: Core Cell Created -> Index: {index} at ({i}, {j}, {k})")
    
    # 4. Initialize physical fields via property setters (Rule 9)
    cell.vx, cell.vy, cell.vz = init.velocity
    cell.p = init.pressure
    
    return cell

def build_ghost_cell(i: int, j: int, k: int, state: SolverState) -> Cell:
    """
    Creates a View-based virtual cell on the perimeter with sentinel indexing.
    """
    # Sentinel index -1 for ghost regions (Rule 9)
    cell = Cell(index=-1, fields_buffer=state.fields.data)
    
    if DEBUG:
        print(f"DEBUG [Step 2.1]: Ghost Cell Created -> ({i}, {j}, {k}) mapped to sentinel index -1")
    
    # Initialize physical fields to zero state (Explicit zeroing per Rule 5)
    cell.vx, cell.vy, cell.vz = 0.0, 0.0, 0.0
    cell.p = 0.0
    
    return cell