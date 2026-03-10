# src/step2/factory.py

import numpy as np

from src.common.cell import Cell
from src.core.solver_state import SolverState


def get_initialization_context(state: SolverState) -> dict:
    """
    Step 2 Context Provider:
    Hoists physical constants and initial conditions out of the 3D loop.
    """
    init_cond = state.config.initial_conditions 
    init_v = init_cond["velocity"]
    init_p = init_cond["pressure"]

    return {
        "vx": float(init_v[0]),
        "vy": float(init_v[1]),
        "vz": float(init_v[2]),
        "p": float(init_p),
        "dx": state.grid.dx,
        "dy": state.grid.dy,
        "dz": state.grid.dz,
        "x_min": state.grid.x_min,
        "y_min": state.grid.y_min,
        "z_min": state.grid.z_min
    }

def build_core_cell(i: int, j: int, k: int, state: SolverState, ctx: dict, fields_buffer: np.ndarray) -> Cell:
    """
    Creates a View-based Cell inside the nx * ny * nz domain.
    """
    # 1. Calculate linear index for buffer mapping
    nx, ny = state.grid.nx, state.grid.ny
    index = i + nx * (j + ny * k)
    
    # 2. Topology
    mask = int(state.masks.mask[i, j, k])
    
    # 3. Instantiate Cell (The Viewer)
    cell = Cell(
        index=index,
        fields_buffer=fields_buffer,
        x=ctx["x_min"] + (i + 0.5) * ctx["dx"],
        y=ctx["y_min"] + (j + 0.5) * ctx["dy"],
        z=ctx["z_min"] + (k + 0.5) * ctx["dz"],
        mask=mask,
        is_ghost=False
    )
    
    # 4. Map Physics (Writing to the Foundation buffer via property setters)
    cell.vx, cell.vy, cell.vz, cell.p = ctx["vx"], ctx["vy"], ctx["vz"], ctx["p"]
    cell.vx_star, cell.vy_star, cell.vz_star = 0.0, 0.0, 0.0
    cell.p_next = 0.0
    
    return cell

def build_ghost_cell(i: int, j: int, k: int, ctx: dict, fields_buffer: np.ndarray) -> Cell:
    """
    Creates a View-based virtual cell on the perimeter.
    """
    # Ghost cells utilize a sentinel index
    cell = Cell(
        index=-1, 
        fields_buffer=fields_buffer,
        x=ctx["x_min"] + (i + 0.5) * ctx["dx"],
        y=ctx["y_min"] + (j + 0.5) * ctx["dy"],
        z=ctx["z_min"] + (k + 0.5) * ctx["dz"],
        mask=-1,
        is_ghost=True
    )
    
    cell.vx, cell.vy, cell.vz, cell.p = ctx["vx"], ctx["vy"], ctx["vz"], ctx["p"]
    cell.vx_star, cell.vy_star, cell.vz_star, cell.p_next = 0.0, 0.0, 0.0, 0.0
    
    return cell