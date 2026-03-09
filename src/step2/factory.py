# src/step2/factory.py

from src.core.solver_state import SolverState
# Updated import to point to common location
from src.common.cell import Cell 


def get_initialization_context(state: SolverState) -> dict:
    """
    Step 2 Context Provider:
    Hoists physical constants and initial conditions out of the 3D loop.
    This prevents O(N^3) dictionary lookups.
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

def build_core_cell(i: int, j: int, k: int, state: SolverState, ctx: dict) -> Cell:
    """
    Creates a real cell inside the nx * ny * nz domain.
    Maps physical coordinates and topology from the input MaskData.
    """
    # Instantiate with no args to minimize __init__ logic; 
    # validation occurs via property assignment.
    cell = Cell()
    
    # 1. Physical Center Coordinates
    cell.x = ctx["x_min"] + (i + 0.5) * ctx["dx"]
    cell.y = ctx["y_min"] + (j + 0.5) * ctx["dy"]
    cell.z = ctx["z_min"] + (k + 0.5) * ctx["dz"]

    # 2. Map Physics from the hoisted Context
    cell.vx, cell.vy, cell.vz, cell.p = ctx["vx"], ctx["vy"], ctx["vz"], ctx["p"]
    
    # 3. Topology from Step 1 Mask
    cell.mask = int(state.masks.mask[i, j, k])
    cell.is_ghost = False 
    
    return cell

def build_ghost_cell(i: int, j: int, k: int, ctx: dict) -> Cell:
    """
    Creates a virtual cell on the perimeter.
    Uses -1 mask to signify boundary conditions.
    """
    cell = Cell()
    
    # 1. Physical coordinates
    cell.x = ctx["x_min"] + (i + 0.5) * ctx["dx"]
    cell.y = ctx["y_min"] + (j + 0.5) * ctx["dy"]
    cell.z = ctx["z_min"] + (k + 0.5) * ctx["dz"]

    # 2. Initial Physics
    cell.vx, cell.vy, cell.vz, cell.p = ctx["vx"], ctx["vy"], ctx["vz"], ctx["p"]
    
    # 3. Compliance: Using -1 as the boundary/ghost indicator
    cell.mask = -1  
    cell.is_ghost = True
    
    return cell