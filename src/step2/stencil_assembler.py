# src/step2/stencil_assembler.py

from src.common.field_schema import FI
from src.common.solver_state import SolverState
from src.common.stencil_block import StencilBlock

from .factory import build_core_cell, build_ghost_cell

# Rule 7: Granular Traceability
DEBUG = True

def assemble_stencil_matrix(state: SolverState) -> list:
    """
    Assembles a flattened list of StencilBlocks. 
    Directly accesses SSoT containers (Rule 4) to maintain architectural integrity.
    """
    local_stencil_list = []
    
    # 1. Foundation Verification (Rule 9 & Rule 5)
    # Ensure the buffer is fully compliant with the current FI Schema
    if state.fields.data.shape[1] != FI.num_fields():
        raise RuntimeError(f"Foundation Mismatch: Buffer width {state.fields.data.shape[1]} "
                           f"!= Schema requirement {FI.num_fields()}.")

    # 2. Local caching of SSoT pointers for performance (Rule 0)
    grid = state.grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    
    # 3. Physics & Geometry parameters cached from SSoT (Rule 4 & 5)
    # No hardcoded fallbacks; every value must exist in the validated State
    physics_params = {
        "dx": (grid.x_max - grid.x_min) / nx,
        "dy": (grid.y_max - grid.y_min) / ny,
        "dz": (grid.z_max - grid.z_min) / nz,
        "dt": state.simulation_parameters.time_step,
        "rho": state.fluid_properties.density,
        "mu": state.fluid_properties.viscosity,
        "f_vals": tuple(state.external_forces.force_vector)
    }

    # Cache for Cell objects to prevent redundant object creation (Flyweight pattern)
    cell_cache = {}

    def get_cell(ix, iy, iz):
        coord = (ix, iy, iz)
        if coord in cell_cache:
            return cell_cache[coord]
            
        # Factory call uses state directly (Rule 4: Hierarchy over convenience)
        if (0 <= ix < nx) and (0 <= iy < ny) and (0 <= iz < nz):
            cell = build_core_cell(ix, iy, iz, state)
        else:
            cell = build_ghost_cell(ix, iy, iz, state)
            
        cell_cache[coord] = cell
        return cell

    if DEBUG:
        print(f"DEBUG [Step 2.2]: Stencil Assembly Started for {nx}x{ny}x{nz} Domain")
        print(f"  > Physics Bundle: {physics_params}")

    # 4. Iterate through the Core domain to build the wiring (Logic Layer)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                block = StencilBlock(
                    center=get_cell(i, j, k),
                    i_minus=get_cell(i-1, j, k), i_plus=get_cell(i+1, j, k),
                    j_minus=get_cell(i, j-1, k), j_plus=get_cell(i, j+1, k),
                    k_minus=get_cell(i, j, k-1), k_plus=get_cell(i, j, k+1),
                    **physics_params
                )
                local_stencil_list.append(block)
    
    if DEBUG:
        print(f"DEBUG [Step 2.2]: Assembly Complete. Total StencilBlocks: {len(local_stencil_list)}")
                
    return local_stencil_list