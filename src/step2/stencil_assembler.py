# src/step2/stencil_assembler.py

from src.common.field_schema import FI
from src.common.stencil_block import StencilBlock
from src.core.solver_state import SolverState
from .factory import build_core_cell, build_ghost_cell

# Rule 7: Granular Traceability
DEBUG = True

def assemble_stencil_matrix(state: SolverState) -> list:
    """
    Assembles a flattened list of StencilBlocks. 
    Directly accesses SSoT containers (Rule 4) to maintain architectural integrity.
    """
    local_stencil_list = []
    
    # Foundation Verification (Rule 9)
    if state.fields.data.shape[1] != FI.num_fields():
        raise RuntimeError(f"Foundation Mismatch: Buffer width {state.fields.data.shape[1]} "
                           f"!= Schema requirement {FI.num_fields()}.")

    # Local caching of SSoT pointers for performance (Rule 0)
    grid = state.grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    
    # Physics parameters cached from state.config (Rule 5)
    sim_params = state.config.simulation_parameters
    fluid_props = state.config.fluid_properties
    ext_forces = state.config.external_forces
    
    # Prepare parameter bundle for StencilBlock (Rule 5: No defaults)
    physics_params = {
        "dx": grid.dx,
        "dy": grid.dy,
        "dz": grid.dz,
        "dt": sim_params["time_step"],
        "rho": fluid_props["density"],
        "mu": fluid_props["viscosity"],
        "f_vals": tuple(ext_forces["force_vector"])
    }

    # Cache for Cell objects to prevent redundant object creation
    cell_cache = {}

    def get_cell(ix, iy, iz):
        coord = (ix, iy, iz)
        if coord in cell_cache:
            return cell_cache[coord]
            
        # Factory call uses state directly (Rule 4)
        if (0 <= ix < nx) and (0 <= iy < ny) and (0 <= iz < nz):
            cell = build_core_cell(ix, iy, iz, state)
        else:
            cell = build_ghost_cell(ix, iy, iz, state)
            
        cell_cache[coord] = cell
        return cell

    if DEBUG:
        print(f"DEBUG [Step 2.2]: Stencil Assembly Started for {nx}x{ny}x{nz} Domain")

    # Iterate through the Core domain to build the wiring (Logic Layer)
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