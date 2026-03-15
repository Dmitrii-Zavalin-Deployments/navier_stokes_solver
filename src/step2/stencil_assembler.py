# src/step2/stencil_assembler.py

from src.common.field_schema import FI
from src.common.solver_state import SolverState
from src.common.stencil_block import StencilBlock

from .factory import get_cell

# Rule 7: Granular Traceability
DEBUG = True

def assemble_stencil_matrix(state: SolverState) -> list:
    """
    Assembles a flattened list of StencilBlocks. 
    Delegates Cell creation to the factory's Flyweight cache to ensure 
    topological identity and memory efficiency.
    """
    local_stencil_list = []
    
    # 1. Foundation Verification
    if state.fields.data.shape[-1] != FI.num_fields():
        raise RuntimeError(f"Foundation Mismatch: Buffer width {state.fields.data.shape[-1]} "
                           f"!= Schema requirement {FI.num_fields()}.")

    # 2. Physics & Geometry parameters
    grid = state.grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    
    physics_params = {
        "dx": grid.dx,
        "dy": grid.dy,
        "dz": grid.dz,
        "dt": state.simulation_parameters.time_step,
        "rho": state.fluid_properties.density,
        "mu": state.fluid_properties.viscosity,
        "f_vals": tuple(state.external_forces.force_vector)
    }

    if DEBUG:
        print(f"DEBUG [Step 2.2]: Stencil Assembly Started for {nx}x{ny}x{nz} Domain")

    # 3. Iterate through the Core domain
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # The Factory now handles coordinate-context injection
                c_center = get_cell(i, j, k, state)
                print(f"DEBUG: Checking ({i},{j},{k}) | ID: {id(c_center)}")
                c_i_m    = get_cell(i - 1, j, k, state)
                c_i_p    = get_cell(i + 1, j, k, state)
                c_j_m    = get_cell(i, j - 1, k, state)
                c_j_p    = get_cell(i, j + 1, k, state)
                c_k_m    = get_cell(i, j, k - 1, state)
                c_k_p    = get_cell(i, j, k + 1, state)
                
                block = StencilBlock(
                    center=c_center,
                    i_minus=c_i_m, 
                    i_plus=c_i_p,
                    j_minus=c_j_m, 
                    j_plus=c_j_p,
                    k_minus=c_k_m, 
                    k_plus=c_k_p,
                    **physics_params
                )
                local_stencil_list.append(block)
    
    if DEBUG:
        print(f"DEBUG [Step 2.2]: Successfully assembled {len(local_stencil_list)} StencilBlocks.")
    
    return local_stencil_list