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
    if state.fields.data.shape[1] != FI.num_fields():
        raise RuntimeError(f"Foundation Mismatch: Buffer width {state.fields.data.shape[1]} "
                           f"!= Schema requirement {FI.num_fields()}.")

    # 2. Physics & Geometry parameters
    grid = state.grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    
    physics_params = {
        "dx": (grid.x_max - grid.x_min) / nx,
        "dy": (grid.y_max - grid.y_min) / ny,
        "dz": (grid.z_max - grid.z_min) / nz,
        "dt": state.simulation_parameters.time_step,
        "rho": state.fluid_properties.density,
        "mu": state.fluid_properties.viscosity,
        "f_vals": tuple(state.external_forces.force_vector)
    }

    if DEBUG:
        print(f"DEBUG [Step 2.2]: Stencil Assembly Started for {nx}x{ny}x{nz} Domain")
        print(f"DEBUG: Using SolverState instance at {id(state)}")

    # 3. Iterate through the Core domain
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Retrieve cells for the stencil
                c_center = get_cell(i, j, k, state)
                c_i_plus = get_cell(i+1, j, k, state)
                
                # Trace identity before assignment
                if DEBUG and (i == 0 and j == 0 and k == 0):
                    print(f"DEBUG [Wiring]: Block (0,0,0) center ID: {id(c_center)}")
                    print(f"DEBUG [Wiring]: Block (0,0,0) i_plus ID: {id(c_i_plus)}")

                block = StencilBlock(
                    center=c_center,
                    i_minus=get_cell(i-1, j, k, state), 
                    i_plus=c_i_plus,
                    j_minus=get_cell(i, j-1, k, state), 
                    j_plus=get_cell(i, j+1, k, state),
                    k_minus=get_cell(i, j, k-1, state), 
                    k_plus=get_cell(i, j, k+1, state),
                    **physics_params
                )
                local_stencil_list.append(block)
    
    if DEBUG:
        print(f"DEBUG [Step 2.2]: Assembly Complete. Total StencilBlocks: {len(local_stencil_list)}")
                
    return local_stencil_list