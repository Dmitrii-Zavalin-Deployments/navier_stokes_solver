# src/step2/stencil_assembler.py

from src.common.field_schema import FI
from src.common.solver_state import SolverState
from src.common.stencil_block import StencilBlock

from .factory import get_cell

# Rule 7: Granular Traceability
DEBUG = True

class CellRegistry:
    """Manages the lifecycle and topological identity of Cell instances."""
    def __init__(self):
        self._cache = {}

    def get_or_create(self, i: int, j: int, k: int, state: SolverState):
        key = ((int(i), int(j), int(k)), id(state))
        if key not in self._cache:
            self._cache[key] = get_cell(i, j, k, state)
        return self._cache[key]

    def clear(self):
        self._cache.clear()

# Global registry instance
registry = CellRegistry()

def assemble_stencil_matrix(state: SolverState) -> list:
    """
    Assembles a flattened list of StencilBlocks using the local registry 
    to ensure topological identity.
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
                # Retrieve unique, registry-managed cells
                c_center = registry.get_or_create(i, j, k, state)
                c_i_m    = registry.get_or_create(i - 1, j, k, state)
                c_i_p    = registry.get_or_create(i + 1, j, k, state)
                c_j_m    = registry.get_or_create(i, j - 1, k, state)
                c_j_p    = registry.get_or_create(i, j + 1, k, state)
                c_k_m    = registry.get_or_create(i, j, k - 1, state)
                c_k_p    = registry.get_or_create(i, j, k + 1, state)
                
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