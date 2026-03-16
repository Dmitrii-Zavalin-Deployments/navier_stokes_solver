# src/step2/stencil_assembler.py

from src.common.field_schema import FI
from src.common.grid_math import get_flat_index
from src.common.solver_state import SolverState
from src.common.stencil_block import StencilBlock

from .factory import get_cell

# Rule 7: Granular Traceability
DEBUG = True

class CellRegistry:
    def __init__(self, nx: int, ny: int, nz: int):
        # Store these as instance attributes
        self.nx = nx
        self.ny = ny
        self.nz = nz
        
        # 2-Tier Architecture: Padding set to +2
        self.nx_dim = nx + 2
        self.ny_dim = ny + 2
        self.nz_dim = nz + 2
        self._cache = [None] * (self.nx_dim * self.ny_dim * self.nz_dim)

    def _get_idx(self, i: int, j: int, k: int) -> int:
        # Access them via self.nx, self.ny, self.nz
        if not (-1 <= i < self.nx + 1 and -1 <= j < self.ny + 1 and -1 <= k < self.nz + 1):
             raise IndexError(f"Stencil accessing out-of-bounds: ({i}, {j}, {k})")
        return get_flat_index(i, j, k, self.nx_dim, self.ny_dim, offset=1)

    def get_or_create(self, i: int, j: int, k: int, state: SolverState):
        idx = self._get_idx(i, j, k)
        if self._cache[idx] is None:
            # We pass the original (i, j, k) to the factory 
            # so the Cell object retains its actual grid position.
            self._cache[idx] = get_cell(i, j, k, state)
        
        # Insert this right before the line causing the error:
        print(f"DEBUG: Requesting ({i}, {j}, {k}) -> Index {idx} (Cache size: {len(self._cache)})")

        return self._cache[idx]

def assemble_stencil_matrix(state: SolverState) -> list:
    """
    Assembles a flattened list of StencilBlocks using a deterministic
    Flat Index Engine.
    """
    if state.fields.data.shape[-1] != FI.num_fields():
        raise RuntimeError(f"Foundation Mismatch: Buffer width {state.fields.data.shape[-1]} "
                           f"!= Schema requirement {FI.num_fields()}.")

    grid = state.grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    
    registry = CellRegistry(nx, ny, nz)
    
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

    local_stencil_list = []
    
    # Iterate through the Core domain and its immediate Ghost layer.
    for k in range(-1, nz + 1):
        for j in range(-1, ny + 1):
            for i in range(-1, nx + 1):
                block = StencilBlock(
                    center=registry.get_or_create(i, j, k, state),
                    i_minus=registry.get_or_create(i - 1, j, k, state),
                    i_plus=registry.get_or_create(i + 1, j, k, state),
                    j_minus=registry.get_or_create(i, j - 1, k, state),
                    j_plus=registry.get_or_create(i, j + 1, k, state),
                    k_minus=registry.get_or_create(i, j, k - 1, state),
                    k_plus=registry.get_or_create(i, j, k + 1, state),
                    **physics_params
                )
                local_stencil_list.append(block)
    
    if DEBUG:
        print(f"DEBUG [Step 2.2]: Successfully assembled {len(local_stencil_list)} StencilBlocks.")
    
    return local_stencil_list