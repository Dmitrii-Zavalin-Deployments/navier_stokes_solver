# tests/helpers/solver_step2_output_dummy.py

"""
Archivist Testing: Snapshot-based Test Baseline (Step 2).

Compliance:
- Rule 6: Zero-Redundancy (This is a snapshot, not an orchestration).
- Rule 7: Atomic Numerical Truth (Use fixed values for verification).
"""

from src.common.stencil_block import StencilBlock
from src.step2.factory import build_core_cell, build_ghost_cell
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy


def make_step2_output_dummy(nx: int = 4, ny: int = 4, nz: int = 4):
    """
    Returns a 'frozen' prototype representing a successful Step 2 completion.
    The stencil_matrix is populated with static, verified StencilBlocks, 
    correctly identifying ghost vs core cells to prevent IndexErrors.
    """
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # 1. Define Fixed Physics Baseline (Atomic Numerical Truth)
    physics_params = {
        "dx": 0.25, "dy": 0.25, "dz": 0.25,
        "dt": 0.001, "rho": 1000.0, "mu": 0.001,
        "f_vals": (0.0, 0.0, -9.81)
    }
    
    # 2. Populate with Static Blocks (Accounting for Ghost Cells)
    state.stencil_matrix = []
    
    # Loop over the extended grid (N+2)
    for i in range(nx + 2):
        for j in range(ny + 2):
            for k in range(nz + 2):
                # Identify if current index is a ghost cell
                is_ghost = (i == 0 or i == nx + 1 or 
                            j == 0 or j == ny + 1 or 
                            k == 0 or k == nz + 1)
                
                if is_ghost:
                    cell = build_ghost_cell(i, j, k, state)
                else:
                    # Core cells are indexed 1 to nx/ny/nz relative to ghost start
                    cell = build_core_cell(i - 1, j - 1, k - 1, state)
                
                # Mocking the block connectivity for the test baseline
                block = StencilBlock(
                    center=cell,
                    i_minus=cell, i_plus=cell,
                    j_minus=cell, j_plus=cell,
                    k_minus=cell, k_plus=cell,
                    **physics_params
                )
                state.stencil_matrix.append(block)
    
    # 3. Explicitly mark as ready
    state.ready_for_time_loop = True
    
    return state