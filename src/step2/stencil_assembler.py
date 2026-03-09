# src/step2/stencil_assembler.py

from .factory import build_core_cell, build_ghost_cell
from .stencil_block import StencilBlock


def assemble_stencil_matrix(state, nx, ny, nz, ctx, physics_params):
    """
    Assembles a flattened list of StencilBlocks containing only Core cells.
    Neighbors (including ghost cells) are resolved via a shared helper.
    """
    # 1. Initialize a standard Python list
    # For object iteration, lists are slightly more efficient than NumPy object arrays.
    local_stencil_list = []

    # 2. Define the cell-fetcher ONCE outside the loops
    # This prevents thousands of function redefinitions during execution.
    def get_cell(ix, iy, iz):
        if (0 <= ix < nx) and (0 <= iy < ny) and (0 <= iz < nz):
            return build_core_cell(ix, iy, iz, state, ctx)
        return build_ghost_cell(ix, iy, iz, ctx)

    # 3. Iterate through the Core domain only
    # Note: range(nx) corresponds to core cells 0 to nx-1.
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                
                # Assemble the 7-point stencil for the current core cell
                # Neighbors automatically fall into ghost zones via get_cell logic
                block = StencilBlock(
                    center=get_cell(i, j, k),
                    i_minus=get_cell(i-1, j, k), i_plus=get_cell(i+1, j, k),
                    j_minus=get_cell(i, j-1, k), j_plus=get_cell(i, j+1, k),
                    k_minus=get_cell(i, j, k-1), k_plus=get_cell(i, j, k+1),
                    **physics_params
                )
                
                local_stencil_list.append(block)
                
    return local_stencil_list