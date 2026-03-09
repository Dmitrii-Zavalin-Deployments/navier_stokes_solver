# src/step2/stencil_assembler.py

from .factory import build_core_cell, build_ghost_cell
# Updated import to point to the new common location
from src.common.stencil_block import StencilBlock


def assemble_stencil_matrix(state, nx, ny, nz, ctx, physics_params):
    """
    Assembles a flattened list of StencilBlocks. 
    Uses a cache to ensure shared cells are instantiated only once.
    """
    local_stencil_list = []
    
    # Cache for Cell objects to prevent redundant object creation
    # Mapping (i, j, k) -> Cell object
    cell_cache = {}

    def get_cell(ix, iy, iz):
        # Check if we already created this cell
        coord = (ix, iy, iz)
        if coord in cell_cache:
            return cell_cache[coord]
            
        # Otherwise, build it and cache it
        if (0 <= ix < nx) and (0 <= iy < ny) and (0 <= iz < nz):
            cell = build_core_cell(ix, iy, iz, state, ctx)
        else:
            cell = build_ghost_cell(ix, iy, iz, ctx)
            
        cell_cache[coord] = cell
        return cell

    # 3. Iterate through the Core domain
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                
                # We assemble the block using the cached cell instances
                block = StencilBlock(
                    center=get_cell(i, j, k),
                    i_minus=get_cell(i-1, j, k), i_plus=get_cell(i+1, j, k),
                    j_minus=get_cell(i, j-1, k), j_plus=get_cell(i, j+1, k),
                    k_minus=get_cell(i, j, k-1), k_plus=get_cell(i, j, k+1),
                    **physics_params
                )
                
                local_stencil_list.append(block)
                
    return local_stencil_list