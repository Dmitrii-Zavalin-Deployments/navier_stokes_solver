# src/step2/stencil_assembler.py

from src.common.stencil_block import StencilBlock

from .factory import build_core_cell, build_ghost_cell


def assemble_stencil_matrix(state, nx, ny, nz, ctx, physics_params):
    """
    Assembles a flattened list of StencilBlocks. 
    Uses a cache to ensure shared cells are instantiated only once.
    """
    local_stencil_list = []
    
    # Extract the Foundation Buffer once for efficiency
    fields_buffer = state.fields.data
    
    # Cache for Cell objects to prevent redundant object creation
    cell_cache = {}

    def get_cell(ix, iy, iz):
        coord = (ix, iy, iz)
        if coord in cell_cache:
            return cell_cache[coord]
            
        # Determine if we are building a Core or Ghost cell
        if (0 <= ix < nx) and (0 <= iy < ny) and (0 <= iz < nz):
            # Pass the fields_buffer to the Cell factory so it can 'view' the foundation
            cell = build_core_cell(ix, iy, iz, state, ctx, fields_buffer)
        else:
            # Ghost cells typically don't hold physical data, 
            # but we pass the buffer for interface consistency if needed
            cell = build_ghost_cell(ix, iy, iz, ctx, fields_buffer)
            
        cell_cache[coord] = cell
        return cell

    # Iterate through the Core domain to build the wiring
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                
                # Assemble the block using the cached cell instances (The Wiring)
                block = StencilBlock(
                    center=get_cell(i, j, k),
                    i_minus=get_cell(i-1, j, k), i_plus=get_cell(i+1, j, k),
                    j_minus=get_cell(i, j-1, k), j_plus=get_cell(i, j+1, k),
                    k_minus=get_cell(i, j, k-1), k_plus=get_cell(i, j, k+1),
                    **physics_params
                )
                
                local_stencil_list.append(block)
                
    return local_stencil_list