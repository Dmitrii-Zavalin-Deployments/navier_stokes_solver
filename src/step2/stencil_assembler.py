# src/step2/stencil_assembler.py

from src.common.stencil_block import StencilBlock
from src.common.field_schema import FI  # Import the Single Source of Truth
from .factory import build_core_cell, build_ghost_cell

# Rule 7: Granular Traceability
DEBUG = True

def assemble_stencil_matrix(state, nx, ny, nz, ctx, physics_params):
    """
    Assembles a flattened list of StencilBlocks. 
    Uses a cache to ensure shared cells are instantiated only once.
    """
    local_stencil_list = []
    
    # Ensure the Foundation Buffer is initialized to the correct width
    # Based on the Schema definition
    if state.fields.data.shape[1] != FI.num_fields():
        raise RuntimeError(f"Foundation Mismatch: Buffer width {state.fields.data.shape[1]} "
                           f"does not match Schema requirement {FI.num_fields()}.")

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

    if DEBUG:
        print(f"DEBUG [Step 2.2]: Stencil Assembly Started for {nx}x{ny}x{nz} Domain")
        print(f"DEBUG [Step 2.2]: Foundation Schema Width: {FI.num_fields()} fields")

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
    
    if DEBUG:
        print(f"DEBUG [Step 2.2]: Assembly Complete")
        print(f"  > Total StencilBlocks: {len(local_stencil_list)}")
        print(f"  > Unique Cell Cache Size: {len(cell_cache)}")
                
    return local_stencil_list