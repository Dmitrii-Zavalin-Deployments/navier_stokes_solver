# src/step3/ops/ghost_handler.py

from src.common.stencil_block import StencilBlock


def sync_ghost_trial_buffers(block: StencilBlock) -> None:
    """
    Rule 7 & 9: Direct Buffer Alignment for Ghost Cells.
    
    Ensures Trial buffers (STAR, NEXT) are synchronized with Foundation buffers
    (VX, VY, VZ, P) directly in the NumPy memory space. 
    
    Constraint: No temporary arrays, dicts, or heap reallocations.
    """
    # Rule 9: Accessing the center cell's logic-pointer
    cell = block.center
    
    # Rule 9: Pointer-Based Access via @property 
    # This writes directly to the underlying fields_buffer[index, FI.VX_STAR]
    cell.vx_star = cell.vx
    cell.vy_star = cell.vy
    cell.vz_star = cell.vz
    
    # Sync Pressure Trial Buffer
    cell.p_next = cell.p