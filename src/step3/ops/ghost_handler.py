# src/step3/ops/ghost_handler.py

import logging
from src.common.stencil_block import StencilBlock

logger = logging.getLogger(__name__)

def sync_ghost_trial_buffers(block: StencilBlock) -> None:
    """
    Rule 7 & 9: Direct Buffer Alignment for Ghost Cells.
    
    Ensures Trial buffers (STAR, NEXT) are synchronized with Foundation buffers
    (VX, VY, VZ, P) directly in the NumPy memory space. 
    
    Constraint: No temporary arrays, dicts, or heap reallocations.
    """
    # Rule 9: Accessing the center cell's logic-pointer
    cell = block.center
    
    try:
        # Rule 9: Pointer-Based Access via @property 
        # These properties must write directly to the underlying fields_buffer
        cell.vx_star = cell.vx
        cell.vy_star = cell.vy
        cell.vz_star = cell.vz
        
        # Sync Pressure Trial Buffer
        cell.p_next = cell.p

        # Rule 7: High-resolution debug log for traceability
        logger.debug(f"GHOST SYNC [Success]: Cell {getattr(block, 'id', 'N/A')} trial buffers aligned.")
        
    except AttributeError as e:
        # Rule 5: Explicit Error on structural failure
        logger.critical(f"CONTRACT VIOLATION: Block center lacks Rule 9 field properties. {e}")
        raise RuntimeError("Ghost sync failed: Block center is not a ValidatedContainer.") from e