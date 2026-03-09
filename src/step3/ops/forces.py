# src/step3/ops/forces.py

from src.common.stencil_block import StencilBlock

def get_local_body_force(block: StencilBlock) -> tuple:
    """
    Returns the body force vector (Fx, Fy, Fz) for the current stencil block.
    
    Accesses the tuple stored in the StencilBlock instance.
    """
    # f_vals is stored as (Fx, Fy, Fz) in the block
    return block.f_vals