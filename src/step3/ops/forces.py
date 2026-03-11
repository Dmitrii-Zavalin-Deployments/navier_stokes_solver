# src/step3/ops/forces.py

from src.common.stencil_block import StencilBlock


def get_local_body_force(block: StencilBlock) -> tuple:
    """
    Returns the body force vector (Fx, Fy, Fz) for the current stencil block.
    
    Compliance:
    - Accesses simulation-wide constant forces stored in the StencilBlock.
    - These values are static for the duration of the time-loop, ensuring
      zero overhead during physics execution.
    """
    # f_vals is defined at assembly time in src/step2/orchestrate_step2.py
    # and passed through the StencilBlock during construction.
    return block.f_vals