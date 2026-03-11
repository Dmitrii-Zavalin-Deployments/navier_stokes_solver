# src/step3/ops/forces.py

from src.common.stencil_block import StencilBlock

def get_local_body_force(block: StencilBlock) -> tuple:
    """
    Returns the body force vector (Fx, Fy, Fz) for the current stencil block.
    
    Compliance:
    - Strictly follows Rule 4 (SSoT): Logic objects do not own configuration data.
    - Forces are retrieved via the immutable physics parameters passed to the 
      block during assembly, which are derived from state.config.
    """
    # The StencilBlock stores these as immutable properties defined at 
    # initialization time (Step 2). This ensures no overhead in the 
    # time-stepping loop while maintaining architectural purity.
    return block.f_vals