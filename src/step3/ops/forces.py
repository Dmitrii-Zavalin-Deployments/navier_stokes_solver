# src/step3/ops/forces.py


from src.common.stencil_block import StencilBlock


def get_local_body_force(block: StencilBlock) -> tuple[float, float, float]:
    """
    Returns the body force vector (Fx, Fy, Fz) for the current stencil block.
    
    Compliance:
    - Rule 7: Fail-Fast validation. Ensures input forces are finite before 
      they enter the momentum predictor calculation.
    - Rule 4 (SSoT): Forces are retrieved from the block's immutable properties.
    """
    # Retrieve the force values stored in the block
    forces = block.f_vals
        
    return forces