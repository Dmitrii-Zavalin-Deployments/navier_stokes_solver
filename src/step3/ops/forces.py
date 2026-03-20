# src/step3/ops/forces.py

import math

from src.common.stencil_block import StencilBlock


def get_local_body_force(block: StencilBlock) -> tuple[float, float, float]:
    """
    Returns the body force vector (Fx, Fy, Fz) for the current stencil block.
    
    Compliance:
    - Rule 7: Fail-Fast validation. Ensures input forces are finite before 
      they enter the momentum predictor calculation.
    - Rule 4 (SSoT): Forces are retrieved from the block's immutable properties.
    """
    # 1. Retrieve the force values stored in the block
    forces = block.f_vals
    
    # 2. Rule 7: Numerical Integrity Audit
    # This catches "poisoned" configuration data (NaN/Inf in JSON inputs)
    if not all(math.isfinite(f) for f in forces):
        # We raise ArithmeticError to trigger the same recovery/exit logic
        # as the more complex physics operators.
        raise ArithmeticError(
            f"Non-finite body force detected: {forces} at "
            f"cell ({block.center.i}, {block.center.j}, {block.center.k})"
        )
        
    return forces