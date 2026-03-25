# src/step3/corrector.py

import logging
import numpy as np

from src.common.field_schema import FI
from src.common.stencil_block import StencilBlock
from src.step3.ops.gradient import compute_local_gradient_p
from src.step3.ops.scaling import get_dt_over_rho

logger = logging.getLogger("Solver.Corrector")

def apply_local_velocity_correction(block: StencilBlock) -> None:
    """
    Projects the intermediate velocity field v* onto a divergence-free space.
    
    Formula: v^{n+1} = v^* - (dt/rho) * grad(p^{n+1})
    
    Compliance:
    - Rule 7: Fail-Fast math audit. 
    - Rule 9: In-place update of Trial Fields (VX_STAR, etc.)
    """
    
    # 1. Compute the pressure gradient at p^{n+1} (FI.P_NEXT)
    grad_p = compute_local_gradient_p(block, field_id=FI.P_NEXT)
    
    # 2. Scaling factor (dt/rho)
    scaling = get_dt_over_rho(block)
    
    # 3. Retrieve intermediate star-velocity
    v_star = (
        block.center.get_field(FI.VX_STAR),
        block.center.get_field(FI.VY_STAR),
        block.center.get_field(FI.VZ_STAR)
    )
    
    # 4. Calculate new velocities
    # Stencil operations here often promote results to NumPy arrays
    v_new_raw = [
        v_star[0] - (scaling * grad_p[0]),
        v_star[1] - (scaling * grad_p[1]),
        v_star[2] - (scaling * grad_p[2])
    ]
    
    # --- FORENSIC DNA AUDIT & SCALAR ENFORCEMENT ---
    v_final = []
    for i, val in enumerate(v_new_raw):
        # Identify if this is the "Array Leak" culprit
        if hasattr(val, "shape") and val.shape != ():
            logger.debug(
                f"AUDIT [Correction]: Detected array leak in Block {block.id} | "
                f"Component {i} | Shape {val.shape}. Extracting scalar..."
            )
            v_final.append(val.item())
        else:
            v_final.append(float(val))

    # --- RULE 7: FAIL-FAST MATH CHECK ---
    if not np.isfinite(v_final).all():
        logger.error(f"CORRECTOR CRITICAL: Non-finite velocity in block {block.id} | {v_final=}")
        raise ArithmeticError(f"Instability detected during velocity correction at block {block.id}")

    # 5. Apply velocity correction in-place to the STAR buffer
    # v_final is now guaranteed to be a list of Python floats
    block.center.set_field(FI.VX_STAR, v_final[0])
    block.center.set_field(FI.VY_STAR, v_final[1])
    block.center.set_field(FI.VZ_STAR, v_final[2])

    logger.debug(f"CORRECT [Success]: Block {block.id} updated with corrected velocities.")