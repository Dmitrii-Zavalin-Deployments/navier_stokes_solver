# src/step3/predictor.py

import logging

import numpy as np

from src.common.field_schema import FI
from src.common.stencil_block import StencilBlock
from src.step3.ops.advection import compute_local_advection_vector
from src.step3.ops.forces import get_local_body_force
from src.step3.ops.gradient import compute_local_gradient_p
from src.step3.ops.laplacian import compute_local_laplacian_v_n
from src.step3.ops.scaling import get_dt_over_rho

# Initialize logger for this module
logger = logging.getLogger("Solver.Predictor")

def compute_local_predictor_step(block: StencilBlock) -> None:
    """
    Computes the intermediate star-velocity field v*.
    
    Formula: v* = v^n + (dt/rho) * (mu * lap(v^n) - rho * (v^n ⋅ ∇)v^n + F - grad(p^n))
    
    Compliance:
    - Rule 7: Fail-Fast math audit.
    - Rule 9: In-place update via Schema-Locked Accessors (Sovereign Scalars).
    """
    
    # 1. Local Operator calls
    # Returns tuples/lists of scalars enforced by the Cell foundation
    lap_v = compute_local_laplacian_v_n(block)    
    adv_v = compute_local_advection_vector(block) 
    force = get_local_body_force(block)           
    grad_p = compute_local_gradient_p(block, field_id=FI.P) 
    
    # 2. Scaling factor
    dt_over_rho = get_dt_over_rho(block)
    
    # 3. Retrieve current velocities (Sovereign Foundation Access)
    v_n = (
        block.center.get_field(FI.VX),
        block.center.get_field(FI.VY),
        block.center.get_field(FI.VZ)
    )
    
    # 4. Compute and apply v_star components
    star_fields = [FI.VX_STAR, FI.VY_STAR, FI.VZ_STAR]
    
    for i, field_id in enumerate(star_fields):
        # 4. Calculate intermediate value (Pure scalar math)
        # Physics implementation is now readable and decoupled from buffer logic.
        v_star_val = v_n[i] + dt_over_rho * (
            block.mu * lap_v[i] - block.rho * adv_v[i] + force[i] - grad_p[i]
        )
        
        # --- RULE 7: FAIL-FAST MATH CHECK ---
        # Catch numerical explosions before they propagate to the PPE step.
        if not np.isfinite(v_star_val):
            logger.critical(f"PREDICTOR FAILURE: Non-finite v_star in {block.id} | Component {i}")
            raise ArithmeticError(f"Instability detected in momentum predictor for {field_id}")

        # --- [STRATEGIC DEBUG LOG] ---
        # Log the first component (VX_STAR) to monitor simulation health at a glance.
        if i == 0:
            logger.debug("DEBUG [Predictor]: Type=Sovereign | Block %s | VX_STAR: %s", block.id, v_star_val)

        # 5. Commit to the Trial (Star) buffer
        # Cell.set_field handles final type normalization to float.
        block.center.set_field(field_id, v_star_val)

    logger.debug(f"PREDICT [Success]: {block.id} updated with v_star.")