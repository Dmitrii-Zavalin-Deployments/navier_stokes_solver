# src/step3/predictor.py

import math

from src.common.field_schema import FI
from src.common.stencil_block import StencilBlock
from src.step3.ops.advection import compute_local_advection_vector
from src.step3.ops.forces import get_local_body_force
from src.step3.ops.gradient import compute_local_gradient_p
from src.step3.ops.laplacian import compute_local_laplacian_v_n
from src.step3.ops.scaling import get_dt_over_rho


def compute_local_predictor_step(block: StencilBlock) -> None:
    """
    Computes the intermediate star-velocity field v*.
    
    Formula: v* = v^n + (dt/rho) * (mu * lap(v^n) - rho * (v^n ⋅ ∇)v^n + F - grad(p^n))

    Compliance:
    - Rule 7: Fail-Fast math audit. If v* calculation results in NaN/Inf, 
      raises ArithmeticError to trigger Elasticity recovery.
    - Rule 9: Direct modification of FI.VX_STAR buffers via StencilBlock.
    """
    
    # 1. Local Operator calls (Physics gradients)
    lap_v = compute_local_laplacian_v_n(block)    
    adv_v = compute_local_advection_vector(block) 
    force = get_local_body_force(block)           
    grad_p = compute_local_gradient_p(block, field_id=FI.P) 
    
    # 2. Scaling factor (dt/rho)
    dt_over_rho = get_dt_over_rho(block)
    
    # 3. Retrieve current velocities from Foundation
    v_n = (
        block.center.get_field(FI.VX),
        block.center.get_field(FI.VY),
        block.center.get_field(FI.VZ)
    )
    
    # 4. Compute and audit v_star components
    star_fields = [FI.VX_STAR, FI.VY_STAR, FI.VZ_STAR]
    
    for i, field_id in enumerate(star_fields):
        # Calculate intermediate value
        v_star_val = v_n[i] + dt_over_rho * (
            block.mu * lap_v[i] - block.rho * adv_v[i] + force[i] - grad_p[i]
        )
        
        # Rule 7: Immediate Numerical Audit
        if not math.isfinite(v_star_val):
            # Provide high-res telemetry for GitHub Actions
            physics_context = (
                f"lap={lap_v[i]:.2e}, adv={adv_v[i]:.2e}, "
                f"force={force[i]:.2e}, grad_p={grad_p[i]:.2e}"
            )
            raise ArithmeticError(
                f"Predictor Divergence at {field_id.name}: value={v_star_val} | "
                f"Context: {physics_context}"
            )

        # Commit to the Trial (Star) buffer
        block.center.set_field(field_id, v_star_val)