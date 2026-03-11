# src/step3/predictor.py

from src.common.stencil_block import StencilBlock
from src.common.field_schema import FI
from src.step3.ops.advection import compute_local_advection_vector
from src.step3.ops.forces import get_local_body_force
from src.step3.ops.gradient import compute_local_gradient_p
from src.step3.ops.laplacian import compute_local_laplacian_v_n
from src.step3.ops.scaling import get_dt_over_rho


def compute_local_predictor_step(block: StencilBlock) -> None:
    """
    Computes the intermediate velocity v* = (u*, v*, w*).
    
    Formula: v* = v^n + (dt/rho) * (mu * lap(v^n) - rho * (v^n ⋅ ∇)v^n + F - grad(p^n))

    Compliance:
    - Rule 9 (Hybrid Memory): Uses set_field(FI) for direct Foundation modification.
    - Rule 0 (Performance): Avoids property-setter overhead by using buffer-locked index mapping.
    """
    
    # 1. Local Operator calls
    lap_v = compute_local_laplacian_v_n(block)    
    adv_v = compute_local_advection_vector(block) 
    force = get_local_body_force(block)           
    grad_p = compute_local_gradient_p(block, field_id=FI.P) 
    
    # 2. Scaling factor
    dt_over_rho = get_dt_over_rho(block)
    
    # 3. Compute and store in-place using FI schema
    # Retrieve current velocities (FI.VX, etc.) from Foundation
    v_n = (
        block.center.get_field(FI.VX),
        block.center.get_field(FI.VY),
        block.center.get_field(FI.VZ)
    )
    
    # Apply predictor update to Star buffer (FI.VX_STAR, etc.)
    for i, field_id in enumerate([FI.VX_STAR, FI.VY_STAR, FI.VZ_STAR]):
        v_star_val = v_n[i] + dt_over_rho * (
            block.mu * lap_v[i] - block.rho * adv_v[i] + force[i] - grad_p[i]
        )
        block.center.set_field(field_id, v_star_val)