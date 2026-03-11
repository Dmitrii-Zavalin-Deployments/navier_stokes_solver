# src/step3/predictor.py

from src.common.stencil_block import StencilBlock
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
    - Performs direct mutation of the underlying fields_buffer.
    - Each assignment (e.g., .vx_star = ...) triggers an in-place write 
      to the Foundation, ensuring zero heap reallocation during the time-loop.
    """
    
    # 1. Local Operator calls
    lap_v = compute_local_laplacian_v_n(block)    # (lap_u, lap_v, lap_w)
    adv_v = compute_local_advection_vector(block) # (adv_u, adv_v, adv_w)
    force = get_local_body_force(block)           # (Fx, Fy, Fz)
    grad_p = compute_local_gradient_p(block, use_next=False) 
    
    # 2. Scaling factor
    dt_over_rho = get_dt_over_rho(block)
    
    # 3. Direct mutation of the block's center cell state
    # These assignments write directly into the Foundation via the Cell's property setters.
    block.center.vx_star = block.center.vx + dt_over_rho * (
        block.mu * lap_v[0] - block.rho * adv_v[0] + force[0] - grad_p[0]
    )
    block.center.vy_star = block.center.vy + dt_over_rho * (
        block.mu * lap_v[1] - block.rho * adv_v[1] + force[1] - grad_p[1]
    )
    block.center.vz_star = block.center.vz + dt_over_rho * (
        block.mu * lap_v[2] - block.rho * adv_v[2] + force[2] - grad_p[2]
    )