# src/step3/predictor.py

from src.common.stencil_block import StencilBlock
from src.step3.ops.advection import compute_local_advection_vector
from src.step3.ops.forces import get_local_body_force
from src.step3.ops.gradient import compute_local_gradient_p
from src.step3.ops.laplacian import compute_local_laplacian_v_n
from src.step3.ops.scaling import get_dt_over_rho

def compute_local_predictor_step(block: StencilBlock) -> tuple:
    """
    Computes the intermediate velocity v* = (u*, v*, w*) for a single StencilBlock.
    Formula: v* = v^n + (dt/rho) * (mu * lap(v^n) - rho * (v^n ⋅ ∇)v^n + F + grad(p^n))
    """
    
    # 1. Access components
    mu = block.mu
    rho = block.rho
    
    # 2. Local Operator calls (The Logic-Layer)
    lap_v = compute_local_laplacian_v_n(block)    # (lap_u, lap_v, lap_w)
    adv_v = compute_local_advection_vector(block) # (adv_u, adv_v, adv_w)
    force = get_local_body_force(block)           # (Fx, Fy, Fz)
    grad_p = compute_local_gradient_p(block, use_next=False) # (dpdx, dpdy, dpdz)
    
    # 3. Scaling factor
    dt_over_rho = get_dt_over_rho(block)
    
    # 4. Compute components of v* (u*, v*, w*)
    # v_star_i = v_n_i + (dt/rho) * (mu * lap_i - rho * adv_i + F_i + grad_i)
    u_star = block.center.vx + dt_over_rho * (mu * lap_v[0] - rho * adv_v[0] + force[0] + grad_p[0])
    v_star = block.center.vy + dt_over_rho * (mu * lap_v[1] - rho * adv_v[1] + force[1] + grad_p[1])
    w_star = block.center.vz + dt_over_rho * (mu * lap_v[2] - rho * adv_v[2] + force[2] + grad_p[2])
    
    return u_star, v_star, w_star