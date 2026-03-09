# src/step3/corrector.py

from src.common.stencil_block import StencilBlock
from src.step3.ops.gradient import compute_local_gradient_p
from src.step3.ops.scaling import get_dt_over_rho


def apply_local_velocity_correction(block: StencilBlock) -> None:
    """
    Projects the intermediate velocity field v* onto a divergence-free space.
    Formula: v^{n+1} = v^* - (dt/rho) * grad(p^{n+1})
    
    Directly updates block.center.vx, vy, vz to their n+1 values.
    """
    
    # 1. Compute the pressure gradient at p^{n+1}
    # We pass use_next=True to access p_next values from neighbors
    grad_p = compute_local_gradient_p(block, use_next=True)
    
    # 2. Scaling factor (dt/rho)
    scaling = get_dt_over_rho(block)
    
    # 3. Apply velocity correction in-place
    # v^{n+1} = v^* - (scaling * ∇p^{n+1})
    block.center.vx = block.center.vx_star - (scaling * grad_p[0])
    block.center.vy = block.center.vy_star - (scaling * grad_p[1])
    block.center.vz = block.center.vz_star - (scaling * grad_p[2])