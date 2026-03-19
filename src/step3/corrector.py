# src/step3/corrector.py

from src.common.field_schema import FI
from src.common.stencil_block import StencilBlock
from src.step3.ops.gradient import compute_local_gradient_p
from src.step3.ops.scaling import get_dt_over_rho


def apply_local_velocity_correction(block: StencilBlock) -> None:
    """
    Projects the intermediate velocity field v* onto a divergence-free space
    by modifying the Foundation buffer in-place.
    
    Formula: v^{n+1} = v^* - (dt/rho) * grad(p^{n+1})
    
    Returns:
        None: Operation is performed in-place on block.fields_buffer.
    """
    
    # 1. Compute the pressure gradient at p^{n+1} (FI.P_NEXT)
    grad_p = compute_local_gradient_p(block, field_id=FI.P_NEXT)
    
    # 2. Scaling factor (dt/rho)
    scaling = get_dt_over_rho(block)
    
    # 3. Apply velocity correction in-place
    # We update the primary velocity buffer (FI.VX, FI.VY, FI.VZ)
    # using values retrieved from the intermediate star-buffer.
    
    v_star = (
        block.center.get_field(FI.VX_STAR),
        block.center.get_field(FI.VY_STAR),
        block.center.get_field(FI.VZ_STAR)
    )
    
    block.center.set_field(FI.VX_STAR, v_star[0] - (scaling * grad_p[0]))
    block.center.set_field(FI.VY_STAR, v_star[1] - (scaling * grad_p[1]))
    block.center.set_field(FI.VZ_STAR, v_star[2] - (scaling * grad_p[2]))