# src/step3/ops/gradient.py

from src.common.stencil_block import StencilBlock


def compute_local_gradient_p(block: StencilBlock, use_next: bool = False) -> tuple:
    """
    Computes the pressure gradient: ∇p = (dp/dx, dp/dy, dp/dz)
    
    Formula:
    ∇p \approx ((p_{i+1} - p_{i-1}) / 2dx, ...)
    
    Args:
        block: The StencilBlock containing neighbor references.
        use_next: If True, uses p^{n+1}; if False, uses p^n.
    """
    
    # 1. Select source field
    if not use_next:
        p_im, p_ip = block.i_minus.p, block.i_plus.p
        p_jm, p_jp = block.j_minus.p, block.j_plus.p
        p_km, p_kp = block.k_minus.p, block.k_plus.p
    else:
        p_im, p_ip = block.i_minus.p_next, block.i_plus.p_next
        p_jm, p_jp = block.j_minus.p_next, block.j_plus.p_next
        p_km, p_kp = block.k_minus.p_next, block.k_plus.p_next
        
    # 2. Central difference: (dp/dx, dp/dy, dp/dz)
    # NO negative sign here! We return exactly ∇p.
    grad_x = (p_ip - p_im) / (2.0 * block.dx)
    grad_y = (p_jp - p_jm) / (2.0 * block.dy)
    grad_z = (p_kp - p_km) / (2.0 * block.dz)
    
    return grad_x, grad_y, grad_z