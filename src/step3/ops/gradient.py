# src/step3/ops/gradient.py

from src.common.stencil_block import StencilBlock


def compute_local_gradient_p(block: StencilBlock, use_next: bool = False) -> tuple:
    """
    Computes the pressure gradient: ∇p = (dp/dx, dp/dy, dp/dz)
    
    Compliance:
    - Uses schema-locked property getters (e.g., .p, .p_next).
    - Accesses the Foundation buffer via the Cell object-pointer graph.
    """
    
    # 1. Select source field using schema-locked properties
    if not use_next:
        # These properties map internally to FI.P
        p_im, p_ip = block.i_minus.p, block.i_plus.p
        p_jm, p_jp = block.j_minus.p, block.j_plus.p
        p_km, p_kp = block.k_minus.p, block.k_plus.p
    else:
        # These properties map internally to FI.P_NEXT
        p_im, p_ip = block.i_minus.p_next, block.i_plus.p_next
        p_jm, p_jp = block.j_minus.p_next, block.j_plus.p_next
        p_km, p_kp = block.k_minus.p_next, block.k_plus.p_next
        
    # 2. Central difference: (dp/dx, dp/dy, dp/dz)
    grad_x = (p_ip - p_im) / (2.0 * block.dx)
    grad_y = (p_jp - p_jm) / (2.0 * block.dy)
    grad_z = (p_kp - p_km) / (2.0 * block.dz)
    
    return grad_x, grad_y, grad_z