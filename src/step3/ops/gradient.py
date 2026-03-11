# src/step3/ops/gradient.py

from src.common.stencil_block import StencilBlock
from src.common.field_schema import FI

def compute_local_gradient_p(block: StencilBlock, field_id: FI = FI.P) -> tuple:
    """
    Computes the pressure gradient: ∇p = (dp/dx, dp/dy, dp/dz)
    
    Compliance:
    - Uses schema-locked field_id mapping (Rule 9).
    - Eliminates conditional branch overhead (Rule 0).
    """
    
    # 1. Access field values via explicit schema-locked lookup
    # This keeps the logic clean and performance-oriented by avoiding 
    # branches inside the inner loop (Rule 0).
    p_im = block.i_minus.get_field(field_id)
    p_ip = block.i_plus.get_field(field_id)
    
    p_jm = block.j_minus.get_field(field_id)
    p_jp = block.j_plus.get_field(field_id)
    
    p_km = block.k_minus.get_field(field_id)
    p_kp = block.k_plus.get_field(field_id)
        
    # 2. Central difference: (dp/dx, dp/dy, dp/dz)
    grad_x = (p_ip - p_im) / (2.0 * block.dx)
    grad_y = (p_jp - p_jm) / (2.0 * block.dy)
    grad_z = (p_kp - p_km) / (2.0 * block.dz)
    
    return grad_x, grad_y, grad_z