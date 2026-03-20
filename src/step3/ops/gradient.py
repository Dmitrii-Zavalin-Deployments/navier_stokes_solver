# src/step3/ops/gradient.py

import math

from src.common.field_schema import FI
from src.common.stencil_block import StencilBlock


def compute_local_gradient_p(block: StencilBlock, field_id: FI = FI.P) -> tuple[float, float, float]:
    """
    Computes the pressure gradient: ∇p = (dp/dx, dp/dy, dp/dz)
    
    Compliance:
    - Rule 7: Fail-Fast math audit. If the pressure gradient is non-finite, 
      raises ArithmeticError to signal the Elasticity Manager.
    - Rule 9: Uses schema-locked field_id mapping for buffer access.
    """
    
    # 1. Access field values via explicit schema-locked lookup
    p_im = block.i_minus.get_field(field_id)
    p_ip = block.i_plus.get_field(field_id)
    
    p_jm = block.j_minus.get_field(field_id)
    p_jp = block.j_plus.get_field(field_id)
    
    p_km = block.k_minus.get_field(field_id)
    p_kp = block.k_plus.get_field(field_id)
        
    # 2. Central difference: (dp/dx, dp/dy, dp/dz)
    # Rule 7: Guard against division by zero in geometry
    try:
        grad_x = (p_ip - p_im) / (2.0 * block.dx)
        grad_y = (p_jp - p_jm) / (2.0 * block.dy)
        grad_z = (p_kp - p_km) / (2.0 * block.dz)
    except ZeroDivisionError:
        raise ValueError(f"Zero grid spacing at ({block.center.i}, {block.center.j})")
    
    grad = (grad_x, grad_y, grad_z)

    # 3. Rule 7: Numerical Integrity Audit
    # If the pressure field diverges, it manifests as a massive gradient.
    if not all(math.isfinite(g) for g in grad):
        raise ArithmeticError(
            f"Pressure Gradient explosion for {field_id.name}: {grad} | "
            f"Cell: ({block.center.i}, {block.center.j}, {block.center.k})"
        )
    
    return grad