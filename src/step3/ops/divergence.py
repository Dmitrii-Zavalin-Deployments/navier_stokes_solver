# src/step3/ops/divergence.py

import math

from src.common.field_schema import FI
from src.common.stencil_block import StencilBlock


def compute_local_divergence_v_star(block: StencilBlock) -> float:
    """
    Computes the local scalar divergence ∇ ⋅ v* for the Pressure Poisson Equation.
    
    Formula:
    ∇ ⋅ v* \approx (u_{i+1} - u_{i-1}) / (2*dx) + 
                   (v_{j+1} - v_{j-1}) / (2*dy) + 
                   (w_{k+1} - w_{k-1}) / (2*dz)
    
    Compliance:
    - Rule 7: Fail-Fast math audit. Raises ArithmeticError if star-velocities 
      or derivatives are non-finite, preventing PPE poisoning.
    - Rule 9: Uses schema-locked get_field() for foundation buffers.
    """
    
    # 1. Access intermediate velocity components (FI.VX_STAR, etc.)
    u_ip = block.i_plus.get_field(FI.VX_STAR)
    u_im = block.i_minus.get_field(FI.VX_STAR)
    
    v_jp = block.j_plus.get_field(FI.VY_STAR)
    v_jm = block.j_minus.get_field(FI.VY_STAR)
    
    w_kp = block.k_plus.get_field(FI.VZ_STAR)
    w_km = block.k_minus.get_field(FI.VZ_STAR)
    
    # 2. Central difference: ∂u/∂x + ∂v/∂y + ∂w/∂z
    # Rule 7: Guard against division by zero in geometry
    try:
        div_x = (u_ip - u_im) / (2.0 * block.dx)
        div_y = (v_jp - v_jm) / (2.0 * block.dy)
        div_z = (w_kp - w_km) / (2.0 * block.dz)
    except ZeroDivisionError as e:
        raise ValueError(f"Zero grid spacing at ({block.center.i}, {block.center.j})") from e
    
    divergence_val = div_x + div_y + div_z

    # 3. Rule 7: Numerical Integrity Audit
    if not math.isfinite(divergence_val):
        # We log the specific components to see which axis exploded
        raise ArithmeticError(
            f"Divergence explosion: val={divergence_val} | "
            f"Components: [dx:{div_x:.2e}, dy:{div_y:.2e}, dz:{div_z:.2e}]"
        )
    
    return divergence_val