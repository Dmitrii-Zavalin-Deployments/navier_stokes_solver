# src/step3/ops/divergence.py

from src.common.stencil_block import StencilBlock
from src.common.field_schema import FI

def compute_local_divergence_v_star(block: StencilBlock) -> float:
    """
    Computes the local scalar divergence ∇ ⋅ v* for the Pressure Poisson Equation.
    
    Formula:
    ∇ ⋅ v* \approx (u_{i+1} - u_{i-1}) / (2*dx) + 
                   (v_{j+1} - v_{j-1}) / (2*dy) + 
                   (w_{k+1} - w_{k-1}) / (2*dz)
    
    Compliance:
    - Uses schema-locked method get_field() to access foundation buffers (Rule 9).
    - Eliminates implicit property lookups, ensuring O(1) performance (Rule 0).
    """
    
    # Access intermediate velocity components from neighbors via Enum-locked mapping
    # This maintains the Hybrid Memory Foundation integrity.
    u_ip = block.i_plus.get_field(FI.VX_STAR)
    u_im = block.i_minus.get_field(FI.VX_STAR)
    
    v_jp = block.j_plus.get_field(FI.VY_STAR)
    v_jm = block.j_minus.get_field(FI.VY_STAR)
    
    w_kp = block.k_plus.get_field(FI.VZ_STAR)
    w_km = block.k_minus.get_field(FI.VZ_STAR)
    
    # Central difference: ∇ ⋅ v* \approx ∂u/∂x + ∂v/∂y + ∂w/∂z
    # Direct extraction of grid spacing from block (Hybrid Memory Wiring)
    div_x = (u_ip - u_im) / (2.0 * block.dx)
    div_y = (v_jp - v_jm) / (2.0 * block.dy)
    div_z = (w_kp - w_km) / (2.0 * block.dz)
    
    return div_x + div_y + div_z