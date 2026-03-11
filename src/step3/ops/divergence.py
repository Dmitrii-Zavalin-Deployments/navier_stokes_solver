# src/step3/ops/divergence.py

from src.common.stencil_block import StencilBlock


def compute_local_divergence_v_star(block: StencilBlock) -> float:
    """
    Computes the local scalar divergence ∇ ⋅ v* for the Pressure Poisson Equation.
    
    Formula:
    ∇ ⋅ v* \approx (u_{i+1} - u_{i-1}) / (2*dx) + 
                   (v_{j+1} - v_{j-1}) / (2*dy) + 
                   (w_{k+1} - w_{k-1}) / (2*dz)
    
    Compliance:
    - Uses schema-locked property getters (e.g., .vx_star) to access
      intermediate velocity fields from the foundation buffer.
    """
    
    # Access intermediate velocity components from neighbors
    # Each access retrieves data directly from the foundation buffer via Cell.vx_star
    u_ip, u_im = block.i_plus.vx_star, block.i_minus.vx_star
    v_jp, v_jm = block.j_plus.vy_star, block.j_minus.vy_star
    w_kp, w_km = block.k_plus.vz_star, block.k_minus.vz_star
    
    # Central difference: ∇ ⋅ v* \approx ∂u/∂x + ∂v/∂y + ∂w/∂z
    div_x = (u_ip - u_im) / (2.0 * block.dx)
    div_y = (v_jp - v_jm) / (2.0 * block.dy)
    div_z = (w_kp - w_km) / (2.0 * block.dz)
    
    return div_x + div_y + div_z