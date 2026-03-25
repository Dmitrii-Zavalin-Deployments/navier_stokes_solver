# src/step3/ops/laplacian.py


from src.common.field_schema import FI
from src.common.stencil_block import StencilBlock


def compute_local_laplacian(block: StencilBlock, field_id: FI) -> float:
    """
    Computes the discrete Laplacian ∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z²
    
    Compliance:
    - Rule 7: Fail-Fast math audit. Catching division by zero or non-finite 
      results immediately to prevent "poisoning" the predictor/solver.
    - Rule 8: Centralized logic prevents drift between velocity and pressure ops.
    """
    # 1. Access center and neighbors via Foundation schema (Rule 9)
    f_c = block.center.get_field(field_id)
    
    f_ip, f_im = block.i_plus.get_field(field_id), block.i_minus.get_field(field_id)
    f_jp, f_jm = block.j_plus.get_field(field_id), block.j_minus.get_field(field_id)
    f_kp, f_km = block.k_plus.get_field(field_id), block.k_minus.get_field(field_id)
    
    # 2. Geometry Setup (Rule 4: SSoT from block)
    dx2, dy2, dz2 = block.dx**2, block.dy**2, block.dz**2

    # 3. Discrete Laplacian Calculation
    lap_val = (
        (f_ip - 2.0 * f_c + f_im) / dx2 +
        (f_jp - 2.0 * f_c + f_jm) / dy2 +
        (f_kp - 2.0 * f_c + f_km) / dz2
    )

    return lap_val

def compute_local_laplacian_v_n(block: StencilBlock) -> tuple[float, float, float]:
    """Computes Laplacian for primary velocity components (v^n)."""
    return (
        compute_local_laplacian(block, FI.VX),
        compute_local_laplacian(block, FI.VY),
        compute_local_laplacian(block, FI.VZ)
    )

def compute_local_laplacian_p_next(block: StencilBlock) -> float:
    """Computes Laplacian for trial pressure p^{n+1}."""
    return compute_local_laplacian(block, FI.P_NEXT)