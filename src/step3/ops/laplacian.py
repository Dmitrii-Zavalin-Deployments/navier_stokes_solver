# src/step3/ops/laplacian.py

from src.common.stencil_block import StencilBlock
from src.common.field_schema import FI

def compute_local_laplacian(block: StencilBlock, field_id: FI) -> float:
    """
    Computes the discrete Laplacian ∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z²
    for a given field_id using the Foundation buffer.
    
    Compliance:
    - Uses get_field(FI) for O(1) buffer access (Rule 0).
    - Single implementation for all fields prevents logic drift (Rule 8).
    """
    # Access neighbors via the wiring layer (StencilBlock) and Foundation (Buffer)
    f_c = block.center.get_field(field_id)
    
    f_ip, f_im = block.i_plus.get_field(field_id), block.i_minus.get_field(field_id)
    f_jp, f_jm = block.j_plus.get_field(field_id), block.j_minus.get_field(field_id)
    f_kp, f_km = block.k_plus.get_field(field_id), block.k_minus.get_field(field_id)
    
    return (
        (f_ip - 2.0 * f_c + f_im) / (block.dx**2) +
        (f_jp - 2.0 * f_c + f_jm) / (block.dy**2) +
        (f_kp - 2.0 * f_c + f_km) / (block.dz**2)
    )

def compute_local_laplacian_v_n(block: StencilBlock) -> tuple:
    """Computes Laplacian for primary velocity components."""
    return (
        compute_local_laplacian(block, FI.VX),
        compute_local_laplacian(block, FI.VY),
        compute_local_laplacian(block, FI.VZ)
    )

def compute_local_laplacian_p_next(block: StencilBlock) -> float:
    """Computes Laplacian for pressure p^{n+1}."""
    return compute_local_laplacian(block, FI.P_NEXT)