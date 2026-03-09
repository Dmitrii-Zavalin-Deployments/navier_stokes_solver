# src/step3/ops/advection.py

from src.common.stencil_block import StencilBlock


def compute_local_advection(block: StencilBlock, field: str) -> float:
    """
    Computes local (v^n ⋅ ∇) * field for a specific component (field).
    
    Formula: 
    u_c * df/dx + v_c * df/dy + w_c * df/dz
    
    Args:
        block: The StencilBlock instance.
        field: The attribute name in Cell to advect (e.g., 'vx', 'vy', 'vz').
    """
    # 1. Access components for central difference (df/dx, df/dy, df/dz)
    # Using getattr to dynamically fetch the attribute (vx, vy, or vz) from neighbors
    f_ip, f_im = getattr(block.i_plus, field), getattr(block.i_minus, field)
    f_jp, f_jm = getattr(block.j_plus, field), getattr(block.j_minus, field)
    f_kp, f_km = getattr(block.k_plus, field), getattr(block.k_minus, field)
    
    df_dx = (f_ip - f_im) / (2.0 * block.dx)
    df_dy = (f_jp - f_jm) / (2.0 * block.dy)
    df_dz = (f_kp - f_km) / (2.0 * block.dz)
    
    # 2. Compute cell-centered velocities (average of neighbors)
    u_c = (block.i_plus.vx + block.i_minus.vx) / 2.0
    v_c = (block.j_plus.vy + block.j_minus.vy) / 2.0
    w_c = (block.k_plus.vz + block.k_minus.vz) / 2.0
    
    # 3. Assemble advection term: (v ⋅ ∇)f
    return u_c * df_dx + v_c * df_dy + w_c * df_dz

def compute_local_advection_vector(block: StencilBlock) -> tuple:
    """
    Computes the full advective term for the momentum equation:
    (v^n ⋅ ∇) * v^n = ((v ⋅ ∇)u, (v ⋅ ∇)v, (v ⋅ ∇)w)
    """
    return (
        compute_local_advection(block, 'vx'),
        compute_local_advection(block, 'vy'),
        compute_local_advection(block, 'vz')
    )