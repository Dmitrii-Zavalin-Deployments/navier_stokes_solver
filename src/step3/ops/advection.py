# src/step3/ops/advection.py

from src.common.stencil_block import StencilBlock
from src.common.field_schema import FI

def compute_local_advection(block: StencilBlock, field_id: FI) -> float:
    """
    Computes local (v^n ⋅ ∇) * field using schema-locked accessors.

    Formula: 
    u_c * (df/dx) + v_c * (df/dy) + w_c * (df/dz)
    
    Compliance:
    - Rule 9 (Hybrid Memory): Uses get_field(FI) for all buffer access.
    - Rule 0 (Performance): Direct indexing into Foundation buffers.
    """
    
    # 1. Compute spatial derivatives (Central Differencing)
    # Accessing f_neighbors via explicit FI schema
    f_ip = block.i_plus.get_field(field_id)
    f_im = block.i_minus.get_field(field_id)
    f_jp = block.j_plus.get_field(field_id)
    f_jm = block.j_minus.get_field(field_id)
    f_kp = block.k_plus.get_field(field_id)
    f_km = block.k_minus.get_field(field_id)
    
    df_dx = (f_ip - f_im) / (2.0 * block.dx)
    df_dy = (f_jp - f_jm) / (2.0 * block.dy)
    df_dz = (f_kp - f_km) / (2.0 * block.dz)
    
    # 2. Compute cell-centered velocities
    # Accessing velocity Foundation buffers via FI schema constants
    u_c = (block.i_plus.get_field(FI.VX) + block.i_minus.get_field(FI.VX)) / 2.0
    v_c = (block.j_plus.get_field(FI.VY) + block.j_minus.get_field(FI.VY)) / 2.0
    w_c = (block.k_plus.get_field(FI.VZ) + block.k_minus.get_field(FI.VZ)) / 2.0
    
    # 3. Assemble advection term: (v ⋅ ∇)f
    return u_c * df_dx + v_c * df_dy + w_c * df_dz

def compute_local_advection_vector(block: StencilBlock) -> tuple:
    """
    Computes the full advective term for the momentum equation.
    """
    return (
        compute_local_advection(block, FI.VX),
        compute_local_advection(block, FI.VY),
        compute_local_advection(block, FI.VZ)
    )