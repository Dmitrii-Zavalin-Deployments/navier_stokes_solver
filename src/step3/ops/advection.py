# src/step3/ops/advection.py

import math

from src.common.field_schema import FI
from src.common.stencil_block import StencilBlock


def compute_local_advection(block: StencilBlock, field_id: FI) -> float:
    """
    Computes local (v^n ⋅ ∇) * field using schema-locked accessors.

    Formula: 
    u_c * (df/dx) + v_c * (df/dy) + w_c * (df/dz)
    
    Compliance:
    - Rule 7: Fail-Fast math audit. Raises ArithmeticError if gradients 
      or the dot product are non-finite.
    - Rule 9 (Hybrid Memory): Uses get_field(FI) for all buffer access.
    """
    
    # 1. Compute spatial derivatives (Central Differencing)
    # Accessing f_neighbors via explicit FI schema
    f_ip = block.i_plus.get_field(field_id)
    f_im = block.i_minus.get_field(field_id)
    f_jp = block.j_plus.get_field(field_id)
    f_jm = block.j_minus.get_field(field_id)
    f_kp = block.k_plus.get_field(field_id)
    f_km = block.k_minus.get_field(field_id)
    
    # Rule 7: Guard against division by zero in geometry
    # (Though usually caught in higher orchestrators, we keep it here for ops-level safety)
    try:
        df_dx = (f_ip - f_im) / (2.0 * block.dx)
        df_dy = (f_jp - f_jm) / (2.0 * block.dy)
        df_dz = (f_kp - f_km) / (2.0 * block.dz)
    except ZeroDivisionError as e:
        raise ValueError(f"Zero grid spacing detected at {block.center.i}, {block.center.j}") from e

    # 2. Compute cell-centered velocities
    u_c = block.center.get_field(FI.VX)
    v_c = block.center.get_field(FI.VY)
    w_c = block.center.get_field(FI.VZ)
    
    # 3. Assemble advection term: (v ⋅ ∇)f
    advection_val = (u_c * df_dx) + (v_c * df_dy) + (w_c * df_dz)

    # 4. Rule 7: Immediate Numerical Audit
    if not math.isfinite(advection_val):
        raise ArithmeticError(
            f"Advection divergence for {field_id.name}: val={advection_val} | "
            f"Gradients: [{df_dx:.2e}, {df_dy:.2e}, {df_dz:.2e}]"
            f"Derivatives: ({df_dx:.2e}, {df_dy:.2e}, {df_dz:.2e})"
        )

    return advection_val

def compute_local_advection_vector(block: StencilBlock) -> tuple[float, float, float]:
    """
    Computes the full advective term for the momentum equation (3D Vector).
    
    Note: Exceptions from compute_local_advection bubble up to 
    orchestrate_step3 for ElasticManager handling.
    """
    return (
        compute_local_advection(block, FI.VX),
        compute_local_advection(block, FI.VY),
        compute_local_advection(block, FI.VZ)
    )