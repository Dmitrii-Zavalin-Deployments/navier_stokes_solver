# src/step3/ppe_solver.py

import numpy as np
import logging
from src.common.field_schema import FI
from src.common.stencil_block import StencilBlock
from src.step3.ops.divergence import compute_local_divergence_v_star
from src.step3.ops.scaling import get_rho_over_dt

logger = logging.getLogger(__name__)

def solve_pressure_poisson_step(block: StencilBlock, omega: float) -> float:
    """
    Consolidated PPE Solver using SOR iteration with Fail-Fast Math.
    
    Compliance:
    - Rule 7: Immediate math audit. If p_new or delta is non-finite, 
      raises ArithmeticError to signal Elasticity Manager for a retry.
    - Rule 9: Performs in-place updates via schema-locked accessors.
    """
    # 1. Geometry Setup
    dx2, dy2, dz2 = block.dx**2, block.dy**2, block.dz**2
    stencil_denom = 2.0 * (1.0/dx2 + 1.0/dy2 + 1.0/dz2)
    
    # 2. Compute Rhie-Chow Stabilization (Access FI.P Foundation - Stable State)
    # This term is dt-scaled, so it must be physically consistent with t_n
    lap_p_n = (
        (block.i_plus.get_field(FI.P) - 2.0 * block.center.get_field(FI.P) + block.i_minus.get_field(FI.P)) / dx2 +
        (block.j_plus.get_field(FI.P) - 2.0 * block.center.get_field(FI.P) + block.j_minus.get_field(FI.P)) / dy2 +
        (block.k_plus.get_field(FI.P) - 2.0 * block.center.get_field(FI.P) + block.k_minus.get_field(FI.P)) / dz2
    )
    rhie_chow_term = block.dt * lap_p_n
    
    # 3. Compute RHS
    div_v_star = compute_local_divergence_v_star(block)
    rho_over_dt = get_rho_over_dt(block)
    rhs = rho_over_dt * (div_v_star - rhie_chow_term)
    
    # 4. SOR Update (using FI.P_NEXT Trial Buffer)
    sum_neighbors = (
        (block.i_plus.get_field(FI.P_NEXT) + block.i_minus.get_field(FI.P_NEXT)) / dx2 +
        (block.j_plus.get_field(FI.P_NEXT) + block.j_minus.get_field(FI.P_NEXT)) / dy2 +
        (block.k_plus.get_field(FI.P_NEXT) + block.k_minus.get_field(FI.P_NEXT)) / dz2
    )
    
    p_old = block.center.get_field(FI.P_NEXT)

    # --- RULE 7: PRE-UPDATE AUDIT ---
    # Catching the "1/dt slingshot" before it writes to the buffer
    if not np.isfinite(p_old) or abs(p_old) > 1e12:
        logger.error(f"PPE CRITICAL: Poisoned p_old detected in block {block.id} | "
                     f"Value: {p_old:.4e} | dt: {block.dt:.4e}")
        raise ArithmeticError(f"Poisoned Pressure Trial: {p_old}")

    # 5. Calculate Trial Pressure
    p_new = (1.0 - omega) * p_old + (omega / stencil_denom) * (sum_neighbors - rhs)
    
    # --- RULE 7: POST-UPDATE AUDIT ---
    if not np.isfinite(p_new):
        logger.error(f"PPE MATH ERROR: Non-finite p_new in block {block.id} | "
                     f"rhs: {rhs:.4e} | div: {div_v_star:.4e}")
        raise ArithmeticError("Non-finite pressure generated in SOR step")

    delta = abs(p_new - p_old)
    
    # 6. Direct write-back via schema-locked accessor
    block.center.set_field(FI.P_NEXT, p_new)
    
    return delta