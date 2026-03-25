# src/step3/ppe_solver.py

import logging
import numpy as np

from src.common.field_schema import FI
from src.common.stencil_block import StencilBlock
from src.step3.ops.divergence import compute_local_divergence_v_star
from src.step3.ops.scaling import get_rho_over_dt

logger = logging.getLogger("Solver.PPE")

def solve_pressure_poisson_step(
    block: StencilBlock, 
    divergence_threshold: float, 
    omega: float
) -> float:
    """
    Consolidated PPE Solver using SOR iteration with Fail-Fast Math.
    
    Compliance:
    - Rule 7: Immediate math audit (Atomic Verification).
    - Rule 9: Performs in-place updates via schema-locked accessors.
    """
    # 1. Geometry Setup
    dx2, dy2, dz2 = block.dx**2, block.dy**2, block.dz**2
    stencil_denom = 2.0 * (1.0/dx2 + 1.0/dy2 + 1.0/dz2)
    
    # 2. Compute Rhie-Chow Stabilization
    # FORENSIC: We wrap the stencil in float() or .item() because adding 
    # multiple get_field() results often triggers NumPy array promotion.
    lap_p_n = (
        (block.i_plus.get_field(FI.P) - 2.0 * block.center.get_field(FI.P) + block.i_minus.get_field(FI.P)) / dx2 +
        (block.j_plus.get_field(FI.P) - 2.0 * block.center.get_field(FI.P) + block.j_minus.get_field(FI.P)) / dy2 +
        (block.k_plus.get_field(FI.P) - 2.0 * block.center.get_field(FI.P) + block.k_minus.get_field(FI.P)) / dz2
    )
    
    # Force scalar context
    if hasattr(lap_p_n, "item"): lap_p_n = lap_p_n.item()
    
    rhie_chow_term = block.dt * lap_p_n
    
    # 3. Compute RHS
    div_v_star = compute_local_divergence_v_star(block)
    if hasattr(div_v_star, "item"): div_v_star = div_v_star.item()
    
    # --- RULE 7: ATOMIC DIVERGENCE GATE ---
    if not np.isfinite(div_v_star):
        logger.error(f"PPE MATH ERROR: Non-finite divergence in block {block.id}")
        raise ArithmeticError(f"NaN detected in divergence source term: {div_v_star}")

    rho_over_dt = get_rho_over_dt(block)
    if hasattr(rho_over_dt, "item"): rho_over_dt = rho_over_dt.item()
    
    rhs = rho_over_dt * (div_v_star - rhie_chow_term)
    
    # 4. SOR Update (Phase 2: Solve & Correct)
    sum_neighbors = (
        (block.i_plus.get_field(FI.P_NEXT) + block.i_minus.get_field(FI.P_NEXT)) / dx2 +
        (block.j_plus.get_field(FI.P_NEXT) + block.j_minus.get_field(FI.P_NEXT)) / dy2 +
        (block.k_plus.get_field(FI.P_NEXT) + block.k_minus.get_field(FI.P_NEXT)) / dz2
    )
    if hasattr(sum_neighbors, "item"): sum_neighbors = sum_neighbors.item()
    
    p_old = block.center.get_field(FI.P_NEXT)
    if hasattr(p_old, "item"): p_old = p_old.item()
    
    # --- RULE 7: PRE-UPDATE AUDIT ---
    if not np.isfinite(p_old) or abs(p_old) > divergence_threshold:
        logger.error(f"PPE CRITICAL: Poisoned p_old in block {block.id} | Val: {p_old:.2e}")
        raise ArithmeticError(f"Poisoned Pressure Trial detected at {block.id}")
    
    # 5. Calculate Trial Pressure via SOR Formula
    p_new = (1.0 - omega) * p_old + (omega / stencil_denom) * (sum_neighbors - rhs)
    
    # --- FINAL DNA CHECK ---
    # This is the "Root Cause" logger. It will tell us if p_new is still an array.
    if hasattr(p_new, "shape") and p_new.shape != ():
        logger.warning(f"DNA AUDIT: p_new leaked as array {p_new.shape} in block {block.id}. Correcting...")
        p_new = p_new.item()

    if not np.isfinite(p_new):
        logger.error(f"PPE MATH ERROR: Non-finite p_new in block {block.id}")
        raise ArithmeticError("Non-finite pressure generated in SOR step")
    
    # Rule 0: Robust delta calculation
    delta = float(abs(p_new - p_old))
    
    # 6. Direct write-back to foundation (Now guaranteed to be a scalar)
    block.center.set_field(FI.P_NEXT, p_new)
    
    return delta