# src/step3/ppe_solver.py

from src.common.stencil_block import StencilBlock
from src.step3.ops.divergence import compute_local_divergence_v_star
from src.step3.ops.scaling import get_rho_over_dt


def solve_pressure_poisson_step(block: StencilBlock, omega: float) -> float:
    """
    Consolidated PPE Solver using SOR iteration.
    
    Compliance:
    - Performs in-place SOR updates directly on the Foundation via 
      schema-locked property setters (e.g., .p_next).
    - Ensures zero memory allocation during iteration by leveraging 
      the pre-allocated StencilBlock pointer graph.
    """
    # 1. Geometry Setup
    dx2, dy2, dz2 = block.dx**2, block.dy**2, block.dz**2
    stencil_denom = 2.0 * (1.0/dx2 + 1.0/dy2 + 1.0/dz2)
    
    # 2. Compute Rhie-Chow Stabilization (dt * lap(p^n))
    # Properties access the Foundation's P column (FI.P)
    lap_p_n = (
        (block.i_plus.p - 2.0 * block.center.p + block.i_minus.p) / dx2 +
        (block.j_plus.p - 2.0 * block.center.p + block.j_minus.p) / dy2 +
        (block.k_plus.p - 2.0 * block.center.p + block.k_minus.p) / dz2
    )
    rhie_chow_term = block.dt * lap_p_n
    
    # 3. Compute RHS
    rho_over_dt = get_rho_over_dt(block)
    div_v_star = compute_local_divergence_v_star(block)
    rhs = rho_over_dt * (div_v_star - rhie_chow_term)
    
    # 4. SOR Update
    # Properties access the Foundation's P_NEXT column (FI.P_NEXT)
    sum_neighbors = (
        (block.i_plus.p_next + block.i_minus.p_next) / dx2 +
        (block.j_plus.p_next + block.j_minus.p_next) / dy2 +
        (block.k_plus.p_next + block.k_minus.p_next) / dz2
    )
    
    p_old = block.center.p_next
    p_new = (1.0 - omega) * p_old + (omega / stencil_denom) * (sum_neighbors - rhs)
    
    # Direct write-back to the Foundation via property setter
    block.center.p_next = p_new
    
    return abs(p_new - p_old)