# src/step3/ppe_solver.py

from src.common.stencil_block import StencilBlock
from src.common.field_schema import FI
from src.step3.ops.divergence import compute_local_divergence_v_star
from src.step3.ops.scaling import get_rho_over_dt


def solve_pressure_poisson_step(block: StencilBlock, omega: float) -> float:
    """
    Consolidated PPE Solver using SOR iteration.
    
    Compliance:
    - Rule 9: Performs in-place updates via schema-locked accessors.
    - Rule 4: Uses block-local geometry and configuration contexts.
    """
    # 1. Geometry Setup
    dx2, dy2, dz2 = block.dx**2, block.dy**2, block.dz**2
    stencil_denom = 2.0 * (1.0/dx2 + 1.0/dy2 + 1.0/dz2)
    
    # 2. Compute Rhie-Chow Stabilization
    # Access P via FI.P (Foundation Buffer)
    lap_p_n = (
        (block.i_plus.get_field(FI.P) - 2.0 * block.center.get_field(FI.P) + block.i_minus.get_field(FI.P)) / dx2 +
        (block.j_plus.get_field(FI.P) - 2.0 * block.center.get_field(FI.P) + block.j_minus.get_field(FI.P)) / dy2 +
        (block.k_plus.get_field(FI.P) - 2.0 * block.center.get_field(FI.P) + block.k_minus.get_field(FI.P)) / dz2
    )
    rhie_chow_term = block.dt * lap_p_n
    
    # 3. Compute RHS
    div_v_star = compute_local_divergence_v_star(block)
    rhs = get_rho_over_dt(block) * (div_v_star - rhie_chow_term)
    
    # 4. SOR Update (using FI.P_NEXT)
    sum_neighbors = (
        (block.i_plus.get_field(FI.P_NEXT) + block.i_minus.get_field(FI.P_NEXT)) / dx2 +
        (block.j_plus.get_field(FI.P_NEXT) + block.j_minus.get_field(FI.P_NEXT)) / dy2 +
        (block.k_plus.get_field(FI.P_NEXT) + block.k_minus.get_field(FI.P_NEXT)) / dz2
    )
    
    p_old = block.center.get_field(FI.P_NEXT)
    p_new = (1.0 - omega) * p_old + (omega / stencil_denom) * (sum_neighbors - rhs)
    
    # Direct write-back via schema-locked accessor
    block.center.set_field(FI.P_NEXT, p_new)
    
    return abs(p_new - p_old)