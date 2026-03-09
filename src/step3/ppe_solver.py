# src/step3/ppe_solver.py

import json
from pathlib import Path

from src.common.stencil_block import StencilBlock
from src.step3.ops.divergence import compute_local_divergence_v_star
from src.step3.ops.scaling import get_rho_over_dt


def _load_ppe_config() -> dict:
    config_path = Path(__file__).resolve().parents[2] / "config.json"
    with open(config_path) as f:
        return json.load(f)["solver_settings"]

def solve_pressure_poisson_step(block: StencilBlock) -> float:
    """
    Consolidated PPE Solver with integrated Rhie-Chow stabilization.
    Uses the 7-point Laplacian stencil to perform an in-place SOR update.
    """
    cfg = _load_ppe_config()
    omega = cfg["ppe_omega"]
    
    # 1. Geometry Setup (The 7-point Stencil denominator)
    dx2, dy2, dz2 = block.dx**2, block.dy**2, block.dz**2
    stencil_denom = 2.0 * (1.0/dx2 + 1.0/dy2 + 1.0/dz2)
    
    # 2. Compute Rhie-Chow Stabilization Term (dt * lap(p^n))
    # We use block.center.p to represent p^n (the stable previous pressure field)
    lap_p_n = (
        (block.i_plus.p - 2.0 * block.center.p + block.i_minus.p) / dx2 +
        (block.j_plus.p - 2.0 * block.center.p + block.j_minus.p) / dy2 +
        (block.k_plus.p - 2.0 * block.center.p + block.k_minus.p) / dz2
    )
    rhie_chow_term = block.dt * lap_p_n
    
    # 3. Compute RHS (Stabilized)
    # RHS = (rho/dt) * (div(v*) - Rhie_Chow_Term)
    rho_over_dt = get_rho_over_dt(block)
    div_v_star = compute_local_divergence_v_star(block)
    rhs = rho_over_dt * (div_v_star - rhie_chow_term)
    
    # 4. Sum of Neighbors (p^{n+1})
    sum_neighbors = (
        (block.i_plus.p_next + block.i_minus.p_next) / dx2 +
        (block.j_plus.p_next + block.j_minus.p_next) / dy2 +
        (block.k_plus.p_next + block.k_minus.p_next) / dz2
    )
    
    # 5. SOR Update (In-place mutation)
    p_old = block.center.p_next
    p_new = (1.0 - omega) * p_old + (omega / stencil_denom) * (sum_neighbors - rhs)
    
    block.center.p_next = p_new
    
    return abs(p_new - p_old)