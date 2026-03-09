# src/step3/ops/sor_stencil.py

from src.common.stencil_block import StencilBlock

def perform_sor_iteration_step(block: StencilBlock, rhs: float, omega: float) -> float:
    """
    Performs one SOR iteration step for a single StencilBlock.
    
    Formula:
    p_new = (1 - omega) * p_old + (omega / stencil_denom) * (Sum_neighbors - RHS)
    
    Args:
        block: StencilBlock with p_next values.
        rhs: The local divergence source term at this cell.
        omega: Relaxation factor (usually 1.0 < omega < 2.0).
        
    Returns:
        float: The updated pressure value at the center cell.
    """
    
    # 1. Access neighbors (p_next)
    p_ip, p_im = block.i_plus.p_next, block.i_minus.p_next
    p_jp, p_jm = block.j_plus.p_next, block.j_minus.p_next
    p_kp, p_km = block.k_plus.p_next, block.k_minus.p_next
    
    # 2. Precomputed coefficients from your Section 6.1
    dx2, dy2, dz2 = block.dx**2, block.dy**2, block.dz**2
    stencil_denom = 2.0 * (1.0/dx2 + 1.0/dy2 + 1.0/dz2)
    
    # 3. Sum of neighbors
    sum_neighbors = (p_ip + p_im)/dx2 + (p_jp + p_jm)/dy2 + (p_kp + p_km)/dz2
    
    # 4. SOR Update formula
    # p_new = (1-w)p_old + (w/denom) * (sum_neighbors - rhs)
    p_old = block.center.p_next
    p_new = (1.0 - omega) * p_old + (omega / stencil_denom) * (sum_neighbors - rhs)
    
    # Update the block's center cell with the new pressure
    block.center.p_next = p_new
    
    return p_new