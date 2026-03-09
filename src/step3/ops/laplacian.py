# src/step3/ops/laplacian.py

from src.common.stencil_block import StencilBlock

def compute_local_laplacian_v_n(block: StencilBlock) -> tuple:
    """
    Computes the discrete Laplacian ∇²v^n = (∇²u, ∇²v, ∇²w) for the Predictor Step.
    
    Formula (centered at i,j,k):
    ∇²f \approx (f_{i+1} - 2f_{i,j,k} + f_{i-1}) / dx² + ... (for y and z)
    """
    
    def compute_comp(f_ip, f_im, f_jp, f_jm, f_kp, f_km, f_c):
        return (
            (f_ip - 2.0 * f_c + f_im) / (block.dx**2) +
            (f_jp - 2.0 * f_c + f_jm) / (block.dy**2) +
            (f_kp - 2.0 * f_c + f_km) / (block.dz**2)
        )
    
    # 1. Laplacian for U (vx)
    lap_u = compute_comp(
        block.i_plus.vx, block.i_minus.vx,
        block.j_plus.vx, block.j_minus.vx,
        block.k_plus.vx, block.k_minus.vx, block.center.vx
    )
    
    # 2. Laplacian for V (vy)
    lap_v = compute_comp(
        block.i_plus.vy, block.i_minus.vy,
        block.j_plus.vy, block.j_minus.vy,
        block.k_plus.vy, block.k_minus.vy, block.center.vy
    )
    
    # 3. Laplacian for W (vz)
    lap_w = compute_comp(
        block.i_plus.vz, block.i_minus.vz,
        block.j_plus.vz, block.j_minus.vz,
        block.k_plus.vz, block.k_minus.vz, block.center.vz
    )
    
    return lap_u, lap_v, lap_w

def compute_local_laplacian_p_next(block: StencilBlock) -> float:
    """
    Computes ∇²p^{n+1} for the Pressure Poisson Equation.
    """
    p_c = block.center.p_next
    
    return (
        (block.i_plus.p_next - 2.0 * p_c + block.i_minus.p_next) / (block.dx**2) +
        (block.j_plus.p_next - 2.0 * p_c + block.j_minus.p_next) / (block.dy**2) +
        (block.k_plus.p_next - 2.0 * p_c + block.k_minus.p_next) / (block.dz**2)
    )