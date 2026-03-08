# src/step3/rhie_chow.py

from .ops.laplacian import laplacian_p_n_plus_1


def compute_rhie_chow_term(p_n, dx, dy, dz, dt):
    """
    Computes: div(M_rc * p_n) = div(dt * grad(p_n)) = dt * laplacian(p_n)
    
    Returns:
    Scalar field (nx-2, ny-2, nz-2)
    """
    lap_p = laplacian_p_n_plus_1(p_n, dx, dy, dz)
    return dt * lap_p