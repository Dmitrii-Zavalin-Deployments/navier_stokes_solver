# src/step3/ops/laplacian.py
import numpy as np

def laplacian_v_n(v_n, dx, dy, dz):
    """
    Computes \nabla^2 \vec{v}^n for the Predictor Step.
    Returns (lap_u, lap_v, lap_w).
    """
    def compute(f):
        return (
            (f[2:, 1:-1, 1:-1] - 2*f[1:-1, 1:-1, 1:-1] + f[:-2, 1:-1, 1:-1]) / dx**2 +
            (f[1:-1, 2:, 1:-1] - 2*f[1:-1, 1:-1, 1:-1] + f[1:-1, :-2, 1:-1]) / dy**2 +
            (f[1:-1, 1:-1, 2:] - 2*f[1:-1, 1:-1, 1:-1] + f[1:-1, 1:-1, :-2]) / dz**2
        )
    
    return compute(v_n[0]), compute(v_n[1]), compute(v_n[2])

def laplacian_p_n_plus_1(p_n_plus_1, dx, dy, dz):
    """
    Computes \nabla^2 p^{n+1} for the Pressure Poisson Equation.
    """
    return (
        (p_n_plus_1[2:, 1:-1, 1:-1] - 2*p_n_plus_1[1:-1, 1:-1, 1:-1] + p_n_plus_1[:-2, 1:-1, 1:-1]) / dx**2 +
        (p_n_plus_1[1:-1, 2:, 1:-1] - 2*p_n_plus_1[1:-1, 1:-1, 1:-1] + p_n_plus_1[1:-1, :-2, 1:-1]) / dy**2 +
        (p_n_plus_1[1:-1, 1:-1, 2:] - 2*p_n_plus_1[1:-1, 1:-1, 1:-1] + p_n_plus_1[1:-1, 1:-1, :-2]) / dz**2
    )