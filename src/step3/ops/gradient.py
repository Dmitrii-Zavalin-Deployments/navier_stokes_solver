# src/step3/ops/gradient.py


def gradient_p_n(p_n, dx, dy, dz):
    """
    Computes -nabla p^n for the Predictor Step (Section 5.1).
    
    Formula:
    - (dp/dx, dp/dy, dp/dz)
    
    Audit Reference:
    - Central difference stencil for pressure gradient
    """
    # --- PRODUCTION (Optimized) ---
    # Using numpy slicing for performance
    grad_x = -(p_n[2:, 1:-1, 1:-1] - p_n[:-2, 1:-1, 1:-1]) / (2 * dx)
    grad_y = -(p_n[1:-1, 2:, 1:-1] - p_n[1:-1, :-2, 1:-1]) / (2 * dy)
    grad_z = -(p_n[1:-1, 1:-1, 2:] - p_n[1:-1, 1:-1, :-2]) / (2 * dz)
    
    return grad_x, grad_y, grad_z

def gradient_p_n_plus_1(p_n_plus_1, dx, dy, dz):
    """
    Computes -nabla p^{n+1} for the Velocity Correction Step (Section 5.3).
    
    Formula:
    - (dp/dx, dp/dy, dp/dz)
    """
    # --- PRODUCTION (Optimized) ---
    grad_x = -(p_n_plus_1[2:, 1:-1, 1:-1] - p_n_plus_1[:-2, 1:-1, 1:-1]) / (2 * dx)
    grad_y = -(p_n_plus_1[1:-1, 2:, 1:-1] - p_n_plus_1[1:-1, :-2, 1:-1]) / (2 * dy)
    grad_z = -(p_n_plus_1[1:-1, 1:-1, 2:] - p_n_plus_1[1:-1, 1:-1, :-2]) / (2 * dz)
    
    return grad_x, grad_y, grad_z