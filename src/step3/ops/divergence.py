# src/step3/ops/divergence.py


def divergence_v_star(v_star, dx, dy, dz):
    """
    Computes the scalar divergence \nabla \cdot \vec{v}^* for the 
    Pressure Poisson Equation source term.
    
    Formula:
    \nabla \cdot \vec{v}^* \approx (u_{i+1}-u_{i-1})/2dx + (v_{j+1}-v_{j-1})/2dy + (w_{k+1}-w_{k-1})/2dz
    """
    u, v, w = v_star[0], v_star[1], v_star[2]
    
    # --- PRODUCTION (Optimized) ---
    div = (
        (u[2:, 1:-1, 1:-1] - u[:-2, 1:-1, 1:-1]) / (2 * dx) +
        (v[1:-1, 2:, 1:-1] - v[1:-1, :-2, 1:-1]) / (2 * dy) +
        (w[1:-1, 1:-1, 2:] - w[1:-1, 1:-1, :-2]) / (2 * dz)
    )
    return div