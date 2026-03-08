# src/step3/ops/advection.py


def apply_advection_operator(v_n, field, dx, dy, dz):
    """
    Computes (v^n * nabla) * field.
    
    Formula:
    u * df/dx + v * df/dy + w * df/dz
    """
    u, v, w = v_n[0], v_n[1], v_n[2]
    
    # Central difference for gradients
    df_dx = (field[2:, 1:-1, 1:-1] - field[:-2, 1:-1, 1:-1]) / (2 * dx)
    df_dy = (field[1:-1, 2:, 1:-1] - field[1:-1, :-2, 1:-1]) / (2 * dy)
    df_dz = (field[1:-1, 1:-1, 2:] - field[1:-1, 1:-1, :-2]) / (2 * dz)
    
    # Velocity at cell centers (simple average for consistency with cell-centered pressure)
    u_c = (u[2:, 1:-1, 1:-1] + u[:-2, 1:-1, 1:-1]) / 2 # Simplified
    v_c = (v[1:-1, 2:, 1:-1] + v[1:-1, :-2, 1:-1]) / 2
    w_c = (w[1:-1, 1:-1, 2:] + w[1:-1, 1:-1, :-2]) / 2
    
    return u_c * df_dx + v_c * df_dy + w_c * df_dz

def advective_term_v_n(v_n, dx, dy, dz):
    """
    Computes (v^n * nabla) * v^n (The Advective Term).
    """
    return (
        apply_advection_operator(v_n, v_n[0], dx, dy, dz),
        apply_advection_operator(v_n, v_n[1], dx, dy, dz),
        apply_advection_operator(v_n, v_n[2], dx, dy, dz)
    )