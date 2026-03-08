# src/step3/predictor.py
import numpy as np

from .core.extract import get_interior_field
from .ops.advection import advective_term_v_n
from .ops.forces import get_body_forces_interior
from .ops.gradient import gradient_p
from .ops.laplacian import laplacian_v
from .ops.scaling import get_dt_over_rho


def compute_predictor_step(v_n, p_n, dx, dy, dz, dt, rho, mu, F_vals):
    """
    The Predictor Step, now located directly in src/step3/
    """
    nx, ny, nz = v_n.shape[1:]
    
    # Generate Force Field (interior)
    F_int = get_body_forces_interior(nx, ny, nz, *F_vals)
    
    # Physics Terms (all returned as 3, interior_dims)
    v_n_int = get_interior_field(v_n)
    diff = laplacian_v(v_n, dx, dy, dz)
    adv = np.stack(advective_term_v_n(v_n, dx, dy, dz))
    grad_p = gradient_p(p_n, dx, dy, dz)
    
    # Scaling and final update
    scaling = get_dt_over_rho(dt, rho)
    v_star = v_n_int + scaling * (mu * diff - rho * adv + F_int + grad_p)
    
    return v_star