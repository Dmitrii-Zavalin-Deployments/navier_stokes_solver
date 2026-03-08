# src/step3/solver/predictor.py

import numpy as np

from src.step3.core.extract import get_interior_field
from src.step3.ops.advection import advective_term_v_n
from src.step3.ops.gradient import gradient_p
from src.step3.ops.laplacian import laplacian_v
from src.step3.ops.scaling import get_dt_over_rho


def compute_predictor_step(v_n, p_n, F_int, dx, dy, dz, dt, rho, mu):
    """
    v_n: (3, nx, ny, nz)
    p_n: (nx, ny, nz)
    F_int: (3, nx-2, ny-2, nz-2)
    """
    # 1. Extraction: Get all terms into (3, nx-2, ny-2, nz-2)
    v_n_int = get_interior_field(v_n)
    
    # 2. Physics Terms
    diff = laplacian_v(v_n, dx, dy, dz)
    adv = np.stack(advective_term_v_n(v_n, dx, dy, dz))
    grad_p = gradient_p(p_n, dx, dy, dz)
    
    # 3. Scaling
    scaling = get_dt_over_rho(dt, rho)
    
    # 4. Final Equation: v* = vn + (dt/rho) * [mu*lap(v) - rho*(v.grad)v + F - grad(p)]
    # Note: v_n_int is the interior velocity corresponding to the result shape
    v_star = v_n_int + scaling * (mu * diff - rho * adv + F_int + grad_p)
    
    return v_star