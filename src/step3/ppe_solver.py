# src/step3/ppe.py

from .ops.divergence import divergence_v_star
from .ops.scaling import get_rho_over_dt
from .rhie_chow import compute_rhie_chow_term


def compute_ppe_rhs(v_star, p_n, dx, dy, dz, rho, dt):
    """
    Assembles the RHS of the stabilized Pressure Poisson Equation.
    """
    div_v_star = divergence_v_star(v_star, dx, dy, dz)
    div_rc = compute_rhie_chow_term(p_n, dx, dy, dz, dt)
    
    scaling = get_rho_over_dt(dt, rho)
    
    return scaling * (div_v_star - div_rc)