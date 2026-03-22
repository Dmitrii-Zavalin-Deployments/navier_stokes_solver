# src/step3/ops/scaling.py

import math

from src.common.stencil_block import StencilBlock


def get_dt_over_rho(block: StencilBlock) -> float:
    """
    Returns the scaling factor (dt / rho) for the Predictor and Corrector Steps.
    
    Compliance:
    - Rule 7: Fail-Fast math audit. Guards against vacuum density (rho=0)
      which would cause infinite acceleration.
    - Rule 4 (SSoT): Derived from immutable block properties.
    """

    scaling = block.dt / block.rho

    return scaling

def get_rho_over_dt(block: StencilBlock) -> float:
    """
    Returns the scaling factor (rho / dt) for the Pressure Poisson Equation.
    
    Compliance:
    - Rule 7: Fail-Fast math audit. Guards against dt=0 which would 
      cause infinite pressure source terms.
    - Zero-Debt Mandate: Explicit arithmetic; no intermediate caching.
    """

    scaling = block.rho / block.dt

    return scaling