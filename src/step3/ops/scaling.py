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
    # Rule 7: Physical Integrity Guard
    if block.rho <= 0:
        raise ValueError(
            f"Physical Violation: Non-positive density (rho={block.rho}) "
            f"at block ({block.center.i}, {block.center.j}, {block.center.k})"
        )

    scaling = block.dt / block.rho

    if not math.isfinite(scaling):
        raise ArithmeticError(f"Predictor scaling (dt/rho) exploded: {scaling}")

    return scaling

def get_rho_over_dt(block: StencilBlock) -> float:
    """
    Returns the scaling factor (rho / dt) for the Pressure Poisson Equation.
    
    Compliance:
    - Rule 7: Fail-Fast math audit. Guards against dt=0 which would 
      cause infinite pressure source terms.
    - Zero-Debt Mandate: Explicit arithmetic; no intermediate caching.
    """
    # Rule 7: Time-Step Integrity Guard
    if block.dt <= 0:
        # Note: We raise ArithmeticError so the main_solver knows 
        # the simulation state has become physically invalid.
        raise ValueError(
            f"Time-Step Violation: Non-positive dt ({block.dt}) "
            f"at block ({block.center.i}, {block.center.j}, {block.center.k})"
        )

    scaling = block.rho / block.dt

    if not math.isfinite(scaling):
        raise ArithmeticError(f"PPE scaling (rho/dt) exploded: {scaling}")

    return scaling