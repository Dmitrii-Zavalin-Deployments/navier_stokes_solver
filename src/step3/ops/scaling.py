# src/step3/ops/scaling.py

from src.common.stencil_block import StencilBlock


def get_dt_over_rho(block: StencilBlock) -> float:
    """
    Returns the scaling factor (dt / rho) for the Predictor Step.
    
    Compliance:
    - Accesses simulation-wide constants stored in StencilBlock metadata.
    - These values are immutable during the time-step loop, ensuring
      stability for the predictor projection.
    """
    return block.dt / block.rho

def get_rho_over_dt(block: StencilBlock) -> float:
    """
    Returns the scaling factor (rho / dt) for the Pressure Poisson Equation.
    
    Compliance:
    - Accesses simulation-wide constants stored in StencilBlock metadata.
    - Used to normalize the divergence of intermediate velocity fields
      against the pressure correction.
    """
    return block.rho / block.dt