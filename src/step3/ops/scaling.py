# src/step3/ops/scaling.py

from src.common.stencil_block import StencilBlock

def get_dt_over_rho(block: StencilBlock) -> float:
    """
    Returns the scaling factor (dt / rho) for the Predictor Step.
    
    Args:
        block: The StencilBlock containing simulation parameters.
    """
    return block.dt / block.rho

def get_rho_over_dt(block: StencilBlock) -> float:
    """
    Returns the scaling factor (rho / dt) for the Pressure Poisson Equation.
    
    Args:
        block: The StencilBlock containing simulation parameters.
    """
    return block.rho / block.dt