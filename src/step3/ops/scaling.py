# src/step3/ops/scaling.py

from src.common.stencil_block import StencilBlock

def get_dt_over_rho(block: StencilBlock) -> float:
    """
    Returns the scaling factor (dt / rho) for the Predictor Step.
    
    Compliance:
    - Rule 4 (SSoT): Scaling factors are derived from immutable config-based
      parameters pinned to the StencilBlock during construction.
    - Rule 0: No dynamic attribute lookups; constant-time access via 
      pre-cached block properties.
    """
    # These attributes are strictly initialized during assembly (Step 2)
    return block.dt / block.rho

def get_rho_over_dt(block: StencilBlock) -> float:
    """
    Returns the scaling factor (rho / dt) for the Pressure Poisson Equation.
    
    Compliance:
    - Rule 4 (SSoT): Inherits the same architectural rigor as get_dt_over_rho.
    - Zero-Debt Mandate: Explicit arithmetic; no intermediate caching or
      state-mutation.
    """
    return block.rho / block.dt