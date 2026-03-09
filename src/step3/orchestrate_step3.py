# src/step3/orchestrate_step3.py

from src.common.stencil_block import StencilBlock
from src.step3.corrector import apply_local_velocity_correction
from src.step3.predictor import compute_local_predictor_step


def orchestrate_step3(block: StencilBlock):
    """
    Step 3 Orchestrator: Atomic Projection Method pipeline.
    Processes a single StencilBlock through the Predictor-Corrector stages.
    
    Note: The PPE solver (SOR iterations) is managed by the caller 
    (main.py) to allow for global convergence checks.
    """
    
    # 1. PREDICT: Calculate intermediate velocity (v*)
    # Directly updates block.center.v*_star
    if not block.center.is_ghost:
        compute_local_predictor_step(block)
    
    # 2. SOLVE: PPE SOR step
    # Note: This is now called externally via solve_pressure_poisson_step(block)
    # in the caller's convergence loop.
    
    # 3. CORRECT: Project v* onto divergence-free space (v^{n+1})
    # Directly updates block.center.v components
    if not block.center.is_ghost:
        apply_local_velocity_correction(block)
    
    return block