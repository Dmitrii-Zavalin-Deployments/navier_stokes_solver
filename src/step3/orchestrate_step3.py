# src/step3/orchestrate_step3.py

from src.common.field_schema import FI
from src.common.simulation_context import SimulationContext
from src.common.stencil_block import StencilBlock
from src.step3.corrector import apply_local_velocity_correction
from src.step3.ppe_solver import solve_pressure_poisson_step
from src.step3.predictor import compute_local_predictor_step


def orchestrate_step3(
    block: StencilBlock, 
    context: SimulationContext, 
    is_first_pass: bool = False
) -> tuple[StencilBlock, float]:
    """
    Step 3 Orchestrator: Projection Method pipeline (Phase C Compliant).
    
    Compliance:
    - Rule 4 (SSoT): Numerical settings (omega) are pulled from context.config.
    - Rule 9 (Hybrid Memory): Logic-data (StencilBlock) mediates buffer access.
    """
    if block.center.is_ghost:
        return block, 0.0

    # 1. PREDICT: Run on first pass only
    if is_first_pass:
        compute_local_predictor_step(block)
        return block, 0.0

    # 2. SOLVE: PPE SOR step
    # Numerical omega is now strictly sourced from the SimulationConfig container
    omega = context.config.ppe_omega
    delta = solve_pressure_poisson_step(block, omega)

    # 3. CORRECT: Project v* -> v^{n+1}
    # Performs in-place mutation of the velocity Foundation buffers
    apply_local_velocity_correction(block)
    
    return block, delta