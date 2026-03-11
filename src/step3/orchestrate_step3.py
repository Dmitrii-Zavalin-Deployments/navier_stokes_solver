# src/step3/orchestrate_step3.py

from src.common.stencil_block import StencilBlock
from src.common.field_schema import FI
from src.step3.corrector import apply_local_velocity_correction
from src.step3.ppe_solver import solve_pressure_poisson_step
from src.step3.predictor import compute_local_predictor_step


def orchestrate_step3(block: StencilBlock, omega: float, is_first_pass: bool = False) -> tuple[StencilBlock, float]:
    """
    Step 3 Orchestrator: Projection Method pipeline.
    
    Compliance:
    - Rule 9 (Hybrid Memory): Uses schema-locked getters/setters for buffer access.
    - Rule 4 (SSoT): State synchronization is handled explicitly via the FI schema.
    """
    if block.center.is_ghost:
        return block, 0.0

    # 1. PREDICT: Run on first pass only
    if is_first_pass:
        compute_local_predictor_step(block)
        return block, 0.0

    # 2. SOLVE: PPE SOR step
    # Updates the P_NEXT buffer in-place
    delta = solve_pressure_poisson_step(block, omega)

    # 3. CORRECT: Project v* -> v^{n+1}
    # Performs in-place mutation of the velocity Foundation buffers
    apply_local_velocity_correction(block)

    # 4. SYNCHRONIZE: Update p^n = p^{n+1}
    # Explicit schema-locked assignment ensures Foundation integrity
    p_next_val = block.center.get_field(FI.P_NEXT)
    block.center.set_field(FI.P, p_next_val)
    
    return block, delta