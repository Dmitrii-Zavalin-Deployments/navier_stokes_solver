# src/step3/orchestrate_step3.py

from src.common.stencil_block import StencilBlock
from src.step3.corrector import apply_local_velocity_correction
from src.step3.ppe_solver import solve_pressure_poisson_step
from src.step3.predictor import compute_local_predictor_step


def orchestrate_step3(block: StencilBlock, omega: float, is_first_pass: bool = False) -> tuple[StencilBlock, float]:
    """
    Step 3 Orchestrator: Projection Method pipeline.
    
    Compliance:
    - Acts as the state controller for the Foundation buffers.
    - Synchronizes the P_NEXT buffer back to the P buffer only after 
      successful pressure convergence.
    """
    if block.center.is_ghost:
        return block, 0.0

    # 1. PREDICT (Run only on the first pass of the time step)
    if is_first_pass:
        compute_local_predictor_step(block)
        return block, 0.0

    # 2. SOLVE: PPE SOR step
    # Updates the P_NEXT buffer in-place
    delta = solve_pressure_poisson_step(block, omega)

    # 3. CORRECT: Project v* -> v^{n+1}
    # Performs in-place mutation of the velocity Foundation buffers
    apply_local_velocity_correction(block)

    # 4. SYNCHRONIZE: Update physical state p^n = p^{n+1}
    # Direct property assignment updates the underlying P column in the Foundation.
    block.center.p = block.center.p_next
    
    return block, delta