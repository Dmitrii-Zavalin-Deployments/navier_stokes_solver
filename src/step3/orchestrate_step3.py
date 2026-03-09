# src/step3/orchestrate_step3.py

from src.common.stencil_block import StencilBlock
from src.step3.corrector import apply_local_velocity_correction
from src.step3.ppe_solver import solve_local_ppe
from src.step3.predictor import compute_local_predictor_step


def orchestrate_step3(block: StencilBlock):
    """
    Step 3 Orchestrator: Explicit Projection Method pipeline.
    Processes a single StencilBlock, updating the 'center' cell properties directly.
    """
    
    # 1. PREDICT: Calculate intermediate velocity (v*) for the center cell
    # The block object provides all necessary neighbor references and physical constants.
    v_star = compute_local_predictor_step(block)
    
    # 2. SOLVE: Solve local PPE contribution for p^{n+1}
    # Operates on the neighborhood gradient to find the pressure adjustment for this cell.
    p_next = solve_local_ppe(block)
    
    # 3. CORRECT: Project local velocity v* to v^{n+1}
    # Directly updates block.center.vx, vy, vz using the pressure gradient.
    apply_local_velocity_correction(block, v_star, p_next)
    
    return block