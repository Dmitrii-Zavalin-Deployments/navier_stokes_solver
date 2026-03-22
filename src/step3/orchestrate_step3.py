# src/step3/orchestrate_step3.py

import math

from src.common.elasticity import ElasticManager
from src.common.simulation_context import SimulationContext
from src.common.stencil_block import StencilBlock
from src.step3.corrector import apply_local_velocity_correction
from src.step3.ppe_solver import solve_pressure_poisson_step
from src.step3.predictor import compute_local_predictor_step

# Rule 7: Granular Traceability for GitHub Actions
DEBUG = False

def orchestrate_step3(
    block: StencilBlock, 
    context: SimulationContext, 
    elasticity: ElasticManager, 
    is_first_pass: bool = False
) -> tuple[StencilBlock, float]:
    """
    Step 3 Orchestrator: Projection Method with Split Exception Handling.
    
    Compliance:
    - Rule 4 (SSoT): Temporarily overrides block.dt with elasticity.dt, ensuring 
      restoration via 'finally' to prevent time-step skew across the grid.
    - Rule 7 (Traceability): Dumps failure telemetry directly to GitHub Actions logs.
    """
    if block.center.is_ghost:
        return block, 0.0

    # Rule 4: Sync the Block's DT with the Elastic Manager
    original_dt = block.dt
    block.dt = elasticity.dt

    try:
        # 1. PREDICT: Intermediate star-velocity calculation
        if is_first_pass:
            compute_local_predictor_step(block)
            return block, 0.0

        # 2. SOLVE: Iterative Pressure Poisson (SOR)
        delta = solve_pressure_poisson_step(block, context.config.ppe_omega)
        
        # Rule 7: Immediate math audit to catch NaN/Inf divergence
        if not math.isfinite(delta):
            raise ArithmeticError(f"Non-finite delta ({delta})")

        # 3. CORRECT: Final velocity projection
        apply_local_velocity_correction(block)
        
        return block, delta

    except ArithmeticError as e:
        # Expected Numerical Instability: Signal Main Solver to retry with Panic Mode
        if DEBUG:
            print(f"DEBUG [Step 3]: Math instability at ({block.center.i}, {block.center.j}, {block.center.k}) -> {e}")
        raise

    except Exception as e:
        # Unexpected Code/Logic Bug: Print telemetry and EXIT simulation
        loc = f"({block.center.i}, {block.center.j}, {block.center.k})"
        print(f"\nFATAL CODE BUG at Block {loc}")
        print(f"  > Error Type: {type(e).__name__}")
        print(f"  > Details: {str(e)}")
        
        # Re-raising without converting to ArithmeticError ensures the main_solver
        # does NOT retry, effectively terminating the pipeline.
        raise 
    
    finally:
        # Mandatory Restoration of block state integrity
        block.dt = original_dt