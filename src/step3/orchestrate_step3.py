# src/step3/orchestrate_step3.py

from src.common.simulation_context import SimulationContext
from src.common.stencil_block import StencilBlock
from src.common.elasticity import ElasticManager  # Now in common!
from src.step3.corrector import apply_local_velocity_correction
from src.step3.ppe_solver import solve_pressure_poisson_step
from src.step3.predictor import compute_local_predictor_step

def orchestrate_step3(
    block: StencilBlock, 
    context: SimulationContext, 
    elasticity: ElasticManager, # Explicit secondary state
    is_first_pass: bool = False
) -> tuple[StencilBlock, float]:
    """
    Step 3 Orchestrator. 
    Uses ElasticManager to override block-level time-stepping.
    """
    if block.center.is_ghost:
        return block, 0.0

    # Rule 4: Sync the Block's DT with the Elastic Manager
    # We save the original to keep the StencilBlock object 're-usable'
    original_dt = block.dt
    block.dt = elasticity.dt

    if is_first_pass:
        compute_local_predictor_step(block)
        block.dt = original_dt
        return block, 0.0

    # SOLVE using elastic omega
    delta = solve_pressure_poisson_step(block, elasticity.omega)

    # CORRECT using synced block.dt
    apply_local_velocity_correction(block)
    
    block.dt = original_dt
    return block, delta