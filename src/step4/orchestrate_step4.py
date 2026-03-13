# src/step4/orchestrate_step4.py

from src.common.simulation_context import SimulationContext
from src.common.stencil_block import StencilBlock
from src.step4.boundary_applier import apply_boundary_values
from src.step4.boundary_dispatcher import get_applicable_boundary_configs


def orchestrate_step4(
    block: StencilBlock, 
    context: SimulationContext, 
    state_grid: object, 
    state_bc_manager: object
) -> StencilBlock:
    """
    Step 4: Boundary Enforcement Orchestration.
    
    Compliance:
    - Rule 4 (SSoT): Uses the SSoT components (grid/bc_manager) passed from the state.
    - Rule 9 (Hybrid Memory): Orchestrates in-place mutation of Foundation buffers.
    """
    
    # 1. Identify applicable boundary rules
    # We pass the domain type from the context for rule dispatching.
    rules = get_applicable_boundary_configs(
        block, 
        {bc.location: bc for bc in state_bc_manager.conditions},
        state_grid, 
        context.input_data.domain_configuration.type
    )
    
    # 2. Apply updates
    # The applier performs direct in-place mutation of the Foundation buffer (Rule 9).
    for rule in rules:
        apply_boundary_values(block, rule)
        
    return block