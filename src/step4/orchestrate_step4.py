# src/step4/orchestrate_step4.py

from src.common.stencil_block import StencilBlock
from src.step4.boundary_applier import apply_boundary_values
from src.step4.boundary_dispatcher import get_applicable_boundary_configs

def orchestrate_step4(block: StencilBlock, boundary_cfg: list, grid, domain_cfg: dict) -> StencilBlock:
    """
    Step 4: Boundary Enforcement Orchestration.
    
    Compliance:
    - Rule 0 (Performance): Operates as a thin orchestration layer over Foundation-mutating ops.
    - Rule 4 (SSoT): Relies on dispatcher/applier for configuration and memory access.
    """
    
    # 1. Identify applicable boundary rules
    # The dispatcher uses schema-locked properties to filter the block.
    # No local logic or "guesses" permitted (Rule 5: Deterministic Initialization).
    rules = get_applicable_boundary_configs(block, boundary_cfg, grid, domain_cfg)
    
    # 2. Apply updates
    # The applier performs direct in-place mutation of the Foundation buffer (Rule 9).
    for rule in rules:
        apply_boundary_values(block, rule)
        
    return block