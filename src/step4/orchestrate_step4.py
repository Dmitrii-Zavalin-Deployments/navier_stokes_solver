# src/step4/orchestrate_step4.py

from src.common.stencil_block import StencilBlock
from src.step4.boundary_applier import apply_boundary_values
from src.step4.boundary_dispatcher import get_applicable_boundary_configs

def orchestrate_step4(block: StencilBlock, boundary_cfg: list, grid, domain_cfg: dict) -> StencilBlock:
    """
    Step 4: Boundary Enforcement.
    Coordinates the identification and application of boundary conditions
    to a specific stencil block, using domain configuration to select the 
    physics strategy (Internal vs. External).
    """
    # Ghost cells are managed outside the physics loop or skipped here
    if block.center.is_ghost:
        return block

    # 1. Identify which boundaries apply based on location and the domain strategy
    # We now pass domain_cfg to the dispatcher to determine BC enforcement logic
    rules = get_applicable_boundary_configs(block, boundary_cfg, grid, domain_cfg)
    
    # 2. Apply updates for each rule found (Selective partial updates)
    for rule in rules:
        apply_boundary_values(block, rule)
        
    return block