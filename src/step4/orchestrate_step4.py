# src/step4/orchestrate_step4.py

from src.common.stencil_block import StencilBlock
from src.step4.boundary_dispatcher import get_applicable_boundary_configs
from src.step4.boundary_applier import apply_boundary_values

def orchestrate_step4(block: StencilBlock, boundary_cfg: list, grid) -> StencilBlock:
    """
    Step 4: Boundary Enforcement.
    Coordinates the identification and application of boundary conditions
    to a specific stencil block.
    """
    # Ghost cells are managed outside the physics loop or skipped here
    if block.center.is_ghost:
        return block

    # 1. Identify which boundaries apply to this block (e.g., wall, solid, or domain face)
    rules = get_applicable_boundary_configs(block, boundary_cfg, grid)
    
    # 2. Apply updates for each rule found (Selective partial updates)
    for rule in rules:
        apply_boundary_values(block, rule)
        
    return block