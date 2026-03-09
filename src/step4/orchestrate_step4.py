# src/step4/orchestrate_step4.py

from src.step4.boundary_applier import apply_boundary_values
from src.step4.boundary_dispatcher import get_applicable_boundary_configs


def orchestrate_step4(block, boundary_cfg: list) -> StencilBlock:
    if block.center.is_ghost:
        return block

    # 1. Identify which boundaries apply to this block
    rules = get_applicable_boundary_configs(block, boundary_cfg)
    
    # 2. Apply updates for each rule found
    for rule in rules:
        apply_boundary_values(block, rule)
        
    return block