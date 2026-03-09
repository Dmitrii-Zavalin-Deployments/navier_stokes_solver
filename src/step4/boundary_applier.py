# src/step4/boundary_applier.py

from src.common.stencil_block import StencilBlock


def apply_boundary_values(block: StencilBlock, rule: dict):
    """
    Applies boundary conditions to the block's center cell.
    Updates only the fields explicitly provided in the rule's 'values' dict.
    """
    values = rule.get("values", {})
    
    # Selective update logic
    # We only update if the key exists in the 'values' dictionary.
    if "u" in values:
        block.center.vx = values["u"]
    if "v" in values:
        block.center.vy = values["v"]
    if "w" in values:
        block.center.vz = values["w"]
    if "p" in values:
        block.center.p = values["p"]
        
    # Note: If your simulation uses the 'star' velocity intermediate fields, 
    # you may want to sync them here as well if the BC is strictly enforced.
    # e.g., if "u" in values: block.center.vx_star = values["u"]