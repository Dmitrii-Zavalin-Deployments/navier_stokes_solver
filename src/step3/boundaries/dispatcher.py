# src/step3/boundaries/dispatcher.py

import logging

from src.common.stencil_block import StencilBlock

logger = logging.getLogger(__name__)

def get_applicable_boundary_configs(block: StencilBlock, boundary_cfg: list, grid, domain_cfg: dict) -> list:
    """
    Unified boundary dispatcher with Spatial-First Priority.
    """
    # 1. Check Domain Boundaries FIRST (Rule: Geometry overrides Mask)
    b_type = _get_domain_location_type(block, grid)
    if b_type != "none":
        logger.debug(f"DISPATCH [Spatial]: Block {block.id} caught by {b_type}")
        
        if domain_cfg.get("type") == "EXTERNAL":
            ref_v = domain_cfg["reference_velocity"]
            return [{
                'location': b_type, 
                'type': 'free-stream', 
                'values': {'u': ref_v[0], 'v': ref_v[1], 'w': ref_v[2]}
            }]
        
        return _find_config(boundary_cfg, b_type)

    # 2. Check Material Masks SECOND
    mask = block.center.mask
    
    if mask == -1:
        logger.debug(f"DISPATCH [Mask]: Block {block.id} identified as Wall (mask -1)")
        return _find_config(boundary_cfg, "wall")
        
    if mask == 0:
        logger.debug(f"DISPATCH [Mask]: Block {block.id} identified as Solid (mask 0)")
        return [{
            'location': 'solid', 
            'type': 'no-slip', 
            'values': {'u': 0.0, 'v': 0.0, 'w': 0.0}
        }]
        
    # 3. Interior Fluid
    logger.debug(f"DISPATCH [Interior]: Block {block.id} treated as interior fluid")
    return [{'location': 'interior', 'type': 'fluid_gas', 'values': {}}]

def _find_config(boundary_cfg: list, location: str) -> list:
    for bc in boundary_cfg:
        if bc["location"] == location:
            return [bc]
    logger.error(f"DISPATCH FAILURE: No config for {location}")
    raise KeyError(f"No boundary configuration found for location: '{location}'")

def _get_domain_location_type(block, grid) -> str:
    x, y, z = block.center.i, block.center.j, block.center.k
    if x == 0: return "x_min"
    if x == grid.nx - 1: return "x_max"
    if y == 0: return "y_min"
    if y == grid.ny - 1: return "y_max"
    if z == 0: return "z_min"
    if z == grid.nz - 1: return "z_max"
    return "none"