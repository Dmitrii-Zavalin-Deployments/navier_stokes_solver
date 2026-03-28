# src/step3/boundaries/dispatcher.py

import logging
from src.common.stencil_block import StencilBlock

logger = logging.getLogger(__name__)

def get_applicable_boundary_configs(block: StencilBlock, boundary_cfg: list, grid, domain_cfg: dict) -> list:
    """
    Unified boundary dispatcher with Spatial-First Priority and Mask Fallback.
    
    Rule 1: Neighbor-State detection (is_ghost) overrides coordinate math.
    Rule 5: Explicit classification—every cell returns a typed rule.
    """
    # 1. Attempt Domain Boundary Check (Spatial & Neighbor-Aware)
    b_type = _get_domain_location_type(block, grid)
    
    if b_type != "none":
        try:
            # Handle EXTERNAL flow configurations
            if domain_cfg and domain_cfg.get("type") == "EXTERNAL":
                # Rule 5: Direct access, no defaults.
                ref_v = domain_cfg["reference_velocity"]
                return [{
                    'location': b_type, 
                    'type': 'free-stream', 
                    'values': {'u': ref_v[0], 'v': ref_v[1], 'w': ref_v[2]}
                }]
            
            # Map spatial location to user-provided config list (e.g., inlet/outlet)
            return _find_config(boundary_cfg, b_type)
            
        except KeyError:
            logger.error(f"FATAL: Boundary configuration missing for face {b_type}.")
            raise KeyError(f"Missing boundary definition for face: {b_type}") from None

    # 2. Material Mask Logic (Internal Obstructions)
    mask = block.center.mask
    
    if mask == -1:
        # User must provide a "wall" entry in boundary_cfg for masked cells
        return _find_config(boundary_cfg, "wall")
        
    if mask == 0:
        return [{
            'location': 'solid', 
            'type': 'no-slip', 
            'values': {'u': 0.0, 'v': 0.0, 'w': 0.0}
        }]
        
    # 3. FINAL FALLBACK: Interior Fluid
    # Explicitly typed to ensure the orchestrator loop always has a target.
    return [{'location': 'interior', 'type': 'fluid_gas', 'values': {}}]

def _find_config(boundary_cfg: list, location: str) -> list:
    """
    Direct lookup: Matches location string exactly as defined in the Schema.
    No normalization. Rule 5: If it's not in the JSON, it's an error.
    """
    for bc in boundary_cfg:
        if bc["location"] == location:
            return [bc]
            
    raise KeyError(f"No boundary configuration found for location: '{location}'")

def _get_domain_location_type(block: StencilBlock, grid) -> str:
    """
    Identifies the boundary face based strictly on Neighbor-State detection.
    Rule 4: Topology defines Logic. Coordinate math is rejected to prevent drift.
    """
    # Primary & Only Authority: Ghost Neighbor Detection
    if block.i_minus.is_ghost: return "x_min"
    if block.i_plus.is_ghost:  return "x_max"
    if block.j_minus.is_ghost: return "y_min"
    if block.j_plus.is_ghost:  return "y_max"
    if block.k_minus.is_ghost: return "z_min"
    if block.k_plus.is_ghost:  return "z_max"
    
    # If no neighbors are ghosts, this is either an interior cell 
    # or an internal masked obstruction (handled by mask logic in caller).
    return "none"