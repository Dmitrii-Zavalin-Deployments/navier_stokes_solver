# src/step3/boundaries/dispatcher.py

import logging

from src.common.stencil_block import StencilBlock

logger = logging.getLogger(__name__)

def get_applicable_boundary_configs(block: StencilBlock, boundary_cfg: list, grid, domain_cfg: dict) -> list:
    """
    Unified boundary dispatcher with Axiomatic Material Priority.
    
    Rule 4: Topology Defines Logic (Neighbor-State detection).
    Rule 5: Explicit Classification (No silent failures for Walls).
    """
    # 1. MATERIAL MASK LOGIC (INTERNAL AXIOMS)
    # Checked first to ensure internal obstructions (tube/stem) are protected
    # even when positioned near domain boundaries.
    mask = block.center.mask
    
    if mask == -1:
        # AXIOM: Wall requiring user-defined physics
        logger.debug(f"DISPATCH [Mask]: Block {block.id} treated as Wall (mask -1)")
        return _find_config(boundary_cfg, "wall")
        
    if mask == 0:
        # AXIOM: Static Solid (Hardcoded No-Slip)
        logger.debug(f"DISPATCH [Mask]: Block {block.id} treated as Solid (mask 0)")
        return [{
            'location': 'solid', 
            'type': 'no-slip', 
            'values': {'u': 0.0, 'v': 0.0, 'w': 0.0}
        }]

    # 2. DOMAIN BOUNDARY CHECK (SPATIAL & NEIGHBOR-AWARE)
    # Only triggered if the cell is fluid (mask != -1 or 0)
    b_type = _get_domain_location_type(block, grid)
    
    if b_type != "none":
        try:
            # Handle EXTERNAL flow configurations (Wind Axiom)
            if domain_cfg and domain_cfg.get("type") == "EXTERNAL":
                ref_v = domain_cfg["reference_velocity"]
                return [{
                    'location': b_type, 
                    'type': 'free-stream', 
                    'values': {'u': ref_v[0], 'v': ref_v[1], 'w': ref_v[2]}
                }]
            
            # Map spatial face to user-provided config (Inlet/Outlet)
            return _find_config(boundary_cfg, b_type)
            
        except KeyError:
            # Explicitly state which face failed to ensure test regex matches
            logger.error(f"FATAL: Boundary configuration missing for face {b_type}.")
            raise KeyError(f"Missing boundary definition for {b_type}") from None
        
    # 3. FINAL FALLBACK: Interior Fluid
    # The default state for the Navier-Stokes physics kernel.
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