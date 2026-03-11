# src/step4/boundary_dispatcher.py

from src.common.stencil_block import StencilBlock


def get_applicable_boundary_configs(block: StencilBlock, boundary_cfg: list, grid, domain_cfg: dict) -> list:
    """
    Unified boundary dispatcher.
    
    Compliance:
    - Rule 8 (Singular Access): Avoids redundant getters; uses standard cell interface.
    - Rule 9 (Hybrid Memory): Accesses mask and geometry directly from pre-allocated memory.
    """
    
    # Accessing mask and index directly from the Foundation via Cell properties
    mask = block.center.mask
    
    # 1. Wall Boundary (Mask -1)
    if mask == -1:
        return _find_config(boundary_cfg, "wall")
        
    # 2. Solid Boundary (Mask 0)
    if mask == 0:
        return [{
            'location': 'solid', 
            'type': 'no-slip', 
            'values': {'u': 0.0, 'v': 0.0, 'w': 0.0}
        }]
        
    # 3. Domain Boundaries
    b_type = _get_domain_location_type(block, grid)
    if b_type != "none":
        # Strategy: EXTERNAL flow uses far-field reference velocity
        # Rule 5: Explicit or Error. Accesses domain_cfg explicitly.
        if domain_cfg["type"] == "EXTERNAL":
            ref_v = domain_cfg["reference_velocity"]
            return [{
                'location': b_type, 
                'type': 'free-stream', 
                'values': {'u': ref_v[0], 'v': ref_v[1], 'w': ref_v[2]}
            }]
        
        return _find_config(boundary_cfg, b_type)
        
    # 4. Interior Fluid
    return [{'location': 'interior', 'type': 'fluid_gas', 'values': {}}]

def _find_config(boundary_cfg: list, location: str) -> list:
    """Returns the config entry from the list based on location."""
    for bc in boundary_cfg:
        if bc["location"] == location:
            return [bc]
    
    # Rule 5: Raise error if configuration is missing, rather than failing silently
    raise KeyError(f"No boundary configuration found for location: '{location}'")

def _get_domain_location_type(block, grid) -> str:
    """Maps cell index logic to boundary location strings."""
    # Use explicit coordinate properties defined in the Cell
    x, y, z = block.center.i, block.center.j, block.center.k
    
    if x == 0: return "x_min"
    if x == grid.nx - 1: return "x_max"
    if y == 0: return "y_min"
    if y == grid.ny - 1: return "y_max"
    if z == 0: return "z_min"
    if z == grid.nz - 1: return "z_max"
    return "none"