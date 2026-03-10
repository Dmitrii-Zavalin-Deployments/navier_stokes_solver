# src/step4/boundary_dispatcher.py

def get_applicable_boundary_configs(block, boundary_cfg: list, grid) -> list:
    """
    Unified dispatcher returning a list of dicts:
    {'location': str, 'type': str, 'values': dict}
    """
    mask = block.center.mask
    
    # 1. Internal Boundary Fluid (-1) - Now referred to as 'wall'
    if mask == -1:
        return _find_config(boundary_cfg, "wall")
        
    # 2. Solid (0) - Physical definition: No-slip (zero velocity)
    if mask == 0:
        return [{'location': 'solid', 'type': 'no-slip', 'values': {'u': 0.0, 'v': 0.0, 'w': 0.0}}]
        
    # 3. Domain Boundaries (1)
    b_type = _get_domain_location_type(block, grid)
    if b_type != "none":
        return _find_config(boundary_cfg, b_type)
        
    # 4. Interior Fluid (1)
    return [{'location': 'interior', 'type': 'fluid_gas', 'values': {}}]

def _find_config(boundary_cfg: list, location: str) -> list:
    """Returns the config entry including location, type, and values."""
    for bc in boundary_cfg:
        if bc["location"] == location:
            return [bc]
    # If not found in config, returns empty list (Strict Policy)
    return []

def _get_domain_location_type(block, grid) -> str:
    """Maps cell x,y,z indices to domain boundary strings."""
    x, y, z = block.center.x, block.center.y, block.center.z
    if x == 0: return "x_min"
    if x == grid.nx - 1: return "x_max"
    if y == 0: return "y_min"
    if y == grid.ny - 1: return "y_max"
    if z == 0: return "z_min"
    if z == grid.nz - 1: return "z_max"
    return "none"