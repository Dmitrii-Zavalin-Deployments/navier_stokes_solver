# src/step4/boundary_dispatcher.py

# Global Debug Toggle
DEBUG = True

def get_applicable_boundary_configs(block, boundary_cfg: list, grid, domain_cfg: dict) -> list:
    """
    Unified dispatcher returning a list of dicts:
    {'location': str, 'type': str, 'values': dict}
    Handles domain-specific strategy (INTERNAL vs EXTERNAL).
    """
    mask = block.center.mask
    x, y, z = block.center.x, block.center.y, block.center.z
    
    # 1. Internal Boundary Fluid (-1) - Referred to as 'wall'
    if mask == -1:
        if DEBUG:
            print(f"DEBUG [Boundary]: Block ({x},{y},{z}) identified as internal boundary (wall).")
        return _find_config(boundary_cfg, "wall")
        
    # 2. Solid (0) - Physical definition: No-slip (zero velocity)
    if mask == 0:
        if DEBUG:
            print(f"DEBUG [Boundary]: Block ({x},{y},{z}) identified as solid mask (0).")
        return [{'location': 'solid', 'type': 'no-slip', 'values': {'u': 0.0, 'v': 0.0, 'w': 0.0}}]
        
    # 3. Domain Boundaries
    b_type = _get_domain_location_type(block, grid)
    if b_type != "none":
        # Strategy: EXTERNAL flow uses far-field reference velocity vector
        if domain_cfg.get("type") == "EXTERNAL":
            # Extract vector, default to [0,0,0] if not provided
            ref_v = domain_cfg.get("reference_velocity") or [0.0, 0.0, 0.0]
            if DEBUG:
                print(f"DEBUG [Boundary]: Applying EXTERNAL far-field BC at {b_type} with vector {ref_v}")
            
            return [{
                'location': b_type, 
                'type': 'free-stream', 
                'values': {'u': ref_v[0], 'v': ref_v[1], 'w': ref_v[2]}
            }]
        
        # Strategy: INTERNAL flow uses user-provided config (e.g., pressure-driven)
        if DEBUG:
            print(f"DEBUG [Boundary]: Applying INTERNAL boundary config at {b_type}")
        return _find_config(boundary_cfg, b_type)
        
    # 4. Interior Fluid (1)
    return [{'location': 'interior', 'type': 'fluid_gas', 'values': {}}]

def _find_config(boundary_cfg: list, location: str) -> list:
    """Returns the config entry including location, type, and values."""
    for bc in boundary_cfg:
        if bc.get("location") == location:
            return [bc]
            
    if DEBUG:
        print(f"DEBUG [Boundary]: WARNING - No config found for required location: '{location}'")
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