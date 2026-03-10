# src/step4/boundary_applier.py

# Global Debug Toggle
DEBUG = True

def apply_boundary_values(block, rule: dict):
    """
    Selective update: only updates attributes present in the rule['values'].
    Maps schema keys (u, v, w) to Cell attributes (vx, vy, vz).
    """
    values = rule.get("values", {})
    x, y, z = block.center.x, block.center.y, block.center.z
    
    if DEBUG and values:
        print(f"DEBUG [Applier]: Applying values to Block ({x},{y},{z}) | Rule: {rule.get('location')} ({rule.get('type')})")

    if "u" in values: 
        block.center.vx = values["u"]
        if DEBUG: print(f"  -> vx set to {values['u']}")
    if "v" in values: 
        block.center.vy = values["v"]
        if DEBUG: print(f"  -> vy set to {values['v']}")
    if "w" in values: 
        block.center.vz = values["w"]
        if DEBUG: print(f"  -> vz set to {values['w']}")
    if "p" in values: 
        block.center.p = values["p"]
        if DEBUG: print(f"  -> p set to {values['p']}")