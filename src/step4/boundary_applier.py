# src/step4/boundary_applier.py

# Global Debug Toggle
DEBUG = True

def apply_boundary_values(block, rule: dict):
    """
    Selective update: only updates attributes present in the rule['values'].
    Enforces strict policy: if a key is expected for a specific boundary type,
    it must be present in the rule, otherwise, raises ValueError.
    """
    values = rule.get("values")
    location = rule.get("location")
    boundary_type = rule.get("type")
    
    if values is None or location is None or boundary_type is None:
        raise ValueError(f"Boundary rule is missing required fields: location={location}, type={boundary_type}, values={values}")

    x, y, z = block.center.x, block.center.y, block.center.z
    
    if DEBUG:
        print(f"DEBUG [Applier]: Applying {boundary_type} at {location} to Block ({x},{y},{z})")

    # Strict application: If we encounter keys we don't recognize or values are missing
    # where physics requires them, we do not guess—we crash.
    
    for key, value in values.items():
        if key == "u":
            block.center.vx = value
            if DEBUG: print(f"  -> vx set to {value}")
        elif key == "v":
            block.center.vy = value
            if DEBUG: print(f"  -> vy set to {value}")
        elif key == "w":
            block.center.vz = value
            if DEBUG: print(f"  -> vz set to {value}")
        elif key == "p":
            block.center.p = value
            if DEBUG: print(f"  -> p set to {value}")
        else:
            raise ValueError(f"Unsupported boundary value key '{key}' encountered at {location}")