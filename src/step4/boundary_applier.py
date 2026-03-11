# src/step4/boundary_applier.py


# Global Debug Toggle
DEBUG = True

# Explicit mapping from boundary config keys to Cell schema-locked properties.
# This ensures that even in Step 4, we are only writing to the Foundation
# via the authorized API defined in the Cell class.
BC_PROPERTY_MAP = {
    "u": "vx",
    "v": "vy",
    "w": "vz",
    "p": "p"
}

def apply_boundary_values(block, rule: dict):
    """
    Applies boundary values to the StencilBlock.
    
    Compliance:
    - Uses a schema-locked map to route boundary values into the Foundation.
    - Each assignment triggers the Cell property setters, which perform 
      in-place writes to the global fields_buffer, ensuring zero heap allocation.
    """
    values = rule.get("values")
    location = rule.get("location")
    boundary_type = rule.get("type")
    
    if values is None or location is None or boundary_type is None:
        raise ValueError(f"Boundary rule is missing required fields: location={location}, type={boundary_type}")

    # Spatial logging: referencing block center coordinates
    x, y, z = block.center.x, block.center.y, block.center.z
    if DEBUG:
        print(f"DEBUG [Applier]: Applying {boundary_type} at {location} to Block index {block.center.index} ({x},{y},{z})")

    for key, value in values.items():
        attr_name = BC_PROPERTY_MAP.get(key)
        
        if attr_name:
            # Direct write-back to Foundation via property setter.
            # This triggers: self.fields_buffer[self.index, FI.PROPERTY]
            setattr(block.center, attr_name, value)
            if DEBUG:
                print(f"  -> {attr_name} set to {value} (Mapped to FI.{attr_name.upper()})")
        else:
            raise ValueError(f"Unsupported boundary value key '{key}' encountered at {location}")