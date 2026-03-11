# src/step4/boundary_applier.py

from src.common.field_schema import FI
from src.common.stencil_block import StencilBlock

# Explicit mapping from configuration keys to FI schema
BC_FIELD_MAP = {
    "u": FI.VX,
    "v": FI.VY,
    "w": FI.VZ,
    "p": FI.P
}

def apply_boundary_values(block: StencilBlock, rule: dict) -> None:
    """
    Applies boundary values to the StencilBlock.
    
    Compliance:
    - Rule 8 (Singular Access): Uses schema-locked mapping rather than getattr/setattr.
    - Rule 9 (Hybrid Memory): Direct Foundation modification via set_field.
    """
    values = rule.get("values")
    location = rule.get("location")
    boundary_type = rule.get("type")
    
    # Rule 5: Explicit or Error. No fallbacks.
    if values is None or location is None or boundary_type is None:
        raise ValueError(f"Boundary rule missing fields: {location=}, {boundary_type=}")

    for key, value in values.items():
        field_id = BC_FIELD_MAP.get(key)
        
        if field_id is not None:
            # Rule 9: Direct in-place update to Foundation
            block.center.set_field(field_id, value)
        else:
            # Rule 5: Immediate failure for invalid configurations
            raise KeyError(f"Unsupported boundary key '{key}' at {location}")