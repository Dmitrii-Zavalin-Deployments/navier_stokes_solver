# src/step3/boundaries/applier.py

from src.common.field_schema import FI
from src.common.stencil_block import StencilBlock

# Rule 9 Compliance: Trial Field Redirection
# We map configuration keys directly to the 'Active' trial buffers (_STAR and _P_NEXT).
# This ensures that BCs are enforced during the Predictor and PPE phases, 
# and are visible to the Rule 7 Physics Audit before the final commit.
BC_FIELD_MAP = {
    "u": FI.VX_STAR,
    "v": FI.VY_STAR,
    "w": FI.VZ_STAR,
    "p": FI.P_NEXT
}

def apply_boundary_values(block: StencilBlock, rule: dict) -> None:
    """
    Applies boundary values to the StencilBlock Trial Fields.
    
    Compliance:
    - Rule 8 (Singular Access): Uses schema-locked mapping to Trial Indices.
    - Rule 9 (Hybrid Memory): Direct modification of the 'Active' star fields.
    - Rule 7 (Traceability): Ensures BCs are caught by the vectorized physics audit.
    """
    values = rule.get("values")
    location = rule.get("location")
    boundary_type = rule.get("type")
    
    # Rule 5: Explicit or Error. No fallbacks for missing configuration data.
    if values is None or location is None or boundary_type is None:
        raise ValueError(
            f"STRATEGIC FAILURE: Boundary rule missing critical fields at {location=}. "
            f"Type: {boundary_type}"
        )

    for key, value in values.items():
        field_id = BC_FIELD_MAP.get(key)
        
        if field_id is not None:
            # Rule 9: Direct in-place update to the Star/Trial Foundation.
            # This makes the boundary condition "active" for the PPE solver.
            block.center.set_field(field_id, value)
        else:
            # Rule 5: Immediate failure for invalid/unmapped configuration keys.
            raise KeyError(f"CONTRACT VIOLATION: Unsupported boundary key '{key}' at {location}")