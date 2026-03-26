# src/step3/boundaries/applier.py

import logging

from src.common.field_schema import FI
from src.common.stencil_block import StencilBlock

logger = logging.getLogger("Solver.Boundaries")

# Rule 9 Compliance: Trial Field Redirection
# Maps logical physics keys (u, v, w, p) to specific Trial/Next buffers.
# This ensures Step 3 iterations never pollute the n-step base fields.
BC_FIELD_MAP = {
    "u": FI.VX_STAR,
    "v": FI.VY_STAR,
    "w": FI.VZ_STAR,
    "p": FI.P_NEXT
}

def apply_boundary_values(block: StencilBlock, rule: dict) -> None:
    """
    Applies boundary values to the StencilBlock Trial Fields.
    Operates directly on Foundation memory via the StencilBlock interface.
    
    Compliance:
    - Rule 7: Fail-Fast logging for contract violations.
    - Rule 9: In-place update via Sovereign Scalar Accessors.
    """
    values = rule.get("values")
    location = rule.get("location")
    boundary_type = rule.get("type")
    
    if values is None or location is None or boundary_type is None:
        logger.error(f"STRATEGIC FAILURE: Missing fields at location={location}")
        raise ValueError(f"Boundary rule missing critical fields at location={location}")

    for key, value in values.items():
        field_id = BC_FIELD_MAP.get(key)
        
        if field_id is not None:
            # Rule 9: Direct in-place update to the block's center point.
            # Cell.set_field handles the cast to native float.
            block.center.set_field(field_id, value)
            
            # Rule 7: Clean diagnostic trace without forensic bloat.
            logger.debug(
                f"APPLY [Success]: Block {block.id} | Location: {location} | "
                f"Field {field_id.name} set to {float(value):.4e}"
            )
            
        else:
            # Rule 7: Catch and halt on configuration drift.
            logger.error(f"CONTRACT VIOLATION: Unsupported key '{key}' at {location}")
            raise KeyError(f"Unsupported boundary key '{key}' at {location}")