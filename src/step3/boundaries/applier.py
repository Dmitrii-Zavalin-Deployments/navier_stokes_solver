# src/step3/boundaries/applier.py

import logging
from src.common.field_schema import FI
from src.common.stencil_block import StencilBlock

logger = logging.getLogger(__name__)

# Rule 9 Compliance: Trial Field Redirection
BC_FIELD_MAP = {
    "u": FI.VX_STAR,
    "v": FI.VY_STAR,
    "w": FI.VZ_STAR,
    "p": FI.P_NEXT
}

def apply_boundary_values(block: StencilBlock, rule: dict) -> None:
    """
    Applies boundary values to the StencilBlock Trial Fields.
    """
    values = rule.get("values")
    location = rule.get("location")
    boundary_type = rule.get("type")
    
    if values is None or location is None or boundary_type is None:
        logger.error(f"STRATEGIC FAILURE: Missing fields at {location=}")
        raise ValueError(f"Boundary rule missing critical fields at {location=}")

    for key, value in values.items():
        field_id = BC_FIELD_MAP.get(key)
        
        if field_id is not None:
            logger.debug(
                f"APPLY [Start]: Block {block.id} | Boundary: {location} | "
                f"Mapping {key} to Field {field_id} with Value {value:.4e}"
            )
            
            # Rule 9: Direct in-place update
            block.center.set_field(field_id, value)
            
            # Verify the write (Verification Log)
            written_val = block.center.get_field(field_id)
            logger.debug(f"APPLY [Success]: Verified Field {field_id} = {written_val:.4e}")
            
        else:
            logger.error(f"CONTRACT VIOLATION: Unsupported key '{key}' at {location}")
            raise KeyError(f"Unsupported boundary key '{key}' at {location}")