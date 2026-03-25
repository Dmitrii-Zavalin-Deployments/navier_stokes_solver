# src/step3/boundaries/applier.py

import logging

from src.common.field_schema import FI
from src.common.stencil_block import StencilBlock

logger = logging.getLogger("Solver.Boundaries")

# Rule 9 Compliance: Trial Field Redirection
# Maps logical physics keys (u, v, w, p) to specific Trial/Next buffers
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
            # --- [FORENSIC AUDIT: PRE-WRITE] ---
            # We log the input value to ensure the 'value' itself isn't an array 
            # coming from the JSON parser or a previous step.
            logger.debug(
                f"APPLY [Start]: Block {block.id} | Boundary: {location} | "
                f"Mapping {key} to Field {field_id} | Input Value: {value}"
            )
            
            # Rule 9: Direct in-place update to the block's center point
            block.center.set_field(field_id, value)
            
            # --- [FORENSIC AUDIT: POST-WRITE VERIFICATION] ---
            # This is where the crash occurs. We capture the raw object first.
            raw_written_val = block.center.get_field(field_id)
            
            # Extract metadata to identify the 'Array Leak' root cause
            val_type = type(raw_written_val)
            val_shape = getattr(raw_written_val, "shape", "N/A (Scalar)")
            
            # Log the Forensic DNA of the object returned by the Foundation
            logger.debug(
                f"VERIFY [Audit]: Field {field_id} | Type: {val_type} | "
                f"Shape: {val_shape} | Raw Object: {raw_written_val}"
            )

            # --- [FAIL-SAFE SCALAR EXTRACTION] ---
            # If it's a NumPy array (likely (1,)), we extract the item to prevent the format error.
            if hasattr(raw_written_val, "item"):
                safe_val = raw_written_val.item()
            else:
                safe_val = raw_written_val

            try:
                # Rule 7: Fail-Fast math audit via format string
                logger.debug(f"APPLY [Success]: Verified Field {field_id} = {safe_val:.4e}")
            except TypeError as e:
                logger.critical(
                    f"FORMAT FAILURE: Could not format {field_id} as float. "
                    f"Object is still non-scalar after extraction: {raw_written_val}"
                )
                raise e
            
        else:
            logger.error(f"CONTRACT VIOLATION: Unsupported key '{key}' at {location}")
            raise KeyError(f"Unsupported boundary key '{key}' at {location}")