# src/common/field_schema.py
from enum import IntEnum

class FI(IntEnum):
    """
    FieldIndex (FI) Schema:
    The Single Source of Truth for the global fields_buffer column mapping.
    
    This ensures that memory access is consistent across the entire solver.
    """
    VX = 0
    VY = 1
    VZ = 2
    VX_STAR = 3
    VY_STAR = 4
    VZ_STAR = 5
    P = 6
    P_NEXT = 7

    @classmethod
    def num_fields(cls) -> int:
        """Returns the total number of fields in the buffer."""
        return len(cls)