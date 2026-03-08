# src/step3/core/extract.py

import numpy as np

def get_interior_field(field):
    """
    Standardizes the conversion of a field (nx, ny, nz) 
    or (3, nx, ny, nz) into its interior representation.
    """
    if field.ndim == 3:
        # Scalar field (e.g., Pressure)
        return field[1:-1, 1:-1, 1:-1]
    elif field.ndim == 4:
        # Vector field (e.g., Velocity, Force)
        return field[:, 1:-1, 1:-1, 1:-1]
    else:
        raise ValueError("Field must be 3D or 4D (vector) array.")