# src/step3/ops/forces.py
import numpy as np


def get_body_forces_interior(nx, ny, nz, Fx_val, Fy_val, Fz_val):
    """
    Returns the body force field F on the interior grid (nx-2, ny-2, nz-2).
    This ensures it is ready for immediate addition to the velocity field.
    """
    # Create the interior-only grid directly
    Fx = np.full((nx-2, ny-2, nz-2), Fx_val, dtype=np.float64)
    Fy = np.full((nx-2, ny-2, nz-2), Fy_val, dtype=np.float64)
    Fz = np.full((nx-2, ny-2, nz-2), Fz_val, dtype=np.float64)
    
    return np.array([Fx, Fy, Fz])