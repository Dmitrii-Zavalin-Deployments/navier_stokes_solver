# src/step3/ops/forces.py

import numpy as np


def get_body_forces(nx, ny, nz, Fx_val, Fy_val, Fz_val):
    """
    Returns the body force field F = (Fx, Fy, Fz).
    
    Arguments are mandatory to ensure the physical environment 
    is explicitly defined by the caller.
    """
    Fx = np.full((nx, ny, nz), Fx_val, dtype=np.float64)
    Fy = np.full((nx, ny, nz), Fy_val, dtype=np.float64)
    Fz = np.full((nx, ny, nz), Fz_val, dtype=np.float64)
    
    return np.array([Fx, Fy, Fz])