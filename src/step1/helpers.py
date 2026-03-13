# src/step1/helpers.py

import numpy as np

from src.common.solver_input import GridInput

# Rule 7: Granular Traceability
DEBUG = True

def generate_3d_masks(mask_data: list[int], grid: GridInput) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transforms flat input into 3D topology arrays and verified boolean masks.
    
    Compliance:
    - Rule 0 (Performance): NumPy used for vectorized topology representation.
    - Rule 5 (Deterministic): Strict size validation; no fallback defaults.
    """
    # Grid access via GridInput properties (SSoT)
    nx, ny, nz = int(grid.nx), int(grid.ny), int(grid.nz)
    
    expected_len = nx * ny * nz
    if len(mask_data) != expected_len:
        # Rule 5: Explicit or Error. Silent truncation or padding is prohibited.
        raise ValueError(f"Mask size mismatch: Expected {expected_len}, got {len(mask_data)}")

    # Fortran-style ('F') ordering maintains (i, j, k) logical indexing for spatial solvers
    mask_3d = np.asarray(mask_data, dtype=np.int8).reshape((nx, ny, nz), order="F")
    
    # Logic-Layer: Identify fluid and boundary regions via vectorized masks
    is_fluid = (mask_3d == 1)
    is_boundary = (mask_3d == -1)
    
    if DEBUG:
        print(f"DEBUG [Step 1.2]: Topology Verification (Mask Generated)")
        print(f"  > Grid Dimensions: {nx}x{ny}x{nz}")
        print(f"  > Fluid Volume: {np.sum(is_fluid)} cells")
        
    return mask_3d, is_fluid, is_boundary