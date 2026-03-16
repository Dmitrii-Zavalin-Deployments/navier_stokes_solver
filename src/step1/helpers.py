# src/step1/helpers.py

import numpy as np

from src.common.grid_math import get_coords_from_index
from src.common.solver_input import GridInput

# Rule 7: Granular Traceability
DEBUG = True


def generate_3d_masks(mask_data: list[int], grid: GridInput) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transforms flat input into 3D topology arrays via explicit SSoT mapping.
    
    Compliance:
    - Rule 4 (SSoT): Mapping logic derived from grid_math.py.
    - Rule 5 (Deterministic): Strict boundary validation; no implicit resizing.
    """
    nx, ny, nz = int(grid.nx), int(grid.ny), int(grid.nz)
    
    # 1. Initialize empty buffer
    mask_3d = np.zeros((nx, ny, nz), dtype=np.int8)
    
    # 2. Explicit mapping (Eliminating "Black Box" reshape)
    # We map the flat input list to the 3D grid using our SSoT logic.
    # Note: We do NOT use 'is_buffered=True' here because the input mask
    # from JSON is a pure interior grid (e.g., 4x4x4) without halo cells.
    for idx, value in enumerate(mask_data):
        # Ensure your grid_math.py version of this function supports 
        # a toggle or doesn't force a -1 offset for interior mapping.
        i, j, k = get_coords_from_index(idx, nx, ny) 
        
        # Guard against index drift (Deterministic validation)
        if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz:
            mask_3d[i, j, k] = value
        else:
            # This is where the overflow was triggered due to (-1, -1, -1) mapping.
            raise ValueError(f"Mask mapping overflow at index {idx} -> ({i}, {j}, {k})")

    # 3. Logic-Layer: Identify fluid and boundary regions via vectorized masks
    is_fluid = (mask_3d == 1)
    is_boundary = (mask_3d == -1)
    
    if DEBUG:
        print(f"DEBUG [Step 1.2]: Topology Verification (Mask Generated)")
        print(f"  > Grid Dimensions: {nx}x{ny}x{nz}")
        print(f"  > Fluid Volume: {np.sum(is_fluid)} cells")
        
    return mask_3d, is_fluid, is_boundary