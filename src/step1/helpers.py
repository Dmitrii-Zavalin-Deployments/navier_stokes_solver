# src/step1/helpers.py

import numpy as np
from typing import Dict, List, Tuple
from src.solver_input import GridInput, BoundaryConditionItem

# Global Debug Toggle - Rule 7: Granular Traceability
DEBUG = True

def allocate_staggered_fields(grid: GridInput) -> Dict[str, np.ndarray]:
    """Allocates memory for the Harlow-Welch staggered grid."""
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    # Staggered Grid Rule: Velocity components have N+1 points in their direction
    fields = {
        "P": np.zeros((nx, ny, nz), dtype=np.float64),
        "U": np.zeros((nx + 1, ny, nz), dtype=np.float64),
        "V": np.zeros((nx, ny + 1, nz), dtype=np.float64),
        "W": np.zeros((nx, ny, nz + 1), dtype=np.float64),
    }

    if DEBUG:
        print(f"DEBUG [Step 1.1]: Harlow-Welch Staggering Check:")
        print(f"  > P-Grid (Cell Center): {fields['P'].shape}")
        print(f"  > U-Face (East-West):   {fields['U'].shape}")
        print(f"  > V-Face (North-South): {fields['V'].shape}")
        print(f"  > W-Face (Front-Back):  {fields['W'].shape}")

    return fields

def generate_3d_masks(mask_data: List[int], grid: GridInput) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transforms the flat JSON mask into 3D topology arrays."""
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    
    expected_len = nx * ny * nz
    if len(mask_data) != expected_len:
        raise ValueError(f"Mask size mismatch: Expected {expected_len}, got {len(mask_data)}")

    # Order 'F' is critical for Fortran-style indexing (i, j, k)
    mask_3d = np.asarray(mask_data, dtype=np.int8).reshape((nx, ny, nz), order="F")
    
    is_fluid = (mask_3d == 1)
    is_boundary = (mask_3d == -1)
    
    if DEBUG:
        print(f"DEBUG [Step 1.2]: Topology Verification:")
        print(f"  > Target Domain: {nx}x{ny}x{nz}")
        print(f"  > Fluid Volume: {np.sum(is_fluid)} cells")
        print(f"  > Solid/Boundary Volume: {np.sum(is_boundary)} cells")
        
    return mask_3d, is_fluid, is_boundary

def parse_bc_lookup(items: List[BoundaryConditionItem]) -> Dict[str, Dict]:
    """
    Converts BC list into a high-speed lookup table.
    Rule 5 Violation Fixed: Removed .get() defaults.
    """
    table = {}
    for item in items:
        # Accessing keys directly: If 'u', 'v', 'w', or 'p' is missing, 
        # it will raise a KeyError, satisfying the "Explicit or Error" mandate.
        table[item.location] = {
            "type": item.type,
            "u": float(item.values["u"]),
            "v": float(item.values["v"]),
            "w": float(item.values["w"]),
            "p": float(item.values["p"])
        }
        if DEBUG:
            print(f"DEBUG [Step 1.3]: BC Map -> {item.location}: type={item.type}")
            
    return table