# src/step1/helpers.py

import numpy as np

from src.solver_input import BoundaryConditionItem, GridInput

# Global Debug Toggle - Rule 7: Granular Traceability
DEBUG = True

def allocate_fields(grid: GridInput) -> dict[str, np.ndarray]:
    """
    Allocates memory for Cell-Centered fields.
    
    Theory Mapping: Section 3 - All fields (P, U, V, W) share the same spatial domain.
    
    Mathematical Trace:
    Let \phi \in \{P, U, V, W\}. 
    The domain \Omega \subset \mathbb{R}^3 is discretized into N = nx * ny * nz cells.
    Every field \phi is mapped to indices (i, j, k) \in [0, nx-1] \times [0, ny-1] \times [0, nz-1].
    """
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    # Collocated Grid: Every field is defined at the cell center (i, j, k).
    # All fields share shape (nx, ny, nz).
    fields = {
        "P": np.zeros((nx, ny, nz), dtype=np.float64),
        "U": np.zeros((nx, ny, nz), dtype=np.float64),
        "V": np.zeros((nx, ny, nz), dtype=np.float64),
        "W": np.zeros((nx, ny, nz), dtype=np.float64),
    }

    if DEBUG:
        print(f"DEBUG [Step 1.1]: Collocated Grid Allocation (Theory Section 3):")
        print(f"  > All Fields (P, U, V, W) shape: {fields['P'].shape}")

    return fields

def generate_3d_masks(mask_data: list[int], grid: GridInput) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transforms the flat JSON mask into 3D topology arrays.
    
    Theory Mapping: Section 6 - Mask-Based Geometry.
    """
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    
    expected_len = nx * ny * nz
    if len(mask_data) != expected_len:
        raise ValueError(f"Mask size mismatch: Expected {expected_len}, got {len(mask_data)}")

    # Order 'F' is critical for Fortran-style indexing (i, j, k)
    mask_3d = np.asarray(mask_data, dtype=np.int8).reshape((nx, ny, nz), order="F")
    
    is_fluid = (mask_3d == 1)
    is_boundary = (mask_3d == -1)
    
    if DEBUG:
        print(f"DEBUG [Step 1.2]: Topology Verification (Theory Section 6):")
        print(f"  > Target Domain: {nx}x{ny}x{nz}")
        print(f"  > Fluid Volume: {np.sum(is_fluid)} cells")
        
    return mask_3d, is_fluid, is_boundary

def parse_bc_lookup(items: list[BoundaryConditionItem]) -> dict[str, dict]:
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