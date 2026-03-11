# src/tests/test_foundation.py

import numpy as np

from src.common.field_schema import FI


def verify_foundation_integrity(state):
    """
    POST: Verifies that the StencilBlock pointer graph is 
    perfectly aligned with the underlying NumPy FieldManager foundation.
    
    Raises:
        RuntimeError: If a memory swap or indexing mismatch is detected.
    """
    # 1. Prime the foundation with the Identity Signature
    # Value = Index + (Field_ID / 10.0)
    # Example: Cell at index 50, Field P (ID 6) -> 50.6
    for f_id in FI:
        state.fields.data[:, f_id] = np.arange(state.n_cells) + (float(f_id) / 10.0)
        
    # 2. Verify via object-pointer graph (The Sentinel Test)
    # We check a sample cell (index 50) and a boundary/corner cell
    test_indices = [50, 0, state.n_cells - 1]
    
    for idx in test_indices:
        cell = state.stencil_matrix[idx].center
        
        # Check primary fields mapping
        expected_p = idx + (FI.P / 10.0)
        if not np.isclose(cell.p, expected_p):
            raise RuntimeError(
                f"CRITICAL: Memory Swap Detected at Cell {idx}! "
                f"Expected P={expected_p}, got {cell.p}"
            )
            
        # Check intermediate fields mapping
        expected_vx_star = idx + (FI.VX_STAR / 10.0)
        if not np.isclose(cell.vx_star, expected_vx_star):
            raise RuntimeError(
                f"CRITICAL: Memory Swap Detected at Cell {idx}! "
                f"Expected VX_STAR={expected_vx_star}, got {cell.vx_star}"
            )

    print("POST SUCCESS: Foundation integrity verified. Architecture Locked.")