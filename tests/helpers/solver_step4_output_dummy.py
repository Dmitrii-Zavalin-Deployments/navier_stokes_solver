# tests/helpers/solver_step4_output_dummy.py

"""
Archivist Testing: Snapshot-based Test Baseline (Step 4).

Compliance:
- Rule 4: SSoT (Accessed via state sub-containers).
- Rule 9: Hybrid Memory Foundation (Manipulation of NumPy buffers).
"""

import numpy as np

from src.common.field_schema import FI
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy


def make_step4_output_dummy(nx: int = 4, ny: int = 4, nz: int = 4):
    """
    Returns a 'frozen' prototype representing the system state after 
    boundary enforcement (Step 4). 
    """
    state = make_step3_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # 1. Foundation Injection (Rule 9)
    # We apply the 'Ground Truth' directly to the Foundation (NumPy buffer).
    # Since Cell objects in state.stencil_matrix hold indices into this buffer,
    # the changes are immediately reflected in the logic-layer.
    data = state.fields.data
    
    # 2. Applying Boundary Conditions (Atomic Truth)
    # Instead of iterating logic (which belongs in the orchestrator),
    # we set the expected result for the boundary indices.
    
    # Mask check: For indices where masks.mask == -1, enforce wall conditions
    # (Assuming the mask is available via state.mask.mask)
    wall_indices = np.where(state.mask.mask.flatten() == -1)[0]
    
    data[wall_indices, FI.VX] = 0.0
    data[wall_indices, FI.VY] = 0.0
    data[wall_indices, FI.VZ] = 0.0
    
    # Inflow check: For x_min (i=0), enforce inflow velocity
    # We target the index slice corresponding to x=0
    inflow_indices = np.where(np.indices((nx, ny, nz))[0].flatten() == 0)[0]
    
    data[inflow_indices, FI.VX] = 1.0
    
    return state