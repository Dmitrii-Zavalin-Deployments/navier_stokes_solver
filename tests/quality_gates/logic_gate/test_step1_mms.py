# tests/quality_gates/logic_gate/test_step1_mms.py

import numpy as np

from src.common.field_schema import FI
from src.step1.orchestrate_step1 import orchestrate_step1


def test_logic_gate_1_padded_ingestion(solver_input_schema_dummy):
    """
    Verification: state.fields.data reflects (nx+2)^3 allocation with FI.MASK.
    Target: src/step1/orchestrate_step1.py
    """
    # 1. Setup Input (nx=2, ny=2, nz=2)
    context = solver_input_schema_dummy 
    nx, ny, nz = context.grid.nx, context.grid.ny, context.grid.nz
    expected_cells = (nx + 2) * (ny + 2) * (nz + 2) # (2+2)^3 = 64

    # 2. Action
    state = orchestrate_step1(context)

    # 3. Verification
    assert state.fields.data.shape[0] == expected_cells, f"Expected {expected_cells} cells, got {state.fields.data.shape[0]}"
    assert FI.MASK < state.fields.data.shape[1], "FI.MASK index out of bounds"
    
    # Verify padding logic: Center should be 1, boundary ghosts should be 0 (from np.pad)
    # This assumes the dummy mask input was all 1s
    assert np.any(state.fields.data[:, FI.MASK] == 1.0)
    assert np.any(state.fields.data[:, FI.MASK] == 0.0)