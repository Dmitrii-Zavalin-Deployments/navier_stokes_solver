# tests/quality_gates/logic_gate/test_step1_mms.py

import numpy as np

from src.common.field_schema import FI
from src.step1.orchestrate_step1 import orchestrate_step1


def test_logic_gate_1_padded_ingestion(solver_input_schema_dummy):
    """
    Logic Gate 1: Padded Ingestion Verification
    
    Analytical Challenge: Padded Masking
    Success Metric: (N+2)^3 Size Parity
    Target: src/step1/orchestrate_step1.py
    Compliance: Rule 9 (Topological Mask Unified into Foundation)
    """
    # 1. Setup Input: SimulationContext with nx=2 (from dummy)
    context = solver_input_schema_dummy 
    grid_cfg = context.input_data.grid
    nx, ny, nz = int(grid_cfg.nx), int(grid_cfg.ny), int(grid_cfg.nz)
    
    # Success Metric Calculation: (2+2)*(2+2)*(2+2) = 64 cells
    expected_cells = (nx + 2) * (ny + 2) * (nz + 2) 

    # 2. Action: orchestrate_step1(context)
    # This hydrates FieldManager and flattens the padded 3D mask into the buffer.
    state = orchestrate_step1(context)

    # 3. Verification: (N+2)^3 Size Parity
    # Verify that the FieldManager allocated the correct buffer size.
    actual_cells = state.fields.data.shape[0]
    assert actual_cells == expected_cells, (
        f"MMS FAILURE [Size Parity]: Expected {expected_cells} cells, got {actual_cells}"
    )
    
    # 4. Verification: Schema Consistency (SSoT)
    # Verify the buffer width matches the FieldIndex (FI) schema.
    assert state.fields.data.shape[1] == FI.num_fields(), (
        f"MMS FAILURE [Schema]: Foundation width {state.fields.data.shape[1]} "
        f"does not match FI schema {FI.num_fields()}"
    )

    # 5. Verification: Padded Masking Logic
    # orchestrate_step1 uses np.pad(mask_3d, pad_width=1, constant_values=0)
    # We verify that the mask column in the flattened foundation reflects this.
    mask_buffer = state.fields.data[:, FI.MASK]
    
    # Check that the mask is flattened (1D) within the foundation
    assert mask_buffer.ndim == 1, "MMS FAILURE: Mask buffer must be flattened"
    
    # Verification: Ghost Cell Padding
    # Since we padded with constant_values=0, zeros MUST exist in the buffer.
    unique_vals = np.unique(mask_buffer)
    assert 0.0 in unique_vals, (
        "MMS FAILURE [Padded Masking]: Ghost cell padding (0.0) missing from Foundation buffer. "
        "Check np.pad logic in orchestrate_step1.py"
    )

    # 6. Verification: Logic-Layer Hydration
    assert hasattr(state, 'fields'), "MMS FAILURE: FieldManager not attached to SolverState"
    assert hasattr(state, 'mask'), "MMS FAILURE: MaskManager not attached to SolverState"