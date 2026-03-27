# tests/quality_gates/logic_gate/test_step1_mms.py

import numpy as np
import pytest

from src.common.field_schema import FI
from src.step1.orchestrate_step1 import orchestrate_step1
from tests.helpers.solver_input_schema_dummy import create_validated_input

def test_logic_gate_1_padded_ingestion():
    """
    Logic Gate 1: Padded Ingestion Verification
    
    Analytical Challenge: Padded Masking
    Success Metric: (N+2)^3 Size Parity
    Compliance: Rule 9 (Topological Mask Unified into Foundation)
    Compliance: Rule 4 (SSoT Hierarchy - No Facade Properties)
    """

    # 1. Setup Input: Explicitly define grid to avoid Hidden Defaults (Rule 5)
    nx, ny, nz = 2, 2, 2
    context = create_validated_input(nx=nx, ny=ny, nz=nz) 
    
    # Success Metric Calculation: (2+2)*(2+2)*(2+2) = 64 cells
    # This reflects the Ghost Cell Padding requirement for the Foundation.
    expected_cells = (nx + 2) * (ny + 2) * (nz + 2) 

    # 2. Action: orchestrate_step1(context)
    # This hydrates the FieldManager (Foundation) and flattens the padded 3D mask.
    state = orchestrate_step1(context)

    # 3. Verification: (N+2)^3 Size Parity (Rule 1: Field Precision/Scale Audit)
    # Verify that the Foundation (NumPy buffer) allocated the correct size.
    # Data is accessed via the Physical Context container (state.fields).
    actual_cells = state.fields.data.shape[0]
    assert actual_cells == expected_cells, (
        f"MMS FAILURE [Size Parity]: Expected {expected_cells} cells, got {actual_cells}. "
        "Check Rule 9 Ghost Padding logic."
    )
    
    # 4. Verification: Schema Consistency (Rule 1 & Rule 7)
    # Verify the buffer width matches the FieldIndex (FI) atomic numerical truth.
    actual_width = state.fields.data.shape[1]
    expected_width = FI.num_fields()
    assert actual_width == expected_width, (
        f"MMS FAILURE [Schema]: Foundation width {actual_width} "
        f"does not match FI schema {expected_width}."
    )

    # 5. Verification: Padded Masking Logic
    # SSoT Check: Mask data must reside in state.fields.data, not a facade property.
    mask_buffer = state.fields.data[:, FI.MASK]
    
    # Check that the mask is flattened (1D) within the foundation
    assert mask_buffer.ndim == 1, "MMS FAILURE: Mask buffer must be a 1D slice of the Foundation."
    
    # Verification: Ghost Cell Padding
    # Since we padded with constant_values=0, zeros MUST exist in the buffer.
    # This validates the "Padded Masking Unified into Foundation" mandate.
    unique_vals = np.unique(mask_buffer)
    assert 0.0 in unique_vals, (
        "MMS FAILURE [Padded Masking]: Ghost cell padding (0.0) missing from Foundation buffer."
    )

    # 6. Verification: Logic-Layer Hydration & SSoT Compliance (Rule 4)
    # RULE: Hierarchy over Convenience. 
    # Ensure managers are attached to their proper sub-containers.
    assert hasattr(state, 'fields'), "SSoT BREACH: state.fields (Physical Context) missing."
    assert hasattr(state, 'grid'), "SSoT BREACH: state.grid (Geometric Context) missing."
    
    # ARCHITECTURAL CHECK: If 'mask' is a logic manager, it should be a sub-component 
    # of the FieldManager or its own Registry, but NEVER a flat alias if it duplicates data.
    # Based on Rule 4: "Adding facade properties... is strictly prohibited."
    # We verify the state object isn't "polluted" with convenience aliases.
    assert not hasattr(state, 'nx'), "Rule 4 Violation: Found 'nx' alias on SolverState. Use state.grid.nx."
    assert not hasattr(state, 'density'), "Rule 4 Violation: Found 'density' alias on SolverState."