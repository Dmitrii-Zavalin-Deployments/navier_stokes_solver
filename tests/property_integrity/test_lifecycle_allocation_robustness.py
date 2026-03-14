# tests/property_integrity/test_lifecycle_allocation_robustness.py

import pytest

from src.common.field_schema import FI
from tests.helpers.solver_output_schema_dummy import make_output_schema_dummy
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy
from tests.helpers.solver_step5_output_dummy import make_step5_output_dummy

LIFECYCLE_STAGES = [
    ("Step 1", make_step1_output_dummy),
    ("Step 2", make_step2_output_dummy),
    ("Step 3", make_step3_output_dummy),
    ("Step 4", make_step4_output_dummy),
    ("Step 5", make_step5_output_dummy),
    ("Final Output", make_output_schema_dummy),
]

@pytest.mark.parametrize("stage_name, factory", LIFECYCLE_STAGES)
def test_lifecycle_grid_dimensions_match_fields(stage_name, factory):
    """
    Robustness: Verifies Arakawa C-Grid staggering using the FieldManager's 
    monolithic data buffer and the FI index schema.
    """
    nx, ny, nz = 8, 6, 4
    state = factory(nx=nx, ny=ny, nz=nz)
    
    data = state.fields.data
    
    # Assertions using FI mapping with descriptive error messages
    assert data[:, FI.P].reshape(nx, ny, nz).shape == (nx, ny, nz), \
        f"{stage_name}: Pressure field shape mismatch"
    assert data[:, FI.VX].reshape(nx + 1, ny, nz).shape == (nx + 1, ny, nz), \
        f"{stage_name}: VX field shape mismatch"
    assert data[:, FI.VY].reshape(nx, ny + 1, nz).shape == (nx, ny + 1, nz), \
        f"{stage_name}: VY field shape mismatch"
    assert data[:, FI.VZ].reshape(nx, ny, nz + 1).shape == (nx, ny, nz + 1), \
        f"{stage_name}: VZ field shape mismatch"

def test_step3_intermediate_field_allocation():
    """
    Validation: Predictor fields (U_STAR, V_STAR, W_STAR) must exist in the buffer.
    """
    nx, ny, nz = 5, 5, 5
    state = make_step3_output_dummy(nx=nx, ny=ny, nz=nz)
    data = state.fields.data
    
    assert data[:, FI.U_STAR].reshape(nx + 1, ny, nz).shape == (nx + 1, ny, nz)
    assert data[:, FI.V_STAR].reshape(nx, ny + 1, nz).shape == (nx, ny + 1, nz)
    assert data[:, FI.W_STAR].reshape(nx, ny, nz + 1).shape == (nx, ny, nz + 1)

def test_ghost_cell_allocation_logic():
    """
    Verify the 'Ghost Cell' expansion logic for staggered velocity faces.
    """
    nx, ny, nz = 10, 10, 10
    
    stages = [
        ("Step 4", make_step4_output_dummy),
        ("Step 5", make_step5_output_dummy),
        ("Final Output", make_output_schema_dummy)
    ]
    
    for stage_name, factory in stages:
        state = factory(nx=nx, ny=ny, nz=nz)
        data = state.fields.data
        
        # Accessing extended (ghost-cell included) fields via FI
        assert data[:, FI.P_EXT].reshape(nx + 2, ny + 2, nz + 2).shape == (nx + 2, ny + 2, nz + 2), \
            f"Failure at {stage_name}: P_ext shape mismatch"
        assert data[:, FI.U_EXT].reshape(nx + 3, ny + 2, nz + 2).shape == (nx + 3, ny + 2, nz + 2), \
            f"Failure at {stage_name}: U_ext shape mismatch"