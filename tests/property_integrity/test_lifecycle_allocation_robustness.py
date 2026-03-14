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
    
    # Access the monolithic data buffer
    data = state.fields.data
    
    # Assertions based on FI index mapping (Standardizing buffer slicing)
    # Note: If your fields are flattened, ensure the shape matches the FI requirements
    assert data[:, FI.P].reshape(nx, ny, nz).shape == (nx, ny, nz)
    assert data[:, FI.VX].reshape(nx + 1, ny, nz).shape == (nx + 1, ny, nz)
    assert data[:, FI.VY].reshape(nx, ny + 1, nz).shape == (nx, ny + 1, nz)
    assert data[:, FI.VZ].reshape(nx, ny, nz + 1).shape == (nx, ny, nz + 1)

def test_step3_intermediate_field_allocation():
    """
    Validation: Predictor fields (e.g., U_star) must exist in the buffer.
    """
    nx, ny, nz = 5, 5, 5
    state = make_step3_output_dummy(nx=nx, ny=ny, nz=nz)
    data = state.fields.data
    
    # Accessing intermediate storage mapped via FI
    assert data[:, FI.U_STAR].reshape(nx + 1, ny, nz).shape == (nx + 1, ny, nz)
    assert data[:, FI.V_STAR].reshape(nx, ny + 1, nz).shape == (nx, ny + 1, nz)
    assert data[:, FI.W_STAR].reshape(nx, ny, nz + 1).shape == (nx, ny, nz + 1)

def test_ghost_cell_allocation_logic():
    """
    Verify the 'Ghost Cell' expansion logic within the monolithic buffer.
    """
    nx, ny, nz = 10, 10, 10
    
    for stage_name, factory in [("Step 4", make_step4_output_dummy), 
                                ("Step 5", make_step5_output_dummy),
                                ("Final Output", make_output_schema_dummy)]:
        state = factory(nx=nx, ny=ny, nz=nz)
        data = state.fields.data
        
        # Accessing extended (ghost-cell included) fields via FI
        assert data[:, FI.P_EXT].reshape(nx + 2, ny + 2, nz + 2).shape == (nx + 2, ny + 2, nz + 2)
        assert data[:, FI.U_EXT].reshape(nx + 3, ny + 2, nz + 2).shape == (nx + 3, ny + 2, nz + 2)