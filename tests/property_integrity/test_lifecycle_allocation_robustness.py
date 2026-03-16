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
    Robustness: Verifies buffer size allocation against grid cell count (nx * ny * nz).
    """
    nx, ny, nz = 8, 6, 4
    n_cells = (nx + 2) * (ny + 2) * (nz + 2)
    state = factory(nx=nx, ny=ny, nz=nz)
    
    data = state.fields.data
    
    # Assertions verify that the monolithic column length matches expected cell count
    for field_idx in [FI.P, FI.VX, FI.VY, FI.VZ]:
        assert data[:, field_idx].size == n_cells, f"{stage_name}: Field index {field_idx} size mismatch"

def test_step3_intermediate_field_allocation():
    """
    Validation: Predictor fields (VX_STAR, VY_STAR, VZ_STAR) exist and are allocated to n_cells.
    """
    nx, ny, nz = 5, 5, 5
    n_cells = (nx + 2) * (ny + 2) * (nz + 2)
    state = make_step3_output_dummy(nx=nx, ny=ny, nz=nz)
    data = state.fields.data
    
    assert data[:, FI.VX_STAR].size == n_cells
    assert data[:, FI.VY_STAR].size == n_cells
    assert data[:, FI.VZ_STAR].size == n_cells

def test_ghost_cell_allocation_logic():
    """
    Verify storage capacity. Note: Ghost cells are typically handled via padding
    or specialized stencils; if they are not columns in the monolithic buffer, 
    this test will need to be refactored to point to the actual ghost-cell storage.
    """
    nx, ny, nz = 10, 10, 10
    stages = [
        ("Step 4", make_step4_output_dummy),
        ("Step 5", make_step5_output_dummy),
        ("Final Output", make_output_schema_dummy)
    ]
    
    for stage_name, factory in stages:
        state = factory(nx=nx, ny=ny, nz=nz)
        # If P_EXT/U_EXT are not in FI, we cannot access them via state.fields.data[:, FI...]
        # Verify the architecture if these fields are supposed to exist in the monolithic buffer
        expected_capacity = (nx + 2) * (ny + 2) * (nz + 2)
        assert state.fields.data.shape[0] == expected_capacity, f"Failure at {stage_name}: Capacity mismatch"