# tests/property_integrity/test_lifecycle_allocation_robustness.py

import pytest
import numpy as np

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
    Robustness: Verifies Arakawa C-Grid staggering using the FieldManager interface.
    """
    nx, ny, nz = 8, 6, 4
    state = factory(nx=nx, ny=ny, nz=nz)
    
    # Grid access via SSoT Manager
    grid = state._grid
    assert grid.nx == nx and grid.ny == ny and grid.nz == nz

    # Access fields via the FieldManager (Rule 9)
    fm = state._fields
    
    assert fm.P.shape == (nx, ny, nz)
    assert fm.U.shape == (nx + 1, ny, nz)
    assert fm.V.shape == (nx, ny + 1, nz)
    assert fm.W.shape == (nx, ny, nz + 1)

def test_step3_intermediate_field_allocation():
    """
    Validation: Predictor fields (U*, V*, W*) must follow staggered face dimensions.
    """
    nx, ny, nz = 5, 5, 5
    state = make_step3_output_dummy(nx=nx, ny=ny, nz=nz)
    fm = state._fields
    
    assert fm.U_star.shape == (nx + 1, ny, nz)
    assert fm.V_star.shape == (nx, ny + 1, nz)
    assert fm.W_star.shape == (nx, ny, nz + 1)

def test_ghost_cell_allocation_logic():
    """
    Verify the 'Ghost Cell' expansion logic for staggered velocity faces.
    """
    nx, ny, nz = 10, 10, 10
    
    # Check all stages that involve ghost cell expansion
    stages = [
        ("Step 4", make_step4_output_dummy),
        ("Step 5", make_step5_output_dummy),
        ("Final Output", make_output_schema_dummy)
    ]
    
    for stage_name, factory in stages:
        state = factory(nx=nx, ny=ny, nz=nz)
        fm = state._fields
        
        # P_ext (Cell-centered): (10+2, 10+2, 10+2)
        assert fm.P_ext.shape == (nx + 2, ny + 2, nz + 2), f"{stage_name} P_ext mismatch"
        
        # U_ext (Face-centered in X + 2 ghost): (10+1+2, 10+2, 10+2)
        assert fm.U_ext.shape == (nx + 3, ny + 2, nz + 2), f"{stage_name} U_ext mismatch"
        assert fm.V_ext.shape == (nx + 2, ny + 3, nz + 2), f"{stage_name} V_ext mismatch"
        assert fm.W_ext.shape == (nx + 2, ny + 2, nz + 3), f"{stage_name} W_ext mismatch"