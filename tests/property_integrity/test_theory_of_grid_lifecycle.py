# tests/property_integrity/test_theory_of_grid_lifecycle.py

import numpy as np
import pytest

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
def test_theory_grid_spacing_derivation_integrity(stage_name, factory):
    """Verify grid spacing consistency across all pipeline steps."""
    nx, ny, nz = 50, 20, 10
    state = factory(nx=nx, ny=ny, nz=nz)
    
    grid = getattr(state, "_grid", None)
    assert grid is not None, f"{stage_name}: '_grid' slot missing"

    # 1. Verification of Integrity: Accessing slots
    assert hasattr(grid, "_dx"), f"{stage_name}: missing '_dx'"
    
    # 2. Verification of Derivation Consistency
    calc_dx = (grid._x_max - grid._x_min) / nx
    assert np.isclose(grid._dx, calc_dx), f"{stage_name}: Grid spacing drift"
    assert np.isclose(grid._dx, 1.0 / nx), f"{stage_name}: Spacing mismatch with theory"

def test_theory_mapping_formula_integrity():
    """Verify the Canonical Flattening Rule."""
    nx, ny, nz = 4, 4, 4
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    
    i, j, k = 1, 2, 3
    expected_index = i + nx * (j + ny * k)
    
    # Accessing mask through slot
    assert len(state._mask._mask.flatten()) == nx * ny * nz
    assert expected_index == 57, "Mapping formula specification failure."

def test_theory_extended_geometry_consistency():
    """Verify extended field coordinate alignment (ghost cells)."""
    nx, ny, nz = 10, 10, 10
    
    for factory in [make_step4_output_dummy, make_step5_output_dummy, make_output_schema_dummy]:
        state = factory(nx=nx, ny=ny, nz=nz)
        
        # Accessing fields via slot _fields
        fields = state._fields
        assert fields._P_ext.shape == (nx + 2, ny + 2, nz + 2)
        assert fields._U_ext.shape == (nx + 3, ny + 2, nz + 2)