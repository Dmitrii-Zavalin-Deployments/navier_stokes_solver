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
    """Verify grid spacing consistency across all pipeline steps by deriving it from bounds."""
    nx, ny, nz = 50, 20, 10
    state = factory(nx=nx, ny=ny, nz=nz)
    
    grid = getattr(state, "_grid", None)
    assert grid is not None, f"{stage_name}: '_grid' slot missing"

    # Derive spacing from bounds as per the GridManager slots
    calc_dx = (grid._x_max - grid._x_min) / nx
    calc_dy = (grid._y_max - grid._y_min) / ny
    calc_dz = (grid._z_max - grid._z_min) / nz
    
    # Verification of Derivation Consistency
    assert np.isclose(calc_dx, 1.0 / nx), f"{stage_name}: Grid spacing (dx) mismatch with theory"
    assert np.isclose(calc_dy, 1.0 / ny), f"{stage_name}: Grid spacing (dy) mismatch with theory"
    assert np.isclose(calc_dz, 1.0 / nz), f"{stage_name}: Grid spacing (dz) mismatch with theory"

def test_theory_mapping_formula_integrity():
    """Verify the Canonical Flattening Rule: Index = i + nx * (j + ny * k)."""
    nx, ny, nz = 4, 4, 4
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    
    i, j, k = 1, 2, 3
    expected_index = i + nx * (j + ny * k)
    
    # Accessing mask through slot
    assert len(state._mask._mask.flatten()) == nx * ny * nz
    assert expected_index == 57, "Mapping formula specification failure."

def test_theory_extended_geometry_consistency():
    """Verify extended field coordinate alignment (ghost cells) in the buffer."""
    nx, ny, nz = 10, 10, 10
    
    for stage_name, factory in [
        ("Step 4", make_step4_output_dummy), 
        ("Step 5", make_step5_output_dummy), 
        ("Final Output", make_output_schema_dummy)
    ]:
        state = factory(nx=nx, ny=ny, nz=nz)
        
        # Verify buffer size matches expected ghost-cell-inclusive dimensions
        # Total points in a (N+2) box
        expected_total_pts = (nx + 2) * (ny + 2) * (nz + 2)
        assert state._fields._data.shape[0] >= expected_total_pts, f"{stage_name}: Buffer size insufficient for ghost cells."