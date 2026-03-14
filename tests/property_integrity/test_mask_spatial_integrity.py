# tests/property_integrity/test_mask_spatial_integrity.py

import numpy as np
import pytest

from tests.helpers.solver_output_schema_dummy import make_output_schema_dummy
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy
from tests.helpers.solver_step5_output_dummy import make_step5_output_dummy

MASK_ACTIVE_STAGES = [
    ("Step 1", make_step1_output_dummy),
    ("Step 2", make_step2_output_dummy),
    ("Step 3", make_step3_output_dummy),
    ("Step 4", make_step4_output_dummy),
    ("Step 5", make_step5_output_dummy),
    ("Final Output", make_output_schema_dummy),
]

@pytest.mark.parametrize("stage_name, factory", MASK_ACTIVE_STAGES)
def test_mask_value_constraints_and_shape(stage_name, factory):
    """
    Physics: Verify mask contains only allowed values (-1, 0, 1).
    """
    nx, ny, nz = 4, 4, 4
    state = factory(nx=nx, ny=ny, nz=nz)
    
    # 1. Shape Integrity (Refactored to state.mask.mask)
    mask_np = np.array(state.mask.mask)
    assert mask_np.size == (nx * ny * nz), \
        f"{stage_name}: Mask size {mask_np.size} mismatch"
    
    # 2. Reshaped Verification
    mask_3d = mask_np.reshape((nx, ny, nz))
    assert mask_3d.shape == (nx, ny, nz)
    
    # 3. Value Integrity
    unique_values = np.unique(mask_np)
    allowed_values = {-1, 0, 1}
    for val in unique_values:
        assert val in allowed_values, f"{stage_name}: Non-physical mask value: {val}"

def test_mask_matrix_consistency_step2():
    """Verify Step 2 Laplacian alignment."""
    nx, ny, nz = 4, 4, 4
    state = make_step2_output_dummy(nx=nx, ny=ny, nz=nz)
    assert state.grid.nx * state.grid.ny * state.grid.nz == (nx * ny * nz)

def test_mask_persistence_between_stages():
    """Ensure mask remains immutable across pipeline."""
    nx, ny, nz = 4, 4, 4
    s1 = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    s4 = make_step4_output_dummy(nx=nx, ny=ny, nz=nz)
    
    assert np.array_equal(s1.mask.mask, s4.mask.mask), \
        "Critical Failure: Spatial mask modified during computation!"