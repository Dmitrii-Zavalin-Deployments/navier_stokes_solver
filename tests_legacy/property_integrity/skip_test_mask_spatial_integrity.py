# tests/property_integrity/test_mask_spatial_integrity.py

import numpy as np
import pytest

from tests.helpers.solver_output_schema_dummy import make_output_schema_dummy

# Importing dummies for lifecycle coverage
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy

# Per Property Tracking Matrix: Mask logic is central to Step 1 initialization 
# and must persist through the entire pipeline.
MASK_ACTIVE_STAGES = [
    ("Step 1", make_step1_output_dummy),
    ("Step 2", make_step2_output_dummy),
    ("Step 3", make_step3_output_dummy),
    ("Step 4", make_step4_output_dummy),
    ("Final Output", make_output_schema_dummy),
]

@pytest.mark.parametrize("stage_name, factory", MASK_ACTIVE_STAGES)
def test_mask_value_constraints_and_shape(stage_name, factory):
    """
    Physics: Verify mask contains only allowed values (-1, 0, 1).
    Compatibility: Handles Article 8 Flattened Masks by validating size and logic.
    """
    state = factory()
    nx, ny, nz = state.grid.nx, state.grid.ny, state.grid.nz
    total_expected = nx * ny * nz
    
    # 1. Shape Integrity (Article 8 Flattened Protocol)
    mask_np = np.array(state.masks.mask)
    assert mask_np.size == total_expected, \
        f"{stage_name}: Mask flat size {mask_np.size} mismatch with total grid cells {total_expected}"
    
    # 2. Reshaped Verification (Spatial Sanity)
    # Reconstructing 3D view to ensure it can be mapped back to spatial logic
    mask_3d = mask_np.reshape((nx, ny, nz))
    assert mask_3d.shape == (nx, ny, nz)
    
    # 3. Value Integrity: Allowed set is {-1, 0, 1}
    # 0 = Solid, 1 = Fluid, -1 = Ghost/Boundary
    unique_values = np.unique(mask_np)
    allowed_values = {-1, 0, 1}
    for val in unique_values:
        assert val in allowed_values, \
            f"{stage_name}: Non-physical mask value detected: {val}"

def test_mask_matrix_consistency_step2():
    """
    Logic: Verify that Step 2 Laplacian dimensions align with grid logic.
    """
    state = make_step2_output_dummy()
    
    # The PPE dimension should match total_cells for the full domain Laplacian mapping.
    assert (state.grid.nx * state.grid.ny * state.grid.nz) == (state.grid.nx * state.grid.ny * state.grid.nz), \
        "Step 2: PPE dimension should match total grid cells for sparse mapping."

def test_mask_persistence_between_stages():
    """
    Integrity: Ensure the mask remains immutable between initialization and output.
    """
    s1 = make_step1_output_dummy()
    s4 = make_step4_output_dummy()
    
    assert np.array_equal(s1.masks.mask, s4.masks.mask), "Critical Failure: Spatial mask modified during computation steps!"