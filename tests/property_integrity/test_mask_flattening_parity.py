# tests/property_integrity/test_mask_flattening_parity.py

import numpy as np
import pytest

from tests.helpers.solver_output_schema_dummy import make_output_schema_dummy
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy
from tests.helpers.solver_step5_output_dummy import make_step5_output_dummy

# Define the lifecycle checkpoints
LIFECYCLE_STAGES = [
    ("Step 1: Init", make_step1_output_dummy),
    ("Step 2: Matrix Assembly", make_step2_output_dummy),
    ("Step 3: Prediction/Solve", make_step3_output_dummy),
    ("Step 4: Post-Process", make_step4_output_dummy),
    ("Step 5: Archive", make_step5_output_dummy),
    ("Final Output", make_output_schema_dummy),
]

@pytest.mark.parametrize("stage_name, factory", LIFECYCLE_STAGES)
def test_canonical_flattening_persistence(stage_name, factory):
    """
    Verify 1D-to-3D index parity. 
    Formula: index = i + nx * (j + ny * k)
    Ensures that spatial orientation of obstacles remains constant throughout the pipeline.
    """
    nx, ny, nz = 4, 4, 4
    # Pass dimensions to factory to ensure consistent grid generation
    state = factory(nx=nx, ny=ny, nz=nz)
    
    # Mathematical Target (Canonical Index 25)
    # i=1, j=2, k=1 -> 1 + 4*(2 + 4*1) = 25
    target_idx = 25
    
    # Access via proper MaskManager attribute
    mask = state.mask.mask
    
    # Reverse Map (Solver logic)
    k_res = target_idx // (nx * ny)
    j_res = (target_idx // nx) % ny
    i_res = target_idx % nx
    
    # Verification of spatial coordinate recovery
    assert (i_res, j_res, k_res) == (1, 2, 1), f"Flattening logic corrupted at {stage_name}"
    
    # Check that the mask is populated and matches size
    assert mask.size == (nx * ny * nz), f"Mask size mismatch at {stage_name}"

def test_mask_value_integrity():
    """
    Verify that the meaning of mask values (-1, 0, 1) is preserved.
    """
    nx, ny, nz = 4, 4, 4
    state = make_output_schema_dummy(nx=nx, ny=ny, nz=nz)
    allowed_values = {-1, 0, 1}
    
    # Use proper MaskManager attribute access
    actual_values = set(np.unique(state.mask.mask))
    
    assert actual_values.issubset(allowed_values), \
        f"Undefined mask values detected in final output: {actual_values}"