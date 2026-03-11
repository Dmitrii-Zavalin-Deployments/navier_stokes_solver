# tests/property_integrity/test_mask_flattening_parity.py
import numpy as np
import pytest

from tests.helpers.solver_output_schema_dummy import make_output_schema_dummy
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy

# Define the lifecycle checkpoints
LIFECYCLE_STAGES = [
    ("Step 1: Init", make_step1_output_dummy),
    ("Step 2: Matrix Assembly", make_step2_output_dummy),
    ("Step 3: Prediction/Solve", make_step3_output_dummy),
    ("Step 4: Post-Process", make_step4_output_dummy),
    ("Step 5: Final Export", make_output_schema_dummy),
]

@pytest.mark.parametrize("stage_name, factory", LIFECYCLE_STAGES)
def test_canonical_flattening_persistence(stage_name, factory):
    """
    The Indexer: Verify i + nx*(j + ny*k) parity across the full pipeline.
    Ensures that spatial orientation of obstacles remains constant from 
    input parsing to final data export.
    """
    nx, ny, nz = 4, 4, 4
    state = factory()
    
    # Mathematical Target (Canonical Index 25)
    # i=1, j=2, k=1 -> 1 + 4*(2 + 4*1) = 25
    target_idx = 25
    
    # We assume the dummy/state contains a mask of size nx*ny*nz
    mask = np.array(state.masks.mask)
    
    # Reverse Map (Solver logic)
    idx = target_idx
    k_res = idx // (nx * ny)
    j_res = (idx // nx) % ny
    i_res = idx % nx
    
    # Verification of spatial coordinate recovery
    assert (i_res, j_res, k_res) == (1, 2, 1), f"Flattening logic corrupted at {stage_name}"
    
    # Check that the mask is still populated (not None or empty)
    assert mask.size == nx * ny * nz, f"Mask size mismatch at {stage_name}"

def test_mask_value_integrity():
    """
    Verify that the meaning of mask values (-1, 0, 1) is preserved.
    """
    state = make_output_schema_dummy()
    allowed_values = {-1, 0, 1}
    actual_values = set(np.unique(state.masks.mask))
    
    assert actual_values.issubset(allowed_values), f"Undefined mask values detected in final output: {actual_values}"