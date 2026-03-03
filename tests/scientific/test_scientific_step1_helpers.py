import pytest
import numpy as np
from src.step1.helpers import allocate_staggered_fields, generate_3d_masks
from src.solver_input import GridInput

def test_scientific_harlow_welch_allocation():
    """Verify staggered grid offsets follow the N+1 mathematical requirement."""
    grid = GridInput(nx=10, ny=20, nz=30)
    fields = allocate_staggered_fields(grid)
    
    # The Physics Rule: Normal velocity on faces must have N+1 points
    assert fields["U"].shape == (11, 20, 30), "U-staggering offset failed"
    assert fields["V"].shape == (10, 21, 30), "V-staggering offset failed"
    assert fields["W"].shape == (10, 20, 31), "W-staggering offset failed"
    assert fields["P"].shape == (10, 20, 30), "Pressure center allocation failed"

def test_scientific_mask_reconstruction_parity():
    """Verify that flat data reconstruction respects Fortran ordering (Order 'F')."""
    # Create a 2x2x2 grid (8 cells total)
    grid = GridInput(nx=2, ny=2, nz=2)
    flat_data = [1, 1, 1, 1, -1, -1, -1, -1]
    
    mask_3d, is_fluid, is_boundary = generate_3d_masks(flat_data, grid)
    
    # In Order 'F', the first 4 elements populate the entire first Z-slice
    np.testing.assert_array_equal(mask_3d[:, :, 0], np.ones((2, 2)))
    np.testing.assert_array_equal(mask_3d[:, :, 1], -1 * np.ones((2, 2)))
    
    assert np.sum(is_fluid) == 4
    assert np.sum(is_boundary) == 4

def test_scientific_staggered_memory_zeroed():
    """Ensure no residual garbage values exist in the allocated physical fields."""
    grid = GridInput(nx=4, ny=4, nz=4)
    fields = allocate_staggered_fields(grid)
    
    for name, arr in fields.items():
        # High-precision check for absolute zero (STS Requirement)
        np.testing.assert_allclose(arr, 0.0, atol=1e-15, err_msg=f"Field {name} has residual noise")
