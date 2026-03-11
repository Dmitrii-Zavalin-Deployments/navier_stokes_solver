# tests/step2/test_mask_semantics.py

import numpy as np
import pytest

from src.step2.enforce_mask_semantics import enforce_mask_semantics
from tests.helpers.solver_step1_output_dummy import (
    make_step1_output_dummy as make_step1_dummy_state,
)


def make_state_with_mask(mask: np.ndarray):
    """
    Create a Step-1 dummy state and override only the mask.
    """
    nx, ny, nz = mask.shape
    # Corrected function name from helper
    state = make_step1_dummy_state(nx=nx, ny=ny, nz=nz)
    state.mask = mask
    return state


# ------------------------------------------------------------
# 1. Valid tri-state mask
# ------------------------------------------------------------
def test_mask_semantics_valid_tristate():
    mask = np.array([[[1, 0, -1]]], dtype=int)
    state = make_state_with_mask(mask)

    # Should not raise
    enforce_mask_semantics(state)

    # Verify attributes were populated
    assert state.is_fluid is not None
    assert state.is_boundary_cell is not None
    assert state.is_fluid.shape == mask.shape
    assert state.is_boundary_cell.shape == mask.shape


# ------------------------------------------------------------
# 2. Invalid values and Float Rejection
# ------------------------------------------------------------
@pytest.mark.parametrize("invalid_mask", [
    np.array([[[2]]], dtype=int),          # Positive out of range
    np.array([[[-5]]], dtype=int),         # Negative out of range
    np.zeros((2, 2, 2), dtype=int),        # All zeros (no fluid)
    np.array([[[1.0, 0.0]]], dtype=float)  # Float mask
])
def test_mask_semantics_rejections(invalid_mask):
    state = make_state_with_mask(invalid_mask)
    with pytest.raises(ValueError):
        enforce_mask_semantics(state)