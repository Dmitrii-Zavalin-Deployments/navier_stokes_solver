# tests/step2/test_fluid_mask.py

import numpy as np
import pytest

from src.step2.create_fluid_mask import create_fluid_mask
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy


def make_state_with_mask(mask: np.ndarray):
    nx, ny, nz = mask.shape
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    state.mask = mask
    return state

def test_fluid_mask_mixed():
    # 0 = Solid, 1 = Fluid, -1 = Boundary Fluid
    mask = np.array([[[0, 1]], [[-1, 0]]], dtype=int)
    state = make_state_with_mask(mask)

    is_fluid, is_boundary = create_fluid_mask(state)

    # FIX: Use implicit truthiness or == True to avoid the 'is' identity trap
    assert is_fluid[0, 0, 1]
    assert is_fluid[1, 0, 0]
    
    assert is_boundary[1, 0, 0]
    assert not is_boundary[0, 0, 1]
    
    # Solid check
    assert not is_fluid[0, 0, 0]
    
    # Verify state persistence
    assert np.array_equal(state.is_fluid, is_fluid)
    assert state.is_solid[0, 0, 0]

def test_fluid_mask_float_rejected():
    mask = np.ones((2, 2, 2), dtype=float)
    state = make_state_with_mask(mask)

    with pytest.raises(ValueError):
        create_fluid_mask(state)