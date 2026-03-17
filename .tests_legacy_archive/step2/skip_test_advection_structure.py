# tests/step2/test_advection_structure.py

import numpy as np

from src.step2.build_advection_structure import build_advection_structure
from src.step2.create_fluid_mask import create_fluid_mask
from tests.helpers.solver_step1_output_dummy import (
    make_step1_output_dummy as make_step1_dummy_state,
)


def make_state(nx=4, ny=4, nz=4, dx=1.0, scheme="upwind"):
    """
    Helper to create a dummy state and ensure the config dict is populated.
    """
    state = make_step1_dummy_state(nx=nx, ny=ny, nz=nz)
    state.grid['dx'] = dx
    state.grid['dy'] = dx
    state.grid['dz'] = dx
    state.mask = np.ones((nx, ny, nz), dtype=int)
    create_fluid_mask(state)
    
    # Fix: Accessing state.config as a dictionary
    state.config['advection_scheme'] = scheme
    return state

# ------------------------------------------------------------
# 1. Verification of Structure Presence
# ------------------------------------------------------------
def test_advection_structure_exists():
    state = make_state()
    build_advection_structure(state)
    
    assert "advection" in state.operators
    adv = state.operators["advection"]
    assert "scheme" in adv
    assert "grid_spacing" in adv
    assert adv["grid_spacing"]["dx"] == 1.0

# ------------------------------------------------------------
# 2. Scheme Persistence
# ------------------------------------------------------------
def test_advection_scheme_config():
    scheme_name = "upwind_1st_order"
    state = make_state(scheme=scheme_name)
    build_advection_structure(state)
    
    # Check that the operator dictionary captured the scheme from config
    assert state.operators["advection"]["scheme"] == scheme_name

# ------------------------------------------------------------
# 3. Scaling Constants
# ------------------------------------------------------------
def test_advection_scaling_math():
    dx = 0.5
    state = make_state(dx=dx)
    build_advection_structure(state)
    
    scaling = state.operators["advection"]["grid_spacing"]
    # 1 / (2 * 0.5) = 1.0
    assert np.isclose(scaling["inv_2dx"], 1.0)