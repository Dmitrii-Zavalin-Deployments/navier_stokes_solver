# tests/step2/test_constants.py

import numpy as np
import pytest

from src.step2.orchestrate_step2 import orchestrate_step2
from tests.helpers.solver_step1_output_dummy import (
    make_step1_output_dummy as make_step1_dummy_state,
)


def make_state(*, dx=1.0, dy=1.0, dz=1.0, dt=0.1, rho=1.0):
    """
    Create a canonical Step‑1 dummy state.
    Ensures boundary_conditions is initialized to prevent AttributeError.
    """
    state = make_step1_dummy_state(nx=1, ny=1, nz=1)

    state.grid['dx'] = dx
    state.grid['dy'] = dy
    state.grid['dz'] = dz

    state.config['dt'] = dt
    state.constants["rho"] = rho
    
    # FIX: Ensure boundary_conditions is a dict, not None
    if state.boundary_conditions is None:
        state.boundary_conditions = {}

    return state


# ------------------------------------------------------------
# 1. Integration Check: Operator Construction
# ------------------------------------------------------------
def test_orchestration_builds_operators():
    state = make_state(dx=0.1, dy=0.2, dz=0.3, dt=0.01)

    orchestrate_step2(state)

    assert "laplacian" in state.operators
    assert "divergence" in state.operators
    assert "gradient" in state.operators
    assert "advection" in state.operators
    
    assert state.grid["dx"] == 0.1
    assert state.grid["dy"] == 0.2


# ------------------------------------------------------------
# 2. Precision Check: Very small dx
# ------------------------------------------------------------
def test_constants_very_small_dx():
    state = make_state(dx=1e-12, dy=1e-12, dz=1e-12)

    orchestrate_step2(state)

    assert np.isfinite(state.grid["dx"])


# ------------------------------------------------------------
# 3. Validation Check: dt = 0
# ------------------------------------------------------------
def test_constants_dt_zero_rejected():
    state = make_state(dt=0.0)

    # Catches non-physical timestep
    with pytest.raises((ValueError, AssertionError)):
        orchestrate_step2(state)


# ------------------------------------------------------------
# 4. State Readiness Check
# ------------------------------------------------------------
def test_step2_readiness_flag():
    state = make_state()
    orchestrate_step2(state)

    # Step 2 should set this to False until Step 3 completes
    assert state.ready_for_time_loop is False