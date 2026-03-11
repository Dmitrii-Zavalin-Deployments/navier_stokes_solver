# tests/step2/test_orchestrate_step2.py

import numpy as np
import pytest

from src.step2.orchestrate_step2 import orchestrate_step2
from tests.helpers.solver_step1_output_dummy import (
    make_step1_output_dummy as make_step1_dummy_state,
)


def make_minimal_state(nx=2, ny=2, nz=2, dx=1.0, dt=0.1, rho=1.0):
    """
    Create a canonical Step-1 dummy state and override only relevant fields.
    """
    # Fix: use the correct helper name
    state = make_step1_dummy_state(nx=nx, ny=ny, nz=nz)

    # Use dictionary access for consistent schema
    state.grid['dx'] = dx
    state.grid['dy'] = dx
    state.grid['dz'] = dx

    state.config['dt'] = dt
    state.constants["rho"] = rho

    state.mask = np.ones((nx, ny, nz), dtype=int)

    state.fields["P"] = np.zeros((nx, ny, nz))
    state.fields["U"] = np.zeros((nx + 1, ny, nz))
    state.fields["V"] = np.zeros((nx, ny + 1, nz))
    state.fields["W"] = np.zeros((nx, ny, nz + 1))

    state.boundary_conditions = {}
    state.health = {}

    return state


def test_orchestrate_step2_minimal_state():
    state = make_minimal_state()

    # Run Step 2
    orchestrate_step2(state)

    # 1. Mask semantics
    assert state.is_fluid is not None
    assert state.is_fluid.shape == state.mask.shape

    # 2. Constants
    assert "inv_dx" in state.constants
    assert state.constants["inv_dx"] == pytest.approx(1.0)

    # 3. Operators
    assert "divergence" in state.operators
    # Note: Depending on your build_divergence_operator, this might be a 
    # scipy sparse matrix or a callable. The orchestrator ensures it exists.
    assert state.operators["divergence"] is not None

    # 4. PPE structure
    assert "rhs_builder" in state.ppe
    assert callable(state.ppe["rhs_builder"])
    assert "ppe_is_singular" in state.ppe

    # 5. Health diagnostics
    assert "divergence_norm" in state.health
    assert np.isfinite(state.health["cfl"])