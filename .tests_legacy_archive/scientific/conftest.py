# tests/scientific/conftest.py

import os

import numpy as np
import pytest

from tests.helpers.solver_input_schema_dummy import make_solver_input_dummy
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy


def setup_grid_3d(state, n=3, length=0.3):
    """
    Utility to inject grid parameters into a SolverState instance.
    Uses _set_safe to bypass read-only property enforcement and ensure
    internal dictionary synchronization.
    """
    # Set dimensions using _set_safe to ensure the ValidatedContainer 
    # internal state is updated consistently.
    for dim in ["nx", "ny", "nz"]:
        state.grid._set_safe(dim, n, int)
    
    spacing = length / n
    state.grid._set_safe("dx", float(spacing), float)
    state.grid._set_safe("dy", float(spacing), float)
    state.grid._set_safe("dz", float(spacing), float)

@pytest.fixture(scope="session", autouse=True)
def configure_scientific_precision():
    """
    STS Global Configuration:
    Forces high-precision printing and strict numerical error handling 
    for all tests within the tests/scientific/ directory.
    """
    np.set_printoptions(precision=15, suppress=False, threshold=np.inf)
    
    os.environ["STS_RTOL"] = "1e-12"
    os.environ["STS_ATOL"] = "1e-15"
    
    yield
    
    # Restore defaults
    np.set_printoptions(edgeitems=3, infstr='inf', linewidth=75, nanstr='nan', precision=8, suppress=False, threshold=1000)

@pytest.fixture
def sts_tolerance():
    """Returns the standard high-precision tolerance for STS assertions."""
    return {"rtol": 1e-12, "atol": 1e-15}

@pytest.fixture
def base_input():
    """Provides a fully hydrated SolverInput object as the scientific baseline."""
    return make_solver_input_dummy()

@pytest.fixture
def state_3d_small():
    """Provides a fresh, hydrated SolverState snapshot for Step 2+ testing."""
    # Start from a baseline state
    state = make_step1_output_dummy(nx=2, ny=2, nz=2)
    # Apply standard scientific grid resolution automatically
    setup_grid_3d(state, n=3, length=0.3)
    return state