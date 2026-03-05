# tests/scientific/conftest.py

import pytest
from tests.helpers.solver_input_schema_dummy import make_solver_input_dummy
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy

@pytest.fixture
def base_input():
    """Provides a fully hydrated SolverInput object as the scientific baseline."""
    return make_solver_input_dummy()

@pytest.fixture
def state_3d_small():
    """Provides a fresh, hydrated SolverState snapshot for Step 2+ testing."""
    return make_step1_output_dummy(nx=2, ny=2, nz=2)