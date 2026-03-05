# tests/scientific/conftest.py

import pytest
from tests.helpers.solver_input_schema_dummy import make_solver_input_dummy

@pytest.fixture
def base_input():
    """Provides a fully hydrated SolverInput object as the scientific baseline."""
    return make_solver_input_dummy()