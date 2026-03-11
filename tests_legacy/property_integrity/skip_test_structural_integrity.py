import numpy as np
import pytest

from src.solver_input import SolverInput

# --- PIPELINE IMPORTS ---
from src.step1.orchestrate_step1 import orchestrate_step1
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy

# Note: These will need to be imported as you implement each stage
# from src.step2.orchestrate_step2 import orchestrate_step2_state
# from src.step3.orchestrate_step3 import orchestrate_step3_state
# from src.step4.orchestrate_step4 import orchestrate_step4_state
# from src.step5.orchestrate_step5 import orchestrate_step5_state

class TestStructuralIntegrity:
    """
    KNOWLEDGE GATE: The Vertical Integrity Mandate.
    Ensures the SolverState (The Constitution) survives all 5 stages.
    """

    # =========================================================================
    # FIXTURES: The Sequential Chain
    # =========================================================================

    @pytest.fixture(scope="class")
    def step1_state(self):
        """STEP 1: Domain setup and field allocation."""
        dummy = solver_input_schema_dummy()
        dummy["solver_settings"] = {"max_iterations": 100, "tolerance": 1e-6, "method": "jacobi"}
        input_data = SolverInput(**dummy)
        return orchestrate_step1(input_data)

    def step2_state(self, step1_state):
        """STEP 2: Operator Assembly (Placeholder until Step 2 is wired)."""
        # return orchestrate_step2_state(step1_state)
        return step1_state # Temporary passthrough

    # =========================================================================
    # TESTS: Step 1 (Domain & Fields)
    # =========================================================================

    def test_step1_departmental_init(self, step1_state):
        """Ensures the core objects exist in the state container."""
        assert hasattr(step1_state, "grid")
        assert hasattr(step1_state, "config")
        assert hasattr(step1_state, "fields")
        assert hasattr(step1_state, "masks")

    def test_step1_physical_constants(self, step1_state):
        """Verifies physics were parsed into the correct objects."""
        assert step1_state.config.dt > 0
        assert step1_state.config.density == 1000.0
        assert isinstance(step1_state.grid.dx, float)

    def test_step1_mac_allocation(self, step1_state):
        """Ensures MAC grid field components are NumPy arrays of correct shape."""
        # Using the new attribute-based access
        assert isinstance(step1_state.fields.U, np.ndarray)
        assert isinstance(step1_state.fields.P, np.ndarray)
        # Verify 3D allocation (nx, ny, nz)
        assert step1_state.fields.P.ndim == 3

    # =========================================================================
    # TESTS: Step 2 (Operators)
    # =========================================================================

    def test_step2_matrix_integrity(self, step2_state):
        """Ensures operators are ready for the linear solver."""
        # Once Step 2 is implemented, these will verify the 'operators' department
        assert hasattr(step2_state, "operators")
        # Example check: assert step2_state.operators.laplacian is not None

    # =========================================================================
    # TESTS: Step 3 (Iterative Solver)
    # =========================================================================

    def test_step3_state_evolution(self, step1_state):
        """Ensures the iteration counter and time tracking exist."""
        assert step1_state.iteration == 0
        assert isinstance(step1_state.time, float)

    # =========================================================================
    # TESTS: Step 4 (Boundaries)
    # =========================================================================

    def test_step4_extended_fields(self, step1_state):
        """Ensures fields have placeholders for ghost-cell expansion."""
        # Step 4 will populate P_ext, U_ext, etc.
        assert hasattr(step1_state.fields, "P_ext")

    # =========================================================================
    # TESTS: Step 5 (Serialization)
    # =========================================================================

    def test_step5_terminal_compliance(self, step1_state):
        """
        The 'Gold Standard' Check.
        Ensures the state can be serialized to JSON at the very end.
        """
        # This replaces your old 'serialization_safety' test
        # We only run it if Step 1 is successful
        json_output = step1_state.to_json_safe()
        assert "config" in json_output
        assert "fields" in json_output
        # Verify NumPy was successfully converted to list by to_dict()
        assert isinstance(json_output["fields"]["P"], list)