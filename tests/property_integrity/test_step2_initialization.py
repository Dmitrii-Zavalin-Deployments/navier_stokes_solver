# tests/property_integrity/test_step2_initialization.py

import pytest

from src.step2.orchestrate_step2 import orchestrate_step2
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy


class TestStep2Initialization:
    """AUDITOR: Step 2 Wiring & Matrix Assembly Verification."""

    @pytest.fixture(scope="class")
    def assembled_state(self):
        """
        Hydrates state via the Step 1 Dummy and runs Step 2 orchestration.
        This verifies the compatibility of the Step 1 Contract with Step 2 logic.
        """
        # 1. Fetch the immutable core baseline (The Dummy)
        input_state = make_step1_output_dummy(nx=4, ny=4, nz=4)
        
        # 2. Execute Orchestration (The Implementation)
        return orchestrate_step2(input_state)

    def test_stencil_matrix_existence(self, assembled_state):
        """Rule 9: Verifies that the matrix was successfully assigned."""
        assert assembled_state.stencil_matrix is not None
        assert isinstance(assembled_state.stencil_matrix, list)
        assert len(assembled_state.stencil_matrix) > 0

    def test_readiness_sentinel_activation(self, assembled_state):
        """Rule 9: Ensures the readiness gate is engaged post-assembly."""
        # This confirms that orchestrate_step2 correctly transitioned the state
        # triggering verify_foundation_integrity() via the setter.
        assert assembled_state.ready_for_time_loop is True

    def test_matrix_dimensions(self, assembled_state):
        """Rule 0 & 5: Matrix length must match the fluid volume."""
        # 4x4x4 grid = 64 cells total.
        expected_cells = assembled_state.grid.nx * assembled_state.grid.ny * assembled_state.grid.nz
        assert len(assembled_state.stencil_matrix) == expected_cells, \
            "Stencil matrix dimension mismatch with grid volume."