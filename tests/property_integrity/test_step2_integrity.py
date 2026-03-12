# tests/property_integrity/test_step2_integrity.py

import pytest

from src.step2.orchestrate_step2 import orchestrate_step2

# Assuming you have a fixture that runs Step 1 to provide the initial state

class TestStep2Integrity:
    """AUDITOR: Step 2 Wiring & Matrix Assembly Verification."""

    @pytest.fixture(scope="class")
    def assembled_state(self, hydrated_state):
        """Runs Step 2 orchestration on the hydrated Step 1 state."""
        return orchestrate_step2(hydrated_state)

    def test_stencil_matrix_existence(self, assembled_state):
        """Rule 9: Verifies that the matrix was successfully assigned."""
        assert assembled_state.stencil_matrix is not None
        assert isinstance(assembled_state.stencil_matrix, list)
        assert len(assembled_state.stencil_matrix) > 0

    def test_readiness_sentinel_activation(self, assembled_state):
        """Rule 9: Ensures the readiness gate is engaged post-assembly."""
        # This will trigger verify_foundation_integrity() via the setter
        assert assembled_state.ready_for_time_loop is True

    def test_matrix_dimensions(self, assembled_state):
        """Rule 0 & 5: Matrix length must match the fluid volume."""
        # 4x4x4 grid = 64 cells total. Assuming a fully fluid domain for the dummy test.
        expected_cells = assembled_state.grid.nx * assembled_state.grid.ny * assembled_state.grid.nz
        assert len(assembled_state.stencil_matrix) == expected_cells, \
            "Stencil matrix dimension mismatch with grid volume."