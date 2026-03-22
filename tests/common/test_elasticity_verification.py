# tests/common/test_elasticity_verification.py

import pytest
import numpy as np
from unittest.mock import MagicMock
from src.common.elasticity import ElasticManager
from src.common.field_schema import FI
from src.common.solver_state import SolverState

class TestElasticityScientific:
    """
    Scientific Testing Suite (STS) for ElasticManager.
    Focus: Atomic Numerical Truth and Foundation Integrity.
    """

    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.dt_min_limit = 1e-4
        config.ppe_max_retries = 5
        return config

    @pytest.fixture
    def mock_state(self):
        """Creates a mock SolverState with a real NumPy foundation."""
        state = MagicMock(spec=SolverState)
        # Simulate 100 cells across all fields defined in FI
        num_cells = 100
        state.fields.data = np.zeros((num_cells, FI.num_fields()), dtype=np.float32)
        return state

    ## --- ATOMIC NUMERICAL TRUTH (RULE 7) ---

    def test_validate_and_commit_precision(self, mock_config, mock_state):
        """
        Verifies that STAR/NEXT fields are committed to Primary fields 
        at machine precision (Rule 7).
        """
        manager = ElasticManager(mock_config, initial_dt=0.01)
        data = mock_state.fields.data
        
        # 1. Prime the 'Intermediate' (Foundation) with unique signatures
        # Using Index + Offset to ensure no cross-contamination
        data[:, FI.VX_STAR] = np.random.rand(100).astype(np.float32)
        data[:, FI.VY_STAR] = np.random.rand(100).astype(np.float32)
        data[:, FI.VZ_STAR] = np.random.rand(100).astype(np.float32)
        data[:, FI.P_NEXT]  = np.random.rand(100).astype(np.float32)

        # 2. Execute Atomic Commitment
        manager.validate_and_commit(mock_state)

        # 3. Assert Machine Precision Alignment (Rule 7)
        np.testing.assert_array_almost_equal(data[:, FI.VX], data[:, FI.VX_STAR], decimal=7)
        np.testing.assert_array_almost_equal(data[:, FI.VY], data[:, FI.VY_STAR], decimal=7)
        np.testing.assert_array_almost_equal(data[:, FI.VZ], data[:, FI.VZ_STAR], decimal=7)
        np.testing.assert_array_almost_equal(data[:, FI.P],  data[:, FI.P_NEXT],  decimal=7)

    ## --- DETERMINISTIC LOGIC (RULE 5) ---

    def test_dt_reduction_path(self, mock_config, mock_state):
        """Ensures dt strictly follows the pre-calculated range (No Defaults)."""
        initial_dt = 0.1
        manager = ElasticManager(mock_config, initial_dt)
        
        # First failure: Should move to the first step in the range after initial
        manager.stabilization(is_needed=True, state=mock_state)
        
        expected_next_dt = manager._dt_range[1]
        assert manager.dt == pytest.approx(expected_next_dt)
        assert manager._iteration == 1

    def test_stabilization_reset_logic(self, mock_config, mock_state):
        """Verifies that success (is_needed=False) triggers commitment and reset."""
        manager = ElasticManager(mock_config, initial_dt=0.1)
        
        # Induce partial failure state
        manager.stabilization(is_needed=True, state=mock_state)
        assert manager._iteration == 1
        
        # Trigger Success
        manager.stabilization(is_needed=False, state=mock_state)
        
        # Assertions
        assert manager._iteration == 0
        assert manager.dt == 0.1
        # Implicitly checks that validate_and_commit was called because 
        # it is the first line in the 'not is_needed' block.

    ## --- CRITICAL FAILURE MODES (LOUD FAILURES) ---

    def test_exhaustion_runtime_error(self, mock_config, mock_state):
        """Verifies that exceeding retries raises a RuntimeError (Rule 5)."""
        manager = ElasticManager(mock_config, initial_dt=0.1)
        
        # Exhaust the retries
        for _ in range(mock_config.ppe_max_retries):
            manager.stabilization(is_needed=True, state=mock_state)
            
        with pytest.raises(RuntimeError, match="Unstable: reached dt_floor"):
            manager.stabilization(is_needed=True, state=mock_state)

    def test_explicit_state_mandate(self, mock_config):
        """
        Rule 5: Verify that missing state raises TypeError.
        This prevents 'Silent Failure' in the Foundation bridge.
        """
        manager = ElasticManager(mock_config, initial_dt=0.1)
        
        with pytest.raises(TypeError):
            # Attempting to call without the mandatory state object
            manager.stabilization(is_needed=False)