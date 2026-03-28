# tests/step2/test_orchestrate_step2.py

from unittest.mock import MagicMock, patch

from src.step2.orchestrate_step2 import orchestrate_step2
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy


def test_orchestrate_step2_full_flow_with_debug():
    """
    Targets lines 14 and 23: Verify the full orchestration logic
    and ensure debug logging is covered.
    """
    # 1. Setup valid state from Step 1 output dummy
    state = make_step1_output_dummy(nx=2, ny=2, nz=2)
    state.ready_for_time_loop = False
    
    # We keep 'as mock_assembler' here because we use it for an assertion
    with patch("src.step2.orchestrate_step2.assemble_stencil_matrix") as mock_assembler:
        mock_matrix = MagicMock()
        mock_assembler.return_value = mock_matrix
        
        with patch("src.step2.orchestrate_step2.DEBUG", True):
            result_state = orchestrate_step2(state)
            
            assert result_state.ready_for_time_loop is True
            assert result_state.stencil_matrix == mock_matrix
            mock_assembler.assert_called_once_with(state)

def test_orchestrate_step2_standard_flow():
    """
    Ensures the standard path (DEBUG=False) still functions correctly.
    Assignment removed to satisfy Ruff F841.
    """
    state = make_step1_output_dummy(nx=2, ny=2, nz=2)
    
    # Removed 'as mock_assembler' since the reference was unused
    with patch("src.step2.orchestrate_step2.assemble_stencil_matrix"):
        orchestrate_step2(state)
        assert state.ready_for_time_loop is True