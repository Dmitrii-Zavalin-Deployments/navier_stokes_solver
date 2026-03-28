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
    state.ready_for_time_loop = False  # Ensure it starts as False
    
    # 2. Mock the assembler to avoid running the heavy 7-point stencil logic here
    # (We only care about orchestration coverage in this specific file)
    with patch("src.step2.orchestrate_step2.assemble_stencil_matrix") as mock_assembler:
        mock_matrix = MagicMock()
        mock_assembler.return_value = mock_matrix
        
        # 3. Patch DEBUG to True to hit lines 14 and 23
        with patch("src.step2.orchestrate_step2.DEBUG", True):
            result_state = orchestrate_step2(state)
            
            # 4. Verify orchestration side-effects
            assert result_state.ready_for_time_loop is True
            assert result_state.stencil_matrix == mock_matrix
            mock_assembler.assert_called_once_with(state)

def test_orchestrate_step2_standard_flow():
    """
    Ensures the standard path (DEBUG=False) still functions correctly.
    """
    state = make_step1_output_dummy(nx=2, ny=2, nz=2)
    
    with patch("src.step2.orchestrate_step2.assemble_stencil_matrix") as mock_assembler:
        orchestrate_step2(state)
        assert state.ready_for_time_loop is True