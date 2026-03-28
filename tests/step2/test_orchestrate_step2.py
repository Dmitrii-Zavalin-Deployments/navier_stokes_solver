# tests/step2/test_orchestrate_step2.py

from unittest.mock import patch

from src.step2.orchestrate_step2 import orchestrate_step2
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy


def test_orchestrate_step2_full_flow_with_debug():
    """
    Targets lines 14 and 23: Verify the full orchestration logic.
    Uses make_step2_output_dummy to satisfy the list type requirement.
    """
    # 1. Setup valid initial state
    state = make_step1_output_dummy(nx=2, ny=2, nz=2)
    state.ready_for_time_loop = False
    
    # 2. Setup a valid return value (a real list of StencilBlocks)
    dummy_output = make_step2_output_dummy(nx=2, ny=2, nz=2)
    valid_matrix = dummy_output.stencil_matrix
    
    # 3. Patch the assembler to return the real list
    with patch("src.step2.orchestrate_step2.assemble_stencil_matrix") as mock_assembler:
        mock_assembler.return_value = valid_matrix
        
        # 4. Patch DEBUG to True for coverage
        with patch("src.step2.orchestrate_step2.DEBUG", True):
            result_state = orchestrate_step2(state)
            
            # 5. Verify logic and side-effects
            assert result_state.ready_for_time_loop is True
            assert isinstance(result_state.stencil_matrix, list)
            assert len(result_state.stencil_matrix) == 8  # 2x2x2
            mock_assembler.assert_called_once_with(state)

def test_orchestrate_step2_standard_flow():
    """
    Ensures the standard path (DEBUG=False) still functions correctly.
    """
    state = make_step1_output_dummy(nx=2, ny=2, nz=2)
    dummy_output = make_step2_output_dummy(nx=2, ny=2, nz=2)
    
    with patch("src.step2.orchestrate_step2.assemble_stencil_matrix") as mock_assembler:
        mock_assembler.return_value = dummy_output.stencil_matrix
        
        orchestrate_step2(state)
        assert state.ready_for_time_loop is True
        assert len(state.stencil_matrix) == 8