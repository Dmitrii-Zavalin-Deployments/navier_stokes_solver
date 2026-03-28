# tests/step2/test_orchestrate_step2.py

from unittest.mock import patch

from src.step2.orchestrate_step2 import orchestrate_step2
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy


def test_orchestrate_step2_full_flow_with_debug():
    """
    Verifies orchestration while maintaining Memory Parity (Rule 9).
    The matrix must point to the SAME buffer as the state.
    """
    # 1. Setup the 'Live' state
    state = make_step1_output_dummy(nx=2, ny=2, nz=2)
    state.ready_for_time_loop = False
    
    # 2. Setup Dummy Output, but swap its matrix to use OUR state's buffer
    # This ensures verify_foundation_integrity sees the pointers correctly.
    dummy_output = make_step2_output_dummy(nx=2, ny=2, nz=2)
    valid_matrix = dummy_output.stencil_matrix
    
    # RELINKING: Ensure the cells in valid_matrix point to state.fields.data
    for block in valid_matrix:
        for cell in [block.center, block.i_minus, block.i_plus, 
                     block.j_minus, block.j_plus, block.k_minus, block.k_plus]:
            cell._fields_buffer = state.fields.data

    # 3. Patch the assembler to return the relinked matrix
    with patch("src.step2.orchestrate_step2.assemble_stencil_matrix") as mock_assembler:
        mock_assembler.return_value = valid_matrix
        
        with patch("src.step2.orchestrate_step2.DEBUG", True):
            result_state = orchestrate_step2(state)
            
            assert result_state.ready_for_time_loop is True
            assert result_state.stencil_matrix == valid_matrix
            mock_assembler.assert_called_once_with(state)

def test_orchestrate_step2_standard_flow():
    """
    Standard flow test with standard memory relinking.
    """
    state = make_step1_output_dummy(nx=2, ny=2, nz=2)
    dummy_output = make_step2_output_dummy(nx=2, ny=2, nz=2)
    valid_matrix = dummy_output.stencil_matrix
    
    # Relink to avoid Memory Swap error
    for block in valid_matrix:
        block.center._fields_buffer = state.fields.data

    with patch("src.step2.orchestrate_step2.assemble_stencil_matrix") as mock_assembler:
        mock_assembler.return_value = valid_matrix
        orchestrate_step2(state)
        assert state.ready_for_time_loop is True