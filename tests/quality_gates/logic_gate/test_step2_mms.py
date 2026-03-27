# tests/quality_gates/logic_gate/test_step2_mms.py

from src.step2.orchestrate_step2 import orchestrate_step2
from tests.helpers.solver_step1_output_dummy import solver_step1_output_dummy


def test_logic_gate_2_stencil_graph_assembly():
    """
    Verification: state.stencil_matrix[idx].k_plus is non-null for core domain indices.
    Target: src/step2/orchestrate_step2.py
    """
    # 1. Setup Input from Step 1 Dummy
    state_in = solver_step1_output_dummy

    # 2. Action
    state_out = orchestrate_step2(state_in)

    # 3. Verification
    assert hasattr(state_out, 'stencil_matrix'), "Step 2 failed to attach stencil_matrix to state"
    
    # Check a known core index (e.g., center of a 2x2x2 grid)
    # The registry should have successfully wired the neighbors
    sample_block = state_out.stencil_matrix[0] # First core block
    assert sample_block.k_plus is not None, "Neighbor wiring failed: k_plus is None"
    assert sample_block.i_minus is not None, "Neighbor wiring failed: i_minus is None"
    assert hasattr(sample_block, 'rho'), "Physics parameters not mapped to StencilBlock"