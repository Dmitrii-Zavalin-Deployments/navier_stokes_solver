# tests/step4/test_step4_alignment.py

from src.step4.orchestrate_step4 import orchestrate_step4
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy


def test_step4_orchestration_alignment():
    """
    PHASE C: Step 3 Output -> Orchestrate Step 4 -> Step 4 Dummy.
    """
    # 1. Setup
    state_in = make_step3_output_dummy()
    
    # 2. Logic
    state_out = orchestrate_step4(state_in)
    
    # 3. Target
    expected = make_step4_output_dummy()
    
    # 4. Validation
    assert state_out.fields.P_ext.shape == expected.fields.P_ext.shape
    assert state_out.diagnostics.memory_footprint_gb > 0
    assert state_out.ready_for_time_loop is True
    
    print("\n✅ Step 4 Orchestration Alignment: PASSED")