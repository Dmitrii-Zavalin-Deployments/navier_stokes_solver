# tests/step3/test_step3_alignment.py

from src.step3.orchestrate_step3 import orchestrate_step3
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy


def test_step3_orchestration_alignment():
    """
    Orchestration Alignment Audit: Step 2 Output -> Step 3 Logic -> Step 3 Dummy.
    """
    # 1. Setup Input from previous frozen step
    state_in = make_step2_output_dummy()
    
    # 2. Execute Logic
    state_out = orchestrate_step3(state_in)
    
    # 3. Load Target Truth
    expected = make_step3_output_dummy()
    
    # 4. Validations
    assert state_out.iteration == 0  # Updated for SSoT logic
    assert state_out.ready_for_time_loop is True
    assert len(state_out.history.times) == len(expected.history.times)
    
    # Verify field initialization
    assert state_out.fields.U_star.shape == expected.fields.U_star.shape
    
    print("\n✅ Step 3 Orchestration Alignment: PASSED")