# tests/quality_gates/logic_gate/test_step4_mms.py

from unittest.mock import patch

from src.step4.orchestrate_step4 import orchestrate_step4
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy


def test_logic_gate_4_archival_trigger_logic(solver_input_schema_dummy):
    """
    Logic Gate 4: Archival Trigger Verification
    
    Analytical Challenge: Snapshot Cadence
    Success Metric: I mod Delta I == 0
    Target: src/step4/orchestrate_step4.py
    """
    # 1. Setup: Use Step 2 dummy to get a full SolverState 
    # (Step 4 needs the state container, not just an individual StencilBlock)
    state = make_step2_output_dummy(nx=4, ny=4, nz=4)
    context = solver_input_schema_dummy
    
    # 2. Scenario: state.iteration = 20, output_interval = 10
    # Rule 4 SSoT: Accessing interval via simulation_parameters
    state.iteration = 20
    context.input_data.simulation_parameters.output_interval = 10 

    # 3. Action & Verification: Positive Trigger
    # Verify that orchestrate_step4 triggers the archivist when modulo is 0
    with patch('src.step4.orchestrate_step4.save_snapshot') as mock_save:
        orchestrate_step4(state, context)
        
        # Success Metric: I mod Delta I == 0 -> Save Triggered
        mock_save.assert_called_once_with(state)
        
    # 4. Action & Verification: Negative Trigger (Scenario 2)
    # Test that iteration 21 does NOT trigger a save
    state.iteration = 21
    with patch('src.step4.orchestrate_step4.save_snapshot') as mock_save_neg:
        orchestrate_step4(state, context)
        
        # Success Metric: 21 % 10 != 0 -> No Save
        mock_save_neg.assert_not_called()

    # 5. Verification: Logic Layer Integrity
    # Ensure orchestrate_step4 returns the state for the next loop iteration
    result = orchestrate_step4(state, context)
    assert isinstance(result, type(state)), "MMS FAILURE: Orchestrator failed to return SolverState"