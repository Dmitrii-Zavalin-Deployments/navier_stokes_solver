# tests/quality_gates/logic_gate/test_step4_mms.py

import pytest
from unittest.mock import patch
from src.step4.orchestrate_step4 import orchestrate_step4
from tests.helpers.solver_step3_output_dummy import solver_step3_output_dummy

def test_logic_gate_4_archival_trigger_logic(solver_input_schema_dummy):
    """
    Verification: save_snapshot is triggered; state.manifest tracks the save.
    Target: src/step4/orchestrate_step4.py
    """
    # 1. Setup: Force iteration to match interval
    state = solver_step3_output_dummy
    context = solver_input_schema_dummy
    
    state.iteration = 20
    context.input_data.simulation_parameters.output_interval = 10 # 20 % 10 == 0

    # 2. Action & Verification
    # Use patch to verify the call to the archivist utility
    with patch('src.step4.orchestrate_step4.save_snapshot') as mock_save:
        orchestrate_step4(state, context)
        
        # Verify modulo logic worked
        mock_save.assert_called_once()
        
    # Test Negative Case: Iteration 21 should NOT trigger save
    state.iteration = 21
    with patch('src.step4.orchestrate_step4.save_snapshot') as mock_save_neg:
        orchestrate_step4(state, context)
        mock_save_neg.assert_not_called()