# tests/step5/test_step5_alignment.py

import os
import shutil

from src.step5.orchestrate_step5 import orchestrate_step5
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy


def test_step5_orchestration_alignment():
    """
    PHASE C: Step 4 Output -> Orchestrate Step 5 -> Gold Standard Schema.
    """
    # 1. Setup Input
    state_in = make_step4_output_dummy()
    # Force state to look like it's at the end
    state_in.time = state_in.config.total_time 
    
    # 2. Logic
    state_out = orchestrate_step5(state_in)
    
    # 3. Validation
    assert state_out.ready_for_time_loop is False  # Loop must terminate
    assert state_out.time == state_out.config.total_time
    assert len(state_out.manifest.saved_snapshots) > 0
    assert os.path.exists(state_out.manifest.output_directory)
    
    # Cleanup artifacts created by the test
    if os.path.exists("output"):
        shutil.rmtree("output")
        
    print("\n✅ Step 5 Orchestration Alignment: PASSED")