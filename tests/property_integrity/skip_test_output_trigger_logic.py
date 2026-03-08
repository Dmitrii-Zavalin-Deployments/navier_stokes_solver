# tests/property_integrity/test_output_trigger_logic.py

import pytest

from tests.helpers.solver_output_schema_dummy import make_output_schema_dummy
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy


@pytest.mark.parametrize("stage_name, factory", [
    ("Step 4", make_step4_output_dummy),
    ("Final Output", make_output_schema_dummy)
])
def test_output_interval_persistence(stage_name, factory):
    """
    Integrity: Verify simulation_parameters.output_interval is 
    inherited and persistent through the late-stage pipeline.
    """
    state = factory()
    
    # Adaptive Access: Check if it's a dict (serialized) or object (live)
    if isinstance(state, dict):
        params = state.get("simulation_parameters", {})
    else:
        params = getattr(state, "config", {})
    
    assert hasattr(params, "output_interval"), f"{stage_name}: Property 'output_interval' lost"
    assert isinstance(params.output_interval, (int, float)), "Interval must be numeric"

def test_step5_write_trigger_logic():
    """
    Logic: Verify that the Step 5 'Snapshot' is only generated 
    when (iter % interval == 0).
    """
    state = make_output_schema_dummy()
    
    # Adaptive Extraction for live object or dictionary
    if isinstance(state, dict):
        iteration = state.get("iteration", 0)
        params = state.get("simulation_parameters", {})
        diagnostics = state.get("step5_diagnostics", {})
    else:
        iteration = getattr(state, "iteration", 0)
        params = getattr(state, "config", {})
        diagnostics = getattr(state, "health", {})
    
    interval = params.output_interval
    
    # Validation
    assert diagnostics, "Step 5 failed to log execution diagnostics"
    
    if diagnostics.get("snapshot_generated"):
        assert iteration % interval == 0, \
            f"Trigger Error: Snapshot generated at iter {iteration} with interval {interval}"