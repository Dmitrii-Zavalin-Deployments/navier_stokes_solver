# tests/property_integrity/test_total_time_persistence.py

import pytest

from tests.helpers.solver_output_schema_dummy import make_output_schema_dummy
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy


@pytest.mark.parametrize("stage_name, factory", [
    ("Step 1", make_step1_output_dummy),
    ("Step 2", make_step2_output_dummy),
    ("Step 3", make_step3_output_dummy),
    ("Step 4", make_step4_output_dummy),
    ("Final Output", make_output_schema_dummy)
])
def test_total_time_persistence_lifecycle(stage_name, factory):
    """
    Integrity: Verify simulation_parameters.total_time survives the pipeline.
    This governs the terminal condition for the main solver loop.
    """
    state = factory()
    
    # Adaptive Access: Extract simulation_parameters
    if isinstance(state, dict):
        params = state.get("simulation_parameters", {})
    else:
        params = getattr(state, "config", {})

    # 1. Existence Check
    assert "total_time" in params, f"Lifecycle Failure: 'total_time' lost at {stage_name}"
    
    # 2. Type & Value Check
    total_time = params["total_time"]
    assert isinstance(total_time, (int, float)), f"Type Error: total_time must be numeric at {stage_name}"
    assert total_time > 0, f"Physics Error: total_time must be positive at {stage_name}"
    
    # 3. Consistency Check (Matched against the Step 1 'Gold Standard' value)
    assert total_time == 1.0, f"Value Drift: total_time changed at {stage_name}"