# tests/property_integrity/test_initial_pressure_persistence.py

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
def test_initial_pressure_persistence_lifecycle(stage_name, factory):
    """
    Integrity: Verify initial_conditions.pressure survives the entire pipeline.
    This satisfies the 'Required' constraint in the JSON Schema.
    """
    state = factory()
    
    # Adaptive Access for Object vs Dict
    if isinstance(state, dict):
        ic = state.get("initial_conditions", {})
    else:
        ic = getattr(state.config, "initial_conditions", {})

    # The Persistence Check
    assert "pressure" in ic, f"Lifecycle Failure: Initial pressure lost at {stage_name}"
    assert isinstance(ic["pressure"], (int, float)), f"Type Failure: Pressure at {stage_name} must be numeric"
    
    # Value Check (Defaulting to 0.0 per Step 1 intent)
    assert ic["pressure"] == 0.0, f"Value Drift: Initial pressure changed at {stage_name}"