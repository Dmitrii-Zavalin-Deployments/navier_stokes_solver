# tests/property_integrity/test_initial_conditions_persistence.py

import pytest

from tests.helpers.solver_output_schema_dummy import make_output_schema_dummy
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy

# All stages must preserve the "Intent" of the simulation (Initial Conditions)
ALL_STAGES = [
    ("Step 1", make_step1_output_dummy),
    ("Step 2", make_step2_output_dummy),
    ("Step 3", make_step3_output_dummy),
    ("Step 4", make_step4_output_dummy),
    ("Final Output", make_output_schema_dummy),
]

@pytest.mark.parametrize("stage_name, factory", ALL_STAGES)
def test_initial_velocity_persistence(stage_name, factory):
    """
    Integrity: Verify that initial_conditions.velocity persists across all steps.
    This ensures we don't lose the 'Starting Point' reference.
    """
    state = factory()
    
    # 1. Department Existence
    assert hasattr(state.config, "initial_conditions"), f"{stage_name}: config.initial_conditions missing"
    
    # 2. Key Existence
    ic = state.config.initial_conditions
    assert "velocity" in ic, f"{stage_name}: 'velocity' missing from initial_conditions"
    
    # 3. Structural Integrity (Must be a 3D vector for U, V, W)
    v0 = ic["velocity"]
    assert len(v0) == 3, f"{stage_name}: Initial velocity must be a 3D vector [u, v, w]"
    
    # 4. Type Safety (Must be JSON-safe list or numeric)
    assert isinstance(v0, list), f"{stage_name}: Initial velocity must be a list for JSON-safe export"

def test_initial_conditions_are_not_overwritten_by_fields():
    """
    Physics: Ensure 'initial_conditions.velocity' remains constant even after 
    'fields.U/V/W' have been updated (Step 3/4).
    """
    # Grab a late-stage state where fields have definitely evolved
    state = make_step3_output_dummy()
    
    # In our dummy, initial velocity is [0.0, 0.0, 0.0]
    # Even if fields.U is populated with results, the IC must remain [0.0, 0.0, 0.0]
    expected_initial = [0.0, 0.0, 0.0]
    assert state.config.initial_conditions["velocity"] == expected_initial, \
        "Step 3: Initial conditions were corrupted by field updates!"