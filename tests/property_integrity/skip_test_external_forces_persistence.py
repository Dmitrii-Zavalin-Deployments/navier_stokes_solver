# tests/property_integrity/test_external_forces_persistence.py

import pytest

from tests.helpers.solver_output_schema_dummy import make_output_schema_dummy
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy

# The Heavyweight must survive the entire journey
LIFECYCLE_STAGES = [
    ("Step 1", make_step1_output_dummy),
    ("Step 2", make_step2_output_dummy),
    ("Step 3", make_step3_output_dummy),
    ("Step 4", make_step4_output_dummy),
    ("Final Output", make_output_schema_dummy),
]

@pytest.mark.parametrize("stage_name, factory", LIFECYCLE_STAGES)
def test_external_forces_persistence_across_lifecycle(stage_name, factory):
    """
    Integrity: Ensure the external_forces vector (Gravity/Wind) 
    survives every transition in the 5-step pipeline.
    """
    state = factory()
    
    # Check for dictionary existence
    assert hasattr(state.config, "external_forces"), f"{stage_name} is config missing external_forces attribute"
    
    # Check for force_vector integrity
    force = state.config.external_forces.get("force_vector")
    assert force is not None, f"{stage_name} lost the force_vector"
    assert len(force) == 3, f"{stage_name} force_vector must have 3 components (x, y, z)"
    
    # Type check for numerical safety
    for component in force:
        assert isinstance(component, (int, float)), f"{stage_name} has non-numeric force component: {component}"

def test_external_forces_immutability_logic():
    """
    Theory: External forces should remain constant unless a step explicitly 
    models time-varying fields (which is outside our current schema).
    """
    s1 = make_step1_output_dummy()
    s5 = make_output_schema_dummy()
    
    # Verification that Step 5 matches Step 1 intent
    assert s1.config.external_forces["force_vector"] == s5.config.external_forces["force_vector"], \
        "External forces diverged between initialization and final output."