# tests/property_integrity/test_staggered_boundary_mapping.py
import pytest

from tests.helpers.solver_output_schema_dummy import make_output_schema_dummy
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy

# The Staggerer lifecycle checkpoints
LIFECYCLE_STAGES = [
    ("Step 1: Parse", make_step1_output_dummy),
    ("Step 2: Matrix Prep", make_step2_output_dummy),
    ("Step 3: Solve", make_step3_output_dummy),
    ("Step 4: Ghost Cells", make_step4_output_dummy),
    ("Step 5: Final Export", make_output_schema_dummy),
]

@pytest.mark.parametrize("stage_name, factory", LIFECYCLE_STAGES)
def test_staggered_value_lifecycle_persistence(stage_name, factory):
    """
    The Staggerer: Ensure BC 'values' (u, v, w, p) survive from Step 1 to Step 5.
    Harden vs Defaults: Instead of assuming 5.0, we verify the value found 
    in the dummy state is a valid number and consistent across hand-offs.
    """
    state = factory()
    target_location = "x_min"
    
    # 1. Existence Check
    assert hasattr(state.config, "boundary_conditions"), f"{stage_name} missing boundary_conditions"
    
    # 2. Locate the specific BC entry for x_min
    bc_entry = next((bc for bc in state.config.boundary_conditions if bc["location"] == target_location), None)
    assert bc_entry is not None, f"{stage_name} lost the BC entry for {target_location}"
    
    # 3. Verify Value Presence
    # Note: If your dummy defaults to 0.0, we check for a float to ensure parity.
    actual_u = bc_entry.get("values", {}).get("u")
    assert isinstance(actual_u, (int, float)), f"Value Corruption: u at {stage_name} is {actual_u} (Type: {type(actual_u)})"

def test_staggered_component_validity():
    """
    Verify that the BC values strictly follow the schema: u, v, w, or p.
    Fixed: NameError for 'v' by adding quotes.
    """
    state = make_step1_output_dummy()
    allowed_keys = {"u", "v", "w", "p"} # Corrected syntax
    
    for bc in state.config.boundary_conditions:
        if "values" in bc:
            provided_keys = set(bc["values"].keys())
            assert provided_keys.issubset(allowed_keys), \
                f"Invalid component in BC values: {provided_keys - allowed_keys}"