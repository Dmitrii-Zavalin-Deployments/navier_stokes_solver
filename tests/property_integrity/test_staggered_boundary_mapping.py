# tests/property_integrity/test_staggered_boundary_mapping.py

import pytest
from tests.helpers.solver_output_schema_dummy import make_output_schema_dummy
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy
from tests.helpers.solver_step5_output_dummy import make_step5_output_dummy

LIFECYCLE_STAGES = [
    ("Step 1", make_step1_output_dummy),
    ("Step 2", make_step2_output_dummy),
    ("Step 3", make_step3_output_dummy),
    ("Step 4", make_step4_output_dummy),
    ("Step 5", make_step5_output_dummy),
    ("Final Output", make_output_schema_dummy),
]

@pytest.mark.parametrize("stage_name, factory", LIFECYCLE_STAGES)
def test_staggered_value_lifecycle_persistence(stage_name, factory):
    """Verify BC values (u, v, w, p) survive across all pipeline steps."""
    nx, ny, nz = 4, 4, 4
    state = factory(nx=nx, ny=ny, nz=nz)
    
    # 1. Existence Check: Access _boundary_conditions slot
    bcs = getattr(state, "_boundary_conditions", None)
    assert bcs is not None, f"{stage_name}: '_boundary_conditions' slot missing"
    
    # 2. Locate specific BC entry
    # Assuming _boundary_conditions is a list of objects/dicts
    bc_entry = next((bc for bc in bcs if getattr(bc, "location", None) == "x_min"), None)
    assert bc_entry is not None, f"{stage_name}: BC entry for 'x_min' lost"
    
    # 3. Verify Value Presence (u component)
    actual_u = getattr(bc_entry, "values", {}).get("u")
    assert isinstance(actual_u, (int, float)), f"{stage_name}: Value corruption for 'u'"

def test_staggered_component_validity():
    """Verify BC values strictly follow the schema: u, v, w, p."""
    state = make_step1_output_dummy(nx=4, ny=4, nz=4)
    allowed_keys = {"u", "v", "w", "p"}
    
    for bc in state._boundary_conditions:
        if hasattr(bc, "values"):
            # Check keys in the values attribute
            provided_keys = set(bc.values.keys())
            assert provided_keys.issubset(allowed_keys), \
                f"Invalid component in BC values: {provided_keys - allowed_keys}"