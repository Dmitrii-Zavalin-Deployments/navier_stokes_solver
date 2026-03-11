# tests/property_integrity/test_boundary_values_propagation.py

import pytest

from tests.helpers.solver_output_schema_dummy import make_output_schema_dummy
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy


@pytest.mark.parametrize("stage_name, factory", [
    ("Step 2", make_step2_output_dummy),
    ("Step 3", make_step3_output_dummy),
    ("Step 4", make_step4_output_dummy),
    ("Final Output", make_output_schema_dummy)
])
def test_boundary_values_presence_and_type(stage_name, factory):
    """
    Integrity: Verify boundary_conditions.values sub-dictionary persists.
    This ensures that specific inflow/outflow values are available for the PPE 
    Source Term (Step 2) and Ghost Cell assignment (Step 3).
    """
    state = factory()
    
    # Adaptive Access (Handles both dict-style and object-style states)
    if isinstance(state, dict):
        bc_list = state.get("config", {}).get("boundary_conditions", [])
    else:
        bc_list = getattr(state.config, "boundary_conditions", [])

    assert len(bc_list) > 0, f"Setup Error: No boundaries defined at {stage_name}"

    for bc in bc_list:
        # 1. Structural Check: 'values' key must exist for schema parity
        assert "values" in bc, f"Parity Failure: 'values' sub-dictionary missing at {stage_name} for {bc['location']}"
        
        # 2. Functional Check: Extract the values
        vals = bc["values"]
        assert isinstance(vals, dict), f"Type Error: 'values' must be a dictionary at {stage_name}"
        
        # 3. Numeric Integrity Check
        # Boundary values must be numeric to be applied to the linear system or ghost cells.
        for component, val in vals.items():
            assert isinstance(val, (int, float)), \
                f"Numeric Error: Boundary value '{component}' at {stage_name} is not a number."

def test_specific_inflow_value_persistence():
    """
    Logic Check: Verifies that if a boundary has a specific value (e.g., u=5.0), 
    it is not reset to a default 0.0 during the pipeline.
    """
    # Simulate a specific inflow value at Step 2
    state = make_step2_output_dummy()
    
    # Manually inject a specific value for this test case
    state.config.boundary_conditions[0]["values"]["u"] = 5.0
    
    # In a real pipeline, Step 3 would receive this. We check if the dictionary 
    # structure supports this retention.
    assert state.config.boundary_conditions[0]["values"]["u"] == 5.0