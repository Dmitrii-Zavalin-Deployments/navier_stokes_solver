# tests/property_integrity/test_boundary_condition_integrity.py

import pytest

from tests.helpers.solver_output_schema_dummy import make_output_schema_dummy
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy
from tests.helpers.solver_step5_output_dummy import make_step5_output_dummy

# Centralizing allowed types to prevent drift
VALID_BC_TYPES = {"no-slip", "free-slip", "inflow", "outflow", "pressure"}

# Updated lifecycle stages to include Step 5 explicitly
BC_INTEGRITY_STAGES = [
    ("Step 2", make_step2_output_dummy),
    ("Step 3", make_step3_output_dummy),
    ("Step 4", make_step4_output_dummy),
    ("Step 5", make_step5_output_dummy),
    ("Final Output", make_output_schema_dummy),
]

@pytest.mark.parametrize("stage_name, factory", BC_INTEGRITY_STAGES)
def test_boundary_type_persistence_and_validity(stage_name, factory):
    """
    Integrity: Verify BC types persist as standardized strings across 
    the computation and output phases within the Constitutional Safes.
    """
    state = factory()
    
    # Adaptive Access: Extract BCs from dict or object (SSoT-compliant)
    if isinstance(state, dict):
        bc_list = state.get("config", {}).get("boundary_conditions", [])
    else:
        bc_list = state.config.boundary_conditions
    
    # Structural Integrity: BC definitions must be a list
    assert isinstance(bc_list, list), f"{stage_name}: BC definitions must be a list."
    assert len(bc_list) > 0, f"{stage_name}: BC list is empty."

    for bc in bc_list:
        # Schema Contract: Ensure mandatory keys exist
        assert "location" in bc, f"{stage_name}: BC missing 'location'."
        assert "type" in bc, f"{stage_name}: BC missing 'type'."
        # Validation: Ensure type matches the Schema allowed values
        assert bc["type"] in VALID_BC_TYPES, f"{stage_name}: Invalid BC type '{bc['type']}'"

def test_step2_matrix_bc_logic():
    """
    Physics: Verify that in Step 2, BCs are available to set the Laplacian A 
    and that the domain closure is complete.
    """
    state = make_step2_output_dummy()
    
    # Verify PPE Matrix exists (Rule 8: Singular Access)
    assert state.ppe.A is not None, "Step 2: PPE Matrix A missing in State."
    
    # Dynamic Domain Closure Check: 
    # For a 3D grid, we expect 6 faces (x_min, x_max, y_min, y_max, z_min, z_max)
    bc_list = state.config.boundary_conditions
    bc_count = len(bc_list)
    
    assert bc_count >= 6, f"Step 2: Insufficient BCs for 3D domain. Found {bc_count}."