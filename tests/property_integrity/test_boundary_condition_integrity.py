# tests/property_integrity/test_boundary_condition_integrity.py

import pytest

from tests.helpers.solver_output_schema_dummy import make_output_schema_dummy
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy
from tests.helpers.solver_step5_output_dummy import make_step5_output_dummy

# Centralizing allowed types to prevent drift
VALID_BC_TYPES = {"no-slip", "free-slip", "inflow", "outflow", "pressure"}

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
    
    # Adaptive Access: Extract BCs from the object (SSoT-compliant)
    # Using domain_configuration (as defined in SolverState slots)
    bc_list = state.boundary_conditions.conditions
    
    # Structural Integrity: BC definitions must be a list
    assert isinstance(bc_list, list), f"{stage_name}: BC definitions must be a list."
    assert len(bc_list) > 0, f"{stage_name}: BC list is empty."

    for bc in bc_list:
        # 1. Verify existence of attributes
        assert hasattr(bc, "location"), f"{stage_name}: BC missing 'location'."
        assert hasattr(bc, "type"), f"{stage_name}: BC missing 'type'."
        
        # 2. Access attributes directly via dot notation
        assert bc.location is not None, f"{stage_name}: BC location is None."
        assert bc.type in VALID_BC_TYPES, f"{stage_name}: Invalid BC type '{bc.type}'"

def test_step2_matrix_bc_logic():
    """
    Physics: Verify that in Step 2, BCs are available to set the Laplacian A 
    and that the domain closure is complete.
    """
    state = make_step2_output_dummy()
    
    # Verify Stencil Matrix exists (Accessing the correct slot from SolverState)
    assert state.stencil_matrix is not None, "Step 2: Stencil Matrix missing in State."
    
    # Access BCs through domain_configuration
    bc_list = state.boundary_conditions.conditions
    bc_count = len(bc_list)
    
    assert bc_count >= 6, f"Step 2: Insufficient BCs for 3D domain. Found {bc_count}."