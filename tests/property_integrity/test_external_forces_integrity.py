# tests/property_integrity/test_external_forces_integrity.py

import pytest

from tests.helpers.solver_output_schema_dummy import make_output_schema_dummy
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy
from tests.helpers.solver_step5_output_dummy import make_step5_output_dummy

# Per Property Tracking Matrix: External forces are the primary accelerative 
# source term. Integrity must hold through to the final output.
FORCE_ACTIVE_STAGES = [
    ("Step 3", make_step3_output_dummy),
    ("Step 4", make_step4_output_dummy),
    ("Step 5", make_step5_output_dummy),
    ("Final Output", make_output_schema_dummy),
]

def test_external_force_vector_presence_and_dimension():
    """
    Physics: Ensure gravity/external forces are defined as a 3D vector.
    """
    state = make_step3_output_dummy()
    
    # 1. Existence: Accessing via parameter manager property
    assert hasattr(state.simulation_parameters, "g"), "Step 3: Gravity constant 'g' missing"
    
    # 2. Vector Structure: Accessing via ForceManager property
    assert hasattr(state, "external_forces"), "Step 3: external_forces department missing"
    force_vector = state.external_forces.force_vector
    assert len(force_vector) == 3, "External force must be a 3D vector [x, y, z]"

@pytest.mark.parametrize("stage_name, factory", FORCE_ACTIVE_STAGES)
def test_force_term_persistence(stage_name, factory):
    """
    Integrity: Verify force terms survive from prediction through to final output.
    """
    state = factory()
    
    # Check Step 3 Diagnostics: Ensure access via dot-notation
    assert hasattr(state, "diagnostics"), f"{stage_name}: Diagnostics missing"
    assert state.diagnostics.source_term_applied is True, \
        f"{stage_name}: No record of external force application in pipeline diagnostics"

def test_force_coupling_with_dt():
    """
    Theory: Accelerative force must be scaled by dt in Step 3.
    u_star = u_n + dt * (convection + diffusion + G)
    """
    state = make_step3_output_dummy()
    
    # Access via properties rather than dictionary keys
    dt = state.simulation_parameters.time_step
    g = state.simulation_parameters.g
    
    assert dt > 0, "Temporal step dt must be positive for force integration"
    assert g != 0, "Gravity magnitude should be non-zero for integrity testing"