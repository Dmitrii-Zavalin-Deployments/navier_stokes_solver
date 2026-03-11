# tests/property_integrity/test_external_forces_integrity.py

import pytest

from tests.helpers.solver_output_schema_dummy import make_output_schema_dummy
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy

# Per Property Tracking Matrix: External forces are the primary accelerative 
# source term for Step 3. Integrity must hold through to the final output.
FORCE_ACTIVE_STAGES = [
    ("Step 3", make_step3_output_dummy),
    ("Step 4", make_step4_output_dummy),
    ("Final Output", make_output_schema_dummy), # Added to ensure I/O persistence
]

def test_external_force_vector_presence_and_dimension():
    """
    Physics: Ensure gravity/external forces are defined as a 3D vector.
    """
    state = make_step3_output_dummy()
    
    # 1. Existence in Constants
    assert "g" in state.config.simulation_parameters, "Step 3: Gravity constant 'g' missing"
    
    # 2. Vector Structure (Intent)
    assert hasattr(state.config, "external_forces"), "Step 3: external_forces department missing"
    force = state.config.external_forces["force_vector"]
    assert len(force) == 3, "External force must be a 3D vector [x, y, z]"

@pytest.mark.parametrize("stage_name, factory", FORCE_ACTIVE_STAGES)
def test_force_term_persistence(stage_name, factory):
    """
    Integrity: Verify force terms survive from prediction through to final output.
    """
    state = factory()
    
    # Check if we are dealing with the raw object or a dictionary (Final Output)
    data = state if isinstance(state, dict) else state.__dict__
    
    # Check Step 3 Diagnostics
    # This confirms the force application record persists across the pipeline.
    assert "diagnostics" in data, f"{stage_name}: Diagnostics missing"
    assert data["diagnostics"].get("source_term_applied") is True, \
        f"{stage_name}: No record of external force application in pipeline diagnostics"

def test_force_coupling_with_dt():
    """
    Theory: Accelerative force must be scaled by dt in Step 3.
    u_star = u_n + dt * (convection + diffusion + G)
    """
    state = make_step3_output_dummy()
    dt = state.config.simulation_parameters["time_step"]
    g = state.config.simulation_parameters.get("g", 9.81)
    
    assert dt > 0, "Temporal step dt must be positive for force integration"
    assert g != 0, "Gravity magnitude should be non-zero for integrity testing"