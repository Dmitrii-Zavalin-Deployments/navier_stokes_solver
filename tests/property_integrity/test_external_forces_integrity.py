# tests/property_integrity/test_external_forces_integrity.py

import numpy as np
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
    Physics: Ensure external forces are defined as a 3D vector.
    """
    state = make_step3_output_dummy()
    
    # Structural check for existence in the force manager
    assert hasattr(state, "external_forces"), "Step 3: external_forces department missing"
    
    # Type and dimension validation
    force_vector = state.external_forces.force_vector
    assert isinstance(force_vector, np.ndarray), "Force vector must be a NumPy array"
    assert force_vector.shape == (3,), "External force must be a 3D vector [x, y, z]"

@pytest.mark.parametrize("stage_name, factory", FORCE_ACTIVE_STAGES)
def test_force_term_persistence(stage_name, factory):
    """
    Integrity: Verify force terms are present in the ExternalForceManager across all stages.
    """
    state = factory()
    
    # Strict container access per Rule 4 (SSoT)
    assert hasattr(state, "external_forces"), f"{stage_name}: ExternalForceManager missing"
    assert state.external_forces.force_vector is not None, \
        f"{stage_name}: Force vector record is None"

def test_force_vector_magnitude_validity():
    """
    Physics Check: Ensure the force vector has been initialized with valid values.
    """
    state = make_step3_output_dummy()
    force_vector = state.external_forces.force_vector
    
    # Scientific integrity: Validate against non-finite values to prevent propagation
    assert not np.any(np.isnan(force_vector)), "Force vector contains NaN values"
    assert not np.any(np.isinf(force_vector)), "Force vector contains Inf values"

def test_external_forces_immutability_logic():
    """
    Theory: Ensure the force vector does not change value from start to finish.
    """
    s1 = make_step1_output_dummy() # Ensure this is imported
    s_final = make_output_schema_dummy()
    
    np.testing.assert_array_equal(
        s1.external_forces.force_vector, 
        s_final.external_forces.force_vector,
        err_msg="External forces diverged between Step 1 and Final Output."
    )