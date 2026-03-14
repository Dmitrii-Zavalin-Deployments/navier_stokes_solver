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
    
    # 1. Existence: Accessing via ForceManager
    assert hasattr(state, "external_forces"), "Step 3: external_forces department missing"
    
    # 2. Vector Structure: Verify 3D vector parity
    force_vector = state.external_forces.force_vector
    assert len(force_vector) == 3, "External force must be a 3D vector [x, y, z]"

@pytest.mark.parametrize("stage_name, factory", FORCE_ACTIVE_STAGES)
def test_force_term_persistence(stage_name, factory):
    """
    Integrity: Verify force application records survive to final output.
    """
    state = factory()
    
    # Check persistence via manifest (the SSoT for pipeline diagnostics)
    assert hasattr(state, "manifest"), f"{stage_name}: Manifest missing"
    assert state.manifest.source_term_applied is True, \
        f"{stage_name}: No record of external force application in pipeline manifest"

def test_force_vector_magnitude_validity():
    """
    Physics Check: Ensure the force vector has been initialized with valid values.
    """
    state = make_step3_output_dummy()
    force_vector = state.external_forces.force_vector
    
    # Ensure no NaNs or Infs in the force vector
    assert not np.any(np.isnan(force_vector)), "Force vector contains NaN values"
    assert not np.any(np.isinf(force_vector)), "Force vector contains Inf values"