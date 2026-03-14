# tests/property_integrity/test_initial_conditions_persistence.py

import numpy as np
import pytest

from tests.helpers.solver_output_schema_dummy import make_output_schema_dummy
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy
from tests.helpers.solver_step5_output_dummy import make_step5_output_dummy

# All stages must preserve the "Intent" of the simulation (Initial Conditions)
ALL_STAGES = [
    ("Step 1", make_step1_output_dummy),
    ("Step 2", make_step2_output_dummy),
    ("Step 3", make_step3_output_dummy),
    ("Step 4", make_step4_output_dummy),
    ("Step 5", make_step5_output_dummy),
    ("Final Output", make_output_schema_dummy),
]

@pytest.mark.parametrize("stage_name, factory", ALL_STAGES)
def test_initial_velocity_persistence(stage_name, factory):
    """
    Integrity: Verify initial_conditions.velocity attribute exists and is a 3D NumPy array.
    """
    state = factory()
    
    # 1. Department Existence (Rule 4: SSoT Architecture)
    assert hasattr(state.config, "initial_conditions"), f"{stage_name}: config.initial_conditions missing"
    
    # 2. Structural & Type Integrity (Rule 9: Hybrid Memory Foundation)
    v0 = state.config.initial_conditions.velocity
    assert v0 is not None, f"{stage_name}: initial velocity is None"
    assert isinstance(v0, np.ndarray), f"{stage_name}: velocity must be a NumPy array"
    assert v0.shape == (3,), f"{stage_name}: velocity must be a 3D vector [u, v, w]"

def test_initial_conditions_immutability():
    """
    Physics: Ensure 'initial_conditions.velocity' remains constant and is not 
    overwritten by evolved field data (U, V, W).
    """
    # Grab a late-stage state where fields have definitely evolved
    state = make_step3_output_dummy()
    
    # Expected immutable value
    expected_initial = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    
    # Direct comparison of the immutable foundation (Rule 9)
    np.testing.assert_array_almost_equal(
        state.config.initial_conditions.velocity, 
        expected_initial,
        err_msg="Initial conditions corrupted by field updates!"
    )