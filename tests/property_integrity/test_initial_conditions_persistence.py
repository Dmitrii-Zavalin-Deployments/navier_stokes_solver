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
def test_initial_conditions_persistence(stage_name, factory):
    """
    Integrity: Verify initial_conditions manager exists, velocity is a 3D NumPy array,
    and pressure is a numeric scalar. (Consolidated per Rule 6).
    """
    state = factory()
    
    # 1. Direct access via the manager (Rule 4: SSoT Architecture)
    assert hasattr(state, "_initial_conditions"), f"{stage_name}: _initial_conditions manager missing"
    ic = state._initial_conditions
    
    # 2. Velocity Integrity (Hybrid Memory Foundation)
    v0 = getattr(ic, "velocity", ic._velocity)
    assert isinstance(v0, np.ndarray), f"{stage_name}: velocity must be a NumPy array"
    assert v0.shape == (3,), f"{stage_name}: velocity must be a 3D vector [u, v, w]"
    
    # 3. Pressure Integrity (Atomic Numerical Truth)
    p0 = getattr(ic, "pressure", ic._pressure)
    assert isinstance(p0, (float, np.float32, np.float64)), f"{stage_name}: Pressure must be numeric"

def test_initial_conditions_immutability():
    """
    Physics: Ensure 'initial_conditions' (velocity and pressure) remain constant
    after field evolution (Rule 9: Hybrid Memory Foundation).
    """
    state = make_step3_output_dummy()
    
    # Expected immutable values
    expected_v = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    expected_p = 0.0
    
    ic = state._initial_conditions
    
    # Validate Velocity (Machine Precision)
    np.testing.assert_array_almost_equal(
        getattr(ic, "velocity", ic._velocity), 
        expected_v,
        err_msg="Initial velocity corrupted by field updates!"
    )
    
    # Validate Pressure (Machine Precision)
    p_val = getattr(ic, "pressure", ic._pressure)
    assert np.isclose(p_val, expected_p), "Initial pressure corrupted by field updates!"