# tests/property_integrity/test_temporal_integrity.py

import pytest

from tests.helpers.solver_output_schema_dummy import make_output_schema_dummy
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy

# Per Property Tracking Matrix: dt is the "Active Operator" from Step 3 onwards
TEMPORAL_ACTIVE_STAGES = [
    ("Step 3", make_step3_output_dummy),
    ("Step 4", make_step4_output_dummy),
    ("Final Output", make_output_schema_dummy),
]

@pytest.mark.parametrize("stage_name, factory", TEMPORAL_ACTIVE_STAGES)
def test_time_step_persistence_and_sync(stage_name, factory):
    """
    Physics: Verify simulation_parameters.time_step (dt) is present, 
    positive, and synchronized with internal constants from Step 3 onwards.
    """
    state = factory()
    
    # 1. Existence Check: Input Schema Department
    assert "simulation_parameters" in state.__dict__ or hasattr(state, "simulation_parameters"), \
        f"{stage_name}: simulation_parameters department missing"
    
    # 2. Key Existence: Time Step
    params = state.simulation_parameters
    assert hasattr(params, "time_step"), f"{stage_name}: 'time_step' missing from simulation_parameters"
    
    dt_input = params["time_step"]
    
    # 3. Internal Sync Check: constants.dt must match simulation_parameters.time_step
    # This ensures the solver math is using the actual user-defined increment.
    assert "dt" in state.config.simulation_parameters, f"{stage_name}: Internal 'dt' missing from constants"
    dt_internal = state.config.simulation_parameters["dt"]
    
    assert dt_input == dt_internal, f"{stage_name}: Desync! Internal dt ({dt_internal}) != Input time_step ({dt_input})"
    
    # 4. Scale Guard: Must be strictly positive
    assert dt_internal > 0, f"{stage_name}: Non-physical time step detected ({dt_internal})"

def test_time_advancement_logic():
    """
    Logic: Verify that Step 3 reflects a state where global time has 
    advanced relative to the Step 1 initialization.
    """
    from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy
    s1 = make_step1_output_dummy()
    s3 = make_step3_output_dummy()
    
    dt = s1.constants["dt"]
    
    # Global time at Step 3 must have moved forward by at least one dt
    assert s3.time >= s1.time + dt, "Step 3: Global clock failed to advance during prediction/correction."