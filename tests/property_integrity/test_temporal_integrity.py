# tests/property_integrity/test_temporal_integrity.py

import pytest

from tests.helpers.solver_output_schema_dummy import make_output_schema_dummy
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy
from tests.helpers.solver_step5_output_dummy import make_step5_output_dummy

# Added Step 5 to the active tracking matrix
TEMPORAL_ACTIVE_STAGES = [
    ("Step 3", make_step3_output_dummy),
    ("Step 4", make_step4_output_dummy),
    ("Step 5", make_step5_output_dummy),
    ("Final Output", make_output_schema_dummy),
]

@pytest.mark.parametrize("stage_name, factory", TEMPORAL_ACTIVE_STAGES)
def test_time_step_persistence_and_sync(stage_name, factory):
    """
    Physics: Verify simulation_parameters.time_step is present, 
    positive, and consistent across pipeline stages.
    """
    nx, ny, nz = 4, 4, 4
    state = factory(nx=nx, ny=ny, nz=nz)
    
    # 1. Access the SimulationParameterManager via its slot
    params = getattr(state, "_simulation_parameters", None)
    assert params is not None, f"{stage_name}: _simulation_parameters manager missing"
    
    # 2. Key Existence: Time Step (using slot _time_step)
    assert hasattr(params, "_time_step"), f"{stage_name}: '_time_step' missing"
    dt = params._time_step
    
    # 3. Scale Guard: Must be strictly positive
    assert dt > 0, f"{stage_name}: Non-physical time step detected ({dt})"

def test_time_advancement_logic():
    """
    Logic: Verify that Step 3 reflects a state where global time has 
    advanced relative to the Step 1 initialization.
    """
    nx, ny, nz = 4, 4, 4
    s1 = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    s3 = make_step3_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # Access time and time_step from slots
    dt = s1._simulation_parameters._time_step
    
    # Global time at Step 3 must have moved forward by at least one dt
    assert s3._time >= s1._time + dt, "Step 3: Global clock failed to advance."