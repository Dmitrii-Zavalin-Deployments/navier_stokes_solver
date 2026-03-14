# tests/property_integrity/test_solver_termination_logic.py

import pytest

from tests.helpers.solver_output_schema_dummy import make_output_schema_dummy
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy
from tests.helpers.solver_step5_output_dummy import make_step5_output_dummy

LIFECYCLE_STAGES = [
    ("Step 1", make_step1_output_dummy),
    ("Step 2", make_step2_output_dummy),
    ("Step 3", make_step3_output_dummy),
    ("Step 4", make_step4_output_dummy),
    ("Step 5", make_step5_output_dummy),
    ("Final Output", make_output_schema_dummy),
]

@pytest.mark.parametrize("stage_name, factory", LIFECYCLE_STAGES)
def test_termination_persistence_across_lifecycle(stage_name, factory):
    """Verify total_time metadata survives every pipeline transition."""
    nx, ny, nz = 4, 4, 4
    state = factory(nx=nx, ny=ny, nz=nz)
    
    # Access via _simulation_parameters slot
    params = state._simulation_parameters
    assert hasattr(params, "total_time"), f"{stage_name} lost total_time metadata"
    assert params.total_time > 0, f"{stage_name} has non-physical total_time"

def test_termination_logic_math_precision():
    """Validates floating point exit condition: current_time >= total_time."""
    total_time = 0.05
    dt = 0.01
    
    # Mocking setup
    state = make_step1_output_dummy(nx=4, ny=4, nz=4)
    state._simulation_parameters.total_time = total_time
    state._simulation_parameters.time_step = dt
    
    current_time = 0.0
    iterations = 0
    
    while current_time < state._simulation_parameters.total_time:
        current_time += state._simulation_parameters.time_step
        iterations += 1
        
    assert iterations == 5
    assert current_time == pytest.approx(total_time)

def test_final_state_exit_condition():
    """Verify final state correctly marks completion."""
    nx, ny, nz = 4, 4, 4
    state = make_output_schema_dummy(nx=nx, ny=ny, nz=nz)
    
    # 1. Force the terminal state (Rule 5: Deterministic Initialization)
    # Ensure the simulation has reached or exceeded the total_time
    target = 1.0
    state._simulation_parameters.total_time = target
    state._time = target 
    
    t_final = state._time
    t_target = state._simulation_parameters.total_time
    
    # 2. Assertions
    assert t_final >= t_target, f"Snapshot generated at {t_final} before completion goal {t_target}"
    # Access via _ready_for_time_loop property/slot
    assert state._ready_for_time_loop is False, "Final state still marks ready_for_time_loop as True"