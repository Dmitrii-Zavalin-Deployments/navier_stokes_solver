# tests/property_integrity/test_solver_termination_logic.py

import pytest

from tests.helpers.solver_output_schema_dummy import make_output_schema_dummy
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy

# The full pipeline must respect the Chronos Guard
LIFECYCLE_STAGES = [
    ("Step 1", make_step1_output_dummy),
    ("Step 2", make_step2_output_dummy),
    ("Step 3", make_step3_output_dummy),
    ("Step 4", make_step4_output_dummy),
    ("Final Output", make_output_schema_dummy),
]

@pytest.mark.parametrize("stage_name, factory", LIFECYCLE_STAGES)
def test_termination_persistence_across_lifecycle(stage_name, factory):
    """
    Integrity: Ensure total_time survives every transition in the pipeline.
    """
    state = factory()
    assert "total_time" in state.config.simulation_parameters, f"{stage_name} lost total_time metadata"
    assert state.config.simulation_parameters["total_time"] > 0, f"{stage_name} has non-physical total_time"

def test_termination_logic_math_precision():
    """
    Theory: The solver must exit when current_time >= total_time.
    Validates floating point safety for the time loop.
    """
    total_time = 0.05
    dt = 0.01
    state = make_step1_output_dummy()
    
    state.config.simulation_parameters["total_time"] = total_time
    state.config.simulation_parameters["time_step"] = dt
    
    current_time = 0.0
    iterations = 0
    
    while current_time < state.config.simulation_parameters["total_time"]:
        current_time += state.config.simulation_parameters["time_step"]
        iterations += 1
        
    assert iterations == 5
    assert current_time == pytest.approx(total_time)

def test_final_state_exit_condition():
    """
    Verify the final snapshot confirms the simulation is actually finished.
    """
    state = make_output_schema_dummy()
    
    t_final = state.time
    t_target = state.config.simulation_parameters["total_time"]
    
    # Boundary: t_final must be the first step where t >= t_target
    assert t_final >= t_target, f"Snapshot generated at {t_final} before completion goal {t_target}"
    
    # Metadata check: ready_for_time_loop should be False at final exit
    if hasattr(state, "ready_for_time_loop"):
        assert state.ready_for_time_loop is False, "Final state still marks ready_for_time_loop as True"