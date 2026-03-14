# tests/property_integrity/test_output_trigger_logic.py

import pytest
from tests.helpers.solver_output_schema_dummy import make_output_schema_dummy
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy
from tests.helpers.solver_step5_output_dummy import make_step5_output_dummy


@pytest.mark.parametrize("stage_name, factory", [
    ("Step 4", make_step4_output_dummy),
    ("Step 5", make_step5_output_dummy),
    ("Final Output", make_output_schema_dummy)
])
def test_output_interval_persistence(stage_name, factory):
    """
    Integrity: Verify simulation_parameters.output_interval is 
    persistent through the late-stage pipeline.
    """
    nx, ny, nz = 4, 4, 4
    state = factory(nx=nx, ny=ny, nz=nz)
    
    # Access the correct internal attribute from SolverState.__slots__
    params = getattr(state, "_simulation_parameters", None)
    
    assert params is not None, f"{stage_name}: '_simulation_parameters' missing in state"
    assert hasattr(params, "output_interval"), f"{stage_name}: 'output_interval' missing"
    assert isinstance(params.output_interval, (int, float)), "Interval must be numeric"

def test_step5_write_trigger_logic():
    """
    Logic: Verify that the Step 5 'Snapshot' generation adheres to 
    the modulo interval defined in the configuration.
    """
    nx, ny, nz = 4, 4, 4
    state = make_step5_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # Access state attributes directly as per __slots__
    iteration = state._iteration
    params = state._simulation_parameters
    interval = params.output_interval
    
    # Access diagnostics via manifest as per your current architecture
    diagnostics = state.manifest 
    
    assert diagnostics is not None, "Step 5 diagnostics (manifest) missing"
    
    # If the system reports a snapshot was generated, verify the math
    # Note: Ensure diagnostics has a 'snapshot_generated' property or similar
    if getattr(diagnostics, "snapshot_generated", False):
        assert iteration % interval == 0, \
            f"Trigger Error: Snapshot generated at iter {iteration} (interval {interval})"