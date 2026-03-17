# tests/step4/test_step4_orchestrator_no_bcs.py

from src.solver_state import SolverState
from src.step4.orchestrate_step4 import orchestrate_step4_state
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy


def test_step4_orchestrator_no_bcs():
    # Step 3 dummy with no BCs
    state = make_step3_output_dummy(nx=2, ny=2, nz=2)
    state.config["boundary_conditions"] = []

    result = orchestrate_step4_state(state)

    # Type
    assert isinstance(result, SolverState)

    # Extended fields exist
    assert result.P_ext is not None
    assert result.U_ext is not None
    assert result.V_ext is not None
    assert result.W_ext is not None

    # Ghost layers remain zero
    assert (result.P_ext[0, :, :] == 0).all()
    assert (result.U_ext[0, :, :] == 0).all()

    # Diagnostics exist
    assert isinstance(result.step4_diagnostics, dict)

    # Ready flag
    assert result.ready_for_time_loop is True
