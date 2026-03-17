# tests/step4/test_step4_orchestrator_minimal_grid.py

from src.solver_state import SolverState
from src.step4.orchestrate_step4 import orchestrate_step4_state
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy


def test_step4_orchestrator_minimal_grid():
    # Minimal grid 1×1×1
    state = make_step3_output_dummy(nx=1, ny=1, nz=1)
    state.config["boundary_conditions"] = []

    result = orchestrate_step4_state(state)

    assert isinstance(result, SolverState)

    # Shapes must be correct
    assert result.P_ext.shape == (1 + 2, 1 + 2, 1 + 2)
    assert result.U_ext.shape == (1 + 3, 1 + 2, 1 + 2)
    assert result.V_ext.shape == (1,     1 + 3, 1 + 2)
    assert result.W_ext.shape == (1,     1,     1 + 3)

    # No index errors → test passes if no exception
    assert result.ready_for_time_loop is True
