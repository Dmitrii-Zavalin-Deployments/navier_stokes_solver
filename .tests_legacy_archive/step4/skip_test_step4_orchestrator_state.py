# tests/step4/test_step4_orchestrator_state.py

from src.solver_state import SolverState
from src.step4.orchestrate_step4 import orchestrate_step4_state
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy


def test_orchestrate_step4_state_minimal():
    # Create a minimal valid SolverState coming out of Step 3
    state = make_step3_output_dummy(nx=2, ny=2, nz=2)

    # Run Step 4
    result = orchestrate_step4_state(state)

    # Basic type and structure checks
    assert isinstance(result, SolverState)

    # Step‑4 must populate extended fields
    assert result.P_ext is not None
    assert result.U_ext is not None
    assert result.V_ext is not None
    assert result.W_ext is not None

    # Step‑4 must mark readiness for the time loop
    assert result.ready_for_time_loop is True

    # Diagnostics must be present and a dict
    assert isinstance(result.step4_diagnostics, dict)
