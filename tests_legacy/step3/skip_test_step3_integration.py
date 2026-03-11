# tests/step3/test_step3_integration.py

from src.solver_state import SolverState
from src.step3.orchestrate_step3 import orchestrate_step3_state
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy


def test_step3_integration_minimal():
    """
    Step‑3 orchestrator must run end‑to‑end on a valid Step‑2 dummy state.
    """
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)

    result = orchestrate_step3_state(
        state=state,
        current_time=0.0,
        step_index=0,
    )

    assert isinstance(result, SolverState)
    assert "P" in result.fields
    assert result.fields["P"].shape == (3, 3, 3)


def test_step3_optional_fields():
    """
    Step‑3 must tolerate optional or empty fields in Step‑2 state.
    """
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)

    # Remove optional fields to ensure Step‑3 handles them gracefully
    state.history = {}
    state.health = {}
    state.boundary_conditions = None

    result = orchestrate_step3_state(
        state=state,
        current_time=0.0,
        step_index=0,
    )

    assert isinstance(result, SolverState)
    assert "P" in result.fields
