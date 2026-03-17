# tests/step3/test_bc_handler_hooks_called.py

from src.step3.orchestrate_step3 import orchestrate_step3_state
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy


def test_bc_handler_hooks_called():
    """
    Step‑3 must call the boundary_conditions handler twice:
        • once in apply_boundary_conditions_pre
        • once in apply_boundary_conditions_post
    """
    state = make_step2_output_dummy(nx=4, ny=4, nz=4)

    calls = {"count": 0}

    def bc_handler(state, fields):
        calls["count"] += 1
        return fields

    state.boundary_conditions = bc_handler

    # Run Step‑3
    orchestrate_step3_state(
        state=state,
        current_time=0.0,
        step_index=0,
    )

    # Expect exactly two calls: pre + post
    assert calls["count"] == 2
