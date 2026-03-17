# tests/step3/test_step3_optional.py

import numpy as np
import pytest

from src.solver_state import SolverState
from src.step3.orchestrate_step3 import orchestrate_step3_state
from src.step3.update_health import update_health
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy


def test_missing_mask_and_fields_raises_runtimeerror():
    """
    Step‑3 must fail cleanly when both mask and fields are missing.
    """
    bad_state = SolverState()
    bad_state.grid = {"nx": 3, "ny": 3, "nz": 3}
    bad_state.fields = None
    bad_state.mask = None

    with pytest.raises(RuntimeError):
        orchestrate_step3_state(bad_state, current_time=0.0, step_index=0)


def test_divergence_operator_raising_exception_is_handled():
    """
    update_health must not crash if divergence op raises.
    """
    # Construct a minimal dict-like state (update_health supports this)
    state = {
        "fields": {
            "P": np.zeros((3, 3, 3)),
            "U": np.zeros((4, 3, 3)),
            "V": np.zeros((3, 4, 3)),
            "W": np.zeros((3, 3, 4)),
        },
        "operators": {
            "divergence": None,
        },
    }

    def bad_div(U, V, W):
        raise RuntimeError("boom")

    state["operators"]["divergence"] = bad_div

    fields = state["fields"]
    P = fields["P"]

    health = update_health(state, fields, P)
    assert health["post_correction_divergence_norm"] == 0.0


def test_pressure_solver_returning_nans_is_handled():
    """
    Step‑3 must not crash if pressure solver returns NaNs.
    """
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)

    def fake_solver(rhs):
        P = np.full_like(rhs, np.nan)
        return P, {"converged": False}

    state.ppe["solver"] = fake_solver

    out = orchestrate_step3_state(state, current_time=0.0, step_index=0)
    assert np.isnan(out.fields["P"]).all()
