# tests/step3/test_update_health.py

import numpy as np

from src.step3.update_health import update_health
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy


def _make_fields(state):
    return {
        "U": state.fields["U"].copy(),
        "V": state.fields["V"].copy(),
        "W": state.fields["W"].copy(),
        "P": state.fields["P"].copy(),
    }


def test_zero_velocity():
    """
    With zero velocity everywhere, divergence norm and max velocity must be zero.
    """
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)

    fields = _make_fields(state)
    P_new = fields["P"]

    health = update_health(state, fields, P_new)

    assert health["post_correction_divergence_norm"] == 0.0
    assert health["max_velocity_magnitude"] == 0.0


def test_uniform_velocity():
    """
    Max velocity magnitude must reflect the largest absolute velocity component.
    """
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)

    fields = _make_fields(state)
    fields["U"].fill(2.0)

    P_new = fields["P"]

    health = update_health(state, fields, P_new)

    assert health["max_velocity_magnitude"] == 2.0


def test_divergent_field():
    """
    If divergence operator returns a nonzero pattern, divergence norm must be > 0.
    """
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)

    pattern = np.ones_like(state.fields["P"])

    def div_op(U, V, W):
        return pattern

    state.operators["divergence"] = div_op

    fields = _make_fields(state)
    P_new = fields["P"]

    health = update_health(state, fields, P_new)

    assert health["post_correction_divergence_norm"] > 0.0


def test_minimal_grid_no_crash():
    """
    Minimal 1×1×1 grid: only checks that the function does not crash.
    """
    state = make_step2_output_dummy(nx=1, ny=1, nz=1)

    def div_zero(U, V, W):
        return np.zeros((1, 1, 1))

    state.operators["divergence"] = div_zero

    fields = _make_fields(state)
    P_new = fields["P"]

    health = update_health(state, fields, P_new)

    assert "post_correction_divergence_norm" in health
    assert "max_velocity_magnitude" in health
    assert "cfl_advection_estimate" in health
