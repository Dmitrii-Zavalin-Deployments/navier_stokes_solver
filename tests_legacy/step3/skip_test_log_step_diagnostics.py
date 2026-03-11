# tests/step3/test_log_step_diagnostics.py

import numpy as np

from src.step3.log_step_diagnostics import log_step_diagnostics
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy


def _make_fields(state):
    return {
        "U": state.fields["U"].copy(),
        "V": state.fields["V"].copy(),
        "W": state.fields["W"].copy(),
        "P": state.fields["P"].copy(),
    }


def test_record_structure():
    """
    log_step_diagnostics must return a complete diagnostic record.
    """
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)

    # Step 3 health fields must exist
    state.health = {
        "post_correction_divergence_norm": 0.0,
        "max_velocity_magnitude": 0.0,
    }

    fields = _make_fields(state)

    rec = log_step_diagnostics(state, fields, current_time=0.1, step_index=1)

    assert "time" in rec
    assert "step_index" in rec
    assert "divergence_norm" in rec
    assert "max_velocity" in rec
    assert "ppe_iterations" in rec
    assert "energy" in rec


def test_energy_positive():
    """
    Energy must be positive and finite when velocities are nonzero.
    """
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)

    U = np.ones_like(state.fields["U"])
    V = 0.5 * np.ones_like(state.fields["V"])
    W = -0.25 * np.ones_like(state.fields["W"])

    fields = {
        "U": U,
        "V": V,
        "W": W,
        "P": state.fields["P"].copy(),
    }

    state.health = {
        "post_correction_divergence_norm": 0.0,
        "max_velocity_magnitude": 1.0,
    }

    rec = log_step_diagnostics(state, fields, current_time=0.1, step_index=1)

    assert rec["energy"] > 0.0
    assert np.isfinite(rec["energy"])


def test_energy_decay():
    """
    If velocities decrease, energy must not increase.
    """
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)

    state.health = {
        "post_correction_divergence_norm": 0.0,
        "max_velocity_magnitude": 1.0,
    }

    # Step 1: higher velocity
    fields1 = _make_fields(state)
    fields1["U"].fill(1.0)
    rec1 = log_step_diagnostics(state, fields1, current_time=0.1, step_index=1)

    # Step 2: lower velocity
    fields2 = _make_fields(state)
    fields2["U"].fill(0.5)
    rec2 = log_step_diagnostics(state, fields2, current_time=0.2, step_index=2)

    assert rec2["energy"] <= rec1["energy"]
