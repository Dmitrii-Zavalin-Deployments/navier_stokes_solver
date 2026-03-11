# tests/step5/test_step5_divergence_free_stability_over_time.py

import types

from src.step5.orchestrate_step5_state import orchestrate_step5_state


def make_state():
    state = types.SimpleNamespace()

    state.time = 0.0
    state.step_index = 0
    state.history = None

    state.constants = {"dt": 0.1}
    state.config = {
        "total_time": 0.5,
        "max_steps": 10,
        "output_interval": None,
    }

    state.health = {
        "post_correction_divergence_norm": 0.01,
        "max_velocity_magnitude": 0.5,
        "cfl_advection_estimate": 0.2,
    }
    state.ppe = {"iterations": 3}

    state.fields = {}
    state.P_ext = None
    state.U_ext = None
    state.V_ext = None
    state.W_ext = None

    return state


def test_step5_divergence_free_stability_over_time(monkeypatch):
    state = make_state()

    # Step 3 stub that reduces divergence over time
    def fake_step3(state, t, step):
        state.health["post_correction_divergence_norm"] *= 0.5

    monkeypatch.setattr("src.step3.orchestrate_step3_state", fake_step3)

    orchestrate_step5_state(state)

    # Divergence should decrease monotonically
    divs = state.history["divergence_norms"]
    assert all(divs[i+1] <= divs[i] for i in range(len(divs)-1))
