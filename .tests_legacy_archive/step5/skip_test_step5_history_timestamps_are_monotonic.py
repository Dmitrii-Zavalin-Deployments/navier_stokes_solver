# tests/step5/test_step5_history_timestamps_are_monotonic.py

import types
from unittest.mock import MagicMock

from src.step5.orchestrate_step5_state import orchestrate_step5_state


def make_state():
    state = types.SimpleNamespace()
    state.time = 0.0
    state.step_index = 0
    state.history = None

    state.constants = {"dt": 0.1}
    state.config = {
        "total_time": 0.4,
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


def test_step5_history_timestamps_are_monotonic(monkeypatch):
    state = make_state()

    monkeypatch.setattr("src.step3.orchestrate_step3_state", MagicMock())

    orchestrate_step5_state(state)

    times = state.history["times"]
    assert all(times[i] < times[i+1] for i in range(len(times)-1))
