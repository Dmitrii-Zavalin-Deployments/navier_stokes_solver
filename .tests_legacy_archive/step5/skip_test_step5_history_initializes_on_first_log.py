# tests/step5/test_step5_history_initializes_on_first_log.py

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
        "total_time": 0.1,
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


def test_step5_history_initializes_on_first_log(monkeypatch):
    state = make_state()

    monkeypatch.setattr("src.step3.orchestrate_step3_state", MagicMock())

    orchestrate_step5_state(state)

    assert state.history is not None
    assert "times" in state.history
    assert "steps" in state.history
    assert "divergence_norms" in state.history
    assert "max_velocity_history" in state.history
    assert "cfl_values" in state.history
    assert "ppe_iterations" in state.history
