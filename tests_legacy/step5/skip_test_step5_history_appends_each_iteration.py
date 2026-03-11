# tests/step5/test_step5_history_appends_each_iteration.py

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
        "total_time": 0.3,
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


def test_step5_history_appends_each_iteration(monkeypatch):
    state = make_state()

    monkeypatch.setattr("src.step3.orchestrate_step3_state", MagicMock())

    orchestrate_step5_state(state)

    # dt = 0.1, total_time = 0.3 → 3 iterations
    assert len(state.history["times"]) == 3
    assert len(state.history["steps"]) == 3
    assert len(state.history["divergence_norms"]) == 3
    assert len(state.history["max_velocity_history"]) == 3
    assert len(state.history["cfl_values"]) == 3
    assert len(state.history["ppe_iterations"]) == 3
