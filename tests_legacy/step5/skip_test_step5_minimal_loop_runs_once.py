# tests/step5/test_step5_minimal_loop_runs_once.py

import types
from unittest.mock import MagicMock

from src.step5.orchestrate_step5_state import orchestrate_step5_state


def make_state(dt=0.1, total_time=0.1, max_steps=10):
    state = types.SimpleNamespace()
    state.time = 0.0
    state.step_index = 0
    state.history = None

    state.constants = {"dt": dt}
    state.config = {
        "total_time": total_time,
        "max_steps": max_steps,
        "output_interval": None,
    }

    # Minimal required structures
    state.health = {}
    state.ppe = {}
    state.fields = {}
    state.P_ext = None
    state.U_ext = None
    state.V_ext = None
    state.W_ext = None

    return state


def test_step5_minimal_loop_runs_once(monkeypatch):
    state = make_state(dt=0.1, total_time=0.1)

    mock_step3 = MagicMock()
    monkeypatch.setattr("src.step3.orchestrate_step3_state", mock_step3)

    orchestrate_step5_state(state)

    assert mock_step3.call_count == 1
    assert state.time == 0.1
    assert state.step_index == 1
