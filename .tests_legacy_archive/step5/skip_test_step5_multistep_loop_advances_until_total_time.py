# tests/step5/test_step5_multistep_loop_advances_until_total_time.py

import types
from unittest.mock import MagicMock

from src.step5.orchestrate_step5_state import orchestrate_step5_state


def make_state(dt=0.1, total_time=0.35, max_steps=100):
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

    state.health = {}
    state.ppe = {}
    state.fields = {}
    state.P_ext = None
    state.U_ext = None
    state.V_ext = None
    state.W_ext = None

    return state


def test_step5_multistep_loop_advances_until_total_time(monkeypatch):
    state = make_state()

    mock_step3 = MagicMock()
    monkeypatch.setattr("src.step3.orchestrate_step3_state", mock_step3)

    orchestrate_step5_state(state)

    # dt = 0.1, total_time = 0.35 → steps at t = 0.0, 0.1, 0.2, 0.3 → 4 steps
    assert mock_step3.call_count == 4
    assert state.step_index == 4
    assert abs(state.time - 0.4) < 1e-12
