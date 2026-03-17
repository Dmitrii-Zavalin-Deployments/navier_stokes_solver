# tests/step5/test_step5_does_not_modify_solverstate_structure.py

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

    # Minimal valid structures
    state.fields = {"P": 1, "U": 2, "V": 3, "W": 4}
    state.P_ext = "P_ext"
    state.U_ext = "U_ext"
    state.V_ext = "V_ext"
    state.W_ext = "W_ext"

    state.health = {}
    state.ppe = {}

    return state


def test_step5_does_not_modify_solverstate_structure(monkeypatch):
    state = make_state()

    # Capture initial keys
    initial_keys = set(state.__dict__.keys())

    monkeypatch.setattr("src.step3.orchestrate_step3_state", MagicMock())

    orchestrate_step5_state(state)

    final_keys = set(state.__dict__.keys())

    # Step 5 may add final_health and history, but nothing else
    allowed_new = {"history", "final_health"}
    assert final_keys - initial_keys <= allowed_new
