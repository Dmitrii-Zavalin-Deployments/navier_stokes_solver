# tests/step5/test_step5_zero_output_interval_produces_no_snapshots.py

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
        "output_interval": None,   # <— zero output interval
    }

    state.health = {}
    state.ppe = {}
    state.fields = {}
    state.P_ext = None
    state.U_ext = None
    state.V_ext = None
    state.W_ext = None

    return state


def test_step5_zero_output_interval_produces_no_snapshots(monkeypatch):
    state = make_state()

    monkeypatch.setattr("src.step3.orchestrate_step3_state", MagicMock())
    monkeypatch.setattr("src.step5.write_output_snapshot", MagicMock())

    orchestrate_step5_state(state)

    # No snapshots should be written
    write_mock = src.step5.write_output_snapshot
    assert write_mock.call_count == 0
