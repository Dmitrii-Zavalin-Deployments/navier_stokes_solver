# tests/step5/test_step5_preserves_extended_field_shapes.py

import types
from unittest.mock import MagicMock

import numpy as np

from src.step5.orchestrate_step5_state import orchestrate_step5_state


def make_state():
    nx, ny, nz = 4, 3, 2

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

    state.fields = {
        "P": np.zeros((nx, ny, nz)),
        "U": np.zeros((nx+1, ny, nz)),
        "V": np.zeros((nx, ny+1, nz)),
        "W": np.zeros((nx, ny, nz+1)),
    }

    state.P_ext = np.zeros((nx+2, ny+2, nz+2))
    state.U_ext = np.zeros((nx+3, ny+2, nz+2))
    state.V_ext = np.zeros((nx, ny+3, nz+2))
    state.W_ext = np.zeros((nx, ny, nz+3))

    state.health = {}
    state.ppe = {}

    return state


def test_step5_preserves_extended_field_shapes(monkeypatch):
    state = make_state()

    P_shape = state.P_ext.shape
    U_shape = state.U_ext.shape
    V_shape = state.V_ext.shape
    W_shape = state.W_ext.shape

    monkeypatch.setattr("src.step3.orchestrate_step3_state", MagicMock())

    orchestrate_step5_state(state)

    assert state.P_ext.shape == P_shape
    assert state.U_ext.shape == U_shape
    assert state.V_ext.shape == V_shape
    assert state.W_ext.shape == W_shape
