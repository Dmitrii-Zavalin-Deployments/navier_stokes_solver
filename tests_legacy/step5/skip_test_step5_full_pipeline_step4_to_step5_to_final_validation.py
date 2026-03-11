# tests/step5/test_step5_full_pipeline_step4_to_step5_to_final_validation.py

import types

import jsonschema
from tests.helpers.final_output_schema import final_output_schema

from src.step4.orchestrate_step4_state import orchestrate_step4_state
from src.step5.orchestrate_step5_state import orchestrate_step5_state


def make_step3_stub(monkeypatch):
    """
    Step 3 stub that simulates a stable projection update.
    """
    def fake_step3(state, t, step):
        state.health = {
            "post_correction_divergence_norm": 0.001,
            "max_velocity_magnitude": 0.5,
            "cfl_advection_estimate": 0.2,
        }
        state.ppe = {"iterations": 5}
    monkeypatch.setattr("src.step3.orchestrate_step3_state", fake_step3)


def make_state():
    state = types.SimpleNamespace()

    # Minimal Step 4-ready state
    state.fields = {
        "P": [[[]]],
        "U": [[[]]],
        "V": [[[]]],
        "W": [[[]]],
    }

    state.config = {
        "grid": {"nx": 1, "ny": 1, "nz": 1},
        "total_time": 0.2,
        "max_steps": 10,
        "output_interval": None,
        "boundary_conditions": [],
    }

    state.constants = {"dt": 0.1}
    state.is_fluid = [[[True]]]
    state.health = {}
    state.ppe = {}

    return state


def test_step5_full_pipeline_step4_to_step5_to_final_validation(monkeypatch):
    state = make_state()

    # Step 3 stub
    make_step3_stub(monkeypatch)

    # Run Step 4
    orchestrate_step4_state(state)

    # Run Step 5
    orchestrate_step5_state(state)

    # Validate final state
    jsonschema.validate(instance=state.__dict__, schema=final_output_schema)
