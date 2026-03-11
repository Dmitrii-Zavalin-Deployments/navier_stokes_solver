# tests/step4/test_step4_output_dummy_schema.py

from tests.helpers.solver_step4_output_schema import step4_output_schema

from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy


def test_step4_dummy_matches_schema():
    state = make_step4_output_dummy()

    for key in step4_output_schema["required"]:
        assert hasattr(state, key), f"Missing field: {key}"
