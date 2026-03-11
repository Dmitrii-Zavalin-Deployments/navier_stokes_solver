# tests/step5/test_step5_output_dummy_schema.py

import jsonschema
from tests.helpers.solver_step5_output_schema import step5_output_schema

from tests.helpers.solver_step5_output_dummy import make_step5_output_dummy


def test_step5_output_dummy_matches_schema():
    dummy = make_step5_output_dummy()
    jsonschema.validate(instance=dummy, schema=step5_output_schema)
