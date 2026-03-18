# tests/property_integrity/test_schema_contracts.py

import json
from pathlib import Path

import jsonschema
import pytest

from tests.helpers.solver_input_schema_dummy import create_validated_input
from tests.helpers.solver_output_schema_dummy import make_output_schema_dummy


# Instead of importing a non-existent utility, we define how to find the schemas
def load_schema(schema_name: str) -> dict:
    """Helper to load schemas from the project's /schema directory."""
    # This finds the 'schema' folder at the root of your repo
    project_root = Path(__file__).parent.parent.parent
    schema_path = project_root / "schema" / schema_name
    
    if not schema_path.exists():
        pytest.fail(f"Schema not found at {schema_path}")
        
    with open(schema_path) as f:
        return json.load(f)

class TestSchemaContracts:
    """
    SYSTEM AUDITOR: Verifies that the SSoT helpers generate data 
    that strictly adheres to the JSON schemas.
    """

    def test_input_dummy_matches_schema(self):
        schema = load_schema("solver_input_schema.json")
        input_obj = create_validated_input(nx=4, ny=4, nz=4)
        payload = input_obj.to_dict()
        
        try:
            jsonschema.validate(instance=payload, schema=schema)
        except jsonschema.exceptions.ValidationError as e:
            pytest.fail(f"Input Contract Violation: {e.message}")

    def test_output_dummy_matches_schema(self):
        schema = load_schema("solver_output_schema.json")
        state = make_output_schema_dummy(nx=4, ny=4, nz=4)
        payload = state.to_json_safe()
        
        try:
            jsonschema.validate(instance=payload, schema=schema)
        except jsonschema.exceptions.ValidationError as e:
            pytest.fail(f"Output Contract Violation: {e.message}")