# tests/test_pipeline_integrity.py

import jsonschema
import pytest

from src.step1.schema_utils import load_schema
from tests.helpers.solver_output_schema_dummy import make_output_schema_dummy


def test_final_state_contract():
    """
    Integration Test: Ensures the SolverState terminal output 
    strictly follows the JSON schema/solver_output_schema.json contract.
    """
    # 1. Load the SSoT schema
    schema = load_schema("solver_output_schema.json")
    
    # 2. Generate the Ground Truth dummy
    state = make_output_schema_dummy()
    
    # 3. Transform state to JSON-safe dict
    # Note: Ensure to_json_safe() converts the NumPy fields.data to list
    payload = state.to_json_safe()
    
    # 4. Validate against schema
    try:
        jsonschema.validate(instance=payload, schema=schema)
    except jsonschema.exceptions.ValidationError as e:
        pytest.fail(f"Contract Violation: {e.message} at {' -> '.join([str(p) for p in e.path])}")

def test_fields_buffer_format():
    """
    Specific check to ensure the Foundation buffer is 
    correctly formatted for JSON serialization.
    """
    state = make_output_schema_dummy()
    payload = state.to_json_safe()
    
    # Verify the Foundation buffer exists as a list (JSON-compatible)
    assert isinstance(payload["fields"]["data"], list), "Foundation buffer must be a list"