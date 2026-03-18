# test_schema_contracts.py

import jsonschema
import pytest
from src.step1.schema_utils import load_schema
from tests.helpers.solver_input_schema_dummy import create_validated_input
from tests.helpers.solver_output_schema_dummy import make_output_schema_dummy

class TestSchemaContracts:
    """
    SYSTEM AUDITOR: Verifies that the SSoT (Single Source of Truth) 
    helpers generate data that strictly adheres to the JSON schemas.
    """

    def test_input_dummy_matches_schema(self):
        # 1. Load the Input Schema
        schema = load_schema("solver_input_schema.json")
        
        # 2. Generate the Input Dummy (Rule 4 Compliance)
        # Using a standard 4x4x4 grid
        input_obj = create_validated_input(nx=4, ny=4, nz=4)
        payload = input_obj.to_dict()
        
        # 3. Validate Contract
        try:
            jsonschema.validate(instance=payload, schema=schema)
        except jsonschema.exceptions.ValidationError as e:
            pytest.fail(f"Input Contract Violation: {e.message}")

    def test_output_dummy_matches_schema(self):
        # 1. Load the Output Schema
        schema = load_schema("solver_output_schema.json")
        
        # 2. Generate the Output Dummy (SolverState)
        state = make_output_schema_dummy(nx=4, ny=4, nz=4)
        
        # 3. Transform to JSON-safe dict (converts NumPy to lists)
        payload = state.to_json_safe()
        
        # 4. Validate Contract
        try:
            jsonschema.validate(instance=payload, schema=schema)
        except jsonschema.exceptions.ValidationError as e:
            pytest.fail(f"Output Contract Violation: {e.message}")