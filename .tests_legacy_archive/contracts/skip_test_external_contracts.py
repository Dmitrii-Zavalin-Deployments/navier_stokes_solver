# tests/contracts/test_external_contracts.py

"""
test_external_contracts.py

THE BOUNDARY GUARD: Validates that the solver's public API remains stable.
Ensures that the 'Genesis Input' and 'Gold Standard Output' dummies fully 
satisfy their corresponding JSON schemas (Draft 2020-12).

Robustness Strategy:
1. Auto-detects Schema Draft versions.
2. Converts SolverState objects to JSON-safe dicts before validation.
3. Reports ALL contract violations simultaneously via iter_errors.
"""

import json
from pathlib import Path

import jsonschema
import pytest

# Helpers and Dummies
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy
from tests.helpers.solver_output_schema_dummy import make_output_schema_dummy

# Resolve the absolute path to the /schema directory at the project root
# Structure assumed: /tests/contracts/test_external_contracts.py -> ../../schema/
SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schema"

def validate_contract(data: dict, schema_filename: str):
    """
    Core validation engine. 
    Uses the validator version specified in the $schema tag of the file.
    """
    schema_path = SCHEMA_DIR / schema_filename
    
    if not schema_path.exists():
        pytest.fail(f"CRITICAL: Schema file '{schema_filename}' missing at {schema_path}")

    with open(schema_path, encoding="utf-8") as f:
        schema = json.load(f)

    # 100% Robustness: Auto-select the correct validator (Draft 2020-12)
    validator_cls = jsonschema.validators.validator_for(schema)
    validator = validator_cls(schema)
    
    # Collect all errors instead of stopping at the first one
    errors = list(validator.iter_errors(data))
    
    if errors:
        error_report = "\n".join([
            f"  - LOCATION: {list(e.path)} | ERROR: {e.message}" 
            for e in sorted(errors, key=lambda e: e.path)
        ])
        pytest.fail(f"Contract Violation in '{schema_filename}':\n{error_report}")


def test_solver_input_dummy_matches_schema():
    """
    Validates the 'Genesis' input dummy.
    Checks that user-provided configuration matches the mathematical requirements.
    """
    # The input dummy is a standard dict, so we validate it directly
    input_data = solver_input_schema_dummy()
    validate_contract(input_data, "solver_input_schema.json")


def test_solver_output_dummy_matches_schema():
    """
    Validates the 'Gold Standard' state.
    Robustness Check: Transforms SolverState -> JSON-safe dict to catch
    NumPy leakage or illegal additionalProperties before validation.
    """
    state_obj = make_output_schema_dummy()
    json_safe_data = state_obj.to_json_safe()
    
    # DEBUG: Check if the required keys are present
    required_keys = ["time", "iteration", "ready_for_time_loop", "config", "grid", "fields", "masks", "fluid", "operators", "advection", "ppe", "health", "history", "diagnostics", "manifest"]
    for key in required_keys:
        if key not in json_safe_data:
            print(f"DEBUG: Missing key: {key}")
    
    validate_contract(json_safe_data, "solver_output_schema.json")