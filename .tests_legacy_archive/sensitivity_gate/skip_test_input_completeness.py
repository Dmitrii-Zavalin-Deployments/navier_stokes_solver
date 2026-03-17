import pytest

from src.step1.orchestrate_step1 import orchestrate_step1
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy


def test_six_face_mandate_violation():
    """
    Gate 1.F: Completeness.
    Code raises ValueError for missing faces during BC parsing.
    """
    invalid_input = solver_input_schema_dummy()
    invalid_input["boundary_conditions"] = [
        bc for bc in invalid_input["boundary_conditions"] 
        if bc["location"] != "z_max"
    ]
    
    # We catch ValueError because src/step1/parse_boundary_conditions.py 
    # raises ValueError: Incomplete Domain...
    with pytest.raises(ValueError, match="Incomplete Domain"):
        orchestrate_step1(invalid_input)

def test_domain_inversion_error():
    """
    Gate 1.B: Inversion.
    Code raises ValueError for spatial inversions (max < min).
    """
    invalid_input = solver_input_schema_dummy()
    invalid_input["grid"]["x_min"] = 10.0
    invalid_input["grid"]["x_max"] = 5.0 
    
    # We catch ValueError because src/step1/initialize_grid.py 
    # raises ValueError: Inverted domain...
    with pytest.raises(ValueError, match="Inverted domain"):
        orchestrate_step1(invalid_input)

def test_zero_volume_resolution_error():
    """
    Gate 1.B: Structural Firewall.
    The JSON Schema catches invalid integers and the orchestrator 
    wraps it in a 'Contract Violation' RuntimeError.
    """
    invalid_input = solver_input_schema_dummy()
    invalid_input["grid"]["nx"] = 0
    
    # We catch RuntimeError because orchestrate_step1.py wraps 
    # schema validation failures.
    with pytest.raises(RuntimeError, match="Contract Violation"):
        orchestrate_step1(invalid_input)
