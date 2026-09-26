"""
Unit Test Module for Strict Schema & Configuration Ingestion Module (src/ingestion.py).

This module provides comprehensive narrative and executable verification for load_and_validate_inputs,
testing explicit argument enforcement, file existence checks, formal JSON schema validations via jsonschema,
and semantic cross-field consistency checks (grid bounds and mask length matching).
"""

import json

import pytest

from src.ingestion import load_and_validate_inputs

# ============================================================================
# NARRATIVE SECTION 1: Argument Presence & File Existence Safeguards
# ============================================================================
# Under strict non-default policies, missing input or configuration paths (None)
# and non-existent files on disk must immediately raise ValueError or FileNotFoundError.
# ============================================================================


def test_ingestion_none_arguments(tmp_path):
    """
    # We verify that passing None for input_path or config_path raises a ValueError.
    """
    dummy_file = tmp_path / "dummy.json"
    dummy_file.write_text("{}")

    # When input_path is None:
    with pytest.raises(ValueError, match="input_path must be explicitly provided"):
        load_and_validate_inputs(None, dummy_file)

    # When config_path is None:
    with pytest.raises(ValueError, match="config_path must be explicitly provided"):
        load_and_validate_inputs(dummy_file, None)


def test_ingestion_file_not_found(tmp_path):
    """
    # We verify that pointing to non-existent input or configuration files raises a FileNotFoundError.
    """
    valid_file = tmp_path / "file.json"
    valid_file.write_text("{}")

    missing_file = tmp_path / "non_existent.json"

    # When input file does not exist:
    with pytest.raises(FileNotFoundError, match="Simulation input file not found"):
        load_and_validate_inputs(missing_file, valid_file)

    # When config file does not exist:
    with pytest.raises(FileNotFoundError, match="Solver configuration file not found"):
        load_and_validate_inputs(valid_file, missing_file)


# ============================================================================
# NARRATIVE SECTION 2: Schema File Presence & Formal Validation Errors
# ============================================================================
# The ingestion engine requires formal schema definition files to be present on disk
# and enforces strict conformance via the jsonschema validator.
# ============================================================================


def test_ingestion_schema_file_not_found(tmp_path, monkeypatch):
    """
    # We verify that missing formal schema files trigger a FileNotFoundError.
    """
    input_file = tmp_path / "input.json"
    config_file = tmp_path / "config.json"
    input_file.write_text("{}")
    config_file.write_text("{}")

    empty_schema_dir = tmp_path / "empty_schema"
    empty_schema_dir.mkdir()

    # We redirect SCHEMA_DIR to an empty directory to test missing input schema:
    import src.ingestion
    monkeypatch.setattr(src.ingestion, "SCHEMA_DIR", empty_schema_dir)

    with pytest.raises(FileNotFoundError, match="Input schema not found"):
        load_and_validate_inputs(input_file, config_file)

    # We create the input schema but omit the config schema to test config schema check:
    (empty_schema_dir / "solver_input_schema.json").write_text('{"type": "object"}')
    with pytest.raises(FileNotFoundError, match="Config schema not found"):
        load_and_validate_inputs(input_file, config_file)


def test_ingestion_input_schema_validation_error(tmp_path, valid_schema_files):
    """
    # We verify that an input instance violating the formal input schema raises a ValueError.
    """
    input_schema_path, config_schema_path, schema_dir = valid_schema_files
    
    input_path = tmp_path / "invalid_input.json"
    config_path = tmp_path / "config.json"

    # We write an input missing required fields (e.g., missing grid):
    input_path.write_text(json.dumps({"invalid_root": True}))
    config_path.write_text(json.dumps({"max_poisson_iterations": 1000, "poisson_tolerance": 1e-6}))

    import src.ingestion
    monkeypatch_schema = pytest.MonkeyPatch()
    monkeypatch_schema.setattr(src.ingestion, "SCHEMA_DIR", schema_dir)

    try:
        with pytest.raises(ValueError, match="Input schema validation failed"):
            load_and_validate_inputs(input_path, config_path)
    finally:
        monkeypatch_schema.undo()


def test_ingestion_config_schema_validation_error(tmp_path, valid_schema_files, sample_valid_input_dict):
    """
    # We verify that a configuration instance violating the formal config schema raises a ValueError.
    """
    input_schema_path, config_schema_path, schema_dir = valid_schema_files

    input_path = tmp_path / "input.json"
    config_path = tmp_path / "invalid_config.json"

    input_path.write_text(json.dumps(sample_valid_input_dict))
    # We write a config missing required fields or having invalid types:
    config_path.write_text(json.dumps({"max_poisson_iterations": "not_an_integer"}))

    import src.ingestion
    monkeypatch_schema = pytest.MonkeyPatch()
    monkeypatch_schema.setattr(src.ingestion, "SCHEMA_DIR", schema_dir)

    try:
        with pytest.raises(ValueError, match="Config schema validation failed"):
            load_and_validate_inputs(input_path, config_path)
    finally:
        monkeypatch_schema.undo()


# ============================================================================
# NARRATIVE SECTION 3: Semantic Cross-Field & Boundary Consistency Checks
# ============================================================================
# Beyond syntactic schema validation, ingestion enforces physical consistency:
#     1. Grid maximum boundaries must strictly exceed minimum boundaries:
#          x_max > x_min, y_max > y_min, z_max > z_min
#     2. Mask array length must equal total volumetric cell count:
#          len(mask) == nx * ny * nz
# ============================================================================


def test_ingestion_invalid_grid_boundaries(tmp_path, valid_schema_files, sample_valid_input_dict):
    """
    # We verify that invalid grid boundaries (where max <= min) raise a ValueError.
    """
    input_schema_path, config_schema_path, schema_dir = valid_schema_files

    input_path = tmp_path / "input.json"
    config_path = tmp_path / "config.json"

    bad_input = sample_valid_input_dict.copy()
    # We set x_max <= x_min:
    bad_input["grid"]["x_max"] = bad_input["grid"]["x_min"]

    input_path.write_text(json.dumps(bad_input))
    config_path.write_text(json.dumps({"max_poisson_iterations": 1000, "poisson_tolerance": 1e-6}))

    import src.ingestion
    monkeypatch_schema = pytest.MonkeyPatch()
    monkeypatch_schema.setattr(src.ingestion, "SCHEMA_DIR", schema_dir)

    try:
        with pytest.raises(ValueError, match="Grid physical maximum boundaries must be strictly greater"):
            load_and_validate_inputs(input_path, config_path)
    finally:
        monkeypatch_schema.undo()


def test_ingestion_mask_length_mismatch(tmp_path, valid_schema_files, sample_valid_input_dict):
    """
    # We verify that a mask length mismatch against nx * ny * nz raises a ValueError.
    """
    input_schema_path, config_schema_path, schema_dir = valid_schema_files

    input_path = tmp_path / "input.json"
    config_path = tmp_path / "config.json"

    bad_input = sample_valid_input_dict.copy()
    # Grid is 2x2x2 = 8 cells, but we provide a mask of length 2:
    bad_input["grid"]["nx"] = 2
    bad_input["grid"]["ny"] = 2
    bad_input["grid"]["nz"] = 2
    bad_input["mask"] = [0, 0]

    input_path.write_text(json.dumps(bad_input))
    config_path.write_text(json.dumps({"max_poisson_iterations": 1000, "poisson_tolerance": 1e-6}))

    import src.ingestion
    monkeypatch_schema = pytest.MonkeyPatch()
    monkeypatch_schema.setattr(src.ingestion, "SCHEMA_DIR", schema_dir)

    try:
        with pytest.raises(ValueError, match="Mask length mismatch"):
            load_and_validate_inputs(input_path, config_path)
    finally:
        monkeypatch_schema.undo()


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_valid_input_dict():
    """
    # We provide a sample valid input dictionary matching the schema requirements.
    """
    return {
        "grid": {
            "nx": 2, "ny": 2, "nz": 2,
            "x_min": 0.0, "x_max": 1.0,
            "y_min": 0.0, "y_max": 1.0,
            "z_min": 0.0, "z_max": 1.0
        },
        "fluid_properties": {"density": 1.0, "viscosity": 0.01},
        "simulation_parameters": {"time_step": 0.01, "total_time": 0.1, "output_interval": 1},
        "boundary_conditions": [{"location": "wall", "type": "no-slip", "values": {"u": 0.0}}],
        "external_forces": {"force_vector": [0.0, 0.0, 0.0]},
        "physical_constraints": {
            "min_velocity": -10.0, "max_velocity": 10.0,
            "min_pressure": -100.0, "max_pressure": 100.0
        },
        "mask": [0, 0, 0, 0, 0, 0, 0, 0]
    }


@pytest.fixture
def valid_schema_files(tmp_path):
    """
    # We set up temporary valid JSON schema files for strict jsonschema validation tests.
    """
    schema_dir = tmp_path / "schema"
    schema_dir.mkdir()

    input_schema = {
        "type": "object",
        "required": ["grid", "fluid_properties", "simulation_parameters", "boundary_conditions", "external_forces", "physical_constraints", "mask"],
        "properties": {
            "grid": {
                "type": "object",
                "required": ["nx", "ny", "nz", "x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]
            },
            "fluid_properties": {"type": "object"},
            "simulation_parameters": {"type": "object"},
            "boundary_conditions": {"type": "array"},
            "external_forces": {"type": "object"},
            "physical_constraints": {"type": "object"},
            "mask": {"type": "array"}
        }
    }

    config_schema = {
        "type": "object",
        "required": ["max_poisson_iterations", "poisson_tolerance"],
        "properties": {
            "max_poisson_iterations": {"type": "integer"},
            "poisson_tolerance": {"type": "number"}
        }
    }

    in_schema_path = schema_dir / "solver_input_schema.json"
    cfg_schema_path = schema_dir / "solver_config_schema.json"

    in_schema_path.write_text(json.dumps(input_schema))
    cfg_schema_path.write_text(json.dumps(config_schema))

    return in_schema_path, cfg_schema_path, schema_dir
