# tests/step1/test_grid_and_allocation.py

import numpy as np
import pytest

from src.solver_state import SolverState
from src.step1.allocate_fields import allocate_fields
from src.step1.assemble_simulation_state import assemble_simulation_state
from src.step1.initialize_grid import initialize_grid
from src.step1.verify_cell_centered_shapes import verify_cell_centered_shapes
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy


@pytest.fixture
def dummy_input():
    """Provides the canonical JSON-safe dummy input (Section 5 Compliance)."""
    return solver_input_schema_dummy()

# --- INITIALIZE GRID TESTS ---

def test_initialize_grid_logic_and_math(dummy_input):
    """Verifies functional contract, converts raw params, and checks spacing precision."""
    raw_grid = dummy_input["grid"].copy()
    grid_result = initialize_grid(raw_grid)
    
    # 1. Assertions on Dimensions
    assert grid_result["nx"] == raw_grid["nx"]
    assert isinstance(grid_result["nx"], int)

    # 2. Assertions on Spacing (Precision)
    expected_dx = (raw_grid["x_max"] - raw_grid["x_min"]) / raw_grid["nx"]
    assert grid_result["dx"] == pytest.approx(expected_dx)
    assert isinstance(grid_result["dx"], (float, np.float64))

def test_initialize_grid_guardrails(dummy_input):
    """Triggers guardrails for missing keys, inverted domains, and non-finite values."""
    
    # Case: Missing keys
    bad_grid = {"nx": 4} 
    with pytest.raises(ValueError, match="Grid initialization failed"):
        initialize_grid(bad_grid)

    # Case: Inverted domain (max <= min)
    grid_inv = dummy_input["grid"].copy()
    grid_inv["x_min"], grid_inv["x_max"] = 10.0, 10.0
    with pytest.raises(ValueError, match="Inverted domain"):
        initialize_grid(grid_inv)

    # Case: Non-positive dimensions
    grid_pos = dummy_input["grid"].copy()
    grid_pos["nx"] = 0
    with pytest.raises(ValueError, match="Grid dimensions must be positive"):
        initialize_grid(grid_pos)

    # Case: Non-finite values
    grid_fin = dummy_input["grid"].copy()
    grid_fin["x_min"] = float("inf")
    with pytest.raises(ValueError, match="(?i)finite"):
        initialize_grid(grid_fin)

# --- ALLOCATE FIELDS TESTS ---

def test_field_shapes_and_staggering(dummy_input):
    """Verifies field allocation shapes for Arakawa C-grid staggering."""
    grid = {"nx": 4, "ny": 4, "nz": 4}
    fields = allocate_fields(grid)
    
    # P is cell-centered (nx, ny, nz)
    # U, V, W are face-centered (staggered +1 on their respective axis)
    assert fields["P"].shape == (4, 4, 4)
    assert fields["U"].shape == (5, 4, 4)
    assert fields["V"].shape == (4, 5, 4)
    assert fields["W"].shape == (4, 4, 5)
    
    # Verify SolverState attribute pointers
    state = SolverState(fields=fields, grid=grid)
    assert state.velocity_u.shape == (5, 4, 4)
    assert np.all(state.pressure == 0.0)



def test_allocate_fields_debt_trigger():
    """Target: allocate_fields.py line 24 (Invalid dims)."""
    with pytest.raises(ValueError, match="Invalid grid dimensions"):
        allocate_fields({"nx": 2, "ny": -1, "nz": 2})

# --- ASSEMBLE STATE DEBT TESTS ---

def test_assemble_state_genesis_and_spatial_errors():
    """Triggers Line 53 (Missing Fields) and Line 58 (Mismatched Mask)."""
    grid = {"nx": 4, "ny": 4, "nz": 4}
    constants = {"rho": 1.0, "mu": 0.1}
    mask_valid = np.zeros((4, 4, 4))
    mask_invalid = np.zeros((2, 2, 2))
    
    fields = {
        "U": np.zeros((5, 4, 4)), "V": np.zeros((4, 5, 4)), 
        "W": np.zeros((4, 4, 5)), "P": np.zeros((4, 4, 4))
    }

    # 1. Missing required field 'W'
    incomplete = {k: v for k, v in fields.items() if k != "W"}
    with pytest.raises(KeyError, match="Genesis Error: Required field 'W' missing"):
        assemble_simulation_state({}, grid, incomplete, mask_valid, constants, {}, mask_valid, mask_valid)

    # 2. Spatial Incoherence (Mask shape mismatch)
    with pytest.raises(ValueError, match="Spatial Incoherence"):
        assemble_simulation_state({}, grid, fields, mask_invalid, constants, {}, mask_invalid, mask_invalid)

# --- SECTION: Initialize Grid Full ---

def test_invalid_x_extent(dummy_input):
    """Verifies that a zero-thickness X-grid triggers a ValueError."""
    grid = dummy_input["grid"].copy()
    grid.update({"x_min": 1.0, "x_max": 1.0})
    with pytest.raises(ValueError, match="x_max"):
        initialize_grid(grid)

def test_invalid_y_extent(dummy_input):
    """Verifies that a zero-thickness Y-grid triggers a ValueError."""
    grid = dummy_input["grid"].copy()
    grid.update({"y_min": 1.0, "y_max": 1.0})
    with pytest.raises(ValueError, match="y_max"):
        initialize_grid(grid)

def test_invalid_z_extent(dummy_input):
    """Verifies that a zero-thickness Z-grid triggers a ValueError."""
    grid = dummy_input["grid"].copy()
    grid.update({"z_min": 2.0, "z_max": 2.0})
    with pytest.raises(ValueError, match="z_max"):
        initialize_grid(grid)

def test_grid_initialization_in_state(dummy_input):
    """
    Integration test: Verifies that initialize_grid output is correctly 
    stored in the SolverState.grid attribute.
    """
    grid_params = dummy_input["grid"]
    grid_dict = initialize_grid(grid_params)
    state = SolverState(grid=grid_dict)

    # 1. Check Object-style access
    assert hasattr(state, "grid")
    assert isinstance(state.grid, dict)

    # 2. Check calculated values (e.g., dx = (1.0 - 0.0) / 2 = 0.5)
    assert state.grid["dx"] == 0.5
    assert state.grid["nx"] == grid_params["nx"]

# --- SECTION: Verify Shapes (Cell-Centered Conversion) ---

@pytest.fixture
def valid_state_dict():
    """Provides a valid Step 1 state dictionary for shape verification."""
    nx, ny, nz = 2, 2, 2
    data_size = nx * ny * nz
    return {
        "grid": {"nx": nx, "ny": ny, "nz": nz},
        "fields": {
            "P": [0.0] * data_size,
            "U": [0.0] * data_size,
            "V": [0.0] * data_size,
            "W": [0.0] * data_size,
            "Mask": [1] * data_size
        }
    }

def test_successful_conversion_and_verification(valid_state_dict):
    """Verifies that lists are converted to arrays and shapes are validated."""
    # Reshape lists to match expected 3D structure for a clean pass
    for key in ["P", "U", "V", "W"]:
        valid_state_dict["fields"][key] = np.zeros((2, 2, 2)).tolist()
    valid_state_dict["fields"]["Mask"] = np.ones((2, 2, 2), dtype=np.int8).tolist()

    verify_cell_centered_shapes(valid_state_dict)

    fields = valid_state_dict["fields"]
    assert isinstance(fields["P"], np.ndarray)
    assert fields["P"].dtype == np.float64
    assert fields["Mask"].dtype == np.int8
    assert fields["P"].shape == (2, 2, 2)

def test_missing_grid_returns_early():
    """Ensures the guard skips processing if 'grid' key is missing."""
    state = {"fields": {}}
    assert verify_cell_centered_shapes(state) is None

def test_malformed_grid_returns_early():
    """Ensures guard skips if grid values are not castable to int."""
    state = {
        "grid": {"nx": "invalid", "ny": 2, "nz": 2},
        "fields": {}
    }
    assert verify_cell_centered_shapes(state) is None

def test_missing_essential_field_raises_keyerror(valid_state_dict):
    """Checks that missing 'P', 'U', 'V', 'W', or 'Mask' raises KeyError."""
    del valid_state_dict["fields"]["P"]
    with pytest.raises(KeyError, match="Missing essential field"):
        verify_cell_centered_shapes(valid_state_dict)

def test_invalid_field_type_raises_typeerror(valid_state_dict):
    """Ensures fields that are neither lists nor arrays raise TypeError."""
    valid_state_dict["fields"]["P"] = "not a list"
    with pytest.raises(TypeError, match="must be a list or numpy array"):
        verify_cell_centered_shapes(valid_state_dict)

def test_dimension_mismatch_raises_value_error(valid_state_dict):
    """Checks that arrays with wrong dimensions trigger the logic firewall."""
    # Grid expects (2, 2, 2), we give (3, 2, 2)
    valid_state_dict["fields"]["P"] = np.zeros((3, 2, 2))
    
    for key in ["U", "V", "W", "Mask"]:
        valid_state_dict["fields"][key] = np.zeros((2, 2, 2))

    with pytest.raises(ValueError, match="Dimension Mismatch"):
        verify_cell_centered_shapes(valid_state_dict)

def test_already_numpy_arrays_pass(valid_state_dict):
    """Ensures the function handles states that already contain numpy arrays."""
    nx, ny, nz = 2, 2, 2
    for key in ["P", "U", "V", "W"]:
        valid_state_dict["fields"][key] = np.zeros((nx, ny, nz))
    valid_state_dict["fields"]["Mask"] = np.zeros((nx, ny, nz), dtype=np.int8)

    verify_cell_centered_shapes(valid_state_dict)
    assert isinstance(valid_state_dict["fields"]["P"], np.ndarray)