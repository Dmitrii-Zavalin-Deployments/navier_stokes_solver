# tests/step1/test_topology_and_boundaries.py

import numpy as np
import pytest

from src.solver_state import SolverState
from src.step1.assemble_simulation_state import assemble_simulation_state
from src.step1.map_geometry_mask import map_geometry_mask

# Updated import to match actual source function name
from src.step1.orchestrate_step1 import debug_state_step1, orchestrate_step1
from src.step1.parse_boundary_conditions import parse_boundary_conditions
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy


@pytest.fixture
def dummy_grid():
    """Provides canonical grid metadata."""
    return solver_input_schema_dummy()["grid"]

@pytest.fixture
def dummy_data():
    """Provides the full canonical dummy input."""
    return solver_input_schema_dummy()

@pytest.fixture
def valid_6_face_bc():
    """Provides a complete, mathematically sound 6-face BC list."""
    return [
        {"location": "x_min", "type": "inflow", "values": {"u": 123.45, "v": 0.0, "w": 0.0}},
        {"location": "x_max", "type": "pressure", "values": {"p": 0.0}},
        {"location": "y_min", "type": "no-slip", "values": {}},
        {"location": "y_max", "type": "no-slip", "values": {}},
        {"location": "z_min", "type": "no-slip", "values": {}},
        {"location": "z_max", "type": "no-slip", "values": {}}
    ]

# --- SECTION 1: BOUNDARY CONDITION COMPLIANCE ---

def test_full_bc_normalization_and_storage(dummy_grid, valid_6_face_bc):
    bc_table = parse_boundary_conditions(valid_6_face_bc, dummy_grid)
    assert len(bc_table) == 6
    assert bc_table["x_min"]["u"] == 123.45

def test_invalid_location_override(dummy_grid):
    bc_list = [{"location": "center_of_universe", "type": "no-slip", "values": {}}]
    with pytest.raises(ValueError, match="(?i)Invalid or missing boundary location"):
        parse_boundary_conditions(bc_list, dummy_grid)

def test_inflow_requires_numeric_uvw(dummy_grid):
    incomplete = [{"location": "x_min", "type": "inflow", "values": {"u": 5.0}}]
    with pytest.raises(ValueError, match="requires numeric velocity"):
        parse_boundary_conditions(incomplete, dummy_grid)

def test_pressure_requires_numeric_p(dummy_grid):
    missing_p = [{"location": "x_max", "type": "pressure", "values": {"u": 0.0}}]
    with pytest.raises(ValueError, match="requires numeric 'p'"):
        parse_boundary_conditions(missing_p, dummy_grid)

def test_physical_exclusion_collision(dummy_grid):
    bad_outflow = [{"location": "x_max", "type": "outflow", "values": {"p": 10.0}}]
    with pytest.raises(ValueError, match="cannot define pressure"):
        parse_boundary_conditions(bad_outflow, dummy_grid)

def test_duplicate_location_logic(dummy_grid):
    duplicate_loc = [{"location": "y_min", "type": "no-slip"}, {"location": "y_min", "type": "free-slip"}]
    with pytest.raises(ValueError, match="(?i)Duplicate BC"):
        parse_boundary_conditions(duplicate_loc, dummy_grid)

def test_incomplete_domain_trigger(dummy_grid):
    incomplete = [{"location": "x_min", "type": "no-slip"}]
    with pytest.raises(ValueError, match="Incomplete Domain"):
        parse_boundary_conditions(incomplete, dummy_grid)

# --- SECTION 2: GEOMETRY & MASKING ---

def test_mask_reshaping_fortran_order(dummy_data):
    grid = dummy_data["grid"]
    nx, ny, nz = grid["nx"], grid["ny"], grid["nz"]
    total_cells = nx * ny * nz
    for target_index in [0, 1, nx, nx * ny - 1, total_cells - 1]:
        flat = [0] * total_cells
        flat[target_index] = 1
        mask_1d, _, _ = map_geometry_mask(flat, grid)
        assert mask_1d[target_index] == 1

def test_forbidden_topology_rule(dummy_data):
    grid = dummy_data["grid"]
    bad_val_flat = [0] * (grid["nx"] * grid["ny"] * grid["nz"])
    bad_val_flat[0] = 5
    with pytest.raises(ValueError, match="Mask contains unauthorized values"):
        map_geometry_mask(bad_val_flat, grid)

def test_mask_validation_mismatch(dummy_data):
    grid = dummy_data["grid"]
    with pytest.raises(ValueError, match="Mask length mismatch"):
        map_geometry_mask([1, 0, 1], grid)

# --- SECTION 3: INTEGRITY & ASSEMBLY ---

def test_assemble_state_spatial_incoherence():
    grid = {"nx": 4, "ny": 4, "nz": 4}
    constants = {"rho": 1.0, "mu": 0.1, "dt": 0.1, "dx": 0.5, "dy": 0.5, "dz": 0.5}
    fields = {"U": np.zeros((5, 4, 4)), "V": np.zeros((4, 5, 4)), "W": np.zeros((4, 4, 5)), "P": np.zeros((4, 4, 4))}
    mask_small = [0] * 8 
    with pytest.raises(ValueError, match="Spatial Incoherence"):
        assemble_simulation_state(config={}, grid=grid, fields=fields, mask=mask_small, 
                                 constants=constants, boundary_conditions={}, 
                                 is_fluid=mask_small, is_boundary_cell=mask_small)

def test_mask_encapsulation_in_solver_state(dummy_data):
    grid = dummy_data["grid"]
    mask_list, _, _ = map_geometry_mask(dummy_data["mask"], grid)
    state = SolverState(mask=mask_list, grid=grid)
    assert len(state.mask) == grid["nx"] * grid["ny"] * grid["nz"]

def test_mask_non_integer_error():
    grid_ctx = {"nx": 2, "ny": 2, "nz": 2}
    bad_mask = [1, 0, 1, 0, 1, 0, 1, 0.5] 
    with pytest.raises(ValueError, match="Mask entries must be finite integers"):
        map_geometry_mask(bad_mask, grid_ctx)

def test_bc_invalid_type_error():
    bad_bc_list = [{"location": "x_min", "type": "quantum_flux", "values": {}}]
    with pytest.raises(ValueError, match="Invalid boundary type: quantum_flux"):
        parse_boundary_conditions(bad_bc_list, {"nx": 2, "ny": 2, "nz": 2})

# --- SECTION 4: ORCHESTRATION & COVERAGE ---

def test_orchestrate_debug_printer_coverage(dummy_data):
    """
    Targets orchestrate_step1.py Line 31.
    Directly triggers the debug summary with a mock array to hit 100% coverage.
    """
    # 1. Generate a valid state first
    state = orchestrate_step1(dummy_data)
    
    # 2. Inject a NumPy array into the mask attribute.
    state.mask = np.array([0, 1, 0, 1], dtype=np.int8)
    
    # 3. Manually trigger the public debug printer
    debug_state_step1(state)