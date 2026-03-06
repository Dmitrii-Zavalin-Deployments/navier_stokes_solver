# tests/scientific/test_scientific_step1_helpers.py

import numpy as np
import pytest

from src.solver_input import BoundaryConditionItem
from src.step1.helpers import (
    allocate_staggered_fields,
    generate_3d_masks,
    parse_bc_lookup,
)


def test_scientific_harlow_welch_allocation(base_input):
    """Rule 1.1: Harlow-Welch Staggering must follow the N+1 face requirement."""
    grid = base_input.grid
    grid.nx, grid.ny, grid.nz = 10, 20, 30
    fields = allocate_staggered_fields(grid)
    
    # Physics check: Pressure is cell-centered, Velocities are face-centered
    assert fields["P"].shape == (10, 20, 30), "Pressure shape mismatch"
    assert fields["U"].shape == (11, 20, 30), "U-velocity (East-West) must be nx+1"
    assert fields["V"].shape == (10, 21, 30), "V-velocity (North-South) must be ny+1"
    assert fields["W"].shape == (10, 20, 31), "W-velocity (Front-Back) must be nz+1"
    assert fields["U"].dtype == np.float64, "Fields must be double precision"

def test_scientific_mask_reconstruction_parity(base_input):
    """Rule 1.2: 3D reconstruction must respect Order F (Fortran/Column-major)."""
    grid = base_input.grid
    grid.nx, grid.ny, grid.nz = 2, 2, 2
    # 8 cells total. In Order F, indices change: X -> Y -> Z
    flat_data = [1, 2, 3, 4, 5, 6, 7, 8]
    mask_3d, _, _ = generate_3d_masks(flat_data, grid)
    
    # Verify the first XY plane (Z=0): [[1, 3], [2, 4]]
    expected_z0 = np.array([[1, 3], [2, 4]], dtype=np.int8)
    np.testing.assert_array_equal(mask_3d[:, :, 0], expected_z0)
    
    # Verify the second XY plane (Z=1): [[5, 7], [6, 8]]
    expected_z1 = np.array([[5, 7], [6, 8]], dtype=np.int8)
    np.testing.assert_array_equal(mask_3d[:, :, 1], expected_z1)

def test_scientific_mask_validation_error(base_input):
    """Ensure the helper catches data volume mismatches immediately."""
    grid = base_input.grid
    grid.nx, grid.ny, grid.nz = 2, 2, 2
    with pytest.raises(ValueError, match="Mask size mismatch"):
        generate_3d_masks([1, 1], grid)

def test_scientific_bc_lookup_mapping():
    """Rule 1.3: BC table must map locations and handle missing velocity components."""
    item = BoundaryConditionItem()
    item.location = "x_min"
    item.type = "inflow"
    item.values = {"u": 5.0, "v": 0.0, "w": 0.0, "p": 101325.0}
    bc_map = parse_bc_lookup([item])
    
    assert "x_min" in bc_map
    assert bc_map["x_min"]["u"] == 5.0
    assert bc_map["x_min"]["v"] == 0.0, "Value must match explicit input"
    assert bc_map["x_min"]["p"] == 101325.0
    assert bc_map["x_min"]["type"] == "inflow"

def test_scientific_staggered_memory_zeroed(base_input, sts_tolerance):
    """Zero-Debt Check: Memory must be pre-zeroed to machine precision."""
    grid = base_input.grid
    grid.nx, grid.ny, grid.nz = 4, 4, 4
    fields = allocate_staggered_fields(grid)
    for name, arr in fields.items():
        np.testing.assert_allclose(
            arr, 0.0, 
            atol=sts_tolerance["atol"], 
            rtol=sts_tolerance["rtol"],
            err_msg=f"Field {name} has residual garbage data"
        )

def test_scientific_bc_lookup_3d_completeness():
    """Rule 1.3 Ext: Verify W-component and full float conversion."""
    item = BoundaryConditionItem()
    item.location = "z_max"
    item.type = "no-slip"
    item.values = {"u": 0.0, "v": 0.0, "w": -1.5, "p": 0.0}
    bc_map = parse_bc_lookup([item])
    
    assert bc_map["z_max"]["w"] == -1.5
    assert isinstance(bc_map["z_max"]["w"], float)

def test_scientific_mask_multi_value_preservation(base_input):
    """Verify that the int8 mask preserves non-binary markers (e.g., obstacles)."""
    grid = base_input.grid
    grid.nx, grid.ny, grid.nz = 2, 1, 1
    # 0 = Void, 1 = Fluid, -1 = Boundary, 2 = Inflow/Outflow marker
    data = [1, 2] 
    mask_3d, _, _ = generate_3d_masks(data, grid)
    assert mask_3d[1, 0, 0] == 2, "Mask value '2' was corrupted during reconstruction"

def test_scientific_step1_debug_handshake(base_input, capsys):
    """Verify the shape handshake in the debug logs."""
    grid = base_input.grid
    grid.nx, grid.ny, grid.nz = 2, 3, 4
    allocate_staggered_fields(grid)
    captured = capsys.readouterr().out
    assert "U-Face (East-West):   (3, 3, 4)" in captured

def test_scientific_bc_lookup_type_safety():
    """Rule 5/7: Ensure the parser enforces numeric types and fails on garbage."""
    item = BoundaryConditionItem()
    item.location = "x_min"
    
    # Initialize the mandatory type field to satisfy the BaseContainer security guard
    item.type = "no-slip" 
    
    # This dictionary contains a string 'invalid', which will trigger 
    # the ValueError during float conversion in the parse_bc_lookup loop.
    item.values = {"u": "invalid", "v": 0.0, "w": 0.0, "p": 0.0}
    
    with pytest.raises(ValueError):
        parse_bc_lookup([item])

def test_scientific_mask_boolean_logic(base_input):
    """Rule 1.2: Explicitly verify the boolean extraction of fluid and boundaries."""
    grid = base_input.grid
    grid.nx, grid.ny, grid.nz = 3, 1, 1
    # 1=Fluid, -1=Boundary, 0=Void
    data = [1, -1, 0]
    _, is_fluid, is_boundary = generate_3d_masks(data, grid)
    
    assert is_fluid[0, 0, 0] == True
    assert is_fluid[1, 0, 0] == False
    assert is_boundary[1, 0, 0] == True
    assert is_boundary[2, 0, 0] == False

def test_scientific_mask_non_symmetric_order_f(base_input):
    """Rule 1.2: Verify Fortran order on a non-symmetric grid to catch index swaps."""
    grid = base_input.grid
    grid.nx, grid.ny, grid.nz = 2, 3, 1  # 6 cells total
    # Order F: X moves fastest, then Y
    # Linear: [0, 1, 2, 3, 4, 5] 
    # Grid: Y0:[0,1], Y1:[2,3], Y2:[4,5]
    data = [0, 1, 2, 3, 4, 5]
    mask_3d, _, _ = generate_3d_masks(data, grid)
    
    # Check Y-axis progression
    assert mask_3d[0, 1, 0] == 2
    assert mask_3d[1, 2, 0] == 5

def test_scientific_parse_bc_missing_key():
    """Rule 5: Ensure direct access raises KeyError when required values are missing."""
    item = BoundaryConditionItem()
    item.location = "x_min"
    item.type = "inflow"
    # 'p' is missing
    item.values = {"u": 0.0, "v": 0.0, "w": 0.0} 
    
    with pytest.raises(KeyError):
        parse_bc_lookup([item])

def test_scientific_mask_invalid_type_rejection(base_input):  # Added base_input here
    """Rule 7: Ensure the mask helper handles non-integer types strictly."""
    grid = base_input.grid
    grid.nx, grid.ny, grid.nz = 1, 1, 1
    with pytest.raises(ValueError):
        generate_3d_masks(["a"], grid)

def test_scientific_allocation_negative_grid(base_input):  # Added base_input here
    """Rule 5: Ensure logic fails immediately on non-physical grid inputs."""
    grid = base_input.grid
    grid.nx, grid.ny, grid.nz = -1, 10, 10
    with pytest.raises(ValueError):
        allocate_staggered_fields(grid)