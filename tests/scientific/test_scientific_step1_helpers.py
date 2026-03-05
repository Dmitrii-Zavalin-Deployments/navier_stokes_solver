# tests/scientific/test_scientific_step1_helpers.py

import pytest
import numpy as np
from src.step1.helpers import allocate_staggered_fields, generate_3d_masks, parse_bc_lookup
from src.solver_input import GridInput, BoundaryConditionItem

def create_scientific_grid(nx, ny, nz):
    """
    Helper to populate the frozen GridInput using public setters.
    This bypasses the underscore-prefixed dataclass constructor.
    """
    grid = GridInput()
    grid.nx, grid.ny, grid.nz = nx, ny, nz
    return grid

def test_scientific_harlow_welch_allocation():
    """Rule 1.1: Harlow-Welch Staggering must follow the N+1 face requirement."""
    grid = create_scientific_grid(10, 20, 30)
    fields = allocate_staggered_fields(grid)
    
    # Physics check: Pressure is cell-centered, Velocities are face-centered
    assert fields["P"].shape == (10, 20, 30), "Pressure shape mismatch"
    assert fields["U"].shape == (11, 20, 30), "U-velocity (East-West) must be nx+1"
    assert fields["V"].shape == (10, 21, 30), "V-velocity (North-South) must be ny+1"
    assert fields["W"].shape == (10, 20, 31), "W-velocity (Front-Back) must be nz+1"
    assert fields["U"].dtype == np.float64, "Fields must be double precision"

def test_scientific_mask_reconstruction_parity():
    """Rule 1.2: 3D reconstruction must respect Order F (Fortran/Column-major)."""
    grid = create_scientific_grid(2, 2, 2)
    # 8 cells total. In Order F, indices change: X -> Y -> Z
    # Data: [X0Y0Z0, X1Y0Z0, X0Y1Z0, X1Y1Z0, ...]
    flat_data = [1, 2, 3, 4, 5, 6, 7, 8]
    mask_3d, is_fluid, is_boundary = generate_3d_masks(flat_data, grid)
    
    # Verify the first XY plane (Z=0)
    # Should look like: [[1, 3], [2, 4]]
    expected_z0 = np.array([[1, 3], [2, 4]], dtype=np.int8)
    np.testing.assert_array_equal(mask_3d[:, :, 0], expected_z0)
    
    # Verify the second XY plane (Z=1)
    # Should look like: [[5, 7], [6, 8]]
    expected_z1 = np.array([[5, 7], [6, 8]], dtype=np.int8)
    np.testing.assert_array_equal(mask_3d[:, :, 1], expected_z1)

def test_scientific_mask_validation_error():
    """Ensure the helper catches data volume mismatches immediately."""
    grid = create_scientific_grid(2, 2, 2)
    with pytest.raises(ValueError, match="Mask size mismatch"):
        generate_3d_masks([1, 1], grid)

def test_scientific_bc_lookup_mapping():
    """Rule 1.3: BC table must map locations and handle missing velocity components."""
    item = BoundaryConditionItem()
    item.location = "x_min"
    item.type = "inflow"
    item.values = {"u": 5.0, "p": 101325.0}
    bc_map = parse_bc_lookup([item])
    
    assert "x_min" in bc_map
    assert bc_map["x_min"]["u"] == 5.0
    assert bc_map["x_min"]["v"] == 0.0, "Missing values must default to 0.0"
    assert bc_map["x_min"]["p"] == 101325.0
    assert bc_map["x_min"]["type"] == "inflow"

def test_scientific_staggered_memory_zeroed(sts_tolerance):
    """Zero-Debt Check: Memory must be pre-zeroed to machine precision."""
    grid = create_scientific_grid(4, 4, 4)
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
    item.values = {"w": -1.5} # Testing negative velocity and W-axis
    bc_map = parse_bc_lookup([item])
    
    assert bc_map["z_max"]["w"] == -1.5
    assert isinstance(bc_map["z_max"]["w"], float)

def test_scientific_mask_multi_value_preservation():
    """Verify that the int8 mask preserves non-binary markers (e.g., obstacles)."""
    grid = create_scientific_grid(2, 1, 1)
    # 0 = Void, 1 = Fluid, -1 = Boundary, 2 = Inflow/Outflow marker
    data = [1, 2] 
    mask_3d, _, _ = generate_3d_masks(data, grid)
    assert mask_3d[1, 0, 0] == 2, "Mask value '2' was corrupted during reconstruction"

def test_scientific_step1_debug_handshake(capsys):
    """Verify the shape handshake in the debug logs."""
    grid = create_scientific_grid(2, 3, 4)
    allocate_staggered_fields(grid)
    captured = capsys.readouterr().out
    assert "U-Face (East-West):   (3, 3, 4)" in captured # nx+1 = 3