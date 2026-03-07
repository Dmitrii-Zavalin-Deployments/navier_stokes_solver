# tests/scientific/test_scientific_step1_helpers.py

import numpy as np
import pytest

from src.solver_input import BoundaryConditionItem
from src.step1.helpers import (
    allocate_fields,
    generate_3d_masks,
    parse_bc_lookup,
)


def test_scientific_collocated_allocation(base_input):
    """Rule: Collocated fields must share exact Nx, Ny, Nz dimensions (Theory Section 3)."""
    grid = base_input.grid
    grid.nx, grid.ny, grid.nz = 10, 20, 30
    fields = allocate_fields(grid)
    
    # Physics check: All fields defined at cell centers (i, j, k)
    assert fields["P"].shape == (10, 20, 30)
    assert fields["U"].shape == (10, 20, 30)
    assert fields["V"].shape == (10, 20, 30)
    assert fields["W"].shape == (10, 20, 30)
    assert fields["P"].dtype == np.float64

def test_scientific_mask_reconstruction_parity(base_input):
    """Rule: 3D reconstruction must respect Order F (Theory Section 6)."""
    grid = base_input.grid
    grid.nx, grid.ny, grid.nz = 2, 2, 2
    # 8 cells total. In Order F: X changes fastest (index 0), then Y, then Z
    flat_data = [1, 2, 3, 4, 5, 6, 7, 8]
    mask_3d, _, _ = generate_3d_masks(flat_data, grid)
    
    # Verify the first XY plane (Z=0)
    # Fortran order: index (i,j,k) -> i + nx*j + nx*ny*k
    assert mask_3d[0, 0, 0] == 1
    assert mask_3d[1, 0, 0] == 2
    assert mask_3d[0, 1, 0] == 3
    assert mask_3d[1, 1, 0] == 4
    
    # Verify the second XY plane (Z=1)
    assert mask_3d[0, 0, 1] == 5
    assert mask_3d[1, 1, 1] == 8

def test_scientific_mask_validation_error(base_input):
    """Ensure helper catches data volume mismatches (Atomic Truth)."""
    grid = base_input.grid
    grid.nx, grid.ny, grid.nz = 2, 2, 2
    with pytest.raises(ValueError, match="Mask size mismatch"):
        generate_3d_masks([1, 1], grid)

def test_scientific_bc_lookup_mapping():
    """Rule: BC table must enforce strict key access (Rule 5)."""
    item = BoundaryConditionItem()
    item.location = "x_min"
    item.type = "inflow"
    item.values = {"u": 5.0, "v": 0.0, "w": 0.0, "p": 101325.0}
    bc_map = parse_bc_lookup([item])
    
    assert bc_map["x_min"]["u"] == 5.0
    assert bc_map["x_min"]["type"] == "inflow"

def test_scientific_memory_zeroed(base_input, sts_tolerance):
    """Zero-Debt Check: Memory must be pre-zeroed to machine precision."""
    grid = base_input.grid
    grid.nx, grid.ny, grid.nz = 4, 4, 4
    fields = allocate_fields(grid)
    for name, arr in fields.items():
        np.testing.assert_allclose(
            arr, 0.0, 
            atol=sts_tolerance["atol"], 
            rtol=sts_tolerance["rtol"],
            err_msg=f"Field {name} has residual garbage"
        )

def test_scientific_step1_debug_handshake(base_input, capsys):
    """Verify Collocated handshake in debug logs."""
    grid = base_input.grid
    grid.nx, grid.ny, grid.nz = 2, 3, 4
    allocate_fields(grid)
    captured = capsys.readouterr().out
    assert "All Fields (P, U, V, W) shape: (2, 3, 4)" in captured

def test_scientific_parse_bc_missing_key():
    """Rule 5: Ensure direct access raises KeyError when required values missing."""
    item = BoundaryConditionItem()
    item.location = "x_min"
    item.type = "inflow"
    item.values = {"u": 0.0, "v": 0.0, "w": 0.0} # Missing 'p'
    
    with pytest.raises(KeyError):
        parse_bc_lookup([item])

def test_scientific_mask_boolean_logic(base_input):
    """Verify boolean extraction of fluid/boundary masks (Section 6)."""
    grid = base_input.grid
    grid.nx, grid.ny, grid.nz = 3, 1, 1
    data = [1, -1, 0] # 1=Fluid, -1=Boundary, 0=Void
    _, is_fluid, is_boundary = generate_3d_masks(data, grid)
    
    assert is_fluid[0, 0, 0] == True
    assert is_fluid[1, 0, 0] == False
    assert is_boundary[1, 0, 0] == True