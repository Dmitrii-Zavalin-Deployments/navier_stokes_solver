# tests/ quality_gates/physics_gate/test_advection.py

import copy

from src.common.field_schema import FI
from src.step3.ops.advection import (
    compute_local_advection,
    compute_local_advection_vector,
)
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy

# --- Local Stencil Utility (Bypass Wedged Dummies) ---

def setup_stencil_data(block):
    """
    Clones cells from the dummy to assign unique spatial data.
    Ensures neighbors aren't the same object as the center.
    """
    # 1. Clone cells to create independent objects (Rule 9 compliant)
    block.center = copy.copy(block.center)
    block.i_plus = copy.copy(block.i_plus);   block.i_minus = copy.copy(block.i_minus)
    block.j_plus = copy.copy(block.j_plus);   block.j_minus = copy.copy(block.j_minus)
    block.k_plus = copy.copy(block.k_plus);   block.k_minus = copy.copy(block.k_minus)

    # 2. Assign unique buffer indices to avoid (val - val) = 0
    base_idx = block.center.index
    block.i_plus.index  = base_idx + 1;  block.i_minus.index = base_idx - 1
    block.j_plus.index  = base_idx + 10; block.j_minus.index = base_idx - 10
    block.k_plus.index  = base_idx + 100; block.k_minus.index = base_idx - 100

    # 3. Assign spatial coordinates (i, j, k) for field setup
    block.center.i, block.center.j, block.center.k = 1, 1, 1
    block.i_plus.i,  block.i_plus.j,  block.i_plus.k  = 2, 1, 1
    block.i_minus.i, block.i_minus.j, block.i_minus.k = 0, 1, 1
    block.j_plus.i,  block.j_plus.j,  block.j_plus.k  = 1, 2, 1
    block.j_minus.i, block.j_minus.j, block.j_minus.k = 1, 0, 1
    block.k_plus.i,  block.k_plus.j,  block.k_plus.k  = 1, 1, 2
    block.k_minus.i, block.k_minus.j, block.k_minus.k = 1, 1, 0
    
    block.dx = block.dy = block.dz = 1.0
    return block

def get_all_cells(block):
    return [block.center, block.i_plus, block.i_minus, 
            block.j_plus, block.j_minus, block.k_plus, block.k_minus]

# --- Scenario 1: Null Field (The Zero-Gate) ---
def test_advection_zero_velocity():
    """Verify (0 ⋅ ∇)f = 0 even with a steep gradient."""
    block = setup_stencil_data(make_step3_output_dummy())
    for cell in get_all_cells(block):
        cell.set_field(FI.VX, 0.0); cell.set_field(FI.VY, 0.0); cell.set_field(FI.VZ, 0.0)
        cell.set_field(FI.P, float(cell.i + cell.j + cell.k))

    result = compute_local_advection(block, FI.P)
    assert result == 0.0, f"Expected 0.0, got {result}"

# --- Scenario 2: Uniform Velocity & Linear Gradient ---
def test_advection_linear_fidelity():
    """Verify (v ⋅ ∇)f for uniform velocity v=(1,1,1) and f=x+y+z."""
    block = setup_stencil_data(make_step3_output_dummy())
    for cell in get_all_cells(block):
        cell.set_field(FI.VX, 1.0); cell.set_field(FI.VY, 1.0); cell.set_field(FI.VZ, 1.0)
        cell.set_field(FI.P, float(cell.i + cell.j + cell.k))

    result = compute_local_advection(block, FI.P)
    # 1*(1) + 1*(1) + 1*(1) = 3.0
    assert abs(result - 3.0) < 1e-12

# --- Scenario 3: Staggered Velocity Alignment (Vector Test) ---
def test_advection_vector_component_isolation():
    """Ensure vector advection maps components to buffers correctly."""
    block = setup_stencil_data(make_step3_output_dummy())
    for cell in get_all_cells(block):
        cell.set_field(FI.VX, 2.0); cell.set_field(FI.VY, 0.0); cell.set_field(FI.VZ, 0.0)
        cell.set_field(FI.VX_STAR, float(cell.i)); cell.set_field(FI.VY_STAR, 0.0); cell.set_field(FI.VZ_STAR, 0.0)

    # Use the specific vector wrapper to test internal component loops
    adv_vec = compute_local_advection_vector(block)
    
    assert adv_vec[0] == 2.0, f"X-component failed. Got {adv_vec[0]}"
    assert adv_vec[1] == 0.0, f"Y-component leaked. Got {adv_vec[1]}"
    assert adv_vec[2] == 0.0, f"Z-component leaked. Got {adv_vec[2]}"

# --- Scenario 4: Edge Case - High Gradient Reversal ---
def test_advection_opposite_directions():
    """Verify v=(1,0,0) with f=-x results in negative advection."""
    block = setup_stencil_data(make_step3_output_dummy())
    for cell in get_all_cells(block):
        cell.set_field(FI.VX, 1.0); cell.set_field(FI.VY, 0.0); cell.set_field(FI.VZ, 0.0)
        cell.set_field(FI.P, -float(cell.i))

    result = compute_local_advection(block, FI.P)
    # 1 * (-1) = -1.0
    assert result == -1.0, f"Expected -1.0, got {result}"