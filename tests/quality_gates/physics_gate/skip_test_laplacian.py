# tests/quality_gates/physics_gate/test_laplacian.py

import copy

import pytest

from src.common.field_schema import FI
from src.step3.ops.laplacian import (
    compute_local_laplacian,
    compute_local_laplacian_p_next,
    compute_local_laplacian_v_n,
)
from tests.helpers.solver_step2_output_dummy import SimpleCellMock
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy


# --- RULE 9 BRIDGE: Alignment with existing Physics Gate Pattern ---
def get_field(self, field_idx):
    """Bypasses missing get_field in SimpleCellMock to satisfy physics operators."""
    return self.fields_buffer[self.index, field_idx]

SimpleCellMock.get_field = get_field

def setup_stencil_data(block):
    """
    Wires the stencil and enforces Rule 0 (Performance) topology.
    Targets protected slots to ensure mathematical independence.
    """
    def force_set(obj, attr, val):
        object.__setattr__(obj, f"_{attr}", val)

    # 1. Clone cells to ensure neighbors are independent objects
    force_set(block, 'center', copy.copy(block.center))
    force_set(block, 'i_plus', copy.copy(block.i_plus));   force_set(block, 'i_minus', copy.copy(block.i_minus))
    force_set(block, 'j_plus', copy.copy(block.j_plus));   force_set(block, 'j_minus', copy.copy(block.j_minus))
    force_set(block, 'k_plus', copy.copy(block.k_plus));   force_set(block, 'k_minus', copy.copy(block.k_minus))

    # 2. Assign unique indices to separate memory locations
    base = block.center.index
    block.i_plus.index, block.i_minus.index = base + 1, base - 1
    block.j_plus.index, block.j_minus.index = base + 10, base - 10
    block.k_plus.index, block.k_minus.index = base + 100, base - 100

    # 3. Explicit Spacing (Rule 5: Zero-Default Policy)
    force_set(block, 'dx', 1.0)
    force_set(block, 'dy', 1.0)
    force_set(block, 'dz', 1.0)
    
    return block

def set_field_values(block, field_id, f_func):
    """Applies an analytical scalar function to a specific Field ID."""
    layout = {
        block.center: (1, 1, 1),
        block.i_plus: (2, 1, 1), block.i_minus: (0, 1, 1),
        block.j_plus: (1, 2, 1), block.j_minus: (1, 0, 1),
        block.k_plus: (1, 1, 2), block.k_minus: (1, 1, 0)
    }
    for cell, (i, j, k) in layout.items():
        cell.set_field(field_id, float(f_func(i, j, k)))

# --- Scenario 1: Harmonic Field (Laplace's Equation: ∇²f = 0) ---
def test_laplacian_harmonic_field():
    block = setup_stencil_data(make_step3_output_dummy())
    # f = x + y + z (Linear fields have zero second derivatives)
    set_field_values(block, FI.P, lambda i, j, k: i + j + k)
    
    result = compute_local_laplacian(block, FI.P)
    
    # ∇²(x+y+z) = 0 + 0 + 0 = 0
    assert result == 0.0

# --- Scenario 2: Quadratic Field (Constant Laplacian) ---
def test_laplacian_quadratic_field():
    block = setup_stencil_data(make_step3_output_dummy())
    # f = x² + y² + z² -> ∂²f/∂x² = 2, ∂²f/∂y² = 2, ∂²f/∂z² = 2 -> ∇²f = 6.0
    set_field_values(block, FI.P, lambda i, j, k: i**2 + j**2 + k**2)
    
    result = compute_local_laplacian(block, FI.P)
    
    # Using dx=1.0: (4 - 2(1) + 0) / 1.0² = 2.0 (for each axis)
    assert result == pytest.approx(6.0, abs=1e-15)

# --- Scenario 3: Velocity Vector Laplacian (Rule 8: Unified Implementation) ---
def test_laplacian_velocity_vector():
    block = setup_stencil_data(make_step3_output_dummy())
    # u = x², v = 0, w = 0 -> ∇²u = 2, ∇²v = 0, ∇²w = 0
    set_field_values(block, FI.VX, lambda i, j, k: i**2)
    set_field_values(block, FI.VY, lambda i, j, k: 0)
    set_field_values(block, FI.VZ, lambda i, j, k: 0)
    
    lap_u, lap_v, lap_w = compute_local_laplacian_v_n(block)
    
    assert lap_u == pytest.approx(2.0, abs=1e-15)
    assert lap_v == 0.0
    assert lap_w == 0.0

# --- Scenario 4: Pressure Next Gate (Rule 4: SSoT) ---
def test_laplacian_p_next_gate():
    block = setup_stencil_data(make_step3_output_dummy())
    # Verify that the wrapper correctly targets the P_NEXT buffer
    set_field_values(block, FI.P_NEXT, lambda i, j, k: j**2)
    
    result = compute_local_laplacian_p_next(block)
    
    # ∂²(j²)/∂y² = 2
    assert result == pytest.approx(2.0, abs=1e-15)

# --- Scenario 5: Stretched Grid Scaling ---
def test_laplacian_stretched_grid():
    block = setup_stencil_data(make_step3_output_dummy())
    # Force dx = 2.0. f = x² -> ∇²f = (4 - 2(0) + 0) / 2² = 1.0 (if center is at x=0)
    # Using our layout: center=1, i_plus=2, i_minus=0. f = i²
    # (4 - 2(1) + 0) / (2.0²) = 2 / 4 = 0.5
    object.__setattr__(block, '_dx', 2.0)
    set_field_values(block, FI.P, lambda i, j, k: i**2)
    
    result = compute_local_laplacian(block, FI.P)
    
    assert result == pytest.approx(0.5, abs=1e-15)