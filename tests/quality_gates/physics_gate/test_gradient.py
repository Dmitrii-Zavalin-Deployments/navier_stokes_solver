# tests/quality_gates/physics_gate/test_gradient.py

import pytest
import copy
from src.common.field_schema import FI
from src.step3.ops.gradient import compute_local_gradient_p
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

def set_pressure_field(block, p_func):
    """
    Applies an analytical scalar function to the Pressure (FI.P) field.
    """
    layout = {
        block.i_plus: (2, 1, 1), block.i_minus: (0, 1, 1),
        block.j_plus: (1, 2, 1), block.j_minus: (1, 0, 1),
        block.k_plus: (1, 1, 2), block.k_minus: (1, 1, 0)
    }
    
    for cell, (i, j, k) in layout.items():
        cell.set_field(FI.P, float(p_func(i, j, k)))

# --- Scenario 1: Hydrostatic Equilibrium (Zero Gradient) ---
def test_gradient_null_field():
    block = setup_stencil_data(make_step3_output_dummy())
    # p = 10.0 everywhere -> grad(p) = (0,0,0)
    set_pressure_field(block, lambda i, j, k: 10.0)
    
    grad_x, grad_y, grad_z = compute_local_gradient_p(block)
    
    assert (grad_x, grad_y, grad_z) == (0.0, 0.0, 0.0)

# --- Scenario 2: Unit Linear Gradient ---
def test_gradient_linear_ascent():
    block = setup_stencil_data(make_step3_output_dummy())
    # p = x + y + z -> dp/dx=1, dp/dy=1, dp/dz=1
    set_pressure_field(block, lambda i, j, k: i + j + k)
    
    grad_x, grad_y, grad_z = compute_local_gradient_p(block)
    
    # (2 - 0) / (2 * 1.0) = 1.0
    assert grad_x == pytest.approx(1.0, abs=1e-15)
    assert grad_y == pytest.approx(1.0, abs=1e-15)
    assert grad_z == pytest.approx(1.0, abs=1e-15)

# --- Scenario 3: Gradient Reversal (Descent) ---
def test_gradient_linear_descent():
    block = setup_stencil_data(make_step3_output_dummy())
    # p = -2x -> dp/dx=-2.0, dp/dy=0, dp/dz=0
    set_pressure_field(block, lambda i, j, k: -2.0 * i)
    
    grad_x, grad_y, grad_z = compute_local_gradient_p(block)
    
    # (-4 - 0) / (2 * 1.0) = -2.0
    assert grad_x == pytest.approx(-2.0, abs=1e-15)
    assert grad_y == 0.0
    assert grad_z == 0.0

# --- Scenario 4: Alternative Field Mapping (Rule 9 Check) ---
def test_gradient_alt_field_id():
    """Verifies the operator can target P_NEXT if requested."""
    block = setup_stencil_data(make_step3_output_dummy())
    # Set P_NEXT to a gradient while keeping FI.P at zero
    block.i_plus.set_field(FI.P_NEXT, 5.0)
    block.i_minus.set_field(FI.P_NEXT, 1.0)
    
    # grad_x = (5 - 1) / 2 = 2.0
    grad_x, _, _ = compute_local_gradient_p(block, field_id=FI.P_NEXT)
    
    assert grad_x == 2.0