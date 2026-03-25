# tests/quality_gates/physics_gate/test_divergence.py

import copy

from src.common.field_schema import FI
from src.step3.ops.divergence import compute_local_divergence_v_star
from tests.helpers.solver_step2_output_dummy import SimpleCellMock
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy


# --- RULE 9 BRIDGE: Alignment with existing Advection Test Pattern ---
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

    # 1. Clone cells to prevent cross-talk during field setting
    force_set(block, 'center', copy.copy(block.center))
    force_set(block, 'i_plus', copy.copy(block.i_plus));   force_set(block, 'i_minus', copy.copy(block.i_minus))
    force_set(block, 'j_plus', copy.copy(block.j_plus));   force_set(block, 'j_minus', copy.copy(block.j_minus))
    force_set(block, 'k_plus', copy.copy(block.k_plus));   force_set(block, 'k_minus', copy.copy(block.k_minus))

    # 2. Assign unique indices to map to separate buffer locations
    base = block.center.index
    block.i_plus.index, block.i_minus.index = base + 1, base - 1
    block.j_plus.index, block.j_minus.index = base + 10, base - 10
    block.k_plus.index, block.k_minus.index = base + 100, base - 100

    # 3. Explicit Spacing (Deterministic Initialization Mandate)
    force_set(block, 'dx', 1.0)
    force_set(block, 'dy', 1.0)
    force_set(block, 'dz', 1.0)
    
    return block

def set_divergent_field(block, u_func, v_func, w_func):
    """
    Applies analytical vector functions to the VX_STAR, VY_STAR, VZ_STAR fields.
    """
    layout = {
        block.i_plus: (2, 1, 1), block.i_minus: (0, 1, 1),
        block.j_plus: (1, 2, 1), block.j_minus: (1, 0, 1),
        block.k_plus: (1, 1, 2), block.k_minus: (1, 1, 0)
    }
    
    for cell, (i, j, k) in layout.items():
        cell.set_field(FI.VX_STAR, float(u_func(i, j, k)))
        cell.set_field(FI.VY_STAR, float(v_func(i, j, k)))
        cell.set_field(FI.VZ_STAR, float(w_func(i, j, k)))

# --- Scenario 1: Null Field (The Zero-Gate) ---
def test_divergence_zero_field():
    block = setup_stencil_data(make_step3_output_dummy())
    # v=(0,0,0) everywhere -> div(v) = 0
    set_divergent_field(block, lambda i,j,k: 0, lambda i,j,k: 0, lambda i,j,k: 0)
    
    result = compute_local_divergence_v_star(block)
    assert result == 0.0

# --- Scenario 2: Unit Divergence (Linear Expansion) ---
def test_divergence_linear_expansion():
    block = setup_stencil_data(make_step3_output_dummy())
    # u=x, v=y, w=z -> du/dx=1, dv/dy=1, dw/dz=1 -> div(v)=3.0
    set_divergent_field(block, lambda i,j,k: i, lambda i,j,k: j, lambda i,j,k: k)
    
    result = compute_local_divergence_v_star(block)
    assert abs(result - 3.0) < 1e-12

# --- Scenario 3: Solenoidal Field (Rotational, Zero Divergence) ---
def test_divergence_solenoidal_flow():
    block = setup_stencil_data(make_step3_output_dummy())
    # u=y, v=-x, w=0 -> du/dx=0, dv/dy=0, dw/dz=0 -> div(v)=0
    set_divergent_field(block, lambda i,j,k: j, lambda i,j,k: -i, lambda i,j,k: 0)
    
    result = compute_local_divergence_v_star(block)
    assert abs(result) < 1e-12

# --- Scenario 4: Asymmetric Gradient ---
def test_divergence_asymmetric_flow():
    block = setup_stencil_data(make_step3_output_dummy())
    # u=2x, v=0, w=0 -> du/dx=2, dv/dy=0, dw/dz=0 -> div(v)=2.0
    set_divergent_field(block, lambda i,j,k: 2*i, lambda i,j,k: 0, lambda i,j,k: 0)
    
    result = compute_local_divergence_v_star(block)
    assert result == 2.0