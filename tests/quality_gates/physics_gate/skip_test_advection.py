# tests/quality_gates/physics_gate/test_advection.py

import copy

from src.common.field_schema import FI
from src.step3.ops.advection import (
    compute_local_advection,
    compute_local_advection_vector,
)
from tests.helpers.solver_step2_output_dummy import SimpleCellMock
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy


# --- RULE 9 BRIDGE: Monkeypatch missing interface into Mock ---
def get_field(self, field_idx):
    """Bypasses missing get_field in SimpleCellMock to satisfy physics operators."""
    return self.fields_buffer[self.index, field_idx]

SimpleCellMock.get_field = get_field


def setup_stencil_data(block):
    """
    Manually wires the stencil to ensure neighbors are independent 
    and mathematically distinct for finite difference testing.
    """
    def force_set(obj, attr, val):
        # Targets protected slots in ValidatedContainer
        object.__setattr__(obj, f"_{attr}", val)

    # 1. Clone cells and force-set them into StencilBlock slots
    force_set(block, 'center', copy.copy(block.center))
    force_set(block, 'i_plus', copy.copy(block.i_plus));   force_set(block, 'i_minus', copy.copy(block.i_minus))
    force_set(block, 'j_plus', copy.copy(block.j_plus));   force_set(block, 'j_minus', copy.copy(block.j_minus))
    force_set(block, 'k_plus', copy.copy(block.k_plus));   force_set(block, 'k_minus', copy.copy(block.k_minus))

    # 2. Assign unique buffer indices to each neighbor
    base = block.center.index
    block.i_plus.index, block.i_minus.index = base + 1, base - 1
    block.j_plus.index, block.j_minus.index = base + 10, base - 10
    block.k_plus.index, block.k_minus.index = base + 100, base - 100

    # 3. Setup Physics Constants
    force_set(block, 'dx', 1.0)
    force_set(block, 'dy', 1.0)
    force_set(block, 'dz', 1.0)
    
    return block


def set_linear_field(block, velocity_vec, scalar_func):
    """
    Applies analytical fields to the stencil.
    scalar_func is used to define the gradient of the field being advected.
    """
    layout = {
        block.center: (1, 1, 1),
        block.i_plus: (2, 1, 1), block.i_minus: (0, 1, 1),
        block.j_plus: (1, 2, 1), block.j_minus: (1, 0, 1),
        block.k_plus: (1, 1, 2), block.k_minus: (1, 1, 0)
    }
    
    for cell, (i, j, k) in layout.items():
        # Setup scalar field (for compute_local_advection tests)
        cell.set_field(FI.P, float(scalar_func(i, j, k)))
        
        # Setup vector fields (for compute_local_advection_vector tests)
        # We need to balance the 'driving' velocity with the 'gradient' velocity
        if cell == block.center:
            # The velocity at the center drives the advection: (u, v, w)
            cell.set_field(FI.VX, velocity_vec[0])
            cell.set_field(FI.VY, velocity_vec[1])
            cell.set_field(FI.VZ, velocity_vec[2])
        else:
            # Neighbors define the gradient of the field being advected.
            # Here we set VX/VY/VZ to the scalar_func values to test vector advection.
            val = float(scalar_func(i, j, k))
            cell.set_field(FI.VX, val)
            cell.set_field(FI.VY, 0.0)
            cell.set_field(FI.VZ, 0.0)


# --- Scenario 1: Null Field (The Zero-Gate) ---
def test_advection_zero_velocity():
    block = setup_stencil_data(make_step3_output_dummy())
    # v=(0,0,0), grad(f)=(1,1,1) -> Result = 0.0
    set_linear_field(block, (0.0, 0.0, 0.0), lambda i, j, k: i + j + k)
    
    result = compute_local_advection(block, FI.P)
    assert result == 0.0


# --- Scenario 2: Uniform Velocity & Linear Gradient ---
def test_advection_linear_fidelity():
    block = setup_stencil_data(make_step3_output_dummy())
    # v=(1,1,1), grad(f)=(1,1,1) -> Result = 1*1 + 1*1 + 1*1 = 3.0
    set_linear_field(block, (1.0, 1.0, 1.0), lambda i, j, k: i + j + k)
    
    result = compute_local_advection(block, FI.P)
    assert abs(result - 3.0) < 1e-12


# --- Scenario 3: Staggered Velocity Alignment (Vector Test) ---
def test_advection_vector_component_isolation():
    block = setup_stencil_data(make_step3_output_dummy())
    # Driving v=(2,0,0). Advected field VX has grad (1,0,0).
    # Result = u*(dVX/dx) = 2.0 * 1.0 = 2.0
    set_linear_field(block, (2.0, 0.0, 0.0), lambda i, j, k: i)
    
    adv_vec = compute_local_advection_vector(block)
    assert adv_vec[0] == 2.0
    assert adv_vec[1] == 0.0
    assert adv_vec[2] == 0.0


# --- Scenario 4: Edge Case - High Gradient Reversal ---
def test_advection_opposite_directions():
    block = setup_stencil_data(make_step3_output_dummy())
    # v=(1,0,0), grad(f)=(-1,0,0) -> Result = 1 * -1 = -1.0
    set_linear_field(block, (1.0, 0.0, 0.0), lambda i, j, k: -i)
    
    result = compute_local_advection(block, FI.P)
    assert result == -1.0