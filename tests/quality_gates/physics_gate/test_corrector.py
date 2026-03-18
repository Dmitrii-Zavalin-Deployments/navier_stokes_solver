# tests/quality_gates/physics_gate/test_corrector.py

import copy

import pytest

from src.common.field_schema import FI
from src.step3.corrector import apply_local_velocity_correction
from tests.helpers.solver_step2_output_dummy import SimpleCellMock
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy


# --- RULE 9 BRIDGE: Integration with real operators ---
def get_field(self, field_idx):
    return self.fields_buffer[self.index, field_idx]

SimpleCellMock.get_field = get_field

def setup_integration_block(block, dt=1.0, rho=1.0, dx=1.0):
    """
    Wires the stencil and injects physical constants for integration testing.
    """
    def force_set(obj, attr, val):
        object.__setattr__(obj, f"_{attr}", val)

    # 1. Clone cells for independence
    force_set(block, 'center', copy.copy(block.center))
    force_set(block, 'i_plus', copy.copy(block.i_plus));   force_set(block, 'i_minus', copy.copy(block.i_minus))
    force_set(block, 'j_plus', copy.copy(block.j_plus));   force_set(block, 'j_minus', copy.copy(block.j_minus))
    force_set(block, 'k_plus', copy.copy(block.k_plus));   force_set(block, 'k_minus', copy.copy(block.k_minus))

    # 2. Physics Constants (Rule 4 SSoT)
    force_set(block, 'dt', float(dt))
    force_set(block, 'rho', float(rho))
    force_set(block, 'dx', float(dx))
    force_set(block, 'dy', float(dx))
    force_set(block, 'dz', float(dx))
    
    return block

# --- Scenario 1: Zero Gradient (No Correction) ---
def test_corrector_zero_gradient_preservation():
    """If pressure is uniform, v_next should equal v_star."""
    block = setup_integration_block(make_step3_output_dummy())
    
    # Setup: v_star = (1.0, 0, 0), Pressure = constant (0.0)
    block.center.set_field(FI.VX_STAR, 1.0)
    for cell in [block.i_plus, block.i_minus, block.j_plus, block.j_minus, block.k_plus, block.k_minus]:
        cell.set_field(FI.P_NEXT, 0.0)
    
    apply_local_velocity_correction(block)
    
    # Result should be exactly v_star
    assert block.center.get_field(FI.VX) == 1.0
    assert block.center.get_field(FI.VY) == 0.0

# --- Scenario 2: Standard Pressure Correction ---
def test_corrector_analytical_correction():
    """
    Verifies: v = v* - (dt/rho) * grad(P)
    v_star = (1.0, 0, 0)
    dt=0.1, rho=1.0
    grad(P)_x = (P_ip - P_im) / (2*dx) = (2.0 - 0.0) / (2*1.0) = 1.0
    Expected v_x = 1.0 - (0.1/1.0)*1.0 = 0.9
    """
    block = setup_integration_block(make_step3_output_dummy(), dt=0.1, rho=1.0)
    
    # Set intermediate velocity
    block.center.set_field(FI.VX_STAR, 1.0)
    
    # Set pressure gradient in X
    block.i_plus.set_field(FI.P_NEXT, 2.0)
    block.i_minus.set_field(FI.P_NEXT, 0.0)
    # Zero gradient in Y and Z
    for c in [block.j_plus, block.j_minus, block.k_plus, block.k_minus]:
        c.set_field(FI.P_NEXT, 0.0)

    apply_local_velocity_correction(block)
    
    assert block.center.get_field(FI.VX) == pytest.approx(0.9, abs=1e-15)
    assert block.center.get_field(FI.VY) == 0.0
    assert block.center.get_field(FI.VZ) == 0.0

# --- Scenario 3: High Density Inertia ---
def test_corrector_density_scaling():
    """
    Higher density (rho) should result in a smaller velocity correction.
    grad(P)_x = 1.0, dt=1.0, rho=10.0 -> correction = 1.0/10.0 = 0.1
    v_x = 1.0 - 0.1 = 0.9
    """
    block = setup_integration_block(make_step3_output_dummy(), dt=1.0, rho=10.0)
    
    block.center.set_field(FI.VX_STAR, 1.0)
    block.i_plus.set_field(FI.P_NEXT, 2.0)
    block.i_minus.set_field(FI.P_NEXT, 0.0)
    
    apply_local_velocity_correction(block)
    
    assert block.center.get_field(FI.VX) == pytest.approx(0.9, abs=1e-15)

# --- Scenario 4: Full 3D Vector Correction ---
def test_corrector_3d_alignment():
    """Checks that all 3 components are corrected simultaneously."""
    block = setup_integration_block(make_step3_output_dummy(), dt=1.0, rho=1.0)
    
    block.center.set_field(FI.VX_STAR, 0.0)
    block.center.set_field(FI.VY_STAR, 0.0)
    block.center.set_field(FI.VZ_STAR, 0.0)
    
    # Setup gradients of 1.0 on all axes
    block.i_plus.set_field(FI.P_NEXT, 2.0); block.i_minus.set_field(FI.P_NEXT, 0.0)
    block.j_plus.set_field(FI.P_NEXT, 2.0); block.j_minus.set_field(FI.P_NEXT, 0.0)
    block.k_plus.set_field(FI.P_NEXT, 2.0); block.k_minus.set_field(FI.P_NEXT, 0.0)
    
    apply_local_velocity_correction(block)
    
    # Expected: 0 - (1.0/1.0)*1.0 = -1.0 for all components
    assert block.center.get_field(FI.VX) == -1.0
    assert block.center.get_field(FI.VY) == -1.0
    assert block.center.get_field(FI.VZ) == -1.0