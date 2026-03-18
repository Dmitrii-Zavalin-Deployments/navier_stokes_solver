# tests/quality_gates/physics_gate/test_corrector.py

import pytest
from src.common.field_schema import FI
from src.step3.corrector import apply_local_velocity_correction
from tests.helpers.solver_step2_output_dummy import SimpleCellMock
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy

# --- RULE 9 BRIDGE: Integration with real operators ---
def get_field(self, field_idx):
    """Provides O(1) buffer access for the corrector integration gate."""
    return self.fields_buffer[self.index, field_idx]

SimpleCellMock.get_field = get_field

def setup_integration_block(block, dt=1.0, rho=1.0, dx=1.0):
    """
    Directly injects physics into the existing StencilBlock and fixes index wiring.
    
    Compliance:
    - Rule 4 (SSoT): Overwrites protected slots to simulate config injection.
    - Rule 9 (Foundation): Manually maps indices to ensure spatial delta exists.
    """
    def force_set(obj, attr, val):
        object.__setattr__(obj, f"_{attr}", float(val))

    # 1. Physics Constants
    force_set(block, 'dt', dt)
    force_set(block, 'rho', rho)
    force_set(block, 'dx', dx)
    force_set(block, 'dy', dx)
    force_set(block, 'dz', dx)
    
    # 2. REWIRE INDICES
    # The dummy defaults all indices to 0, which collapses the gradient to zero.
    # We assign unique indices so each cell looks at a different row in the buffer.
    block.center.index = 10
    block.i_minus.index = 9;  block.i_plus.index = 11
    block.j_minus.index = 8;  block.j_plus.index = 12
    block.k_minus.index = 7;  block.k_plus.index = 13
    
    # 3. Memory Sanitization
    # Wipe the '0.51' values from the dummy for clean testing.
    block.center.fields_buffer.fill(0.0)
    
    return block

# --- Scenario 1: Zero Gradient (No Correction) ---
def test_corrector_zero_gradient_preservation():
    """If pressure is uniform, v_next should equal v_star exactly."""
    block = setup_integration_block(make_step3_output_dummy())
    
    # Setup: v_star = (1.0, 0.5, 0.2)
    block.center.set_field(FI.VX_STAR, 1.0)
    block.center.set_field(FI.VY_STAR, 0.5)
    block.center.set_field(FI.VZ_STAR, 0.2)
    
    # Uniform pressure means grad(P) = 0
    for cell in [block.i_plus, block.i_minus, block.j_plus, block.j_minus, block.k_plus, block.k_minus]:
        cell.set_field(FI.P_NEXT, 10.0)
    
    apply_local_velocity_correction(block)
    
    assert block.center.get_field(FI.VX) == 1.0
    assert block.center.get_field(FI.VY) == 0.5
    assert block.center.get_field(FI.VZ) == 0.2

# --- Scenario 2: Standard Pressure Correction (Analytical Gate) ---
def test_corrector_analytical_correction():
    """
    Verifies: v = v* - (dt/rho) * grad(P)
    v_star = (1.0, 0, 0) | dt=0.1, rho=1.0
    grad(P)_x = (P_ip - P_im) / (2*dx) = (2.0 - 0.0) / (2*1.0) = 1.0
    Expected v_x = 1.0 - (0.1/1.0)*1.0 = 0.9
    """
    block = setup_integration_block(make_step3_output_dummy(), dt=0.1, rho=1.0)
    
    block.center.set_field(FI.VX_STAR, 1.0)
    
    # Set pressure values at unique indices 11 and 9
    block.i_plus.set_field(FI.P_NEXT, 2.0)
    block.i_minus.set_field(FI.P_NEXT, 0.0)

    apply_local_velocity_correction(block)
    
    assert block.center.get_field(FI.VX) == pytest.approx(0.9, abs=1e-15)
    assert block.center.get_field(FI.VY) == 0.0

# --- Scenario 3: High Density Inertia ---
def test_corrector_density_scaling():
    """Higher rho must result in a dampened velocity correction."""
    # rho=10.0, dt=1.0, grad_p=1.0 -> correction = 0.1
    # v_x = 1.0 - 0.1 = 0.9
    block = setup_integration_block(make_step3_output_dummy(), dt=1.0, rho=10.0)
    
    block.center.set_field(FI.VX_STAR, 1.0)
    block.i_plus.set_field(FI.P_NEXT, 2.0)
    block.i_minus.set_field(FI.P_NEXT, 0.0)
    
    apply_local_velocity_correction(block)
    
    assert block.center.get_field(FI.VX) == pytest.approx(0.9, abs=1e-15)

# --- Scenario 4: Full 3D Vector Correction ---
def test_corrector_3d_alignment():
    """Checks that all 3 components are corrected via their respective gradients."""
    block = setup_integration_block(make_step3_output_dummy(), dt=1.0, rho=1.0)
    
    # Start at rest
    block.center.set_field(FI.VX_STAR, 0.0)
    block.center.set_field(FI.VY_STAR, 0.0)
    block.center.set_field(FI.VZ_STAR, 0.0)
    
    # Setup unit gradients (1.0) on all axes by using the rewired indices
    block.i_plus.set_field(FI.P_NEXT, 2.0); block.i_minus.set_field(FI.P_NEXT, 0.0)
    block.j_plus.set_field(FI.P_NEXT, 2.0); block.j_minus.set_field(FI.P_NEXT, 0.0)
    block.k_plus.set_field(FI.P_NEXT, 2.0); block.k_minus.set_field(FI.P_NEXT, 0.0)
    
    apply_local_velocity_correction(block)
    
    # v = 0 - (1/1)*1 = -1.0
    assert block.center.get_field(FI.VX) == -1.0
    assert block.center.get_field(FI.VY) == -1.0
    assert block.center.get_field(FI.VZ) == -1.0