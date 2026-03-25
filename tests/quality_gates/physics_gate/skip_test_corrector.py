# tests/quality_gates/physics_gate/test_corrector.py

import pytest

from src.common.field_schema import FI
from src.step3.corrector import apply_local_velocity_correction
from src.step3.ops.gradient import compute_local_gradient_p
from tests.helpers.solver_step2_output_dummy import SimpleCellMock
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy


# --- RULE 9 BRIDGE: Integration with real operators ---
def get_field(self, field_idx):
    """Provides O(1) buffer access for the corrector integration gate."""
    return self.fields_buffer[self.index, field_idx]

SimpleCellMock.get_field = get_field

def setup_integration_block(block, dt=1.0, rho=1.0):
    """
    Standardizes the block for analytical testing by forcing unit geometry.
    """
    # Use object.__setattr__ to bypass the Read-Only properties of StencilBlock
    object.__setattr__(block, '_dt', float(dt))
    object.__setattr__(block, '_rho', float(rho))
    
    # FORCE UNIT GEOMETRY (The fix for the 4.0 vs 1.0 error)
    object.__setattr__(block, '_dx', 1.0)
    object.__setattr__(block, '_dy', 1.0)
    object.__setattr__(block, '_dz', 1.0)
    
    # Ensure all cells are initialized to zero to prevent 'ghost' data interference
    for cell in [block.center, block.i_plus, block.i_minus, 
                 block.j_plus, block.j_minus, block.k_plus, block.k_minus]:
        cell.fields_buffer[cell.index, :] = 0.0
        
    return block

# --- Scenario 1: Zero Gradient (No Correction) ---
def test_corrector_zero_gradient_preservation():
    """If pressure is uniform, v_next should equal v_star exactly."""
    block = setup_integration_block(make_step3_output_dummy())
    
    block.center.set_field(FI.VX_STAR, 1.0)
    block.center.set_field(FI.VY_STAR, 0.5)
    block.center.set_field(FI.VZ_STAR, 0.2)
    
    # Uniform pressure: all rows in the shared buffer get 10.0 for P_NEXT
    for cell in [block.i_plus, block.i_minus, block.j_plus, block.j_minus, block.k_plus, block.k_minus]:
        cell.set_field(FI.P_NEXT, 10.0)
    
    apply_local_velocity_correction(block)
    
    assert block.center.get_field(FI.VX_STAR) == 1.0
    assert block.center.get_field(FI.VY_STAR) == 0.5
    assert block.center.get_field(FI.VZ_STAR) == 0.2

# --- Scenario 2: Standard Pressure Correction (Analytical Gate) ---
def test_corrector_analytical_correction():
    """
    Verifies: v = v* - (dt/rho) * grad(P)
    Expected v_x = 1.0 - (0.1/1.0)*1.0 = 0.9
    """
    block = setup_integration_block(make_step3_output_dummy(), dt=0.1, rho=1.0)
    
    # --- AUDIT 1: Foundation Check ---
    block.i_plus.set_field(FI.P_NEXT, 2.0)
    block.i_minus.set_field(FI.P_NEXT, 0.0)
    target_idx = block.i_plus.index
    assert block.center.fields_buffer[target_idx, FI.P_NEXT] == 2.0

    # --- AUDIT 2: Operator Check ---
    grad = compute_local_gradient_p(block, FI.P_NEXT)
    assert grad[0] == 1.0, f"OPERATOR BREAK: Grad_x is {grad[0]}, expected 1.0"

    block.center.set_field(FI.VX_STAR, 1.0)
    apply_local_velocity_correction(block)
    
    assert block.center.get_field(FI.VX_STAR) == pytest.approx(0.9, abs=1e-15)

# --- Scenario 3: High Density Inertia ---
def test_corrector_density_scaling():
    """Higher rho must result in a dampened velocity correction."""
    # rho=10.0, dt=1.0, grad_p=1.0 -> correction = 0.1
    block = setup_integration_block(make_step3_output_dummy(), dt=1.0, rho=10.0)
    
    block.center.set_field(FI.VX_STAR, 1.0)
    block.i_plus.set_field(FI.P_NEXT, 2.0)
    block.i_minus.set_field(FI.P_NEXT, 0.0)
    
    apply_local_velocity_correction(block)
    
    assert block.center.get_field(FI.VX_STAR) == pytest.approx(0.9, abs=1e-15)

# --- Scenario 4: Full 3D Vector Correction ---
def test_corrector_3d_alignment():
    """Checks that all 3 components are corrected via their respective gradients."""
    block = setup_integration_block(make_step3_output_dummy(), dt=1.0, rho=1.0)
    
    block.center.set_field(FI.VX_STAR, 0.0)
    block.center.set_field(FI.VY_STAR, 0.0)
    block.center.set_field(FI.VZ_STAR, 0.0)
    
    block.i_plus.set_field(FI.P_NEXT, 2.0); block.i_minus.set_field(FI.P_NEXT, 0.0)
    block.j_plus.set_field(FI.P_NEXT, 2.0); block.j_minus.set_field(FI.P_NEXT, 0.0)
    block.k_plus.set_field(FI.P_NEXT, 2.0); block.k_minus.set_field(FI.P_NEXT, 0.0)
    
    # --- AUDIT 3: 3D Operator Check ---
    grad = compute_local_gradient_p(block, FI.P_NEXT)
    assert grad == (1.0, 1.0, 1.0), f"3D OPERATOR BREAK: Grad is {grad}"
    
    apply_local_velocity_correction(block)
    
    assert block.center.get_field(FI.VX_STAR) == -1.0
    assert block.center.get_field(FI.VY_STAR) == -1.0
    assert block.center.get_field(FI.VZ_STAR) == -1.0