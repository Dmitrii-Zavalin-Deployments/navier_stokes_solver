# tests/quality_gates/physics_gate/test_corrector.py

import numpy as np
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

def setup_integration_block(block, dt=1.0, rho=1.0, dx=1.0):
    """
    Directly injects physics and UNIFIES buffer memory across the stencil.
    """
    def force_set(obj, attr, val):
        object.__setattr__(obj, f"_{attr}", float(val))

    # 1. Create a single, shared Foundation Buffer (Rule 9 Compliance)
    shared_buffer = np.zeros((20, FI.num_fields()))

    # 2. Inject Physics Constants
    force_set(block, 'dt', dt)
    force_set(block, 'rho', rho)
    force_set(block, 'dx', dx)
    force_set(block, 'dy', dx)
    force_set(block, 'dz', dx)
    
    # 3. REWIRE INDICES & UNIFY MEMORY
    stencil_mapping = [
        (block.center, 10),
        (block.i_minus, 9), (block.i_plus, 11),
        (block.j_minus, 8), (block.j_plus, 12),
        (block.k_minus, 7), (block.k_plus, 13)
    ]
    
    for cell, idx in stencil_mapping:
        cell.index = idx
        cell.fields_buffer = shared_buffer
    
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
    
    assert block.center.get_field(FI.VX) == 1.0
    assert block.center.get_field(FI.VY) == 0.5
    assert block.center.get_field(FI.VZ) == 0.2

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
    assert block.center.fields_buffer[11, FI.P_NEXT] == 2.0, "FOUNDATION BREAK: Shared buffer write failed"

    # --- AUDIT 2: Operator Check ---
    grad = compute_local_gradient_p(block, FI.P_NEXT)
    assert grad[0] == 1.0, f"OPERATOR BREAK: Grad_x is {grad[0]}, expected 1.0"

    block.center.set_field(FI.VX_STAR, 1.0)
    apply_local_velocity_correction(block)
    
    assert block.center.get_field(FI.VX) == pytest.approx(0.9, abs=1e-15)

# --- Scenario 3: High Density Inertia ---
def test_corrector_density_scaling():
    """Higher rho must result in a dampened velocity correction."""
    # rho=10.0, dt=1.0, grad_p=1.0 -> correction = 0.1
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
    
    assert block.center.get_field(FI.VX) == -1.0
    assert block.center.get_field(FI.VY) == -1.0
    assert block.center.get_field(FI.VZ) == -1.0