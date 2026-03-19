# tests/quality_gates/physics_gate/test_predictor.py

import pytest
from src.common.field_schema import FI
from src.step3.predictor import compute_local_predictor_step
# Import the sub-operators to diagnose the "leak"
from src.step3.ops.advection import compute_local_advection
from src.step3.ops.laplacian import compute_local_laplacian
from src.step3.ops.gradient import compute_local_gradient
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy

def setup_predictor_block(dt=1.0, rho=1.0, mu=1.0, dx=1.0):
    state = make_step2_output_dummy(nx=4, ny=4, nz=4)
    block = state.stencil_matrix[10]

    params = {
        '_dx': float(dx), '_dy': float(dx), '_dz': float(dx),
        '_dt': float(dt), '_rho': float(rho), '_mu': float(mu),
        '_f_vals': (0.0, 0.0, 10.0)
    }
    for attr, val in params.items():
        object.__setattr__(block, attr, val)

    block.center.fields_buffer.fill(0.0) 
    return block

def test_predictor_teamwork_diffusion_only():
    """
    Verifies that v* = v_n + (dt/rho) * (mu * lap(v_n))
    Expectation: With mu=1, dt=1, rho=1, and lap=2.0 -> v* = 0 + 2.0 = 2.0
    """
    block = setup_predictor_block(mu=1.0, dt=1.0, rho=1.0, dx=1.0)
    
    # Create a local curvature in VX: (ip: 1.0, c: 0.0, im: 1.0)
    # lap = (1 - 2*0 + 1) / 1^2 = 2.0
    block.i_plus.set_field(FI.VX, 1.0)
    block.i_minus.set_field(FI.VX, 1.0)
    
    compute_local_predictor_step(block)
    
    # Calculation: v* = 0 + (1/1) * (1.0 * 2.0) = 2.0
    assert block.center.get_field(FI.VX_STAR) == pytest.approx(2.0)

def test_predictor_teamwork_full_integration():
    """
    DIAGNOSTIC TEST: Decomposes the predictor to find the source of the -5.0.
    """
    block = setup_predictor_block(mu=1.0, dt=1.0, rho=1.0, dx=1.0)
    
    # Setup: v_n = 1.0, grad_u = 1.0, lap_u = 0.0, grad_p = 1.0
    block.center.set_field(FI.VX, 1.0)
    block.i_plus.set_field(FI.VX, 2.0)
    block.i_minus.set_field(FI.VX, 0.0)
    block.i_plus.set_field(FI.P, 2.0)
    block.i_minus.set_field(FI.P, 0.0)

    # --- DIAGNOSTIC ASSERTS ---
    
    # 1. Check Laplacian: (2 - 2*1 + 0) / 1^2 = 0.0
    lap = compute_local_laplacian(block, FI.VX)
    assert lap == pytest.approx(0.0), f"Laplacian failed! Expected 0.0, got {lap}"

    # 2. Check Advection: u * du/dx = 1.0 * (2-0)/(2*1.0) = 1.0
    # If this returns 4.0, advection is using dx=0.25
    adv = compute_local_advection(block, FI.VX)
    assert adv == pytest.approx(1.0), f"Advection failed! Expected 1.0, got {adv}"

    # 3. Check Gradient: dp/dx = (2-0)/(2*1.0) = 1.0
    # If this returns 4.0, gradient is using dx=0.25
    grad_p = compute_local_gradient(block, FI.P)[0] # [0] is the X-component
    assert grad_p == pytest.approx(1.0), f"Pressure Gradient failed! Expected 1.0, got {grad_p}"

    # --- FINAL EXECUTION ---
    compute_local_predictor_step(block)
    
    # v* = 1.0 + (1/1) * [ (1*0) - (1*1) + 0 - 1 ] = -1.0
    obtained = block.center.get_field(FI.VX_STAR)
    assert obtained == pytest.approx(-1.0), f"Final VX_STAR failed! Expected -1.0, got {obtained}"