# tests/quality_gates/physics_gate/test_predictor.py

import pytest

from src.common.field_schema import FI
from src.step3.predictor import compute_local_predictor_step
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy


def setup_predictor_block(dt=1.0, rho=1.0, mu=1.0, dx=1.0):
    """
    Standardizes a StencilBlock for Predictor teamwork testing.
    """
    state = make_step2_output_dummy(nx=4, ny=4, nz=4)
    block = state.stencil_matrix[10] # Central block

    # Bypass read-only slots to force Unit Physics
    object.__setattr__(block, '_dt', float(dt))
    object.__setattr__(block, '_rho', float(rho))
    object.__setattr__(block, '_mu', float(mu))
    object.__setattr__(block, '_dx', float(dx))
    object.__setattr__(block, '_dy', float(dx))
    object.__setattr__(block, '_dz', float(dx))
    
    # Also ensure gravity is set in f_vals (usually a tuple in block)
    object.__setattr__(block, '_f_vals', (0.0, 0.0, 10.0)) # Gravity in Z

    # Clean the buffer
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
    
    # Expected v* = 0 + (1/1) * (1.0 * 2.0 - 0 + 0 - 0) = 2.0
    assert block.center.get_field(FI.VX_STAR) == pytest.approx(2.0)

def test_predictor_teamwork_full_integration():
    """
    A 'Kitchen Sink' test where every operator contributes.
    v* = v_n + (dt/rho) * [ (mu * lap) - (rho * adv) + F - grad_p ]
    """
    block = setup_predictor_block(mu=1.0, dt=1.0, rho=1.0, dx=1.0)
    
    # 1. Current Velocity (v_n) = 1.0
    block.center.set_field(FI.VX, 1.0)
    
    # 2. Laplacian (Diffusion) = 2.0
    block.i_plus.set_field(FI.VX, 2.0)  # Curvature: (2 - 2*1 + 2) = 2.0
    block.i_minus.set_field(FI.VX, 2.0)
    
    # 3. Advection = 1.0
    # adv_x = u * du/dx. If u=1 and du/dx = (2-2)/2 = 0... 
    # Let's set a gradient: ip=2.0, im=0.0 -> du/dx = (2-0)/2 = 1.0
    # adv = 1.0 * 1.0 = 1.0
    block.i_plus.set_field(FI.VX, 2.0)
    block.i_minus.set_field(FI.VX, 0.0)
    # Re-calc Laplacian with these: (2 - 2*1 + 0) = 0.0
    
    # 4. Body Force (F) = 10.0 (from setup f_vals in Z)
    # 5. Pressure Gradient (grad_p) = 1.0
    block.i_plus.set_field(FI.P, 2.0)
    block.i_minus.set_field(FI.P, 0.0) # grad = (2-0)/2 = 1.0
    
    compute_local_predictor_step(block)
    
    # Let's calculate VX_STAR expectation:
    # v_n = 1.0
    # mu * lap = 1.0 * (2 - 2*1 + 0) = 0.0
    # rho * adv = 1.0 * (1.0 * (2-0)/2) = 1.0
    # F = 0.0 (force is in Z)
    # grad_p = 1.0
    # VX_STAR = 1.0 + (1/1) * [ 0.0 - 1.0 + 0.0 - 1.0 ] = -1.0
    
    assert block.center.get_field(FI.VX_STAR) == pytest.approx(-1.0)

    # Let's check VZ_STAR for Force contribution:
    # v_n = 0, lap=0, adv=0, grad_p=0, F=10.0
    # VZ_STAR = 0 + (1/1) * [ 10.0 ] = 10.0
    assert block.center.get_field(FI.VZ_STAR) == pytest.approx(10.0)